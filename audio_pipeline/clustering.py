"""
clustering.py — Agrupamento de músicas por similaridade sonora usando embeddings do ChromaDB.

Funcionalidades:
- Carregamento de embeddings agregados e metadados direto do ChromaDB.
- Normalização L2 e redução opcional de dimensionalidade com PCA (salvo para inferência incremental).
- Agrupamento via K-Means com seleção automática do melhor k (via Coeficiente de Silhueta) ou HDBSCAN.
- Persistência segura no ChromaDB (com backup prévio e validação estrita de metadados).
- Geração de clusters.json com estatísticas e medoides de cada cluster.
- Visualização 2D (UMAP ou t-SNE) e gráficos de avaliação para o TCC.
- Exportação de playlists automáticas para o MPD e arquivos .m3u.
- Atribuição incremental de novas faixas ao centroide mais próximo.
"""

import os
import sys
import re
import json
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# Garante path de importação local
_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

import config
import vector_store
from evaluation import (
    evaluate_clustering_metrics,
    evaluate_clustering_validation_metrics,
    evaluate_k_range,
    plot_k_selection,
    save_clustering_metrics_csv
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("clustering")

DEFAULT_RESULTS_DIR = _SCRIPT_DIR / "results"
DEFAULT_MODELS_DIR = _SCRIPT_DIR / "models"
MODEL_FILE_NAME = "cluster_model.joblib"


# ── CARREGAMENTO DE DADOS DO CHROMADB ─────────────────────────────────────────

def load_embeddings_from_chromadb(collection, batch_size: int = 500) -> Tuple[List[str], np.ndarray, List[Dict[str, Any]]]:
    """
    Carrega todos os IDs, embeddings (512d) e metadados da coleção do ChromaDB
    usando paginação segura.
    """
    total_count = collection.count()
    logger.info(f"Carregando {total_count} embeddings do ChromaDB...")

    all_ids: List[str] = []
    all_embeddings: List[List[float]] = []
    all_metadatas: List[Dict[str, Any]] = []

    offset = 0
    while offset < total_count:
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "metadatas"]
        )
        batch_ids = batch.get("ids", [])
        if not batch_ids:
            break

        all_ids.extend(batch_ids)
        all_embeddings.extend(batch.get("embeddings", []))
        all_metadatas.extend(batch.get("metadatas", []))

        offset += len(batch_ids)

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Embeddings carregados com sucesso: {embeddings_array.shape[0]} faixas, {embeddings_array.shape[1]} dimensões.")
    return all_ids, embeddings_array, all_metadatas


# ── PRÉ-PROCESSAMENTO (L2 + PCA OPCIONAL) ──────────────────────────────────────

def preprocess_embeddings(
    embeddings: np.ndarray,
    n_pca: Optional[int] = None,
    pca_model: Optional[PCA] = None,
    seed: int = 42
) -> Tuple[np.ndarray, Optional[PCA]]:
    """
    Aplica normalização L2 e, opcionalmente, transformação PCA.
    Se pca_model já for fornecido, aplica a transformação já ajustada (para faixas incrementais).
    """
    norm_embeddings = normalize(embeddings, norm="l2")

    if n_pca is not None and n_pca > 0:
        max_components = min(norm_embeddings.shape[0], norm_embeddings.shape[1])
        actual_pca = min(n_pca, max_components)

        if pca_model is None:
            logger.info(f"Ajustando PCA com {actual_pca} componentes (seed={seed})...")
            pca_model = PCA(n_components=actual_pca, random_state=seed)
            transformed = pca_model.fit_transform(norm_embeddings)
            var_explained = np.sum(pca_model.explained_variance_ratio_) * 100
            logger.info(f"PCA concluído. Variância explicada acumulada: {var_explained:.2f}%")
        else:
            transformed = pca_model.transform(norm_embeddings)

        return transformed, pca_model

    return norm_embeddings, None


# ── ALGORITMOS DE CLUSTERIZAÇÃO ───────────────────────────────────────────────

def run_kmeans(
    embeddings: np.ndarray,
    k: int,
    seed: int = 42
) -> Tuple[KMeans, np.ndarray, List[str]]:
    """Executa o K-Means e formata os nomes dos clusters (cluster_01, cluster_02, ...)."""
    logger.info(f"Executando K-Means com k={k} (seed={seed})...")
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    cluster_names = [f"cluster_{label + 1:02d}" for label in labels]
    return kmeans, labels, cluster_names


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None
) -> Tuple[Any, np.ndarray, List[str]]:
    """
    Executa o HDBSCAN.
    Faixas de ruído (rótulo -1) recebem o nome 'ruido'.
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        logger.error("HDBSCAN não disponível no scikit-learn. Instale ou atualize scikit-learn >= 1.3.")
        raise

    logger.info(f"Executando HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    labels = hdb.fit_predict(embeddings)

    cluster_names = []
    for label in labels:
        if label == -1:
            cluster_names.append("ruido")
        else:
            cluster_names.append(f"cluster_{label + 1:02d}")

    return hdb, labels, cluster_names


def sanitize_playlist_name(name: str) -> str:
    """
    Sanitiza o nome do cluster para torná-lo um nome de arquivo e playlist válido no Windows.
    Substitui caracteres proibidos (barra, dois pontos, asterisco, interrogação, aspas, menor, maior, pipe) por hífen.
    """
    sanitized = re.sub(r'[\/\\:*?"<>|]', ' - ', name)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip(' .')
    return sanitized if sanitized else "Cluster"


def determine_cluster_names_and_purity(
    ids: List[str],
    metadatas: List[Dict[str, Any]],
    raw_cluster_ids: List[str]
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Determina o nome de cada cluster a partir do subgênero predominante (voto majoritário)
    das faixas do cluster e calcula a pureza e distribuição de gêneros.

    Regras:
      a) Se dois clusters tiverem o mesmo gênero predominante, diferenciar pelo segundo
         gênero mais frequente (ex: 'Rock / Blues') ou por sufixo numérico ('Rock 1', 'Rock 2').
      b) Se nenhum gênero passar de ~50% do cluster, nomear com os dois mais frequentes (ex.: 'Pop / Eletrônica').
      c) Faixas sem subgênero não entram na votação; se o cluster inteiro não tiver gênero, usar 'Sem gênero N'.
    """
    unique_ids = sorted(list(set(raw_cluster_ids)), key=lambda x: (x == "ruido", x))
    cluster_stats: Dict[str, Dict[str, Any]] = {}

    # 1. Contagem de gêneros por cluster
    for cid in unique_ids:
        if cid == "ruido":
            cluster_stats[cid] = {
                "nome_provisorio": "Ruído",
                "genero_predominante": "Ruído",
                "pureza": 0.0,
                "distribuicao": {},
                "g1": "Ruído",
                "g2": None,
                "pct1": 0.0
            }
            continue

        generos = [
            str(m.get("subgenero", "")).strip()
            for m, c in zip(metadatas, raw_cluster_ids)
            if c == cid
        ]
        generos_validos = [g for g in generos if g and g.lower() not in ["", "desconhecido", "none"]]

        if not generos_validos:
            cluster_stats[cid] = {
                "nome_provisorio": "Sem gênero",
                "genero_predominante": "Sem gênero",
                "pureza": 0.0,
                "distribuicao": {},
                "g1": None,
                "g2": None,
                "pct1": 0.0
            }
        else:
            counter = Counter(generos_validos)
            total = len(generos_validos)
            most_common = counter.most_common()
            g1, count1 = most_common[0]
            pct1 = count1 / total
            g2 = most_common[1][0] if len(most_common) > 1 else None

            # Regra b: se nenhum gênero passar de 50%, usar os dois mais frequentes
            if pct1 >= 0.50 or g2 is None:
                nome_base = g1
            else:
                nome_base = f"{g1} / {g2}"

            cluster_stats[cid] = {
                "nome_provisorio": nome_base,
                "genero_predominante": g1,
                "pureza": float(round(pct1, 4)),
                "distribuicao": {g: c for g, c in most_common},
                "g1": g1,
                "g2": g2,
                "pct1": pct1
            }

    # 2. Resolução de conflitos de nomes (Regras a e c)
    by_name = defaultdict(list)
    for cid, st in cluster_stats.items():
        if cid == "ruido":
            continue
        by_name[st["nome_provisorio"]].append(cid)

    final_names: Dict[str, str] = {}
    if "ruido" in cluster_stats:
        final_names["ruido"] = "Ruído"

    for name_base, cids in by_name.items():
        if len(cids) == 1:
            final_names[cids[0]] = name_base
        else:
            # Caso 1: tentar diferenciar pelo segundo gênero mais frequente (g2)
            tentativa_g2 = {}
            for cid in cids:
                st = cluster_stats[cid]
                if st["g1"] and st["g2"] and "/" not in name_base:
                    tentativa_g2[cid] = f"{st['g1']} / {st['g2']}"
                else:
                    tentativa_g2[cid] = name_base

            # Verifica se os nomes ficaram únicos com g2
            if len(set(tentativa_g2.values())) == len(cids):
                for cid in cids:
                    final_names[cid] = tentativa_g2[cid]
            else:
                # Caso 2: sufixo numérico (ex: "Rock 1", "Rock 2" ou "Sem gênero 1", "Sem gênero 2")
                for idx, cid in enumerate(cids, 1):
                    final_names[cid] = f"{name_base} {idx}"

    for cid in unique_ids:
        cluster_stats[cid]["nome"] = final_names[cid]

    return final_names, cluster_stats


# ── MEDÓIDES E RESUMO DE CLUSTERS ─────────────────────────────────────────────

def compute_clusters_summary(
    ids: List[str],
    metadatas: List[Dict[str, Any]],
    processed_embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_ids: List[str],
    cluster_names_map: Dict[str, str],
    cluster_stats: Dict[str, Dict[str, Any]],
    algorithm: str,
    centroids: Optional[np.ndarray] = None,
    validation_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calcula o resumo dos clusters, incluindo tamanho, lista de faixas, medóide,
    gênero predominante, pureza e distribuição de subgêneros.
    """
    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "algorithm": algorithm,
        "total_tracks": len(ids),
        "validation_metrics": validation_metrics or {},
        "clusters": {}
    }

    unique_cids = sorted(list(set(cluster_ids)), key=lambda x: (x == "ruido", x))

    total_pureza_ponderada = 0.0
    total_amostras_validas = 0

    for cid in unique_cids:
        indices = [i for i, c in enumerate(cluster_ids) if c == cid]
        cluster_embs = processed_embeddings[indices]
        cluster_track_ids = [ids[i] for i in indices]
        cluster_metas = [metadatas[i] for i in indices]

        stats = cluster_stats.get(cid, {})
        c_nome = cluster_names_map.get(cid, cid)

        medoid_info = None
        if cid != "ruido" and len(indices) > 0:
            if algorithm == "kmeans" and centroids is not None:
                c_idx = int(cid.split("_")[-1]) - 1
                c_vec = centroids[c_idx]
            else:
                c_vec = np.mean(cluster_embs, axis=0)

            dists = np.linalg.norm(cluster_embs - c_vec, axis=1)
            medoid_local_idx = int(np.argmin(dists))
            medoid_id = cluster_track_ids[medoid_local_idx]
            medoid_meta = cluster_metas[medoid_local_idx]

            medoid_info = {
                "track_id": medoid_id,
                "filename": medoid_meta.get("filename", ""),
                "subgenero": medoid_meta.get("subgenero", ""),
                "distance_to_centroid": float(dists[medoid_local_idx])
            }

        tracks_list = []
        for tid, meta in zip(cluster_track_ids, cluster_metas):
            tracks_list.append({
                "track_id": tid,
                "filename": meta.get("filename", ""),
                "subgenero": meta.get("subgenero", "")
            })

        pureza_val = stats.get("pureza", 0.0)
        tam = len(indices)

        if cid != "ruido":
            total_pureza_ponderada += pureza_val * tam
            total_amostras_validas += tam

        summary["clusters"][cid] = {
            "cluster_id": cid,
            "nome": c_nome,
            "genero_predominante": stats.get("genero_predominante", "Desconhecido"),
            "pureza": pureza_val,
            "distribuicao_generos": stats.get("distribuicao", {}),
            "size": tam,
            "medoid": medoid_info,
            "tracks": tracks_list
        }

    summary["n_clusters"] = len([c for c in unique_cids if c != "ruido"])
    summary["pureza_media"] = round(total_pureza_ponderada / total_amostras_validas, 4) if total_amostras_validas > 0 else 0.0

    return summary


# ── PERSISTÊNCIA DE MODELO (PARA INFERÊNCIA INCREMENTAL) ───────────────────────

def save_cluster_model(
    output_path: Path,
    algorithm: str,
    centroids: Optional[np.ndarray],
    cluster_ids_unique: List[str],
    cluster_names_map: Dict[str, str],
    pca_model: Optional[PCA],
    seed: int
) -> None:
    """Salva os centroides, mapeamento de nomes de gêneros e PCA ajustado para inferência incremental."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_names = [cluster_names_map.get(cid, cid) for cid in cluster_ids_unique]
    model_payload = {
        "algorithm": algorithm,
        "centroids": centroids,
        "cluster_names": cluster_ids_unique,
        "cluster_display_names": display_names,
        "cluster_names_map": cluster_names_map,
        "pca_model": pca_model,
        "seed": seed,
        "saved_at": datetime.now().isoformat()
    }
    joblib.dump(model_payload, output_path)
    logger.info(f"Modelo de clusterização salvo em: {output_path}")


def load_cluster_model(model_path: Path) -> Optional[Dict[str, Any]]:
    """Carrega o modelo de clusterização salvo."""
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        logger.warning(f"Não foi possível carregar modelo de {model_path}: {e}")
        return None


def assign_cluster_incremental(
    embedding_raw: np.ndarray,
    model_data: Dict[str, Any]
) -> Optional[Tuple[str, str]]:
    """
    Atribui uma nova faixa ao centroide mais próximo usando o modelo salvo.
    Retorna tupla (cluster_id, cluster_nome), ex: ('cluster_01', 'Eletrônica').
    Suporta apenas K-Means.
    """
    if not model_data or model_data.get("algorithm") != "kmeans":
        logger.info("Atribuição incremental aplicável apenas para K-Means.")
        return None

    centroids = model_data.get("centroids")
    cluster_names = model_data.get("cluster_names")
    display_names = model_data.get("cluster_display_names") or cluster_names
    pca_model = model_data.get("pca_model")

    if centroids is None or not cluster_names:
        return None

    # Normalização L2
    vec = normalize(embedding_raw.reshape(1, -1), norm="l2")

    # Aplicação do PCA ajustado previamente se existir
    if pca_model is not None:
        vec = pca_model.transform(vec)

    # Distância Euclidiana para cada centroide
    dists = np.linalg.norm(centroids - vec[0], axis=1)
    best_idx = int(np.argmin(dists))
    cid = cluster_names[best_idx]
    cnome = display_names[best_idx] if best_idx < len(display_names) else cid
    return cid, cnome


# ── BACKUP E GRAVAÇÃO SEGURA NO CHROMADB ──────────────────────────────────────

def backup_chromadb(chroma_dir: str) -> str:
    """Cria uma cópia completa de segurança da pasta do ChromaDB."""
    chroma_path = Path(chroma_dir).resolve()
    if not chroma_path.exists():
        raise FileNotFoundError(f"Pasta do ChromaDB não encontrada em {chroma_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = chroma_path.parent / f"{chroma_path.name}_backup_{timestamp}"
    logger.info(f"Criando backup de segurança do ChromaDB em: {backup_path}...")
    shutil.copytree(chroma_path, backup_path)
    logger.info("Backup do ChromaDB concluído com sucesso!")
    return str(backup_path)


def update_chromadb_with_clusters(
    collection,
    ids: List[str],
    cluster_ids: List[str],
    cluster_names_map: Dict[str, str],
    metadata_json_path: Optional[Path] = None
) -> None:
    """
    Atualiza com segurança os metadados no ChromaDB e no metadata.json:
    1. Lê os metadados atuais para validar.
    2. Mescla explicitamente com cluster_id e cluster_nome sem apagar campos preexistentes.
    3. Executa a atualização no ChromaDB e no metadata.json.
    4. Realiza verificação de integridade pós-atualização em 100% das faixas.
    """
    logger.info("Iniciando atualização segura de metadados no ChromaDB...")

    # Leitura prévia dos metadados para auditoria
    current_data = collection.get(ids=ids, include=["metadatas"])
    current_metas_map = {tid: meta for tid, meta in zip(current_data["ids"], current_data["metadatas"])}

    id_to_cid = {tid: cid for tid, cid in zip(ids, cluster_ids)}
    id_to_cnome = {tid: cluster_names_map.get(cid, cid) for tid, cid in zip(ids, cluster_ids)}

    batch_size = 100
    total = len(ids)

    for i in range(0, total, batch_size):
        b_ids = ids[i:i + batch_size]
        b_metas = []
        for tid in b_ids:
            old_meta = current_metas_map.get(tid, {})
            merged_meta = dict(old_meta) if old_meta else {}
            merged_meta["cluster_id"] = str(id_to_cid[tid])
            merged_meta["cluster_nome"] = str(id_to_cnome[tid])
            b_metas.append(merged_meta)

        collection.update(ids=b_ids, metadatas=b_metas)

    logger.info(f"Metadados de {total} faixas atualizados no ChromaDB.")

    # Atualiza também metadata.json se existir
    if metadata_json_path and metadata_json_path.exists():
        logger.info(f"Atualizando {metadata_json_path} com cluster_id e cluster_nome...")
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        for item in json_data:
            tid = str(item.get("track_id"))
            if tid in id_to_cid:
                item["cluster_id"] = id_to_cid[tid]
                item["cluster_nome"] = id_to_cnome[tid]

        temp_json = metadata_json_path.with_suffix(".tmp")
        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        os.replace(temp_json, metadata_json_path)
        logger.info(f"Arquivo {metadata_json_path} sincronizado com sucesso.")

    # ── Auditoria de Integridade Estrita ───────────────────────────────────────
    logger.info("Executando verificação de integridade pós-gravação nas faixas...")
    verification = collection.get(ids=ids, include=["metadatas"])
    verified_map = {tid: meta for tid, meta in zip(verification["ids"], verification["metadatas"])}

    for tid in ids:
        orig = current_metas_map[tid]
        curr = verified_map[tid]

        # Verifica campos originais
        for key in ["filename", "subgenero", "n_segments", "fingerprint"]:
            if key in orig:
                if curr.get(key) != orig[key]:
                    raise AssertionError(
                        f"Inconsistência detectada na faixa {tid}! "
                        f"Campo '{key}' alterado de '{orig[key]}' para '{curr.get(key)}'."
                    )

        # Verifica cluster_id e cluster_nome
        if curr.get("cluster_id") != id_to_cid[tid]:
            raise AssertionError(
                f"cluster_id incorreto na faixa {tid}: esperado '{id_to_cid[tid]}', obtido '{curr.get('cluster_id')}'"
            )
        if curr.get("cluster_nome") != id_to_cnome[tid]:
            raise AssertionError(
                f"cluster_nome incorreto na faixa {tid}: esperado '{id_to_cnome[tid]}', obtido '{curr.get('cluster_nome')}'"
            )

    logger.info(f"Sucesso na auditoria! Todas as {total} faixas mantiveram metadados intactos e receberam cluster_id e cluster_nome.")


# ── VISUALIZAÇÃO GRÁFICA 2D ───────────────────────────────────────────────────

def generate_2d_visualization(
    embeddings: np.ndarray,
    cluster_names: List[str],
    output_png_path: Path,
    seed: int = 42
) -> str:
    """
    Gera projeção 2D das músicas colorida por cluster.
    Tenta utilizar UMAP; se indisponível, utiliza t-SNE como fallback.
    """
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    method_name = "t-SNE"
    coords = None

    try:
        import umap
        logger.info("Reduzindo dimensionalidade para visualização 2D via UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=seed, metric="euclidean", n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(embeddings)
        method_name = "UMAP"
    except Exception as e:
        logger.info(f"UMAP indisponível ({e}). Utilizando t-SNE para projeção 2D...")
        from sklearn.manifold import TSNE
        perplexity_val = min(30, max(5, len(embeddings) // 10))
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity_val, n_iter_without_progress=300)
        coords = tsne.fit_transform(embeddings)
        method_name = "t-SNE"

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    unique_clusters = sorted(list(set(cluster_names)), key=lambda x: (x == "ruido", x))
    n_colors = max(len(unique_clusters), 1)
    if hasattr(plt, "colormaps"):
        cmap = plt.colormaps["tab20"].resampled(n_colors)
    else:
        cmap = plt.get_cmap("tab20", n_colors)

    for i, c_name in enumerate(unique_clusters):
        idx = [j for j, name in enumerate(cluster_names) if name == c_name]
        pts = coords[idx]

        if c_name == "ruido":
            ax.scatter(pts[:, 0], pts[:, 1], c="#999999", marker="x", alpha=0.5, s=25, label=f"Ruído ({len(idx)})")
        else:
            color = cmap(i % 20)
            ax.scatter(pts[:, 0], pts[:, 1], color=[color], alpha=0.75, s=35, label=f"{c_name} ({len(idx)})")

    ax.set_title(f"Mapeamento Sonoro 2D ({method_name}) — Clusters Musicais", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method_name} Componente 1", fontsize=11)
    ax.set_ylabel(f"{method_name} Componente 2", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Legenda inteligente se não houver clusters em excesso
    if len(unique_clusters) <= 25:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()

    logger.info(f"Gráfico 2D ({method_name}) salvo com sucesso em: {output_png_path}")
    return method_name


# ── INTEGRAÇÃO COM MPD E EXPORTAÇÃO DE PLAYLISTS ───────────────────────────────

def export_playlists_to_mpd(
    clusters_summary: Dict[str, Any],
    id_to_file_map: Dict[str, str],
    host: str = config.MPD_HOST,
    port: int = config.MPD_PORT,
    m3u_dir: Optional[Path] = None
) -> Tuple[int, int]:
    """
    Cria ou atualiza playlists no MPD com os nomes de gênero dos clusters (ex: 'Rock', 'Eletrônica', etc.).
    Sanitiza caracteres inválidos no Windows e remove playlists legadas (ex: cluster_01, cluster_02).
    """
    if m3u_dir:
        m3u_dir.mkdir(parents=True, exist_ok=True)
        # Remove arquivos .m3u legados do padrão cluster_*.m3u
        for old_m3u in m3u_dir.glob("cluster_*.m3u"):
            try:
                old_m3u.unlink()
            except Exception:
                pass

    mpd_client = None
    try:
        from mpd import MPDClient
        mpd_client = MPDClient()
        mpd_client.connect(host, port)
        logger.info(f"Conectado ao MPD ({host}:{port}) para exportação de playlists.")
        # Remove playlists legadas do MPD
        existing_playlists = [p.get("playlist") for p in mpd_client.listplaylists()]
        for p_name in existing_playlists:
            if p_name.startswith("cluster_"):
                try:
                    mpd_client.rm(p_name)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Não foi possível conectar ao MPD ({e}). Playlists serão salvas em arquivos .m3u.")
        mpd_client = None

    playlists_criadas_mpd = 0
    arquivos_m3u_salvos = 0

    for cid, c_data in clusters_summary.get("clusters", {}).items():
        if cid == "ruido" or c_data.get("nome") == "Ruído":
            logger.info("Ignorando cluster 'ruido' na criação de playlists.")
            continue

        cluster_display_name = c_data.get("nome", cid)
        playlist_safe_name = sanitize_playlist_name(cluster_display_name)

        tracks = c_data.get("tracks", [])
        if not tracks:
            continue

        file_uris = []
        for t in tracks:
            tid = t.get("track_id")
            file_uri = id_to_file_map.get(tid, tid)
            if file_uri:
                file_uris.append(file_uri)

        # 1. Criação no MPD
        if mpd_client is not None:
            try:
                existing_playlists = [p.get("playlist") for p in mpd_client.listplaylists()]
                if playlist_safe_name in existing_playlists:
                    mpd_client.rm(playlist_safe_name)

                for uri in file_uris:
                    try:
                        mpd_client.playlistadd(playlist_safe_name, uri)
                    except Exception:
                        pass
                playlists_criadas_mpd += 1
            except Exception as e:
                logger.warning(f"Erro ao criar playlist '{playlist_safe_name}' no MPD: {e}")

        # 2. Criação em arquivo .m3u
        if m3u_dir is not None:
            m3u_file = m3u_dir / f"{playlist_safe_name}.m3u"
            with open(m3u_file, "w", encoding="utf-8") as f:
                pureza_pct = c_data.get("pureza", 0.0) * 100
                f.write(f"# Playlist: {cluster_display_name} (ID: {cid}) - {len(file_uris)} faixas - Pureza: {pureza_pct:.1f}%\n")
                for uri in file_uris:
                    f.write(f"{uri}\n")
            arquivos_m3u_salvos += 1

    if mpd_client:
        try:
            mpd_client.disconnect()
        except Exception:
            pass

    logger.info(f"Exportação concluída: {playlists_criadas_mpd} playlists no MPD, {arquivos_m3u_salvos} arquivos .m3u salvos.")
    return playlists_criadas_mpd, arquivos_m3u_salvos


# ── PIPELINE PRINCIPAL DE CLUSTERIZAÇÃO ────────────────────────────────────────

def run_clustering_pipeline(
    k_arg: str = "auto",
    k_min: int = 4,
    k_max: int = 30,
    algorithm: str = "kmeans",
    n_pca: Optional[int] = None,
    seed: int = 42,
    export_playlists: bool = False,
    generate_graph: bool = False,
    recalculate: bool = False,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    skip_backup: bool = False
) -> Dict[str, Any]:
    """Executa o pipeline completo de clusterização."""
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILE_NAME

    # Conectar ao ChromaDB
    client = vector_store.get_client(config.CHROMA_PERSIST_DIR)
    collection = vector_store.get_or_create_collection(client, config.COLLECTION_NAME)

    # 1. Carregar embeddings do ChromaDB
    ids, raw_embeddings, metadatas = load_embeddings_from_chromadb(collection)
    if len(ids) == 0:
        raise ValueError("Coleção do ChromaDB está vazia!")

    # 2. Backup de segurança do banco antes de qualquer alteração
    if not skip_backup:
        backup_chromadb(config.CHROMA_PERSIST_DIR)

    # 3. Pré-processamento (L2 + PCA opcional)
    processed_embs, pca_model = preprocess_embeddings(
        raw_embeddings,
        n_pca=n_pca,
        seed=seed
    )

    chosen_k = None
    df_k_metrics = None
    centroids = None
    model_obj = None

    if algorithm == "kmeans":
        if k_arg.lower() == "auto":
            logger.info(f"Buscando melhor k entre {k_min} e {k_max} via Coeficiente de Silhueta...")
            df_k_metrics = evaluate_k_range(processed_embs, k_min=k_min, k_max=k_max, seed=seed)

            # Salva métricas de k para o TCC
            k_eval_csv = results_dir / "k_selection_metrics.csv"
            save_clustering_metrics_csv(df_k_metrics, str(k_eval_csv))
            logger.info(f"Métricas de avaliação k x silhueta salvas em: {k_eval_csv}")

            # Identifica o melhor k
            best_row = df_k_metrics.loc[df_k_metrics["silhouette"].idxmax()]
            chosen_k = int(best_row["k"])
            logger.info(f"Melhor k encontrado: {chosen_k} (Silhueta: {best_row['silhouette']:.4f})")

            # Salva gráfico de seleção de k
            k_eval_png = results_dir / "k_selection_silhouette.png"
            plot_k_selection(df_k_metrics, str(k_eval_png), best_k=chosen_k)
            logger.info(f"Gráfico de seleção k x silhueta salvo em: {k_eval_png}")
        else:
            chosen_k = int(k_arg)
            logger.info(f"Usando k fixo configurado: {chosen_k}")

        model_obj, labels, raw_cluster_ids = run_kmeans(processed_embs, k=chosen_k, seed=seed)
        centroids = model_obj.cluster_centers_

    elif algorithm == "hdbscan":
        model_obj, labels, raw_cluster_ids = run_hdbscan(processed_embs, min_cluster_size=5)
    else:
        raise ValueError(f"Algoritmo desconhecido: '{algorithm}'. Opções: 'kmeans', 'hdbscan'")

    # 4. Determinação dos nomes dos clusters por gênero musical e cálculo de pureza
    cluster_names_map, cluster_stats = determine_cluster_names_and_purity(ids, metadatas, raw_cluster_ids)
    cluster_display_names = [cluster_names_map.get(cid, cid) for cid in raw_cluster_ids]

    # Validação supervisionada contra subgêneros (TCC)
    subgeneros_ground_truth = [m.get("subgenero") for m in metadatas]
    val_metrics = evaluate_clustering_validation_metrics(subgeneros_ground_truth, cluster_display_names)

    # Avaliação das métricas intrínsecas
    metrics = evaluate_clustering_metrics(processed_embs, labels)
    logger.info("\n=== Métricas Finais de Agrupamento ===")
    logger.info(f"Algoritmo: {algorithm.upper()}")
    logger.info(f"Clusters Formados: {metrics['n_clusters']}")
    if metrics["silhouette"] is not None:
        logger.info(f"Coeficiente de Silhueta: {metrics['silhouette']:.4f}")
        logger.info(f"Índice Davies-Bouldin:   {metrics['davies_bouldin']:.4f}")
        logger.info(f"Índice Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")
    if val_metrics.get("purity") is not None:
        logger.info(f"Pureza Média (Purity):   {val_metrics['purity']:.4f} ({val_metrics['purity']*100:.1f}%)")
        logger.info(f"Adjusted Rand Index (ARI): {val_metrics['ari']:.4f}")
        logger.info(f"Normalized Mutual Info (NMI): {val_metrics['nmi']:.4f}")
    if metrics["noise_count"] > 0:
        logger.info(f"Faixas classificadas como Ruído: {metrics['noise_count']} ({metrics['noise_ratio']*100:.1f}%)")
    logger.info("======================================\n")

    logger.info("--- Detalhamento dos Clusters por Gênero ---")
    for cid, st in cluster_stats.items():
        tam = sum(1 for c in raw_cluster_ids if c == cid)
        pct_pur = st.get("pureza", 0.0) * 100
        logger.info(f"  [{cid}] {st['nome']} - {tam} faixas (Pureza: {pct_pur:.1f}%, Gênero Predominante: {st.get('genero_predominante')})")
    logger.info("--------------------------------------------\n")

    # Salva métricas finais em CSV para o TCC
    final_metrics_df = pd.DataFrame([{
        "algorithm": algorithm,
        "n_clusters": metrics["n_clusters"],
        "silhouette": metrics["silhouette"],
        "davies_bouldin": metrics["davies_bouldin"],
        "calinski_harabasz": metrics["calinski_harabasz"],
        "pureza_media": val_metrics.get("purity"),
        "ari": val_metrics.get("ari"),
        "nmi": val_metrics.get("nmi"),
        "noise_count": metrics["noise_count"],
        "noise_ratio": metrics["noise_ratio"],
        "pca_components": n_pca if n_pca else "None",
        "seed": seed
    }])
    final_metrics_csv = results_dir / "clustering_evaluation.csv"
    save_clustering_metrics_csv(final_metrics_df, str(final_metrics_csv))
    logger.info(f"Métricas finais exportadas para: {final_metrics_csv}")

    # 5. Resumo e Medoides (clusters.json)
    summary = compute_clusters_summary(
        ids=ids,
        metadatas=metadatas,
        processed_embeddings=processed_embs,
        labels=labels,
        cluster_ids=raw_cluster_ids,
        cluster_names_map=cluster_names_map,
        cluster_stats=cluster_stats,
        algorithm=algorithm,
        centroids=centroids,
        validation_metrics=val_metrics
    )
    clusters_json_path = results_dir / "clusters.json"
    with open(clusters_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Resumo dos clusters salvo em: {clusters_json_path}")

    # 6. Salvar modelo para inferência incremental
    unique_cluster_ids = sorted(list(set(raw_cluster_ids)), key=lambda x: (x == "ruido", x))
    save_cluster_model(
        output_path=model_path,
        algorithm=algorithm,
        centroids=centroids,
        cluster_ids_unique=[c for c in unique_cluster_ids if c != "ruido"],
        cluster_names_map=cluster_names_map,
        pca_model=pca_model,
        seed=seed
    )

    # 7. Atualização segura no ChromaDB e no metadata.json
    metadata_json_path = results_dir / "metadata.json"
    update_chromadb_with_clusters(
        collection=collection,
        ids=ids,
        cluster_ids=raw_cluster_ids,
        cluster_names_map=cluster_names_map,
        metadata_json_path=metadata_json_path
    )

    # 8. Visualização 2D
    if generate_graph:
        graph_png_path = results_dir / "clusters_visualization.png"
        generate_2d_visualization(processed_embs, cluster_display_names, graph_png_path, seed=seed)

    # 9. Exportar Playlists para MPD
    if export_playlists:
        id_to_file = {}
        if metadata_json_path.exists():
            try:
                with open(metadata_json_path, "r", encoding="utf-8") as f:
                    for item in json.load(f):
                        tid = str(item.get("track_id"))
                        fpath = item.get("filepath", "")
                        if fpath:
                            id_to_file[tid] = fpath
            except Exception:
                pass

        playlists_dir = _SCRIPT_DIR / "playlists"
        export_playlists_to_mpd(
            clusters_summary=summary,
            id_to_file_map=id_to_file,
            host=config.MPD_HOST,
            port=config.MPD_PORT,
            m3u_dir=playlists_dir
        )

    return {
        "algorithm": algorithm,
        "n_clusters": metrics["n_clusters"],
        "chosen_k": chosen_k,
        "metrics": metrics,
        "validation_metrics": val_metrics,
        "summary": summary
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clusterização de faixas musicais por similaridade vetorial")
    parser.add_argument("--k", type=str, default="auto", help="Número de clusters (int) ou 'auto' para buscar melhor silhueta (padrão: auto)")
    parser.add_argument("--k-min", type=int, default=4, help="Valor mínimo de k para busca automática (padrão: 4)")
    parser.add_argument("--k-max", type=int, default=30, help="Valor máximo de k para busca automática (padrão: 30)")
    parser.add_argument("--algoritmo", type=str, choices=["kmeans", "hdbscan"], default="kmeans", help="Algoritmo de agrupamento (padrão: kmeans)")
    parser.add_argument("--pca", type=int, default=None, help="Número de componentes principais (PCA) para redução de dimensionalidade")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade (padrão: 42)")
    parser.add_argument("--exportar-playlists", action="store_true", help="Cria playlists por cluster no MPD e em arquivos .m3u")
    parser.add_argument("--grafico", action="store_true", help="Gera gráfico de visualização 2D (UMAP/t-SNE)")
    parser.add_argument("--recalcular", action="store_true", help="Força recálculo completo da clusterização")
    parser.add_argument("--skip-backup", action="store_true", help="Pula criação de backup do ChromaDB (uso em testes)")

    args = parser.parse_args()

    run_clustering_pipeline(
        k_arg=args.k,
        k_min=args.k_min,
        k_max=args.k_max,
        algorithm=args.algoritmo,
        n_pca=args.pca,
        seed=args.seed,
        export_playlists=args.exportar_playlists,
        generate_graph=args.grafico,
        recalculate=args.recalcular,
        skip_backup=args.skip_backup
    )

