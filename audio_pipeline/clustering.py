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
import json
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

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


# ── MEDÓIDES E RESUMO DE CLUSTERS ─────────────────────────────────────────────

def compute_clusters_summary(
    ids: List[str],
    metadatas: List[Dict[str, Any]],
    processed_embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_names: List[str],
    algorithm: str,
    centroids: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calcula o resumo dos clusters, incluindo tamanho, lista de faixas e medóide
    (faixa mais próxima do centroide geométrico).
    """
    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "algorithm": algorithm,
        "total_tracks": len(ids),
        "clusters": {}
    }

    unique_names = sorted(list(set(cluster_names)), key=lambda x: (x == "ruido", x))

    for c_name in unique_names:
        indices = [i for i, name in enumerate(cluster_names) if name == c_name]
        cluster_embs = processed_embeddings[indices]
        cluster_ids = [ids[i] for i in indices]
        cluster_metas = [metadatas[i] for i in indices]

        medoid_info = None
        if c_name != "ruido" and len(indices) > 0:
            # Centroide do cluster
            if algorithm == "kmeans" and centroids is not None:
                # O índice numérico correspondente
                c_idx = int(c_name.split("_")[-1]) - 1
                c_vec = centroids[c_idx]
            else:
                c_vec = np.mean(cluster_embs, axis=0)

            # Encontra a faixa com menor distância euclidiana ao centroide
            dists = np.linalg.norm(cluster_embs - c_vec, axis=1)
            medoid_local_idx = int(np.argmin(dists))
            medoid_id = cluster_ids[medoid_local_idx]
            medoid_meta = cluster_metas[medoid_local_idx]

            medoid_info = {
                "track_id": medoid_id,
                "filename": medoid_meta.get("filename", ""),
                "subgenero": medoid_meta.get("subgenero", ""),
                "distance_to_centroid": float(dists[medoid_local_idx])
            }

        tracks_list = []
        for tid, meta in zip(cluster_ids, cluster_metas):
            tracks_list.append({
                "track_id": tid,
                "filename": meta.get("filename", ""),
                "subgenero": meta.get("subgenero", "")
            })

        summary["clusters"][c_name] = {
            "size": len(indices),
            "medoid": medoid_info,
            "tracks": tracks_list
        }

    summary["n_clusters"] = len([c for c in unique_names if c != "ruido"])
    return summary


# ── PERSISTÊNCIA DE MODELO (PARA INFERÊNCIA INCREMENTAL) ───────────────────────

def save_cluster_model(
    output_path: Path,
    algorithm: str,
    centroids: Optional[np.ndarray],
    cluster_names_unique: List[str],
    pca_model: Optional[PCA],
    seed: int
) -> None:
    """Salva os centroides, lista de nomes de clusters e PCA ajustado para inferência incremental."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "algorithm": algorithm,
        "centroids": centroids,
        "cluster_names": cluster_names_unique,
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
) -> Optional[str]:
    """
    Atribui uma nova faixa ao centroide mais próximo usando o modelo salvo.
    Suporta apenas K-Means (conforme especificação, HDBSCAN não suporta atribuição por centroide).
    """
    if not model_data or model_data.get("algorithm") != "kmeans":
        logger.info("Atribuição incremental aplicável apenas para K-Means.")
        return None

    centroids = model_data.get("centroids")
    cluster_names = model_data.get("cluster_names")
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
    return cluster_names[best_idx]


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
    cluster_names: List[str],
    metadata_json_path: Optional[Path] = None
) -> None:
    """
    Atualiza com segurança os metadados no ChromaDB e no metadata.json:
    1. Lê os metadados atuais para validar.
    2. Mescla explicitamente com cluster_id sem apagar campos preexistentes.
    3. Executa a atualização no ChromaDB e no metadata.json.
    4. Realiza verificação de integridade pós-atualização em 100% das faixas.
    """
    logger.info("Iniciando atualização segura de metadados no ChromaDB...")

    # Leitura prévia dos metadados para auditoria
    current_data = collection.get(ids=ids, include=["metadatas"])
    current_metas_map = {tid: meta for tid, meta in zip(current_data["ids"], current_data["metadatas"])}

    cluster_map = {tid: c_name for tid, c_name in zip(ids, cluster_names)}

    batch_size = 100
    total = len(ids)

    for i in range(0, total, batch_size):
        b_ids = ids[i:i + batch_size]
        b_metas = []
        for tid in b_ids:
            old_meta = current_metas_map.get(tid, {})
            merged_meta = dict(old_meta) if old_meta else {}
            merged_meta["cluster_id"] = str(cluster_map[tid])
            b_metas.append(merged_meta)

        collection.update(ids=b_ids, metadatas=b_metas)

    logger.info(f"Metadados de {total} faixas atualizados no ChromaDB.")

    # Atualiza também metadata.json se existir
    if metadata_json_path and metadata_json_path.exists():
        logger.info(f"Atualizando {metadata_json_path} com cluster_id...")
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        for item in json_data:
            tid = str(item.get("track_id"))
            if tid in cluster_map:
                item["cluster_id"] = cluster_map[tid]

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

        # Verifica cluster_id
        if curr.get("cluster_id") != cluster_map[tid]:
            raise AssertionError(
                f"cluster_id incorreto na faixa {tid}: esperado '{cluster_map[tid]}', obtido '{curr.get('cluster_id')}'"
            )

    logger.info(f"Sucesso na auditoria! Todas as {total} faixas mantiveram metadados intactos e receberam cluster_id.")


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
    Cria ou atualiza uma playlist por cluster no MPD (ex: cluster_01, cluster_02, ...).
    Também grava arquivos .m3u na pasta especificada como fallback universal.
    Faixas de ruído ('ruido') são explicitamente excluídas das playlists.
    """
    if m3u_dir:
        m3u_dir.mkdir(parents=True, exist_ok=True)

    mpd_client = None
    try:
        from mpd import MPDClient
        mpd_client = MPDClient()
        mpd_client.connect(host, port)
        logger.info(f"Conectado ao MPD ({host}:{port}) para exportação de playlists.")
    except Exception as e:
        logger.warning(f"Não foi possível conectar ao MPD ({e}). Playlists serão salvas em arquivos .m3u.")
        mpd_client = None

    playlists_criadas_mpd = 0
    arquivos_m3u_salvos = 0

    for c_name, c_data in clusters_summary.get("clusters", {}).items():
        if c_name == "ruido":
            logger.info("Ignorando cluster 'ruido' na criação de playlists.")
            continue

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
                # Verifica se playlist já existe e remove
                existing_playlists = [p.get("playlist") for p in mpd_client.listplaylists()]
                if c_name in existing_playlists:
                    mpd_client.rm(c_name)

                # Adiciona cada faixa
                for uri in file_uris:
                    try:
                        mpd_client.playlistadd(c_name, uri)
                    except Exception:
                        pass
                playlists_criadas_mpd += 1
            except Exception as e:
                logger.warning(f"Erro ao criar playlist '{c_name}' no MPD: {e}")

        # 2. Criação em arquivo .m3u
        if m3u_dir is not None:
            m3u_file = m3u_dir / f"{c_name}.m3u"
            with open(m3u_file, "w", encoding="utf-8") as f:
                f.write(f"# Playlist gerada por clusterizacao ({c_name}) - {len(file_uris)} faixas\n")
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

        model_obj, labels, cluster_names = run_kmeans(processed_embs, k=chosen_k, seed=seed)
        centroids = model_obj.cluster_centers_

    elif algorithm == "hdbscan":
        model_obj, labels, cluster_names = run_hdbscan(processed_embs, min_cluster_size=5)
    else:
        raise ValueError(f"Algoritmo desconhecido: '{algorithm}'. Opções: 'kmeans', 'hdbscan'")

    # 4. Avaliação das métricas finais
    metrics = evaluate_clustering_metrics(processed_embs, labels)
    logger.info("\n=== Métricas Finais de Agrupamento ===")
    logger.info(f"Algoritmo: {algorithm.upper()}")
    logger.info(f"Clusters Formados: {metrics['n_clusters']}")
    if metrics["silhouette"] is not None:
        logger.info(f"Coeficiente de Silhueta: {metrics['silhouette']:.4f}")
        logger.info(f"Índice Davies-Bouldin:   {metrics['davies_bouldin']:.4f}")
        logger.info(f"Índice Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")
    if metrics["noise_count"] > 0:
        logger.info(f"Faixas classificadas como Ruído: {metrics['noise_count']} ({metrics['noise_ratio']*100:.1f}%)")
    logger.info("======================================\n")

    # Salva métricas finais em CSV para o TCC
    final_metrics_df = pd.DataFrame([{
        "algorithm": algorithm,
        "n_clusters": metrics["n_clusters"],
        "silhouette": metrics["silhouette"],
        "davies_bouldin": metrics["davies_bouldin"],
        "calinski_harabasz": metrics["calinski_harabasz"],
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
        cluster_names=cluster_names,
        algorithm=algorithm,
        centroids=centroids
    )
    clusters_json_path = results_dir / "clusters.json"
    with open(clusters_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Resumo dos clusters salvo em: {clusters_json_path}")

    # 6. Salvar modelo para inferência incremental
    unique_cluster_names = sorted(list(set(cluster_names)), key=lambda x: (x == "ruido", x))
    save_cluster_model(
        output_path=model_path,
        algorithm=algorithm,
        centroids=centroids,
        cluster_names_unique=[c for c in unique_cluster_names if c != "ruido"],
        pca_model=pca_model,
        seed=seed
    )

    # 7. Atualização segura no ChromaDB e no metadata.json
    metadata_json_path = results_dir / "metadata.json"
    update_chromadb_with_clusters(
        collection=collection,
        ids=ids,
        cluster_names=cluster_names,
        metadata_json_path=metadata_json_path
    )

    # 8. Visualização 2D
    if generate_graph:
        graph_png_path = results_dir / "clusters_visualization.png"
        generate_2d_visualization(processed_embs, cluster_names, graph_png_path, seed=seed)

    # 9. Exportar Playlists para MPD
    if export_playlists:
        # Carrega mapa de arquivos para URIs do MPD
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

