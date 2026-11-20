"""
Script de geração automática da métrica Precision@K.
Gera: CSV por faixa, JSON de resumo, histograma PNG e histórico acumulado.
Deve ser executado a partir da pasta audio_pipeline/:
    python results/generate_precision_k.py --k 5
"""

import argparse
import json
import os
import sys
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Garante que o diretório pai (audio_pipeline) está no path para importar evaluation.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PARENT_DIR)

import config
import vector_store
from evaluation import (
    load_ground_truth,
    precision_at_k_for_track,
    random_baseline_precision_at_k,
)

RESULTS_DIR = _SCRIPT_DIR


def gerar_metricas(k: int, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    metadata_path = os.path.join(_PARENT_DIR, "results", "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"[ERRO] metadata.json não encontrado em: {metadata_path}")
        sys.exit(1)

    print(f"Carregando ground truth de {metadata_path}...")
    ground_truth = load_ground_truth(metadata_path)
    n_faixas = len(ground_truth)
    print(f"  {n_faixas} faixas válidas encontradas.")

    print("Conectando ao ChromaDB para carregar embeddings...")
    chroma_client = vector_store.get_client(config.CHROMA_PERSIST_DIR)
    collection = vector_store.get_or_create_collection(chroma_client, config.COLLECTION_NAME)

    # Carrega embeddings e metadados uma única vez para processamento vetorial rápido em memória
    all_data = collection.get(include=["embeddings", "metadatas"])
    valid_indices = []
    track_ids_list = []
    subgeneros_list = []
    cluster_ids_list = []
    cluster_nomes_list = []
    filenames_list = []
    embeddings_list = []

    for tid, emb, meta in zip(all_data["ids"], all_data["embeddings"], all_data["metadatas"]):
        if tid in ground_truth:
            track_ids_list.append(tid)
            subgeneros_list.append(ground_truth[tid])
            cluster_ids_list.append(str(meta.get("cluster_id", "cluster_01")))
            cluster_nomes_list.append(str(meta.get("cluster_nome", meta.get("subgenero", "Desconhecido"))))
            filenames_list.append(str(meta.get("filename", tid)))
            embeddings_list.append(emb)

    X = np.array(embeddings_list, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X_norm = X / norms
    N = len(track_ids_list)

    cluster_ids_arr = np.array(cluster_ids_list)
    subgeneros_arr = np.array(subgeneros_list)

    # --- Calcular P@K Global, P@K Restrito ao Cluster e Coerência de Cluster ---
    print(f"Calculando Precision@{k} (Global vs Restrito ao Cluster e Coerência) para {N} faixas...")
    registros = []
    scores_global = []
    scores_cluster = []
    scores_coherence = []

    for i in range(N):
        seed_id = track_ids_list[i]
        seed_sg = subgeneros_list[i]
        seed_cid = cluster_ids_list[i]
        seed_cnome = cluster_nomes_list[i]
        seed_fname = filenames_list[i]

        # Similaridade de cosseno com todas as faixas
        sims = np.dot(X_norm, X_norm[i])
        sims[i] = -1e9  # Exclui a própria faixa

        # 1. K-NN Global
        top_k_global = np.argsort(sims)[::-1][:k]
        p_global = float(np.mean([subgeneros_arr[idx] == seed_sg for idx in top_k_global])) if len(top_k_global) > 0 else 0.0
        coherence = float(np.mean([cluster_ids_arr[idx] == seed_cid for idx in top_k_global])) if len(top_k_global) > 0 else 0.0

        # 2. K-NN Restrito ao Cluster
        same_cluster_mask = (cluster_ids_arr == seed_cid)
        same_cluster_mask[i] = False
        n_same_cluster = int(np.sum(same_cluster_mask))

        if n_same_cluster > 0:
            sims_cluster = np.copy(sims)
            sims_cluster[~same_cluster_mask] = -1e9
            k_eff = min(k, n_same_cluster)
            top_k_cluster = np.argsort(sims_cluster)[::-1][:k_eff]
            p_cluster = float(np.mean([subgeneros_arr[idx] == seed_sg for idx in top_k_cluster]))
        else:
            p_cluster = p_global

        scores_global.append(p_global)
        scores_cluster.append(p_cluster)
        scores_coherence.append(coherence)

        registros.append({
            "track_id": seed_id,
            "filename": seed_fname,
            "subgenero": seed_sg,
            "cluster_id": seed_cid,
            "cluster_nome": seed_cnome,
            f"precision_at_{k}": round(p_global, 4),
            f"precision_at_{k}_cluster": round(p_cluster, 4),
            f"cluster_coherence_at_{k}": round(coherence, 4),
        })

    df = pd.DataFrame(registros)
    csv_path = os.path.join(RESULTS_DIR, f"precision_at_{k}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV salvo com novas colunas de cluster: {csv_path}")

    media_sistema = float(np.mean(scores_global))
    desvio_sistema = float(np.std(scores_global))

    media_cluster = float(np.mean(scores_cluster))
    desvio_cluster = float(np.std(scores_cluster))

    media_coerencia = float(np.mean(scores_coherence))
    desvio_coerencia = float(np.std(scores_coherence))

    # --- Baseline aleatória ---
    print("Calculando baseline aleatória...")
    baseline_scores = []
    track_ids = list(ground_truth.keys())
    for seed_id in track_ids:
        seed_sg = ground_truth[seed_id]
        candidates = [t for t in track_ids if t != seed_id]
        trial_scores = []
        for _ in range(100):
            sample = random.sample(candidates, min(k, len(candidates)))
            matches = sum(1 for t in sample if ground_truth[t] == seed_sg)
            trial_scores.append(matches / len(sample))
        baseline_scores.append(np.mean(trial_scores))

    media_baseline = float(np.mean(baseline_scores))
    desvio_baseline = float(np.std(baseline_scores))
    ganho_global = media_sistema - media_baseline
    ganho_cluster = media_cluster - media_baseline

    # --- Estatísticas por Cluster ---
    por_cluster = {}
    for cnome, group in df.groupby("cluster_nome"):
        c_p_global = float(group[f"precision_at_{k}"].mean())
        c_p_cluster = float(group[f"precision_at_{k}_cluster"].mean())
        c_coherence = float(group[f"cluster_coherence_at_{k}"].mean())
        por_cluster[cnome] = {
            "cluster_id": str(group["cluster_id"].iloc[0]),
            "tamanho": len(group),
            "media_p_at_k": round(c_p_global, 4),
            "media_p_at_k_cluster": round(c_p_cluster, 4),
            "ganho_global": round(c_p_global - media_baseline, 4),
            "ganho_cluster": round(c_p_cluster - media_baseline, 4),
            "coerencia_media": round(c_coherence, 4),
        }

    # --- JSON de resumo ---
    timestamp = datetime.now().isoformat()
    resumo = {
        "timestamp": timestamp,
        "k": k,
        "faixas_avaliadas": n_faixas,
        # Chaves originais mantidas para retrocompatibilidade
        "sistema": {"media": media_sistema, "desvio_padrao": desvio_sistema},
        "baseline": {"media": media_baseline, "desvio_padrao": desvio_baseline},
        "ganho_absoluto": ganho_global,
        # Novas chaves com comparação e clusterização
        "sistema_global": {
            "media": media_sistema,
            "desvio_padrao": desvio_sistema,
            "ganho_absoluto": ganho_global,
        },
        "sistema_cluster": {
            "media": media_cluster,
            "desvio_padrao": desvio_cluster,
            "ganho_absoluto": ganho_cluster,
        },
        "coerencia_cluster": {
            "media": media_coerencia,
            "desvio_padrao": desvio_coerencia,
        },
        "por_cluster": por_cluster,
    }
    json_path = os.path.join(RESULTS_DIR, f"resumo_k_{k}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=4, ensure_ascii=False)
    print(f"  JSON salvo com novas métricas de cluster: {json_path}")

    # --- Histograma de densidade ---
    _gerar_histograma(scores_global, list(baseline_scores), k, n_faixas)

    # --- Histórico acumulado ---
    historico_path = os.path.join(RESULTS_DIR, "historico_execucoes.csv")
    nova_linha = {
        "timestamp": timestamp,
        "k": k,
        "faixas_avaliadas": n_faixas,
        "media_sistema": round(media_sistema, 4),
        "desvio_sistema": round(desvio_sistema, 4),
        "media_baseline": round(media_baseline, 4),
        "ganho_absoluto": round(ganho_global, 4),
    }
    if os.path.exists(historico_path):
        df_hist = pd.read_csv(historico_path)
        df_hist = pd.concat([df_hist, pd.DataFrame([nova_linha])], ignore_index=True)
    else:
        df_hist = pd.DataFrame([nova_linha])
    df_hist.to_csv(historico_path, index=False)
    print(f"  Histórico atualizado: {historico_path}")

    print(f"\n=== Precision@{k} ===")
    print(f"  Sistema (Global)  : {media_sistema:.4f} ± {desvio_sistema:.4f} (Ganho: {ganho_global:+.4f})")
    print(f"  Sistema (Cluster) : {media_cluster:.4f} ± {desvio_cluster:.4f} (Ganho: {ganho_cluster:+.4f})")
    print(f"  Coerência Cluster : {media_coerencia:.4f} ({media_coerencia*100:.1f}% dos vizinhos no mesmo cluster)")
    print(f"  Baseline          : {media_baseline:.4f} ± {desvio_baseline:.4f}")


def _gerar_histograma(scores_sistema, scores_baseline, k: int, n_faixas: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2d2d2d")

    ax.hist(scores_sistema, bins=15, alpha=0.7, color="#4C72B0", label="Sistema", density=True)
    ax.hist(scores_baseline, bins=15, alpha=0.6, color="#DD8452", label="Baseline Aleatória", density=True)

    ax.axvline(np.mean(scores_sistema), color="#4C72B0", linestyle="--", linewidth=1.5,
               label=f"Média Sistema: {np.mean(scores_sistema):.2f}")
    ax.axvline(np.mean(scores_baseline), color="#DD8452", linestyle="--", linewidth=1.5,
               label=f"Média Baseline: {np.mean(scores_baseline):.2f}")

    if n_faixas < 50:
        ax.text(0.5, 0.95,
                f"⚠ Amostra pequena (n={n_faixas}). Resultados indicativos.",
                transform=ax.transAxes, color="#ffcc00", fontsize=9,
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#3a3a00", edgecolor="#ffcc00"))

    ax.set_xlabel(f"Precision@{k} por faixa", color="#cccccc")
    ax.set_ylabel("Densidade", color="#cccccc")
    ax.set_title(f"Distribuição de Precision@{k}: Sistema vs Baseline (n={n_faixas})", color="#ffffff", fontsize=13)
    ax.tick_params(colors="#aaaaaa")
    ax.legend(framealpha=0.3, labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "precision_at_k_histograma.png")
    plt.savefig(png_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Histograma salvo: {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera Precision@K e salva artefatos em results/")
    parser.add_argument("--k", type=int, default=5, help="Valor de K (padrão: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    args = parser.parse_args()
    gerar_metricas(k=args.k, seed=args.seed)
