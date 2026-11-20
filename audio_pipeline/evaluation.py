import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from vector_store import get_client, get_or_create_collection, query_similar

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans

def evaluate_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calcula as principais métricas intrínsecas de qualidade de agrupamento:
    - Coeficiente de Silhueta (Silhouette Score): [-1, 1], quanto maior, melhor
    - Índice Davies-Bouldin: [0, inf), quanto menor, melhor
    - Índice Calinski-Harabasz: [0, inf), quanto maior, melhor

    Para algoritmos com ruído (ex: HDBSCAN com rótulos -1), as métricas são calculadas
    sobre as amostras efetivamente agrupadas (labels >= 0).
    """
    labels = np.asarray(labels)
    unique_labels = set(labels)
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    valid_embeddings = embeddings[valid_mask]
    n_valid_clusters = len(set(valid_labels))

    noise_count = int(np.sum(~valid_mask))
    total_count = len(labels)
    noise_ratio = float(noise_count / total_count) if total_count > 0 else 0.0

    if n_valid_clusters < 2 or len(valid_embeddings) <= n_valid_clusters:
        return {
            "n_clusters": n_valid_clusters,
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "noise_count": noise_count,
            "noise_ratio": noise_ratio
        }

    # Calculando as métricas com distância euclidiana (em vetores normalizados L2 reflete similaridade de cosseno)
    sil = float(silhouette_score(valid_embeddings, valid_labels, metric='euclidean'))
    db = float(davies_bouldin_score(valid_embeddings, valid_labels))
    ch = float(calinski_harabasz_score(valid_embeddings, valid_labels))

    return {
        "n_clusters": n_valid_clusters,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "noise_count": noise_count,
        "noise_ratio": noise_ratio
    }

def evaluate_k_range(embeddings: np.ndarray, k_min: int = 4, k_max: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Avalia uma faixa de valores de k para o K-Means gerando Silhueta, Davies-Bouldin,
    Calinski-Harabasz e Inércia para cada k.
    """
    results = []
    max_k = min(k_max, len(embeddings) - 1)
    min_k = max(2, k_min)

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        metrics = evaluate_clustering_metrics(embeddings, labels)
        results.append({
            "k": k,
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "inertia": float(kmeans.inertia_)
        })

    return pd.DataFrame(results)

def plot_k_selection(df_k_metrics: pd.DataFrame, output_path: str, best_k: int | None = None) -> None:
    """
    Plota gráfico de k x Coeficiente de Silhueta com destaque para o melhor k,
    ideal para inclusão no TCC.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    ax.plot(df_k_metrics["k"], df_k_metrics["silhouette"], marker='o', color='#2b5c8f', linewidth=2, label="Silhueta")

    if best_k is not None and best_k in df_k_metrics["k"].values:
        best_row = df_k_metrics[df_k_metrics["k"] == best_k].iloc[0]
        ax.scatter([best_k], [best_row["silhouette"]], color='#d9534f', s=120, zorder=5, label=f"Melhor k={best_k} (Silhueta: {best_row['silhouette']:.4f})")
        ax.annotate(
            f"k={best_k}\nSilhueta: {best_row['silhouette']:.4f}",
            xy=(best_k, best_row["silhouette"]),
            xytext=(best_k, best_row["silhouette"] + 0.015),
            ha='center',
            arrowprops=dict(facecolor='#d9534f', shrink=0.08, width=1.5, headwidth=6)
        )

    ax.set_title("Escolha de k: Coeficiente de Silhueta por Número de Clusters", fontsize=13, fontweight='bold')
    ax.set_xlabel("Número de Clusters (k)", fontsize=11)
    ax.set_ylabel("Coeficiente de Silhueta Médio", fontsize=11)
    ax.set_xticks(df_k_metrics["k"].values)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_clustering_metrics_csv(df_metrics: pd.DataFrame, output_path: str) -> None:
    """Salva métricas em formato CSV para exportação no TCC."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_metrics.to_csv(output_path, index=False)


def load_ground_truth(metadata_path: str) -> Dict[str, str]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Arquivo metadata não encontrado em: {metadata_path}")
        
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    ground_truth = {}
    missing_subgenre = []
    
    if isinstance(metadata, dict):
        iterator = metadata.items()
    elif isinstance(metadata, list):
        iterator = []
        for i, m in enumerate(metadata):
            track_id = str(m.get('track_id', str(i)))
            iterator.append((track_id, m))
    else:
        raise ValueError("Formato de metadata não suportado. Esperado dict ou list.")
    
    for track_id, info in iterator:
        subgenero = info.get("subgenero", "").strip()
        if not subgenero:
            filename = info.get("filename", "desconhecido")
            missing_subgenre.append(f"ID: {track_id} | Arquivo: {filename}")
        else:
            ground_truth[track_id] = subgenero
            
    if missing_subgenre:
        raise AssertionError(f"Erro de sanidade: {len(missing_subgenre)} faixas estão sem subgenero preenchido. Isso não deveria acontecer com a inferência automática.")
        
    return ground_truth

def precision_at_k_for_track(collection, track_id: str, ground_truth: Dict[str, str], k: int = 5) -> float:
    result = collection.get(ids=[track_id], include=["embeddings"])
    embeddings = result.get('embeddings')
    if embeddings is None or len(embeddings) == 0:
        return 0.0
        
    query_emb = np.array(result['embeddings'][0])
    
    similares = query_similar(collection, query_emb, k=k, exclude_id=track_id)
    
    if not similares:
        return 0.0
        
    seed_subgenero = ground_truth.get(track_id)
    if not seed_subgenero:
        return 0.0
        
    matches = sum(1 for res in similares if res.get('subgenero') == seed_subgenero)
    return matches / len(similares)

def precision_at_k_average(collection, ground_truth: Dict[str, str], k: int = 5) -> float:
    scores = []
    for track_id in ground_truth.keys():
        score = precision_at_k_for_track(collection, track_id, ground_truth, k)
        scores.append(score)
    
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def random_baseline_precision_at_k(ground_truth: Dict[str, str], k: int = 5, n_trials: int = 100) -> float:
    track_ids = list(ground_truth.keys())
    if len(track_ids) <= k:
        return 0.0
        
    total_score = 0.0
    
    for seed_track_id in track_ids:
        seed_subgenero = ground_truth[seed_track_id]
        
        candidates = [t_id for t_id in track_ids if t_id != seed_track_id]
        
        track_score = 0.0
        for _ in range(n_trials):
            sampled_ids = random.sample(candidates, min(k, len(candidates)))
            matches = sum(1 for t_id in sampled_ids if ground_truth[t_id] == seed_subgenero)
            track_score += matches / len(sampled_ids)
            
        total_score += track_score / n_trials
        
    return total_score / len(track_ids)

def run_full_evaluation(collection, ground_truth: Dict[str, str], k_values: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
    results = []
    
    for k in k_values:
        print(f"Avaliando para K={k}...")
        precisao_sistema = precision_at_k_average(collection, ground_truth, k)
        precisao_baseline = random_baseline_precision_at_k(ground_truth, k)
        
        ganho = 0.0
        if precisao_baseline > 0:
            ganho = ((precisao_sistema - precisao_baseline) / precisao_baseline) * 100
            
        results.append({
            "k": k,
            "precisao_sistema": precisao_sistema,
            "precisao_baseline": precisao_baseline,
            "ganho_relativo": ganho
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação do sistema de recomendação")
    parser.add_argument("--metadata", type=str, default="./output/metadata.json", help="Caminho do metadata.json")
    parser.add_argument("--k_values", type=str, default="1,3,5,10", help="Valores de K separados por vírgula")
    parser.add_argument("--output_dir", type=str, default="./results", help="Pasta de saída dos resultados")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    
    print("Carregando ground truth e checando subgêneros...")
    try:
        ground_truth = load_ground_truth(args.metadata)
    except ValueError as e:
        print(e)
        exit(1)
        
    print(f"Total de faixas válidas para avaliação: {len(ground_truth)}")
    
    client = get_client()
    collection = get_or_create_collection(client)
    
    df_results = run_full_evaluation(collection, ground_truth, k_values)
    
    csv_path = os.path.join(args.output_dir, "precisao_at_k.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")
    
    plt.figure(figsize=(10, 6))
    
    bar_width = 0.35
    index = np.arange(len(k_values))
    
    plt.bar(index, df_results["precisao_sistema"], bar_width, label="Sistema", color="#4C72B0")
    plt.bar(index + bar_width, df_results["precisao_baseline"], bar_width, label="Baseline Aleatória", color="#DD8452")
    
    plt.xlabel("K (Número de Recomendações)", fontsize=12)
    plt.ylabel("Precisão@K Média", fontsize=12)
    plt.title("Precisão@K: Sistema vs. Baseline Aleatória", fontsize=14)
    plt.xticks(index + bar_width / 2, k_values)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, row in df_results.iterrows():
        sys_val = row["precisao_sistema"]
        base_val = row["precisao_baseline"]
        
        plt.text(i, sys_val + 0.01, f"{sys_val:.2f}", ha='center', va='bottom', fontsize=10)
        plt.text(i + bar_width, base_val + 0.01, f"{base_val:.2f}", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    img_path = os.path.join(args.output_dir, "precisao_at_k.png")
    plt.savefig(img_path, dpi=300)
    print(f"Gráfico salvo em: {img_path}")
    
    print("\nResumo dos Resultados:")
    print(df_results.to_string(index=False))
    
    k_resumo = 5 if 5 in k_values else k_values[-1]
    resumo_row = df_results[df_results["k"] == k_resumo].iloc[0]
    
    print(f"\nPrecisão@{k_resumo} do sistema: {resumo_row['precisao_sistema']:.2f} vs baseline aleatória: {resumo_row['precisao_baseline']:.2f} — ganho de {resumo_row['ganho_relativo']:.0f}%")
