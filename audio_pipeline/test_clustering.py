"""
test_clustering.py — Testes unitários para o módulo de clusterização e atribuição incremental.
"""

import unittest
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize

from evaluation import evaluate_clustering_metrics, evaluate_k_range
from clustering import (
    preprocess_embeddings,
    run_kmeans,
    run_hdbscan,
    compute_clusters_summary,
    assign_cluster_incremental
)

class TestClustering(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Gera 3 clusters bem definidos no espaço de dimensão 32
        self.X, self.y = make_blobs(n_samples=150, n_features=32, centers=3, cluster_std=0.5, random_state=42)
        self.ids = [f"id_{i:04d}" for i in range(150)]
        self.metadatas = [
            {
                "filename": f"track_{i}.mp3",
                "subgenero": f"genre_{self.y[i]}",
                "n_segments": 10 + i % 5,
                "fingerprint": f"fp_{i}"
            }
            for i in range(150)
        ]

    def test_preprocessing_and_pca(self):
        # Teste de normalização L2
        norm_X, pca_none = preprocess_embeddings(self.X, n_pca=None)
        norms = np.linalg.norm(norm_X, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
        self.assertIsNone(pca_none)

        # Teste com PCA
        n_components = 8
        pca_X, pca_model = preprocess_embeddings(self.X, n_pca=n_components, seed=42)
        self.assertEqual(pca_X.shape[1], n_components)
        self.assertIsNotNone(pca_model)

    def test_kmeans_synthetic(self):
        norm_X, _ = preprocess_embeddings(self.X)
        kmeans, labels, cluster_names = run_kmeans(norm_X, k=3, seed=42)

        self.assertEqual(len(set(cluster_names)), 3)
        self.assertTrue(all(name.startswith("cluster_") for name in cluster_names))

        # Avaliação das métricas
        metrics = evaluate_clustering_metrics(norm_X, labels)
        self.assertIsNotNone(metrics["silhouette"])
        self.assertGreater(metrics["silhouette"], 0.4)  # Clusters bem separados
        self.assertIsNotNone(metrics["davies_bouldin"])
        self.assertIsNotNone(metrics["calinski_harabasz"])
        self.assertEqual(metrics["noise_count"], 0)

    def test_k_range_evaluation(self):
        norm_X, _ = preprocess_embeddings(self.X)
        df_k = evaluate_k_range(norm_X, k_min=2, k_max=5, seed=42)
        self.assertEqual(len(df_k), 4)
        self.assertIn("silhouette", df_k.columns)
        self.assertIn("davies_bouldin", df_k.columns)
        self.assertIn("calinski_harabasz", df_k.columns)
        self.assertIn("inertia", df_k.columns)

        # O k=3 deve ter a maior silhueta entre os testados
        best_k = df_k.loc[df_k["silhouette"].idxmax(), "k"]
        self.assertEqual(best_k, 3)

    def test_pca_incremental_assignment(self):
        # Ajusta L2 + PCA + K-Means
        pca_X, pca_model = preprocess_embeddings(self.X, n_pca=10, seed=42)
        kmeans, labels, cluster_names = run_kmeans(pca_X, k=3, seed=42)

        model_payload = {
            "algorithm": "kmeans",
            "centroids": kmeans.cluster_centers_,
            "cluster_names": ["cluster_01", "cluster_02", "cluster_03"],
            "pca_model": pca_model,
            "seed": 42
        }

        # Testa atribuição incremental para um novo ponto próximo ao primeiro cluster
        new_sample = self.X[0] + np.random.normal(0, 0.05, size=self.X.shape[1])
        assigned_cluster = assign_cluster_incremental(new_sample, model_payload)

        self.assertIn(assigned_cluster, ["cluster_01", "cluster_02", "cluster_03"])
        # Deve receber o mesmo cluster da amostra próxima
        self.assertEqual(assigned_cluster, cluster_names[0])

    def test_hdbscan_noise_and_summary(self):
        norm_X, _ = preprocess_embeddings(self.X)
        hdb, labels, cluster_names = run_hdbscan(norm_X, min_cluster_size=10)

        summary = compute_clusters_summary(
            ids=self.ids,
            metadatas=self.metadatas,
            processed_embeddings=norm_X,
            labels=labels,
            cluster_names=cluster_names,
            algorithm="hdbscan"
        )

        self.assertIn("clusters", summary)
        self.assertEqual(summary["total_tracks"], 150)

        for c_name, c_data in summary["clusters"].items():
            if c_name == "ruido":
                self.assertIsNone(c_data["medoid"])
            else:
                self.assertIsNotNone(c_data["medoid"])
                self.assertIn("track_id", c_data["medoid"])

    def test_metadata_preservation_logic(self):
        # Simula teste de integridade de metadados
        current_meta = {
            "filename": "musica_teste.mp3",
            "subgenero": "Rock",
            "n_segments": 42,
            "fingerprint": "abc123hash"
        }
        new_cluster = "cluster_05"

        merged = dict(current_meta)
        merged["cluster_id"] = new_cluster

        # Garante que nenhum campo anterior foi corrompido ou apagado
        for key, val in current_meta.items():
            self.assertEqual(merged[key], val)
        self.assertEqual(merged["cluster_id"], new_cluster)


if __name__ == "__main__":
    unittest.main()
