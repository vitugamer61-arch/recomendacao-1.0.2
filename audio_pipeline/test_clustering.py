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
    assign_cluster_incremental,
    determine_cluster_names_and_purity,
    sanitize_playlist_name,
    refine_impure_clusters,
    run_by_genre_mode
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

        # Monta o cluster_names_map e display_names como o pipeline real
        raw_cluster_ids = cluster_names
        cluster_names_map, cluster_stats = determine_cluster_names_and_purity(
            self.ids, self.metadatas, raw_cluster_ids
        )
        unique_cids = sorted(list(set(raw_cluster_ids)), key=lambda x: (x == "ruido", x))
        display_names = [cluster_names_map.get(cid, cid) for cid in unique_cids]

        model_payload = {
            "algorithm": "kmeans",
            "centroids": kmeans.cluster_centers_,
            "cluster_names": unique_cids,
            "cluster_display_names": display_names,
            "cluster_names_map": cluster_names_map,
            "pca_model": pca_model,
            "seed": 42
        }

        # Testa atribuição incremental para um novo ponto próximo ao primeiro cluster
        new_sample = self.X[0] + np.random.normal(0, 0.05, size=self.X.shape[1])
        result = assign_cluster_incremental(new_sample, model_payload)

        # Resultado é tupla (cluster_id, cluster_nome)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        assigned_cid, assigned_cnome = result
        self.assertIn(assigned_cid, unique_cids)
        # Deve receber o mesmo cluster da amostra próxima
        self.assertEqual(assigned_cid, cluster_names[0])
        # O nome de gênero deve corresponder ao mapeamento
        self.assertEqual(assigned_cnome, cluster_names_map.get(assigned_cid, assigned_cid))

    def test_hdbscan_noise_and_summary(self):
        norm_X, _ = preprocess_embeddings(self.X)
        hdb, labels, raw_cluster_ids = run_hdbscan(norm_X, min_cluster_size=10)

        cluster_names_map, cluster_stats = determine_cluster_names_and_purity(
            self.ids, self.metadatas, raw_cluster_ids
        )

        summary = compute_clusters_summary(
            ids=self.ids,
            metadatas=self.metadatas,
            processed_embeddings=norm_X,
            labels=labels,
            cluster_ids=raw_cluster_ids,
            cluster_names_map=cluster_names_map,
            cluster_stats=cluster_stats,
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
                # Verifica que o nome do cluster está preenchido
                self.assertTrue(len(c_data.get("nome", "")) > 0)

    def test_metadata_preservation_logic(self):
        # Simula teste de integridade de metadados
        current_meta = {
            "filename": "musica_teste.mp3",
            "subgenero": "Rock",
            "n_segments": 42,
            "fingerprint": "abc123hash"
        }
        new_cluster = "cluster_05"
        new_cluster_nome = "Rock"

        merged = dict(current_meta)
        merged["cluster_id"] = new_cluster
        merged["cluster_nome"] = new_cluster_nome

        # Garante que nenhum campo anterior foi corrompido ou apagado
        for key, val in current_meta.items():
            self.assertEqual(merged[key], val)
        self.assertEqual(merged["cluster_id"], new_cluster)
        self.assertEqual(merged["cluster_nome"], new_cluster_nome)

    def test_cluster_naming_by_genre(self):
        """Testa a nomenclatura de clusters por gênero predominante (voto majoritário)."""
        # 3 clusters: 0 = todo Rock, 1 = misto Pop/Eletrônica, 2 = todo Sertanejo
        ids = [f"id_{i}" for i in range(9)]
        metadatas = [
            {"subgenero": "Rock"}, {"subgenero": "Rock"}, {"subgenero": "Rock"},
            {"subgenero": "Pop"}, {"subgenero": "Eletrônica"}, {"subgenero": "Pop"},
            {"subgenero": "Sertanejo"}, {"subgenero": "Sertanejo"}, {"subgenero": "Sertanejo"},
        ]
        raw_cluster_ids = [
            "cluster_01", "cluster_01", "cluster_01",
            "cluster_02", "cluster_02", "cluster_02",
            "cluster_03", "cluster_03", "cluster_03",
        ]
        names_map, stats = determine_cluster_names_and_purity(ids, metadatas, raw_cluster_ids)

        # cluster_01 = 100% Rock → nome "Rock"
        self.assertEqual(names_map["cluster_01"], "Rock")
        self.assertAlmostEqual(stats["cluster_01"]["pureza"], 1.0)

        # cluster_02 = Pop 66%, Eletrônica 33% → nome "Pop" (> 50%)
        self.assertEqual(names_map["cluster_02"], "Pop")

        # cluster_03 = 100% Sertanejo → nome "Sertanejo"
        self.assertEqual(names_map["cluster_03"], "Sertanejo")

    def test_cluster_naming_conflict_resolution(self):
        """Testa desambiguação quando dois clusters têm o mesmo gênero predominante."""
        ids = [f"id_{i}" for i in range(6)]
        metadatas = [
            {"subgenero": "Rock"}, {"subgenero": "Rock"}, {"subgenero": "Blues"},
            {"subgenero": "Rock"}, {"subgenero": "Rock"}, {"subgenero": "Pop"},
        ]
        raw_cluster_ids = [
            "cluster_01", "cluster_01", "cluster_01",
            "cluster_02", "cluster_02", "cluster_02",
        ]
        names_map, stats = determine_cluster_names_and_purity(ids, metadatas, raw_cluster_ids)

        # Ambos são majoritariamente Rock, devem ser diferenciados por g2
        self.assertNotEqual(names_map["cluster_01"], names_map["cluster_02"])
        # Um deve ser "Rock / Blues" e outro "Rock / Pop"
        self.assertIn("Rock", names_map["cluster_01"])
        self.assertIn("Rock", names_map["cluster_02"])

    def test_cluster_naming_no_genre(self):
        """Testa clusters onde nenhuma faixa tem subgênero preenchido."""
        ids = ["id_0", "id_1"]
        metadatas = [{"subgenero": ""}, {"subgenero": ""}]
        raw_cluster_ids = ["cluster_01", "cluster_01"]
        names_map, stats = determine_cluster_names_and_purity(ids, metadatas, raw_cluster_ids)

        self.assertIn("Sem gênero", names_map["cluster_01"])

    def test_sanitize_playlist_name(self):
        """Testa sanitização de nomes com caracteres proibidos do Windows."""
        self.assertEqual(sanitize_playlist_name("Rock / Blues"), "Rock - Blues")
        self.assertEqual(sanitize_playlist_name('Pop "Remix"'), "Pop - Remix -")
        self.assertNotEqual(sanitize_playlist_name("Eletrônica"), "")
        self.assertEqual(sanitize_playlist_name(""), "Cluster")

    def test_cluster_naming_numeric_suffix_no_compound_names(self):
        """Testa que não são gerados nomes compostos com barra e repetições usam sufixo numérico."""
        ids = [f"id_{i}" for i in range(4)]
        metadatas = [
            {"subgenero": "Rock"},
            {"subgenero": "Rock"},
            {"subgenero": "Rock"},
            {"subgenero": "Rock"},
        ]
        raw_cluster_ids = ["cluster_01", "cluster_01", "cluster_02", "cluster_02"]
        names_map, stats = determine_cluster_names_and_purity(ids, metadatas, raw_cluster_ids)

        self.assertEqual(names_map["cluster_01"], "Rock 1")
        self.assertEqual(names_map["cluster_02"], "Rock 2")
        self.assertNotIn("/", names_map["cluster_01"])
        self.assertNotIn("/", names_map["cluster_02"])

    def test_refine_impure_clusters_splits_mixed(self):
        """Testa se o refinamento recursivo divide um cluster impuro com dois gêneros distintos."""
        # Cria dois blobs separados no mesmo cluster inicial
        X_sub1, _ = make_blobs(n_samples=25, n_features=16, centers=[[2.0]*16], cluster_std=0.2, random_state=42)
        X_sub2, _ = make_blobs(n_samples=25, n_features=16, centers=[[-2.0]*16], cluster_std=0.2, random_state=42)
        X_mixed = np.vstack([X_sub1, X_sub2])
        X_norm = normalize(X_mixed, norm="l2")

        ids = [f"mix_{i}" for i in range(50)]
        metadatas = [{"subgenero": "Rock"} for _ in range(25)] + [{"subgenero": "MPB"} for _ in range(25)]
        raw_cluster_ids = ["cluster_01"] * 50

        refined_ids, centroids, unique_finals = refine_impure_clusters(
            embeddings=X_norm,
            raw_cluster_ids=raw_cluster_ids,
            ids=ids,
            metadatas=metadatas,
            seed=42,
            pureza_minima=0.80,
            tamanho_minimo=15,
            profundidade_maxima=2
        )

        # O cluster impuro de 50 faixas deve ter sido dividido
        self.assertGreater(len(unique_finals), 1)
        names_map, stats = determine_cluster_names_and_purity(ids, metadatas, refined_ids)
        # Nomes devem conter os gêneros (Rock e MPB) separados
        g_names = list(names_map.values())
        self.assertTrue(any("Rock" in n for n in g_names))
        self.assertTrue(any("MPB" in n for n in g_names))

    def test_by_genre_mode(self):
        """Testa a geração de sub-clusters pelo modo por-genero."""
        X_blob, _ = make_blobs(n_samples=40, n_features=16, centers=2, cluster_std=0.3, random_state=42)
        X_norm = normalize(X_blob, norm="l2")

        ids = [f"pg_{i}" for i in range(40)]
        metadatas = [{"subgenero": "Rock"} for _ in range(40)]

        c_ids, centroids, unique_cids = run_by_genre_mode(
            embeddings=X_norm,
            ids=ids,
            metadatas=metadatas,
            seed=42,
            tamanho_minimo=15
        )

        self.assertEqual(len(c_ids), 40)
        self.assertGreaterEqual(len(unique_cids), 1)
        names_map, _ = determine_cluster_names_and_purity(ids, metadatas, c_ids)
        for c_nome in names_map.values():
            self.assertIn("Rock", c_nome)


if __name__ == "__main__":
    unittest.main()
