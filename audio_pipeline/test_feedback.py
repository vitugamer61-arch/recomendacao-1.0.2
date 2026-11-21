"""
test_feedback.py — Testes unitários para o módulo de feedback implícito (skip/completa).
"""

import unittest
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

import feedback_config as fcfg
from feedback import FeedbackStorage, FeedbackDetector, FeedbackEngine


class MockCollection:
    """Mock do ChromaDB para testes rápidos e isolados."""

    def __init__(self, embeddings_dict: dict = None, metadatas_dict: dict = None):
        self.embeddings = embeddings_dict or {}
        self.metadatas = metadatas_dict or {}

    def get(self, ids=None, where=None, include=None):
        if ids is None:
            ids = list(self.embeddings.keys())
        found_ids = [tid for tid in ids if tid in self.embeddings]
        found_embs = [self.embeddings[tid].tolist() if isinstance(self.embeddings[tid], np.ndarray) else self.embeddings[tid] for tid in found_ids]
        found_metas = [self.metadatas.get(tid, {}) for tid in found_ids]
        return {
            "ids": found_ids,
            "embeddings": found_embs,
            "metadatas": found_metas
        }


class TestFeedback(unittest.TestCase):

    def setUp(self):
        self.storage = FeedbackStorage(":memory:")
        self.detector = FeedbackDetector(self.storage)
        self.engine = FeedbackEngine(self.storage)

    def tearDown(self):
        self.storage.close()

    def test_skip_real_detection(self):
        """Skip manual detectado quando a música é pulada antes de min(30s, 30% da duração)."""
        prev_song = {"file": "Rock/musica1.mp3", "pos": 0, "duration": 200.0}
        prev_status = {"song": "0", "elapsed": 12.0, "duration": 200.0, "state": "play"}
        current_song = {"file": "Rock/musica2.mp3", "pos": 1}
        current_status = {"song": "1", "state": "play"}

        ev = self.detector.evaluate_transition(
            prev_song=prev_song,
            prev_status=prev_status,
            current_song=current_song,
            current_status=current_status
        )
        self.assertEqual(ev, "skip")

        # Verifica persistência no SQLite
        skips = self.storage.get_recent_skips()
        self.assertEqual(len(skips), 1)
        self.assertEqual(skips[0]["tipo"], "skip")
        self.assertEqual(skips[0]["elapsed"], 12.0)

    def test_natural_end_and_completed_detection(self):
        """Faixa ouvida até >= 80% é detectada como 'completa' e não como skip."""
        prev_song = {"file": "MPB/musica1.mp3", "pos": 0, "duration": 100.0}
        prev_status = {"song": "0", "elapsed": 85.0, "duration": 100.0, "state": "play"}
        current_song = {"file": "MPB/musica2.mp3", "pos": 1}
        current_status = {"song": "1", "state": "play"}

        ev = self.detector.evaluate_transition(
            prev_song=prev_song,
            prev_status=prev_status,
            current_song=current_song,
            current_status=current_status
        )
        self.assertEqual(ev, "completa")

        completed = self.storage.get_recent_completed()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["tipo"], "completa")

    def test_backwards_navigation_ignored(self):
        """Voltar para a faixa anterior (new_pos < old_pos) é ignorado e não conta como skip."""
        prev_song = {"file": "Pop/musica2.mp3", "pos": 5, "duration": 200.0}
        prev_status = {"song": "5", "elapsed": 10.0, "duration": 200.0, "state": "play"}
        current_song = {"file": "Pop/musica1.mp3", "pos": 4}
        current_status = {"song": "4", "state": "play"}

        ev = self.detector.evaluate_transition(
            prev_song=prev_song,
            prev_status=prev_status,
            current_song=current_song,
            current_status=current_status
        )
        self.assertIsNone(ev)
        self.assertEqual(len(self.storage.get_recent_events()), 0)

    def test_automated_trigger_ignored(self):
        """Trocas sinalizadas como disparadas pelo próprio recomendador não contam como skip."""
        prev_song = {"file": "Eletronica/musica1.mp3", "pos": 0, "duration": 200.0}
        prev_status = {"song": "0", "elapsed": 5.0, "duration": 200.0}
        current_song = {"file": "Eletronica/musica2.mp3", "pos": 1}
        current_status = {"song": "1"}

        ev = self.detector.evaluate_transition(
            prev_song=prev_song,
            prev_status=prev_status,
            current_song=current_song,
            current_status=current_status,
            is_automated_trigger=True
        )
        self.assertIsNone(ev)
        self.assertEqual(len(self.storage.get_recent_events()), 0)

    def test_rocchio_normalization_and_repulsion(self):
        """Valida que q' gerado pelo Rocchio é unitário (L2=1.0) e se afasta das faixas puladas."""
        dim = 16
        np.random.seed(42)
        q = np.random.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)

        # Vetor negativo (pulado) muito próximo de q
        neg_vec = q + np.random.normal(0, 0.05, size=dim).astype(np.float32)
        neg_vec /= np.linalg.norm(neg_vec)

        mock_col = MockCollection(
            embeddings_dict={"neg_01": neg_vec},
            metadatas_dict={"neg_01": {"subgenero": "Funk"}}
        )

        # Sem histórico, q' deve ser idêntico a q
        q_clean, stats_clean = self.engine.compute_rocchio_query(q, mock_col)
        self.assertFalse(stats_clean["applied"])
        np.testing.assert_allclose(np.linalg.norm(q_clean), 1.0, atol=1e-5)
        np.testing.assert_allclose(q_clean, q, atol=1e-5)

        # Insere skip da faixa negativa
        self.storage.record_event("neg_01", "Funk/neg.mp3", "Funk", "c_01", "skip", 10.0, 180.0)

        q_prime, stats_prime = self.engine.compute_rocchio_query(q, mock_col)
        self.assertTrue(stats_prime["applied"])

        # 1. Norma L2 deve ser estritamente 1.0
        norm_prime = float(np.linalg.norm(q_prime))
        self.assertAlmostEqual(norm_prime, 1.0, places=5)

        # 2. Similaridade com o negativo deve ter diminuído (repulsão)
        sim_original = float(np.dot(q, neg_vec))
        sim_prime = float(np.dot(q_prime, neg_vec))
        self.assertLess(sim_prime, sim_original)

    def test_genre_cooldown_and_recovery(self):
        """Valida bloqueio por cooldown após 3 skips do mesmo gênero e recuperação após faixa completa."""
        genero = "Funk"

        # 1. Menos de 3 skips: sem cooldown
        self.storage.record_event("f1", "Funk/1.mp3", genero, "c_01", "skip", 10.0, 180.0)
        self.storage.record_event("f2", "Funk/2.mp3", genero, "c_01", "skip", 8.0, 180.0)
        self.assertFalse(self.storage.is_genre_in_cooldown(genero))

        # 2. Terceiro skip consecutivo: entra em cooldown
        self.storage.record_event("f3", "Funk/3.mp3", genero, "c_01", "skip", 5.0, 180.0)
        self.assertTrue(self.storage.is_genre_in_cooldown(genero))

        # 3. Ouvir uma faixa completa interrompe a rejeição contínua
        self.storage.record_event("f4", "Funk/4.mp3", genero, "c_01", "completa", 175.0, 180.0)
        self.assertFalse(self.storage.is_genre_in_cooldown(genero))

    def test_rerank_penalties(self):
        """Valida que o re-ranking penaliza faixas similares às puladas e faixas de gêneros rejeitados."""
        dim = 8
        q = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Faixa pulada
        skip_emb = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        skip_emb /= np.linalg.norm(skip_emb)
        self.storage.record_event("cand_bad", "Rock/bad.mp3", "Rock", "c_01", "skip", 5.0, 180.0)

        # Candidatas: cand_bad (pulada recentemente), cand_similar_to_bad, cand_neutral
        cand_similar = np.array([0.85, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cand_similar /= np.linalg.norm(cand_similar)

        cand_neutral = np.array([0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cand_neutral /= np.linalg.norm(cand_neutral)

        mock_col = MockCollection(
            embeddings_dict={
                "cand_bad": skip_emb,
                "cand_similar": cand_similar,
                "cand_neutral": cand_neutral
            }
        )

        candidates = [
            {"track_id": "cand_bad", "subgenero": "Rock"},
            {"track_id": "cand_similar", "subgenero": "Rock"},
            {"track_id": "cand_neutral", "subgenero": "Jazz"}
        ]

        reranked = self.engine.rerank_candidates(
            query_emb=q,
            candidates=candidates,
            collection=mock_col,
            id_to_file_map={"cand_bad": "b.mp3", "cand_similar": "s.mp3", "cand_neutral": "n.mp3"},
            queued_files=set()
        )

        # cand_bad deve ser excluída por ser um skip recente
        reranked_ids = [c["track_id"] for c in reranked]
        self.assertNotIn("cand_bad", reranked_ids)

        # cand_neutral (sem penalidade de gênero Rock nem proximidade acústica ao skip) fica acima
        self.assertEqual(reranked[0]["track_id"], "cand_neutral")

    def test_feedback_gradual_weights_and_categories(self):
        """Valida a classificação do feedback gradual: skip (<30s), parcial (30s-80%) e completa (>=80%)."""
        # 1. Skip precoce (< 30s)
        ev_skip = self.detector.evaluate_transition(
            prev_song={"file": "Sertanejo/1.mp3", "pos": 0, "duration": 180.0},
            prev_status={"song": "0", "elapsed": 15.0, "duration": 180.0, "state": "play"},
            current_song={"file": "Sertanejo/2.mp3", "pos": 1},
            current_status={"song": "1", "state": "play"}
        )
        self.assertEqual(ev_skip, "skip")

        # 2. Reprodução parcial (30s até 80%)
        ev_parcial = self.detector.evaluate_transition(
            prev_song={"file": "Sertanejo/2.mp3", "pos": 1, "duration": 180.0},
            prev_status={"song": "1", "elapsed": 45.0, "duration": 180.0, "state": "play"},
            current_song={"file": "Sertanejo/3.mp3", "pos": 2},
            current_status={"song": "2", "state": "play"}
        )
        self.assertEqual(ev_parcial, "parcial")

        # 3. Reprodução completa (>= 80%)
        ev_completa = self.detector.evaluate_transition(
            prev_song={"file": "Sertanejo/3.mp3", "pos": 2, "duration": 180.0},
            prev_status={"song": "2", "elapsed": 160.0, "duration": 180.0, "state": "play"},
            current_song={"file": "Sertanejo/4.mp3", "pos": 3},
            current_status={"song": "3", "state": "play"}
        )
        self.assertEqual(ev_completa, "completa")

        # Verifica que parciais e completas são positivas
        positives = self.storage.get_recent_positives()
        self.assertEqual(len(positives), 2)
        tipos_pos = [p["tipo"] for p in positives]
        self.assertIn("completa", tipos_pos)
        self.assertIn("parcial", tipos_pos)

    def test_regression_play_30s_majority_genre(self):
        """
        Teste de Regressão (Requisito 3):
        Com estado zerado, tocar 30 s de uma faixa do gênero X (ex.: Sertanejo).
        Verifica que a maioria (>= 70%) das próximas K=5 recomendadas é do gênero X.
        """
        self.storage.clear_history()
        genero_alvo = "Sertanejo"

        # Simula biblioteca com clusters de gêneros
        dim = 8
        base_sertanejo = np.array([1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base_sertanejo /= np.linalg.norm(base_sertanejo)

        base_mpb = np.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base_mpb /= np.linalg.norm(base_mpb)

        # Faixa tocada por 30 segundos
        seed_id = "sert_01"
        seed_emb = base_sertanejo.copy()

        # Faixas candidatas no acervo
        embs_dict = {seed_id: seed_emb}
        metas_dict = {seed_id: {"subgenero": genero_alvo, "cluster_id": "c_sert"}}
        id_to_file = {seed_id: "Sertanejo/faixa_01.mp3"}

        candidatas = []
        # 6 faixas de Sertanejo
        for i in range(2, 8):
            tid = f"sert_{i:02d}"
            noise = np.random.normal(0, 0.05, size=dim).astype(np.float32)
            v = base_sertanejo + noise
            v /= np.linalg.norm(v)
            embs_dict[tid] = v
            metas_dict[tid] = {"subgenero": genero_alvo, "cluster_id": "c_sert"}
            id_to_file[tid] = f"Sertanejo/faixa_{i:02d}.mp3"
            sim = float(np.dot(seed_emb, v))
            candidatas.append({"track_id": tid, "subgenero": genero_alvo, "cluster_id": "c_sert", "distance": 1.0 - sim})

        # 4 faixas de MPB
        for i in range(1, 5):
            tid = f"mpb_{i:02d}"
            noise = np.random.normal(0, 0.05, size=dim).astype(np.float32)
            v = base_mpb + noise
            v /= np.linalg.norm(v)
            embs_dict[tid] = v
            metas_dict[tid] = {"subgenero": "MPB", "cluster_id": "c_mpb"}
            id_to_file[tid] = f"MPB/faixa_{i:02d}.mp3"
            sim = float(np.dot(seed_emb, v))
            candidatas.append({"track_id": tid, "subgenero": "MPB", "cluster_id": "c_mpb", "distance": 1.0 - sim})

        mock_col = MockCollection(embs_dict, metas_dict)

        # Tocar 30 segundos da faixa seed de Sertanejo
        ev = self.detector.evaluate_transition(
            prev_song={"file": "Sertanejo/faixa_01.mp3", "pos": 0, "duration": 180.0},
            prev_status={"song": "0", "elapsed": 30.0, "duration": 180.0, "state": "play"},
            current_song={"file": "Sertanejo/faixa_02.mp3", "pos": 1},
            current_status={"song": "1", "state": "play"},
            meta_resolver=lambda f: {"track_id": seed_id, "subgenero": genero_alvo, "cluster_id": "c_sert"}
        )
        self.assertEqual(ev, "parcial")

        # Calcula consulta Rocchio e re-ranking partindo da faixa tocando
        q_prime, _ = self.engine.compute_rocchio_query(seed_emb, mock_col)
        reranked = self.engine.rerank_candidates(
            query_emb=q_prime,
            candidates=candidatas,
            collection=mock_col,
            id_to_file_map=id_to_file,
            queued_files={"Sertanejo/faixa_01.mp3"}
        )

        top_k = reranked[:5]
        sertanejo_count = sum(1 for c in top_k if c.get("subgenero") == genero_alvo)
        fracao = sertanejo_count / 5.0

        # Verifica se pelo menos 70% (4 de 5 ou 100%) das recomendadas são do gênero alvo
        self.assertGreaterEqual(fracao, 0.70, f"Fração de {genero_alvo} foi {fracao:.1%}, esperava >= 70%")


if __name__ == "__main__":
    unittest.main()
