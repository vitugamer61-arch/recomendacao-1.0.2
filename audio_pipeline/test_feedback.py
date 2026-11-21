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

    def test_path_normalization_unquote_and_suffixes(self):
        """Verifica unquote e remoção de sufixos (&time=..., ?cantata=...) após extensão de áudio."""
        from feedback import clean_mpd_filepath
        from mpd_integration import normalize_mpd_filepath

        # 1. Cantata HTTP query string
        url1 = "http://localhost:6600/?file=Rock%2Fbanda%201%2Fmusica1.mp3&cantata=1&time=120&id=42"
        self.assertEqual(clean_mpd_filepath(url1), "Rock/banda 1/musica1.mp3")
        self.assertEqual(normalize_mpd_filepath(url1), "Rock/banda 1/musica1.mp3")

        # 2. Caminho relativo com sufixos MPD/Cantata
        raw2 = "MPB/Chico%20Buarque%20-%20Construcao.flac&time=180&id=99&cantata=1"
        self.assertEqual(clean_mpd_filepath(raw2), "MPB/Chico Buarque - Construcao.flac")
        self.assertEqual(normalize_mpd_filepath(raw2), "MPB/Chico Buarque - Construcao.flac")

        # 3. Caminho com extensão wav e query param
        raw3 = "Sertanejo/Desbloqueado.wav?id=12"
        self.assertEqual(clean_mpd_filepath(raw3), "Sertanejo/Desbloqueado.wav")
        self.assertEqual(normalize_mpd_filepath(raw3), "Sertanejo/Desbloqueado.wav")

    def test_regression_queue_ownership_and_purge_on_seed_change(self):
        """Teste de regressão: sobras de decisão antiga são purgadas na troca de seed e top-1 fica em primeiro."""
        from mpd_integration import (
            mpd_add_recommendation_unified,
            mpd_purge_pending_recommendations,
            RECOMMENDER_SONGID_MAP
        )

        RECOMMENDER_SONGID_MAP.clear()

        class MockMPDClient:
            def __init__(self):
                self.playlist = []
                self.status_dict = {"state": "play", "song": 0}
                self.next_id = 100

            def playlistinfo(self):
                return [dict(it) for it in self.playlist]

            def status(self):
                return dict(self.status_dict)

            def currentsong(self):
                pos = int(self.status_dict.get("song", 0))
                for it in self.playlist:
                    if int(it.get("pos", -1)) == pos:
                        return dict(it)
                return {}

            def addid(self, uri, pos=None):
                self.next_id += 1
                sid = str(self.next_id)
                if pos is not None and pos < len(self.playlist):
                    self.playlist.insert(pos, {"id": sid, "file": uri, "pos": pos})
                    for idx, it in enumerate(self.playlist):
                        it["pos"] = idx
                else:
                    pos = len(self.playlist)
                    self.playlist.append({"id": sid, "file": uri, "pos": pos})
                return sid

            def deleteid(self, songid):
                sid_str = str(songid)
                self.playlist = [it for it in self.playlist if it["id"] != sid_str]
                for idx, it in enumerate(self.playlist):
                    it["pos"] = idx

        client = MockMPDClient()

        # Cenário:
        # Faixa 0: Tocando (Manual do usuário)
        # Faixas 1 e 2: Sobras da Decisão Antiga #1
        # Faixa 3: Manual do usuário aguardando na fila
        client.playlist = [
            {"id": "1", "file": "Gospel/Renato Vianna - 400 Primaveras.mp3", "pos": 0},
            {"id": "2", "file": "Sertanejo/Diego & Victor Hugo - Desbloqueado.mp3", "pos": 1},
            {"id": "3", "file": "Sertanejo/Outra_Sobra_Antiga.mp3", "pos": 2},
            {"id": "4", "file": "MPB/Manual_Chico_Buarque.mp3", "pos": 3}
        ]

        # Registra propriedade das faixas antigas (Decisão #1)
        self.storage.record_song_ownership(songid="2", decision_id=1, origem="recomendada", caminho=client.playlist[1]["file"])
        self.storage.record_song_ownership(songid="3", decision_id=1, origem="exploracao", caminho=client.playlist[2]["file"])
        RECOMMENDER_SONGID_MAP["2"] = {"decision_id": 1, "origem": "recomendada", "file": client.playlist[1]["file"]}
        RECOMMENDER_SONGID_MAP["3"] = {"decision_id": 1, "origem": "exploracao", "file": client.playlist[2]["file"]}

        # 1. Usuário troca manualmente para outra faixa ou pula
        # A faixa atual no player passa a ser a pos 0
        client.status_dict["song"] = 0

        # Purga faixas pendentes da decisão antiga
        purged = mpd_purge_pending_recommendations(
            client=client,
            current_pos=0,
            reason="troca de seed / faixa alterada",
            feedback_storage=self.storage
        )

        self.assertEqual(purged, 2, "Deveria ter purgado exatamente as 2 sobras antigas")

        # Verifica que sobrou apenas a faixa tocando (pos 0) e a manual do usuário
        files_restantes = [it["file"] for it in client.playlist]
        self.assertIn("Gospel/Renato Vianna - 400 Primaveras.mp3", files_restantes)
        self.assertIn("MPB/Manual_Chico_Buarque.mp3", files_restantes)
        self.assertNotIn("Sertanejo/Diego & Victor Hugo - Desbloqueado.mp3", files_restantes)
        self.assertNotIn("Sertanejo/Outra_Sobra_Antiga.mp3", files_restantes)

        # 2. Agora o recomendador gera a Decisão #2 e adiciona o top-1 logo após a faixa atual
        current_pos = int(client.status().get("song", 0))
        top1_file = "Rock/3 Doors Down - Here Without You.mp3"
        insert_pos = current_pos + 1

        top1_sid = mpd_add_recommendation_unified(
            client=client,
            file_path=top1_file,
            position=insert_pos,
            decision_id=2,
            origem="recomendada",
            feedback_storage=self.storage
        )

        # 3. Verifica a ordem da fila:
        # Pos 0: Faixa atual
        # Pos 1: Top-1 da Decisão Atual (Here Without You)
        # Pos 2: Faixa manual do usuário (empurrada para trás sem ser deletada)
        self.assertEqual(client.playlist[0]["file"], "Gospel/Renato Vianna - 400 Primaveras.mp3")
        self.assertEqual(client.playlist[1]["file"], "Rock/3 Doors Down - Here Without You.mp3")
        self.assertEqual(client.playlist[2]["file"], "MPB/Manual_Chico_Buarque.mp3")

        # Ownership do top-1 devidamente registrado com a nova decisão #2
        own = self.storage.get_song_ownership(top1_sid)
        self.assertIsNotNone(own)
        self.assertEqual(own["decision_id"], 2)
        self.assertEqual(own["origem"], "recomendada")

    def test_sync_mpd_queue_state_statuses(self):
        """Verifica status Tocada, Tocando e Na fila na sincronização com feedback.db."""
        from mpd_integration import sync_mpd_queue_state, RECOMMENDER_SONGID_MAP

        RECOMMENDER_SONGID_MAP.clear()

        class MockMPDClient:
            def playlistinfo(self):
                return [
                    {"id": "1", "file": "musica_ja_tocada.mp3", "pos": 0},
                    {"id": "2", "file": "musica_tocando_agora.mp3", "pos": 1},
                    {"id": "3", "file": "musica_na_fila_rec.mp3", "pos": 2},
                    {"id": "4", "file": "musica_na_fila_manual.mp3", "pos": 3}
                ]
            def status(self):
                return {"song": 1, "state": "play"}
            def currentsong(self):
                return {"id": "2", "file": "musica_tocando_agora.mp3", "pos": 1}

        self.storage.record_song_ownership(songid="3", decision_id=10, origem="recomendada", caminho="musica_na_fila_rec.mp3")
        RECOMMENDER_SONGID_MAP["3"] = {"decision_id": 10, "origem": "recomendada", "file": "musica_na_fila_rec.mp3"}

        client = MockMPDClient()
        file_to_id = {"musica_tocando_agora.mp3": "id_2"}
        mock_col = MockCollection()

        sync_mpd_queue_state(client, file_to_id, mock_col, "", self.storage)

        q_state = self.storage.get_queue_state()
        self.assertEqual(len(q_state), 4)

        # Pos 0: Tocada
        self.assertEqual(q_state[0]["status_fila"], "Tocada")
        self.assertFalse(q_state[0]["is_current"])

        # Pos 1: Tocando
        self.assertEqual(q_state[1]["status_fila"], "Tocando")
        self.assertTrue(q_state[1]["is_current"])

        # Pos 2: Na fila (origem recomendada com decision_id)
        self.assertEqual(q_state[2]["status_fila"], "Na fila")
        self.assertEqual(q_state[2]["origem"], "recomendada")
        self.assertEqual(q_state[2]["decision_id"], 10)

        # Pos 3: Na fila (origem manual)
        self.assertEqual(q_state[3]["status_fila"], "Na fila")
        self.assertEqual(q_state[3]["origem"], "manual")
        self.assertIsNone(q_state[3]["decision_id"])


if __name__ == "__main__":
    unittest.main()
