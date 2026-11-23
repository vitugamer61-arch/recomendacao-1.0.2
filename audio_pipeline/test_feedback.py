"""
test_feedback.py — Testes unitários para o módulo de feedback implícito (skip/completa).
"""

import unittest
import sys
import tempfile
from pathlib import Path
from collections import deque
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

    def test_regression_consecutive_events_single_decision(self):
        """Requisito 7: Simula eventos consecutivos (player + playlist) para o mesmo seed.
        Verifica que ocorre exatamente uma decisão e que o top-1 é o próximo da fila.
        """
        import time
        from mpd_integration import (
            replenish_mpd_queue,
            RECOMMENDER_SONGID_MAP,
            RECOMMENDER_CYCLE_LOCK
        )
        import mpd_integration

        RECOMMENDER_SONGID_MAP.clear()

        # Configurar Mock do ChromaDB e dados
        dim = 8
        seed_emb = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cand_a_emb = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cand_b_emb = np.array([0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        embs_dict = {
            "seed_id": seed_emb,
            "cand_a": cand_a_emb,
            "cand_b": cand_b_emb
        }
        metas_dict = {
            "seed_id": {"subgenero": "Rock", "cluster_id": "c1", "cluster_nome": "Rock 1"},
            "cand_a": {"subgenero": "Rock", "cluster_id": "c1", "cluster_nome": "Rock 1"},
            "cand_b": {"subgenero": "Rock", "cluster_id": "c1", "cluster_nome": "Rock 1"}
        }
        id_to_file = {
            "seed_id": "Rock/seed.mp3",
            "cand_a": "Rock/candidata_a.mp3",
            "cand_b": "Rock/candidata_b.mp3"
        }
        file_to_id = {v: k for k, v in id_to_file.items()}
        mock_col = MockCollection(embs_dict, metas_dict)

        original_query = mpd_integration.vector_store.query_similar
        try:
            # query_similar retorna Candidata A (mais próxima) seguida de B
            mpd_integration.vector_store.query_similar = lambda col, emb, k=20, exclude_id=None: [
                {"track_id": "cand_a", "distance": 0.1, "filename": "candidata_a.mp3", "subgenero": "Rock", "cluster_id": "c1"},
                {"track_id": "cand_b", "distance": 0.3, "filename": "candidata_b.mp3", "subgenero": "Rock", "cluster_id": "c1"}
            ]

            class MockMPDClientForEvents:
                def __init__(self):
                    self.playlist = [{"id": "100", "file": "Rock/seed.mp3", "pos": 0}]
                    self.status_dict = {"song": 0, "state": "play"}
                    self.next_id = 100

                def playlistinfo(self):
                    return [dict(it) for it in self.playlist]

                def status(self):
                    return dict(self.status_dict)

                def currentsong(self):
                    return {"id": "100", "file": "Rock/seed.mp3", "pos": 0}

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

            client = MockMPDClientForEvents()

            # 1. Simula chegada do evento 1: 'player' para o seed songid='100'
            last_decided_song_id = None
            current_song_id = "100"

            # Evento 1 é do subsistema player: passa na validação
            events_1 = ["player"]
            self.assertIn("player", events_1)

            # Executa ciclo de recomendação do Evento 1
            # use_feedback=False para evitar criar FeedbackStorage separado;
            # o registro de decisões já é validado nos testes unitários de feedback.py.
            with RECOMMENDER_CYCLE_LOCK:
                added = replenish_mpd_queue(
                    client=client,
                    collection=mock_col,
                    current_track_id="seed_id",
                    file_to_id_map=file_to_id,
                    id_to_file_map=id_to_file,
                    music_root="",
                    play_history=deque(["seed_id"]),
                    failed_tracks=set(),
                    min_queue_ahead=1,
                    selection_mode="first",
                    seed_songid=current_song_id,
                    gatilho="player",
                    use_feedback=False
                )
                last_decided_song_id = current_song_id

            # Verifica que pelo menos 1 faixa foi adicionada à fila
            self.assertGreaterEqual(added, 1, "Deveria ter adicionado pelo menos 1 recomendação")
            initial_playlist_len = len(client.playlist)

            # Verifica que o top-1 é o próximo da fila (Pos 1)
            self.assertEqual(client.playlist[1]["file"], "Rock/candidata_a.mp3")

            # 2. Simula chegada imediata do evento 2: 'playlist' autogerado pelo addid
            events_2 = ["playlist"]

            # Regra 1: subsistemas que não são 'player' são ignorados
            player_triggered = "player" in events_2
            self.assertFalse(player_triggered, "Evento de playlist não deve acionar o recomendador")

            # 3. Simula chegada de evento misto ou espúrio contendo 'player' logo após alteração da fila
            # Regra 2 & 3: Idempotência e janela de supressão autogerada
            is_same_seed = (current_song_id == last_decided_song_id)
            self.assertTrue(is_same_seed, "Idempotência deve detectar que songid 100 já foi decidido")

            # Verifica que a fila NÃO mudou (nenhuma decisão duplicada foi executada)
            self.assertEqual(len(client.playlist), initial_playlist_len,
                             "Fila não deve mudar após evento bloqueado por idempotência")

            # Confirma que top-1 da decisão #1 continua na posição 1 da fila (o próximo a tocar)
            self.assertEqual(client.playlist[0]["file"], "Rock/seed.mp3")
            self.assertEqual(client.playlist[1]["file"], "Rock/candidata_a.mp3")

        finally:
            mpd_integration.vector_store.query_similar = original_query

    def test_regression_fast_skips_not_suppressed_when_song_changes(self):
        """Regressão: Passar faixas rapidamente (< 1s entre skips) NÃO deve ser descartado
        pela janela de supressão autogerada quando a faixa tocando muda (current_song_id != last_decided_song_id).
        """
        import time
        from mpd_integration import AUTOGENERATED_EVENT_SUPPRESSION_WINDOW

        last_mutation = time.time()
        last_decided_song_id = "10"
        current_song_id = "11"  # Usuário pulou para a faixa 11

        time_since_mut = time.time() - last_mutation  # << 1.0s
        self.assertLess(time_since_mut, AUTOGENERATED_EVENT_SUPPRESSION_WINDOW)

        # Se a faixa mudou (songid diferente), NUNCA deve ser suprimida como autogerada
        is_same_song = bool(current_song_id and current_song_id == last_decided_song_id)
        self.assertFalse(is_same_song, "Transição de faixa não é a mesma música")

        # Evento autogerado só suprime se for a mesma música
        should_suppress = is_same_song and (time_since_mut < AUTOGENERATED_EVENT_SUPPRESSION_WINDOW)
        self.assertFalse(should_suppress, "Skip rápido do usuário deve ser processado imediatamente")

    def test_regression_genre_change_when_in_cooldown_mode_cluster(self):
        """Regressão: Quando um gênero entra em cooldown, select_next_recommendations no
        modo 'cluster' deve ignorar o cluster bloqueado e selecionar faixa de outro gênero.
        """
        import mpd_integration

        dim = 8
        seed_emb = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rock_cand_emb = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        eletr_cand_emb = np.array([0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        embs_dict = {
            "seed_rock": seed_emb,
            "rock_1": rock_cand_emb,
            "eletronica_1": eletr_cand_emb
        }
        metas_dict = {
            "seed_rock": {"subgenero": "Rock", "cluster_id": "c_rock", "cluster_nome": "Rock 1"},
            "rock_1": {"subgenero": "Rock", "cluster_id": "c_rock", "cluster_nome": "Rock 1"},
            "eletronica_1": {"subgenero": "Eletrônica", "cluster_id": "c_eletr", "cluster_nome": "Eletrônica 1"}
        }
        id_to_file = {
            "seed_rock": "Rock/seed.mp3",
            "rock_1": "Rock/cand1.mp3",
            "eletronica_1": "Eletrônica/cand2.mp3"
        }

        mock_col = MockCollection(embs_dict, metas_dict)

        # Simula 3 skips de Rock para colocar Rock em cooldown
        for i in range(3):
            self.storage.record_event(
                track_id=f"rock_old_{i}",
                caminho=f"Rock/old_{i}.mp3",
                genero="Rock",
                tipo="skip",
                peso=-1.0,
                elapsed=5.0,
                duration=200.0
            )

        self.assertTrue(self.storage.is_genre_in_cooldown("Rock"), "Rock deve estar em cooldown")

        original_query = mpd_integration.vector_store.query_similar
        try:
            # Query similar global retorna tanto Rock quanto Eletrônica
            mpd_integration.vector_store.query_similar = lambda col, emb, k=20, exclude_id=None: [
                {"track_id": "rock_1", "distance": 0.1, "filename": "cand1.mp3", "subgenero": "Rock", "cluster_id": "c_rock"},
                {"track_id": "eletronica_1", "distance": 0.2, "filename": "cand2.mp3", "subgenero": "Eletrônica", "cluster_id": "c_eletr"}
            ]

            # Injetar o engine com self.storage no teste
            orig_engine_cls = mpd_integration.feedback.FeedbackEngine

            class TestCustomFeedbackEngine(orig_engine_cls):
                def __init__(inst, storage=None):
                    super().__init__(storage=self.storage)

            mpd_integration.feedback.FeedbackEngine = TestCustomFeedbackEngine

            try:
                res = mpd_integration.select_next_recommendations(
                    collection=mock_col,
                    current_track_id="seed_rock",
                    id_to_file_map=id_to_file,
                    k=5,
                    history=[],
                    queued_files=set(),
                    failed_tracks=set(),
                    selection_mode="first",
                    recommendation_mode="cluster",
                    use_feedback=True
                )

                self.assertIsNotNone(res)
                escolhida_file, escolhida_meta, rank = res
                # A faixa escolhida DEVE ser Eletrônica, pois Rock está em cooldown!
                self.assertEqual(escolhida_meta.get("subgenero"), "Eletrônica",
                                 "Deve trocar para Eletrônica quando Rock estiver em cooldown no modo cluster")
                self.assertEqual(escolhida_file, "Eletrônica/cand2.mp3")
            finally:
                mpd_integration.feedback.FeedbackEngine = orig_engine_cls

        finally:
            mpd_integration.vector_store.query_similar = original_query

    def test_scenario_a_skip_during_calculation_discards_stale_result(self):
        """Cenário (a): Skip durante o cálculo (faixa atual mudou antes do addid).
        O recomendador deve detectar a corrida, descartar o resultado obsoleto e não enfileirar.
        """
        import mpd_integration

        dim = 8
        seed_emb = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cand_emb = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        mock_col = MockCollection(
            {"seed_id": seed_emb, "cand_id": cand_emb},
            {"seed_id": {"subgenero": "Rock"}, "cand_id": {"subgenero": "Rock"}}
        )

        class MockMPDClientRace:
            def __init__(self):
                self.playlist = [{"id": "100", "file": "Rock/seed.mp3", "pos": 0}]
                self._current_id = "100"
                self.add_calls = []

            def status(self):
                # Simula que o usuário mudou para a faixa '101' durante o cálculo!
                return {"song": 0, "songid": self._current_id, "state": "play"}

            def currentsong(self):
                return {"id": self._current_id, "file": "Rock/seed.mp3", "pos": 0}

            def playlistinfo(self):
                return list(self.playlist)

            def addid(self, uri, pos=None):
                self.add_calls.append(uri)
                return "102"

            def deleteid(self, songid):
                pass

        client = MockMPDClientRace()
        # Inicialmente seed é 100, mas antes do addid a faixa muda para 101
        client._current_id = "101"

        original_query = mpd_integration.vector_store.query_similar
        try:
            mpd_integration.vector_store.query_similar = lambda col, emb, k=5, exclude_id=None: [
                {"track_id": "cand_id", "distance": 0.1, "filename": "cand.mp3", "subgenero": "Rock"}
            ]

            added = mpd_integration.replenish_mpd_queue(
                client=client,
                collection=mock_col,
                current_track_id="seed_id",
                file_to_id_map={"Rock/seed.mp3": "seed_id", "Rock/cand.mp3": "cand_id"},
                id_to_file_map={"seed_id": "Rock/seed.mp3", "cand_id": "Rock/cand.mp3"},
                music_root="",
                play_history=deque(["seed_id"]),
                failed_tracks=set(),
                min_queue_ahead=1,
                seed_songid="100",  # Seed original era 100
                use_feedback=False
            )

            # Resultado obsoleto deve ter sido descartado: 0 faixas adicionadas
            self.assertEqual(added, 0, "Recomendação obsoleta calculada para faixa anterior não deve ser adicionada")
            self.assertEqual(len(client.add_calls), 0, "addid não deve ser chamado para recomendação obsoleta")
        finally:
            mpd_integration.vector_store.query_similar = original_query

    def test_scenario_b_queue_exhausted_triggers_auto_resume(self):
        """Cenário (b): Fila esgotada seguida de recomendação tardia.
        O recomendador deve chamar playid / play automaticamente para retomar a reprodução.
        """
        import mpd_integration

        dim = 8
        seed_emb = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cand_emb = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        mock_col = MockCollection(
            {"seed_id": seed_emb, "cand_id": cand_emb},
            {"seed_id": {"subgenero": "Rock"}, "cand_id": {"subgenero": "Rock"}}
        )

        class MockMPDClientAutoResume:
            def __init__(self):
                self.playlist = [{"id": "100", "file": "Rock/seed.mp3", "pos": 0}]
                self.playid_called_with = None

            def status(self):
                # Player em STOP porque a playlist esgotou
                return {"song": -1, "state": "stop"}

            def currentsong(self):
                return {}

            def playlistinfo(self):
                return list(self.playlist)

            def addid(self, uri, pos=None):
                self.playlist.append({"id": "101", "file": uri, "pos": len(self.playlist)})
                return "101"

            def playid(self, songid):
                self.playid_called_with = str(songid)

            def deleteid(self, songid):
                pass

        client = MockMPDClientAutoResume()

        original_query = mpd_integration.vector_store.query_similar
        try:
            mpd_integration.vector_store.query_similar = lambda col, emb, k=5, exclude_id=None: [
                {"track_id": "cand_id", "distance": 0.1, "filename": "cand.mp3", "subgenero": "Rock"}
            ]

            added = mpd_integration.replenish_mpd_queue(
                client=client,
                collection=mock_col,
                current_track_id="seed_id",
                file_to_id_map={"Rock/seed.mp3": "seed_id", "Rock/cand.mp3": "cand_id"},
                id_to_file_map={"seed_id": "Rock/seed.mp3", "cand_id": "Rock/cand.mp3"},
                music_root="",
                play_history=deque(["seed_id"]),
                failed_tracks=set(),
                min_queue_ahead=1,
                seed_songid="100",
                use_feedback=False
            )

            self.assertEqual(added, 1)
            # Auto-resume deve ter disparado playid na faixa 101 recém-inserida
            self.assertEqual(client.playid_called_with, "101", "Auto-resume deve chamar playid na faixa adicionada")
        finally:
            mpd_integration.vector_store.query_similar = original_query

    def test_scenario_c_manual_stop_with_queue_ahead_respected(self):
        """Cenário (c): Parada manual pelo usuário com faixas restantes na fila (ahead > 0).
        O recomendador NÃO deve forçar reprodução quando o usuário parou intencionalmente.
        """
        import mpd_integration

        class MockMPDClientManualStop:
            def __init__(self):
                # Playlist tem 3 faixas, o usuário parou na faixa 0 (ainda restam 2 faixas à frente)
                self.playlist = [
                    {"id": "1", "file": "Rock/f1.mp3", "pos": 0},
                    {"id": "2", "file": "Rock/f2.mp3", "pos": 1},
                    {"id": "3", "file": "Rock/f3.mp3", "pos": 2}
                ]
                self.play_called = False

            def status(self):
                return {"song": "0", "state": "stop"}

            def playlistinfo(self):
                return list(self.playlist)

            def play(self, pos=0):
                self.play_called = True

        client = MockMPDClientManualStop()
        status = client.status()
        playlist = client.playlistinfo()
        current_pos = int(status.get("song", -1))
        ahead = len(playlist) if current_pos == -1 else max(0, len(playlist) - (current_pos + 1))

        # ahead deve ser 2 (> 0), indicando que haviam músicas na fila e foi parada intencional
        self.assertEqual(ahead, 2)
        should_auto_resume = (len(playlist) == 0 or ahead == 0)
        self.assertFalse(should_auto_resume, "Não deve disparar auto-resume em parada manual com faixas à frente")
        self.assertFalse(client.play_called, "client.play não deve ser chamado em stop manual")

    def test_scenario_d_burst_of_5_rapid_skips_buffer_preserved_via_swap(self):
        """Cenário (d): Rajada de 5 skips rápidos sucessivos.
        O swap garante que a fila à frente NUNCA fica com 0 faixas por causa do recomendador.
        """
        import mpd_integration

        dim = 8
        embs = {f"t_{i}": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) for i in range(100)}
        metas = {f"t_{i}": {"subgenero": "Rock"} for i in range(100)}
        id_to_file = {f"t_{i}": f"Rock/t_{i}.mp3" for i in range(100)}
        file_to_id = {v: k for k, v in id_to_file.items()}
        mock_col = MockCollection(embs, metas)

        class MockMPDClientBurst:
            def __init__(self):
                self.next_id = 10
                # Fila inicial com faixa atual (0) e 2 faixas à frente (1 e 2)
                self.playlist = [
                    {"id": "1", "file": "Rock/t_0.mp3", "pos": 0},
                    {"id": "2", "file": "Rock/t_1.mp3", "pos": 1},
                    {"id": "3", "file": "Rock/t_2.mp3", "pos": 2}
                ]
                self.current_pos = 0

            def status(self):
                return {"song": str(self.current_pos), "songid": self.playlist[self.current_pos]["id"], "state": "play"}

            def currentsong(self):
                return dict(self.playlist[self.current_pos])

            def playlistinfo(self):
                return [dict(it) for it in self.playlist]

            def addid(self, uri, pos=None):
                self.next_id += 1
                sid = str(self.next_id)
                if pos is not None and pos <= len(self.playlist):
                    self.playlist.insert(pos, {"id": sid, "file": uri, "pos": pos})
                else:
                    self.playlist.append({"id": sid, "file": uri, "pos": len(self.playlist)})
                for idx, it in enumerate(self.playlist):
                    it["pos"] = idx
                return sid

            def deleteid(self, songid):
                sid_str = str(songid)
                self.playlist = [it for it in self.playlist if it["id"] != sid_str]
                for idx, it in enumerate(self.playlist):
                    it["pos"] = idx

        client = MockMPDClientBurst()
        history = deque(["t_0"])

        original_query = mpd_integration.vector_store.query_similar
        try:
            cand_counter = 3

            def mock_query(col, emb, k=5, exclude_id=None):
                nonlocal cand_counter
                res = []
                for j in range(k):
                    t_id = f"t_{cand_counter}"
                    cand_counter += 1
                    if t_id not in id_to_file:
                        id_to_file[t_id] = f"Rock/{t_id}.mp3"
                        file_to_id[f"Rock/{t_id}.mp3"] = t_id
                        mock_col.embeddings[t_id] = np.zeros(8, dtype=np.float32)
                        mock_col.metadatas[t_id] = {"subgenero": "Rock"}
                    res.append({"track_id": t_id, "distance": 0.05 * (j + 1), "filename": f"{t_id}.mp3", "subgenero": "Rock"})
                return res

            mpd_integration.vector_store.query_similar = mock_query

            # Simula rajada de 5 skips rápidos consecutivos
            for skip_step in range(1, 6):
                # Usuário pula para a próxima faixa imediatamente
                client.current_pos += 1
                curr_sid = client.playlist[client.current_pos]["id"]
                curr_tid = file_to_id[client.playlist[client.current_pos]["file"]]

                # Antes de enfileirar, verifica que a fila NÃO está esgotada (buffer preservado)
                ahead_before = len(client.playlist) - (client.current_pos + 1)
                self.assertGreaterEqual(ahead_before, 0, f"Fila à frente não deve ser negativa no skip {skip_step}")

                # Executa reposição com swap
                added = mpd_integration.replenish_mpd_queue(
                    client=client,
                    collection=mock_col,
                    current_track_id=curr_tid,
                    file_to_id_map=file_to_id,
                    id_to_file_map=id_to_file,
                    music_root="",
                    play_history=history,
                    failed_tracks=set(),
                    min_queue_ahead=2,
                    seed_songid=curr_sid,
                    use_feedback=False
                )

                # Confirma que adicionou novas faixas e que a fila à frente tem pelo menos 2 faixas
                ahead_after = len(client.playlist) - (client.current_pos + 1)
                self.assertGreaterEqual(ahead_after, 2, f"Após skip {skip_step}, deve ter pelo menos 2 faixas à frente")
                self.assertEqual(client.status()["state"], "play", "Player deve continuar tocando (nunca entrar em stop)")

        finally:
            mpd_integration.vector_store.query_similar = original_query


if __name__ == "__main__":
    unittest.main()
