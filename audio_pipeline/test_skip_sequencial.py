"""
test_skip_sequencial.py — Regressão do bug "a recomendação para com skips sequenciais".

Causas cobertas:
  1. Ramo `stop` do monitor: ao esgotar a fila (ex.: skips além da última faixa) o MPD vai para `stop`, REMOVE o
     campo `song` do status e mantém a playlist cheia. Isso era confundido com parada manual e a reposição nunca
     mais acontecia. Um stop manual, ao contrário, MANTÉM o campo `song`.
  2. select_next_recommendations devolvia None quando todas as candidatas eram rejeitadas (cooldown de gênero,
     skips recentes, histórico, fila). Agora a busca é ampliada em níveis e nunca deixa a fila sem recomendação.
"""

import os
import sys
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mpd_integration


# ── Dublê do MPD ─────────────────────────────────────────────────────────────

class FakeMPD:
    """MPD mínimo e roteirizado: cada chamada de idle() aplica o próximo passo do roteiro."""

    def __init__(self, script):
        self.script = list(script)
        self.state = {"state": "stop"}
        self.playlist = []
        self.played_positions = []

    def ping(self):
        return None

    def idle(self, *_args):
        if not self.script:
            raise KeyboardInterrupt()
        self.script.pop(0)(self)
        return ["player"]

    def status(self):
        st = dict(self.state)
        st["playlistlength"] = str(len(self.playlist))
        return st

    def currentsong(self):
        pos = self.state.get("song")
        return dict(self.playlist[int(pos)]) if pos is not None else {}

    def playlistinfo(self):
        return [dict(x) for x in self.playlist]

    def play(self, pos):
        self.played_positions.append(int(pos))
        self.state = {"state": "play", "song": str(pos)}


def _item(n):
    return {"file": f"rock/track_{n}.mp3", "id": str(100 + n), "pos": str(n)}


def step_playing(pos, n_items):
    """Usuário tocando a faixa `pos` de uma playlist com `n_items` faixas."""
    def _step(c):
        c.playlist = [_item(i) for i in range(n_items)]
        c.state = {"state": "play", "song": str(pos)}
    return _step


def step_ran_off_end(c):
    """Skip além da última faixa: stop, SEM `song`, playlist intacta."""
    c.state = {"state": "stop"}


def step_manual_stop(pos):
    """Stop manual: o MPD mantém `song`."""
    def _step(c):
        c.state = {"state": "stop", "song": str(pos)}
    return _step


class TestStopBranch(unittest.TestCase):

    def _run(self, script, preload=None):
        client = FakeMPD(script)
        if preload:
            client.playlist = [_item(i) for i in range(preload)]
        calls = {"n": 0}

        def fake_replenish(**kwargs):
            """Simula a reposição: acrescenta 2 faixas ao fim da playlist."""
            calls["n"] += 1
            base = len(client.playlist)
            client.playlist += [_item(base), _item(base + 1)]
            return 2

        ids = {f"rock/track_{i}.mp3": f"T{i}" for i in range(40)}
        with patch.object(mpd_integration, "replenish_mpd_queue", side_effect=fake_replenish), \
             patch.object(mpd_integration, "load_session_history", return_value=deque(["T0"], maxlen=30)), \
             patch.object(mpd_integration, "save_session_history"):
            with self.assertRaises(KeyboardInterrupt):
                mpd_integration.monitor_and_recommend(
                    client=client, collection=MagicMock(), file_to_id_map=ids,
                    id_to_file_map={v: k for k, v in ids.items()}, music_root="", use_feedback=False,
                )
        return client, calls["n"]

    def test_fim_da_fila_sem_song_repoe_e_retoma_na_primeira_faixa_nova(self):
        client, n_repl = self._run([step_playing(2, 3), step_ran_off_end])
        self.assertEqual(n_repl, 2, "deve repor no play e de novo no fim da fila")
        # Ao chegar o stop a playlist tem 5 faixas (3 + 2 da reposição do play): a primeira nova fica na posição 5.
        # Nunca a posição 0, que reiniciaria uma faixa antiga do histórico.
        self.assertEqual(client.played_positions, [5])

    def test_stop_manual_com_song_e_respeitado(self):
        client, n_repl = self._run([step_playing(1, 5), step_manual_stop(1)])
        self.assertEqual(n_repl, 1, "só a reposição do play; nada no stop manual")
        self.assertEqual(client.played_positions, [])

    def test_mpd_ocioso_sem_sessao_nao_e_sequestrado(self):
        """Stop sem `song` com playlist cheia, mas sem nunca ter tocado: não retoma sozinho."""
        client, n_repl = self._run([step_ran_off_end], preload=4)
        self.assertEqual(n_repl, 0)
        self.assertEqual(client.played_positions, [])

    def test_dois_fins_de_fila_seguidos_sem_ver_play_no_meio(self):
        """Skips ultrarrápidos: o MPD esgota a fila de novo antes de o monitor observar a faixa retomada."""
        client, n_repl = self._run([step_playing(2, 3), step_ran_off_end, step_ran_off_end])
        self.assertEqual(n_repl, 3)
        self.assertEqual(client.played_positions, [5, 7])

    def test_stop_manual_encerra_a_sessao(self):
        """Depois de um stop manual, um stop sem `song` não deve retomar sozinho."""
        client, n_repl = self._run([step_playing(1, 5), step_manual_stop(1), step_ran_off_end])
        self.assertEqual(n_repl, 1)
        self.assertEqual(client.played_positions, [])

    def test_fila_vazia_continua_sendo_reposta(self):
        """Comportamento antigo preservado: playlist vazia em stop dispara reposição e play na posição 0."""
        client, n_repl = self._run([lambda c: setattr(c, "state", {"state": "stop"})])
        self.assertEqual(n_repl, 1)
        self.assertEqual(client.played_positions, [0])


# ── Expansão da busca ────────────────────────────────────────────────────────

class FakeCollection:
    def __init__(self, n=10):
        self.n = n

    def count(self):
        return self.n

    def get(self, ids, include=None, where=None):
        return {
            "ids": list(ids),
            "embeddings": [[1.0, 0.0, 0.0] for _ in ids],
            "metadatas": [{"subgenero": "Rock"} for _ in ids],
        }


class FakeStorage:
    def is_genre_in_cooldown(self, genero, **_):
        return True

    def get_genre_penalties(self):
        return {}

    def record_decision(self, **_):
        return 1


class FakeEngine:
    """Motor de feedback que rejeita TUDO (cooldown total), como após vários skips do mesmo gênero."""

    def __init__(self, *_a, **_k):
        self.storage = FakeStorage()

    def compute_rocchio_query(self, query_emb, collection):
        return query_emb, {}

    def rerank_candidates(self, query_emb, candidates, collection, id_to_file_map, queued_files):
        for c in candidates:
            c["rejeitada"] = True
            c["motivo"] = "Rejeitada (cooldown)"
        return []


class TestExpansaoDeBusca(unittest.TestCase):

    def setUp(self):
        self.id_to_file = {x: f"rock/{x}.mp3" for x in "ABCDE"}

    def _select(self, **kw):
        base = dict(collection=FakeCollection(5), current_track_id="A", id_to_file_map=self.id_to_file,
                    history=deque(), selection_mode="first")
        base.update(kw)
        return mpd_integration.select_next_recommendations(**base)

    def test_amplia_a_busca_quando_as_candidatas_iniciais_estao_na_fila(self):
        """Nível 0 só traz B e C (já na fila); com a busca ampliada surge D."""
        def fake_query(col, emb, k=5, exclude_id=None):
            base = [{"track_id": "B", "distance": 0.05}, {"track_id": "C", "distance": 0.10}]
            return base + [{"track_id": "D", "distance": 0.2}] if k >= 100 else base

        queued = {"rock/B.mp3", "rock/C.mp3"}
        with patch.object(mpd_integration.vector_store, "query_similar", side_effect=fake_query):
            res = self._select(queued_files=queued, use_feedback=False)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], "rock/D.mp3")

    def test_cooldown_total_nao_deixa_a_fila_sem_recomendacao(self):
        """Todo o vizinhança rejeitada pelo feedback: o nível de emergência ainda devolve uma faixa."""
        def fake_query(col, emb, k=5, exclude_id=None):
            return [{"track_id": t, "distance": 0.1 * i} for i, t in enumerate("BCDE", start=1)]

        with patch.object(mpd_integration, "HAS_FEEDBACK", True), \
             patch.object(mpd_integration.feedback, "FeedbackEngine", FakeEngine), \
             patch.object(mpd_integration.vector_store, "query_similar", side_effect=fake_query):
            res = self._select(use_feedback=True, record_audit=False)
        self.assertIsNotNone(res, "sem candidatas não pode acontecer com a coleção ainda tendo faixas")
        self.assertEqual(res[0], "rock/B.mp3")

    def test_colecao_realmente_esgotada_devolve_none(self):
        """Se não existe nenhuma faixa elegível (tudo na fila/falha), None é o resultado correto."""
        def fake_query(col, emb, k=5, exclude_id=None):
            return [{"track_id": "B", "distance": 0.1}]

        with patch.object(mpd_integration.vector_store, "query_similar", side_effect=fake_query):
            res = self._select(use_feedback=False, failed_tracks={"B"})
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
