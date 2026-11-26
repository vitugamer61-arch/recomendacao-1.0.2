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
import sqlite3
import sys
import tempfile
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
        self.random_on = False
        self.random_calls = []
        self.extra = {}

    def random(self, valor):
        self.random_calls.append(int(valor))
        self.random_on = bool(int(valor))

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
        st["random"] = "1" if self.random_on else "0"
        st.update(self.extra)
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


# ── Lentidão: cooldown por gênero, nível de busca pegajoso e partida ─────────────

class TestCooldownCacheado(unittest.TestCase):
    """Cada is_genre_in_cooldown abre uma conexão SQLite (cara no Windows): deve ser 1 por gênero, não 1 por candidata."""

    def test_rerank_consulta_o_cooldown_uma_vez_por_genero(self):
        import feedback

        class CountingStorage:
            def __init__(self):
                self.calls = 0

            def get_recent_skips(self, limit=10):
                return []

            def get_genre_penalties(self):
                return {}

            def is_genre_in_cooldown(self, genero, **_):
                self.calls += 1
                return genero == "Rock"

        class Col:
            def get(self, ids, include=None, where=None):
                return {"ids": list(ids), "embeddings": [[1.0, 0.0, 0.0] for _ in ids], "metadatas": [{} for _ in ids]}

        storage = CountingStorage()
        engine = feedback.FeedbackEngine(storage=storage)
        generos = ["Rock", "Funk", "MPB"]
        cands = [{"track_id": f"T{i}", "distance": 0.1, "subgenero": generos[i % 3]} for i in range(300)]
        out = engine.rerank_candidates(np.array([1.0, 0.0, 0.0], dtype=np.float32), cands, Col(), {}, set())
        self.assertEqual(storage.calls, 3, "uma consulta por gênero, e não 300")
        self.assertTrue(all(c.get("subgenero") != "Rock" for c in out), "o gênero em cooldown continua sendo rejeitado")
        self.assertEqual(len(out), 200)


class TestMigracaoCheckAntigo(unittest.TestCase):
    """Bancos antigos têm CHECK(tipo IN ('skip','completa')): eventos 'parcial' eram descartados em silêncio."""

    def _banco_antigo(self, com_peso):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "feedback.db")
        conn = sqlite3.connect(path)
        peso_col = "peso REAL DEFAULT 1.0," if com_peso else ""
        conn.execute(f"""CREATE TABLE feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            track_id TEXT NOT NULL, caminho TEXT, genero TEXT, cluster_id TEXT,
            tipo TEXT NOT NULL CHECK(tipo IN ('skip', 'completa')), {peso_col} elapsed REAL, duration REAL)""")
        conn.execute("INSERT INTO feedback_events (track_id, genero, tipo, elapsed, duration) VALUES ('A','Rock','skip',1,180)")
        conn.execute("INSERT INTO feedback_events (track_id, genero, tipo, elapsed, duration) VALUES ('B','Funk','completa',170,180)")
        conn.commit()
        conn.close()
        return path

    def test_migra_preserva_eventos_e_aceita_parcial(self):
        import feedback
        path = self._banco_antigo(com_peso=True)
        st = feedback.FeedbackStorage(db_path=path)
        st.record_event(track_id="C", genero="MPB", tipo="parcial", peso=0.5, elapsed=60, duration=180)  # antes: CHECK failed
        conn = sqlite3.connect(path)
        tipos = [r[0] for r in conn.execute("SELECT tipo FROM feedback_events ORDER BY id")]
        conn.close()
        self.assertEqual(tipos, ["skip", "completa", "parcial"])

    def test_migra_banco_sem_coluna_peso_com_sinal_correto(self):
        import feedback
        path = self._banco_antigo(com_peso=False)
        feedback.FeedbackStorage(db_path=path)
        conn = sqlite3.connect(path)
        pesos = dict(conn.execute("SELECT tipo, peso FROM feedback_events"))
        conn.close()
        self.assertEqual(pesos["skip"], -1.0)
        self.assertEqual(pesos["completa"], 1.0)

    def test_banco_novo_nao_e_alterado(self):
        import feedback
        path = os.path.join(tempfile.mkdtemp(), "novo.db")
        st = feedback.FeedbackStorage(db_path=path)
        st.record_event(track_id="X", genero="Rock", tipo="skip", peso=-1.0, elapsed=1, duration=100)
        feedback.FeedbackStorage(db_path=path)  # reabrir não deve migrar nem perder dados
        conn = sqlite3.connect(path)
        n = conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
        conn.close()
        self.assertEqual(n, 1)


class TestNivelDeBuscaPegajoso(unittest.TestCase):

    def test_segunda_recomendacao_parte_do_nivel_que_a_primeira_precisou(self):
        class Client:
            def __init__(self):
                self.pl = [{"id": "1", "file": "rock/a.mp3", "pos": 0}]
                self.n = 10

            def status(self):
                return {"song": "0", "songid": "1", "state": "play"}

            def currentsong(self):
                return dict(self.pl[0])

            def playlistinfo(self):
                return [dict(x) for x in self.pl]

            def addid(self, uri, pos=None):
                self.n += 1
                item = {"id": str(self.n), "file": uri, "pos": len(self.pl)}
                self.pl.insert(pos if pos is not None else len(self.pl), item)
                return str(self.n)

            def deleteid(self, sid):
                self.pl = [x for x in self.pl if x["id"] != str(sid)]

        recebidos = []

        def fake_select(**kw):
            recebidos.append(kw["nivel_busca"])
            i = len(recebidos)
            return f"rock/n{i}.mp3", {"track_id": f"N{i}", "distance": 0.1, "decision_id": 1, "nivel_busca": 2}, 0

        ids = {"rock/a.mp3": "A"}
        with patch.object(mpd_integration, "select_next_recommendations", side_effect=fake_select):
            added = mpd_integration.replenish_mpd_queue(
                client=Client(), collection=MagicMock(), current_track_id="A", file_to_id_map=ids,
                id_to_file_map={"A": "rock/a.mp3"}, music_root="", play_history=deque(["A"]),
                failed_tracks=set(), min_queue_ahead=2, use_feedback=False, seed_songid="1")
        self.assertEqual(added, 2)
        self.assertEqual(recebidos, [0, 2], "a 2ª busca não refaz os níveis 0 e 1")


class TestInicioComMpdTocando(unittest.TestCase):

    def test_decide_de_imediato_se_o_mpd_ja_estiver_tocando(self):
        client = FakeMPD([])  # nenhum evento de idle chegará
        client.playlist = [_item(i) for i in range(3)]
        client.state = {"state": "play", "song": "1"}
        ids = {f"rock/track_{i}.mp3": f"T{i}" for i in range(40)}
        chamadas = {"n": 0}

        def fake_replenish(**kw):
            chamadas["n"] += 1
            return 2

        with patch.object(mpd_integration, "replenish_mpd_queue", side_effect=fake_replenish), \
             patch.object(mpd_integration, "load_session_history", return_value=deque(["T0"], maxlen=30)), \
             patch.object(mpd_integration, "save_session_history"):
            with self.assertRaises(KeyboardInterrupt):
                mpd_integration.monitor_and_recommend(
                    client=client, collection=MagicMock(), file_to_id_map=ids,
                    id_to_file_map={v: k for k, v in ids.items()}, music_root="", use_feedback=False)
        self.assertEqual(chamadas["n"], 1, "deve recomendar a partir da faixa atual sem esperar evento")


# ── Modo aleatório do MPD e tempo ouvido ─────────────────────────────────────────

def _roda_monitor(client, use_feedback=False, storage=None, force_sequential=True):
    """Executa o monitor até o roteiro do FakeMPD acabar (idle levanta KeyboardInterrupt)."""
    from contextlib import ExitStack
    ids = {f"rock/track_{i}.mp3": f"T{i}" for i in range(40)}
    with ExitStack() as stack:  # garante a limpeza dos patches mesmo se um deles falhar ao iniciar
        stack.enter_context(patch.object(mpd_integration, "replenish_mpd_queue", return_value=2))
        stack.enter_context(patch.object(mpd_integration, "load_session_history", return_value=deque(["T0"], maxlen=30)))
        stack.enter_context(patch.object(mpd_integration, "save_session_history"))
        stack.enter_context(patch.object(mpd_integration, "FORCE_SEQUENTIAL_PLAYBACK", force_sequential))
        if storage is not None:
            stack.enter_context(patch.object(mpd_integration.feedback, "FeedbackStorage", lambda: storage))
        try:
            mpd_integration.monitor_and_recommend(
                client=client, collection=MagicMock(), file_to_id_map=ids,
                id_to_file_map={v: k for k, v in ids.items()}, music_root="", use_feedback=use_feedback)
        except KeyboardInterrupt:
            pass


class TestModoAleatorio(unittest.TestCase):
    """Com `random` ligado o MPD toca outra faixa que não a enfileirada como próxima: o monitor o desliga."""

    def test_desliga_o_random_quando_ligado(self):
        c = FakeMPD([step_playing(0, 3)])
        c.random_on = True
        _roda_monitor(c)
        self.assertEqual(c.random_calls, [0])
        self.assertFalse(c.random_on)

    def test_nao_mexe_se_ja_estava_desligado(self):
        c = FakeMPD([step_playing(0, 3)])
        _roda_monitor(c)
        self.assertEqual(c.random_calls, [])

    def test_respeita_a_chave_que_permite_shuffle(self):
        c = FakeMPD([step_playing(0, 3)])
        c.random_on = True
        _roda_monitor(c, force_sequential=False)
        self.assertEqual(c.random_calls, [])
        self.assertTrue(c.random_on)


class TestRelogioDeReproducao(unittest.TestCase):

    def test_avanca_com_o_relogio_e_ancora_no_elapsed_do_mpd(self):
        r = mpd_integration.PlaybackClock()
        r.start("5", elapsed=10.0, now=1000.0)
        self.assertAlmostEqual(r.elapsed(now=1030.0), 40.0)
        r.sync(100.0, now=1030.0)  # seek: re-ancora
        self.assertAlmostEqual(r.elapsed(now=1040.0), 110.0)

    def test_pausa_congela_e_retomada_reancora(self):
        r = mpd_integration.PlaybackClock()
        r.start("5", elapsed=0.0, now=1000.0)
        r.pause(20.0, now=1020.0)
        self.assertAlmostEqual(r.elapsed(now=1620.0), 20.0, msg="pausada por 10 min: continua em 20 s")
        r.sync(20.0, now=1620.0)
        self.assertAlmostEqual(r.elapsed(now=1635.0), 35.0)

    def test_sem_faixa_o_tempo_e_zero(self):
        self.assertEqual(mpd_integration.PlaybackClock().elapsed(now=5.0), 0.0)

    def test_status_com_tempo_ouvido_limita_ao_total(self):
        st = mpd_integration._status_com_tempo_ouvido({"time": "0:169", "song": "1"}, {}, 175.0)
        self.assertEqual(st["time"], "169:169")
        self.assertEqual(st["song"], "1")


class TestFeedbackComTempoOuvido(unittest.TestCase):
    """O status guardado no início da faixa tem elapsed ~ 0: tudo era gravado como skip."""

    def _transicao(self, escuta_segundos, pausa_segundos=0.0, pausar_apos=None):
        import feedback
        storage = feedback.FeedbackStorage(db_path=":memory:")
        clock = {"t": 1000.0}

        def evento(n, state="play", elapsed=0.0, avanco=0.0):
            """Um evento do MPD: o relógio de parede avança `avanco` s e o status informa `elapsed` (como o MPD real)."""
            def _step(c):
                clock["t"] += avanco
                c.playlist = [_item(0), _item(1)]
                c.state = {"state": state, "song": str(n)}
                c.extra = {"elapsed": f"{elapsed:.3f}", "time": f"{int(elapsed)}:169", "duration": "169.0"}
            return _step

        script = [evento(0)]
        if pausar_apos is not None:
            # toca `pausar_apos` s, pausa por `pausa_segundos`, retoma (elapsed do MPD congelado) e toca o restante
            script += [evento(0, "pause", pausar_apos, avanco=pausar_apos),
                       evento(0, "play", pausar_apos, avanco=pausa_segundos),
                       evento(1, avanco=escuta_segundos - pausar_apos)]
        else:
            script += [evento(1, avanco=escuta_segundos)]  # troca de faixa após `escuta_segundos` de reprodução
        with patch.object(mpd_integration.time, "time", side_effect=lambda: clock["t"]):
            _roda_monitor(FakeMPD(script), use_feedback=True, storage=storage)
        with storage._get_connection() as conn:
            return [r[0] for r in conn.execute("SELECT tipo FROM feedback_events ORDER BY id").fetchall()]

    def test_ouviu_ate_o_fim_e_completa(self):
        self.assertEqual(self._transicao(169), ["completa"])

    def test_pulou_cedo_e_skip(self):
        self.assertEqual(self._transicao(5), ["skip"])

    def test_ouviu_parte_e_parcial(self):
        self.assertEqual(self._transicao(60), ["parcial"])

    def test_pausa_nao_conta_como_tempo_ouvido(self):
        """20 s + pausa de 10 min + 15 s = 35 s ouvidos (parcial); sem descontar a pausa seria 'completa'."""
        self.assertEqual(self._transicao(35, pausa_segundos=600.0, pausar_apos=20.0), ["parcial"])


if __name__ == "__main__":
    unittest.main()
