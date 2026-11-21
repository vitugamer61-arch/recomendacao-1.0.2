"""
feedback.py — Módulo de feedback implícito por detecção de skip e faixas completadas.

Funcionalidades:
- Persistência de eventos em SQLite (feedback.db).
- Detecção heurística de skips reais vs. fins naturais e navegação anterior.
- Modificação do vetor de consulta via algoritmo de Rocchio com decaimento exponencial.
- Re-ranking de candidatos com penalidade acústica de faixas puladas e penalidade/cooldown de gênero.
"""

import os
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

import feedback_config as fcfg

logger = logging.getLogger("feedback")


# ── 1. PERSISTÊNCIA EM SQLITE ─────────────────────────────────────────────────

class FeedbackStorage:
    """Gerencia a persistência de eventos de feedback no banco SQLite."""

    def __init__(self, db_path: Optional[Any] = None):
        if db_path is None:
            self.db_path = fcfg.DEFAULT_DB_PATH
        elif str(db_path) == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        if self.db_path != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._shared_conn = None
        else:
            self._shared_conn = sqlite3.connect(":memory:")
            self._shared_conn.row_factory = sqlite3.Row

        self._init_db()

    @contextmanager
    def _get_connection(self):
        if self._shared_conn is not None:
            yield self._shared_conn
        else:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def close(self):
        """Fecha conexão compartilhada se existir."""
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None

    def _init_db(self) -> None:
        """Cria as tabelas necessárias se não existirem."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    track_id TEXT NOT NULL,
                    caminho TEXT,
                    genero TEXT,
                    cluster_id TEXT,
                    tipo TEXT CHECK(tipo IN ('skip', 'completa')),
                    elapsed REAL,
                    duration REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_track ON feedback_events(track_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_tipo ON feedback_events(tipo)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_time ON feedback_events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_genero ON feedback_events(genero)")
            conn.commit()

    def record_event(
        self,
        track_id: str,
        caminho: str = "",
        genero: str = "",
        cluster_id: str = "",
        tipo: str = "skip",
        elapsed: float = 0.0,
        duration: float = 0.0
    ) -> int:
        """Registra um evento de feedback (skip ou completa) no banco de dados."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback_events (track_id, caminho, genero, cluster_id, tipo, elapsed, duration)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (track_id, caminho, genero, cluster_id, tipo, elapsed, duration))
            conn.commit()
            event_id = cursor.lastrowid
            logger.info(
                f"[FEEDBACK REGISTRADO] Tipo='{tipo.upper()}' | Faixa='{os.path.basename(caminho or track_id)}' "
                f"| Gênero='{genero}' | Tempo={elapsed:.1f}s/{duration:.1f}s"
            )
            return event_id

    def get_recent_events(
        self,
        tipo: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retorna os eventos mais recentes, ordenados do mais novo para o mais antigo."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if tipo:
                cursor.execute("""
                    SELECT * FROM feedback_events
                    WHERE tipo = ?
                    ORDER BY id DESC LIMIT ?
                """, (tipo, limit))
            else:
                cursor.execute("""
                    SELECT * FROM feedback_events
                    ORDER BY id DESC LIMIT ?
                """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_skips(self, limit: int = fcfg.ROCCHIO_MAX_ITEMS) -> List[Dict[str, Any]]:
        return self.get_recent_events(tipo="skip", limit=limit)

    def get_recent_completed(self, limit: int = fcfg.ROCCHIO_MAX_ITEMS) -> List[Dict[str, Any]]:
        return self.get_recent_events(tipo="completa", limit=limit)

    def get_genre_penalties(self) -> Dict[str, float]:
        """
        Calcula a penalidade atual de cada gênero:
        +1.0 por skip, -0.5 por faixa completa (mínimo 0.0), com decaimento exponencial no tempo.
        """
        penalties: Dict[str, float] = {}
        now = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT genero, tipo, timestamp
                FROM feedback_events
                WHERE genero IS NOT NULL AND genero != '' AND genero != 'desconhecido'
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()

        for row in rows:
            g = row["genero"]
            tipo = row["tipo"]
            ts_str = row["timestamp"]
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                dt_seconds = max(0.0, (now - ts).total_seconds())
            except Exception:
                dt_seconds = 0.0

            # Decaimento temporal contínuo: 2^(-dt / meia_vida)
            half_life = fcfg.GENRE_PENALTY_HALF_LIFE_SECONDS
            decay_factor = 2.0 ** (-dt_seconds / half_life) if half_life > 0 else 1.0

            curr = penalties.get(g, 0.0)
            if tipo == "skip":
                curr += fcfg.PENALTY_SKIP_GENRE * decay_factor
            elif tipo == "completa":
                curr = max(fcfg.GENRE_MIN_PENALTY, curr - fcfg.BONUS_COMPLETE_GENRE * decay_factor)
            penalties[g] = round(max(fcfg.GENRE_MIN_PENALTY, curr), 4)

        return penalties

    def is_genre_in_cooldown(
        self,
        genero: str,
        threshold: int = fcfg.GENRE_COOLDOWN_SKIP_THRESHOLD,
        window_seconds: float = fcfg.GENRE_COOLDOWN_WINDOW_SECONDS,
        cooldown_duration: float = fcfg.GENRE_COOLDOWN_DURATION_SECONDS
    ) -> bool:
        """
        Verifica se um determinado gênero está em cooldown ativo devido a 3 ou mais skips consecutivos
        dentro de uma janela temporal recente.
        """
        if not genero or genero.lower() in ["", "desconhecido", "none"]:
            return False

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Pega os eventos recentes do gênero
            cursor.execute("""
                SELECT tipo, timestamp FROM feedback_events
                WHERE genero = ?
                ORDER BY id DESC LIMIT 10
            """, (genero,))
            rows = cursor.fetchall()

        if len(rows) < threshold:
            return False

        now = datetime.now()
        skips_in_window = 0
        most_recent_skip_time = None

        for row in rows:
            tipo = row["tipo"]
            ts_str = row["timestamp"]
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                age_seconds = (now - ts).total_seconds()
            except Exception:
                continue

            if age_seconds <= (window_seconds + cooldown_duration):
                if tipo == "skip":
                    skips_in_window += 1
                    if most_recent_skip_time is None:
                        most_recent_skip_time = ts
                elif tipo == "completa":
                    # Uma faixa completa no mesmo gênero interrompe a sequência de rejeição total
                    break

        if skips_in_window >= threshold and most_recent_skip_time is not None:
            time_since_last_skip = (now - most_recent_skip_time).total_seconds()
            if time_since_last_skip <= cooldown_duration:
                logger.info(
                    f"[COOLDOWN ATIVO] Gênero '{genero}' temporariamente bloqueado "
                    f"({skips_in_window} skips na janela, restam {cooldown_duration - time_since_last_skip:.0f}s)"
                )
                return True

        return False

    def clear_history(self) -> None:
        """Limpa todo o histórico de feedback (útil para testes e reinicialização)."""
        with self._get_connection() as conn:
            conn.cursor().execute("DELETE FROM feedback_events")
            conn.commit()


# ── 2. DETECTOR DE FEEDBACK (HEURÍSTICA MPD) ───────────────────────────────────

class FeedbackDetector:
    """
    Rastreia transições de faixas no MPD para detectar skips manuais e faixas completadas.
    """

    def __init__(self, storage: FeedbackStorage):
        self.storage = storage

    def evaluate_transition(
        self,
        prev_song: Optional[Dict[str, Any]],
        prev_status: Optional[Dict[str, Any]],
        current_song: Optional[Dict[str, Any]],
        current_status: Optional[Dict[str, Any]],
        meta_resolver: Optional[callable] = None,
        is_automated_trigger: bool = False
    ) -> Optional[str]:
        """
        Avalia o que aconteceu com a faixa anterior (`prev_song`):
        - Retorna 'skip' se foi pulada manualmente antes de min(30s, 30% da duração).
        - Retorna 'completa' se foi ouvida até >= 80% da duração.
        - Retorna None se foi navegação para faixa anterior, fim natural, ou troca pelo recomendador.
        """
        if not prev_song or not prev_status:
            return None

        if is_automated_trigger:
            logger.debug("Troca automática ignorada pelo detector de feedback.")
            return None

        # 1. Posições para detectar se voltou para trás na playlist
        old_pos = int(prev_status.get("song", prev_song.get("pos", 0)))
        new_pos = int(current_status.get("song", current_song.get("pos", 0))) if (current_status and current_song) else old_pos + 1

        if new_pos < old_pos:
            logger.info(f"Navegação para faixa anterior detectada ({old_pos} → {new_pos}). Ignorando skip.")
            return None

        # 2. Extração de tempo reproduzido e duração
        elapsed = 0.0
        duration = 0.0

        time_val = prev_status.get("time")
        if time_val and ":" in str(time_val):
            try:
                parts = str(time_val).split(":")
                elapsed = float(parts[0])
                duration = float(parts[1])
            except Exception:
                pass
        else:
            elapsed = float(prev_status.get("elapsed", 0.0))
            duration = float(prev_status.get("duration", prev_song.get("duration", 0.0)))

        if duration <= 0.0:
            return None

        ratio = elapsed / duration
        prev_file = prev_song.get("file", "")

        # Obter metadados de gênero e cluster
        subgenero = ""
        cluster_id = ""
        track_id = prev_file
        if meta_resolver:
            meta = meta_resolver(prev_file)
            if meta:
                subgenero = meta.get("subgenero", "")
                cluster_id = meta.get("cluster_id", "")
                track_id = meta.get("track_id", prev_file)

        # Regra 1: Faixa ouvida até >= 80% da duração = feedback positivo (completa)
        if ratio >= fcfg.COMPLETE_MIN_DURATION_RATIO or (duration - elapsed) <= fcfg.NATURAL_END_TOLERANCE_SECONDS:
            self.storage.record_event(
                track_id=track_id,
                caminho=prev_file,
                genero=subgenero,
                cluster_id=cluster_id,
                tipo="completa",
                elapsed=elapsed,
                duration=duration
            )
            return "completa"

        # Regra 2: Skip precoce = troca antes de min(30s, 30% da duração)
        skip_threshold_seconds = min(fcfg.SKIP_MAX_ELAPSED_SECONDS, fcfg.SKIP_MAX_DURATION_RATIO * duration)
        if elapsed < skip_threshold_seconds:
            self.storage.record_event(
                track_id=track_id,
                caminho=prev_file,
                genero=subgenero,
                cluster_id=cluster_id,
                tipo="skip",
                elapsed=elapsed,
                duration=duration
            )
            return "skip"

        logger.debug(f"Faixa '{os.path.basename(prev_file)}' trocada em ponto intermediário ({elapsed:.1f}s / {ratio:.1%}). Neutra.")
        return None


# ── 3. MOTOR DE ALGORITMO ROCCHIO & RE-RANKING ────────────────────────────────

class FeedbackEngine:
    """
    Executa a adaptação da consulta via Rocchio e o re-ranking penalizado das recomendações.
    """

    def __init__(self, storage: Optional[FeedbackStorage] = None):
        self.storage = storage or FeedbackStorage()

    def compute_rocchio_query(
        self,
        query_emb: np.ndarray,
        collection
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aplica o algoritmo de Rocchio ao vetor de consulta:
        q' = α·q + β·média(positivas) − γ·média(puladas), normalizado em L2.
        Usa até as últimas 10 de cada tipo com decaimento exponencial.
        """
        recent_pos = self.storage.get_recent_completed(limit=fcfg.ROCCHIO_MAX_ITEMS)
        recent_neg = self.storage.get_recent_skips(limit=fcfg.ROCCHIO_MAX_ITEMS)

        stats = {
            "n_pos": len(recent_pos),
            "n_neg": len(recent_neg),
            "applied": False
        }

        if not recent_pos and not recent_neg:
            norm_q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
            return norm_q, stats

        def _fetch_weighted_average(events: List[Dict[str, Any]]) -> Optional[np.ndarray]:
            if not events:
                return None
            t_ids = [e["track_id"] for e in events]
            db_res = collection.get(ids=t_ids, include=["embeddings"])
            found_ids = db_res.get("ids", [])
            found_embs = db_res.get("embeddings", [])
            emb_map = {tid: np.array(emb, dtype=np.float32) for tid, emb in zip(found_ids, found_embs)}

            weighted_sum = np.zeros_like(query_emb, dtype=np.float32)
            weights_total = 0.0

            for i, ev in enumerate(events):
                tid = ev["track_id"]
                if tid in emb_map:
                    # Decaimento exponencial com a antiguidade do evento
                    w = np.exp(-fcfg.ROCCHIO_DECAY_RATE * i)
                    weighted_sum += w * emb_map[tid]
                    weights_total += w

            if weights_total > 0.0:
                return weighted_sum / weights_total
            return None

        mean_pos = _fetch_weighted_average(recent_pos)
        mean_neg = _fetch_weighted_average(recent_neg)

        # q' = α·q + β·média(pos) - γ·média(neg)
        q_prime = fcfg.ROCCHIO_ALPHA * query_emb.copy()
        if mean_pos is not None:
            q_prime += fcfg.ROCCHIO_BETA * mean_pos
        if mean_neg is not None:
            q_prime -= fcfg.ROCCHIO_GAMMA * mean_neg

        # Normalização L2
        norm_val = np.linalg.norm(q_prime) + 1e-9
        q_prime_normalized = (q_prime / norm_val).astype(np.float32)
        stats["applied"] = True

        logger.info(
            f"[ROCCHIO] Vetor ajustado: α={fcfg.ROCCHIO_ALPHA}, β={fcfg.ROCCHIO_BETA} ({stats['n_pos']} pos), "
            f"γ={fcfg.ROCCHIO_GAMMA} ({stats['n_neg']} skips)"
        )
        return q_prime_normalized, stats

    def rerank_candidates(
        self,
        query_emb: np.ndarray,
        candidates: List[Dict[str, Any]],
        collection,
        id_to_file_map: Dict[str, str],
        queued_files: set
    ) -> List[Dict[str, Any]]:
        """
        Aplica o re-ranking aos candidatos preliminares buscados no ChromaDB:
        score = sim(q, c) − λ · Σ w_i·sim(c, skip_i) − μ · pen_genero(c)

        Regras:
        1. Exclui faixas puladas recentemente e faixas já na fila.
        2. Bloqueia candidatos de gêneros em cooldown ativo (3+ skips recentes).
        3. Peso da faixa específica pulada (λ) é maior que o do gênero (μ).
        """
        if not candidates:
            return []

        recent_skips = self.storage.get_recent_skips(limit=fcfg.EXCLUDE_RECENT_SKIPS_COUNT)
        recent_skip_ids = {e["track_id"] for e in recent_skips}
        genre_penalties = self.storage.get_genre_penalties()

        # Carregar embeddings das faixas puladas recentes para calcular sim(c, skip_i)
        skip_embs: List[Tuple[float, np.ndarray]] = []
        if recent_skips:
            db_skips = collection.get(ids=list(recent_skip_ids), include=["embeddings"])
            found_ids = db_skips.get("ids", [])
            found_embs = db_skips.get("embeddings", [])
            emb_map = {tid: np.array(emb, dtype=np.float32) for tid, emb in zip(found_ids, found_embs)}

            for i, ev in enumerate(recent_skips):
                tid = ev["track_id"]
                if tid in emb_map:
                    w = np.exp(-fcfg.ROCCHIO_DECAY_RATE * i)
                    v = emb_map[tid]
                    norm_v = v / (np.linalg.norm(v) + 1e-9)
                    skip_embs.append((w, norm_v))

        # Obter embeddings dos candidatos caso não estejam presentes
        cand_ids = [c["track_id"] for c in candidates]
        db_cands = collection.get(ids=cand_ids, include=["embeddings", "metadatas"])
        c_emb_map = {tid: np.array(emb, dtype=np.float32) for tid, emb in zip(db_cands.get("ids", []), db_cands.get("embeddings", []))}

        norm_q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        scored_candidates = []

        for cand in candidates:
            cid = cand["track_id"]
            c_file = id_to_file_map.get(cid, cid)
            genero = cand.get("subgenero", "")

            # Exclusão 1: faixa pulada recentemente
            if cid in recent_skip_ids:
                logger.debug(f"[RERANK REJEITADA] Faixa '{cid}' pulada recentemente.")
                continue

            # Exclusão 2: já na fila
            if c_file in queued_files:
                continue

            # Exclusão 3: gênero em cooldown
            if genero and self.storage.is_genre_in_cooldown(genero):
                logger.info(f"[RERANK BLOQUEADA] Faixa '{cid}' ignorada: gênero '{genero}' em cooldown ativo.")
                continue

            c_emb = c_emb_map.get(cid)
            if c_emb is None:
                continue
            norm_c = c_emb / (np.linalg.norm(c_emb) + 1e-9)

            # sim(q, c)
            base_sim = float(np.dot(norm_q, norm_c))

            # Penalidade por similaridade com faixas puladas
            # λ · Σ w_i · sim(c, skip_i)
            skip_sim_penalty = 0.0
            if skip_embs:
                for w, skip_vec in skip_embs:
                    s_sim = max(0.0, float(np.dot(norm_c, skip_vec)))
                    skip_sim_penalty += w * s_sim
                skip_sim_penalty = skip_sim_penalty / len(skip_embs)

            # Penalidade do gênero
            # μ · pen_genero(c)
            g_pen = genre_penalties.get(genero, 0.0)

            # score = sim(q, c) − λ·pen_skip − μ·pen_genero
            final_score = base_sim - (fcfg.LAMBDA_PENALTY_TRACK * skip_sim_penalty) - (fcfg.MU_PENALTY_GENRE * g_pen)

            cand_copy = dict(cand)
            cand_copy["feedback_score"] = float(final_score)
            cand_copy["base_similarity"] = float(base_sim)
            cand_copy["skip_similarity_penalty"] = float(skip_sim_penalty)
            cand_copy["genre_penalty"] = float(g_pen)

            scored_candidates.append(cand_copy)

        # Ordena do maior score para o menor
        scored_candidates.sort(key=lambda x: x["feedback_score"], reverse=True)
        return scored_candidates
