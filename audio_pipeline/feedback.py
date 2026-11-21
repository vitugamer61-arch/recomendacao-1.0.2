"""
feedback.py — Módulo de Feedback Implícito, Adaptação Rocchio e Re-ranking Penalizado.

Implementa:
1. Detecção gradual de feedback de reprodução no MPD:
   - skip: troca manual com elapsed < 30s (peso negativo forte);
   - parcial: troca com 30s até 80% (positivo fraco, proporcional a f = elapsed/duration);
   - completa: ouvida até >= 80% ou fim natural (positivo forte).
2. Persistência SQLite (feedback.db) com WAL mode ativo para concorrência recomendador-monitor.
3. Tabela 'decisoes' para auditoria e acompanhamento em tempo real no monitor.
4. Adaptação vetorial Rocchio (q') com decaimento exponencial e ponderação gradual.
5. Re-ranking de candidatos com penalidades de faixas e gêneros, decaimento por meia-vida e cooldown.
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np

# Configurações do módulo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import feedback_config as fcfg

logger = logging.getLogger("FeedbackSystem")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ── 1. PERSISTÊNCIA SQLITE (FEEDBACK STORAGE) ────────────────────────────────

class FeedbackStorage:
    """
    Gerenciador de persistência SQLite com suporte a WAL mode para leitura concorrente.
    Armazena eventos de feedback, decisões tomadas pelo recomendador e estado da fila do MPD.
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = str(db_path) if db_path is not None else str(fcfg.DEFAULT_DB_PATH)
        self._mem_conn = None
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        else:
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager para gerenciar conexão SQLite de forma segura."""
        if self._mem_conn is not None:
            yield self._mem_conn
        else:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            conn.row_factory = sqlite3.Row
            try:
                # Ativa WAL mode para permitir concorrência limpa de leitura e escrita
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                yield conn
            finally:
                conn.close()

    def close(self):
        """Fecha a conexão em memória caso exista."""
        if self._mem_conn is not None:
            try:
                self._mem_conn.close()
            except Exception:
                pass
            self._mem_conn = None

    def _init_db(self) -> None:
        """Cria as tabelas e índices necessários no SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabela de eventos de reprodução / feedback
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    track_id TEXT NOT NULL,
                    caminho TEXT,
                    genero TEXT,
                    cluster_id TEXT,
                    tipo TEXT NOT NULL,       -- 'skip', 'parcial', 'completa'
                    peso REAL DEFAULT 1.0,    -- negativo para skip (-1.0), positivo fraco (f) ou forte (1.0)
                    elapsed REAL,
                    duration REAL
                )
            """)

            # Migração caso a coluna 'peso' ainda não exista
            cursor.execute("PRAGMA table_info(feedback_events)")
            colunas = [col[1] for col in cursor.fetchall()]
            if "peso" not in colunas:
                try:
                    cursor.execute("ALTER TABLE feedback_events ADD COLUMN peso REAL DEFAULT 1.0")
                except Exception:
                    pass

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_track ON feedback_events(track_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_tipo ON feedback_events(tipo)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_time ON feedback_events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_genero ON feedback_events(genero)")

            # Tabela de decisões tomadas pelo recomendador (para auditoria e monitor ao vivo)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decisoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    seed_track_id TEXT,
                    seed_caminho TEXT,
                    seed_genero TEXT,
                    escolhida_track_id TEXT,
                    escolhida_caminho TEXT,
                    escolhida_genero TEXT,
                    escolhida_cluster TEXT,
                    origem TEXT,              -- 'recomendada', 'exploracao', 'manual'
                    rank_escolhido INTEGER,
                    score_escolhido REAL,
                    penalidades_json TEXT,    -- JSON das penalidades vigentes
                    candidatas_json TEXT      -- JSON das top candidatas avaliadas com scores e motivos
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dec_time ON decisoes(timestamp)")

            # Tabela de estado atual da fila do MPD (para monitor ao vivo)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mpd_queue_state (
                    pos INTEGER PRIMARY KEY,
                    caminho TEXT,
                    track_id TEXT,
                    genero TEXT,
                    cluster TEXT,
                    origem TEXT,              -- 'recomendada', 'exploracao', 'manual'
                    is_current INTEGER DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def record_event(
        self,
        track_id: str,
        caminho: str = "",
        genero: str = "",
        cluster_id: str = "",
        tipo: str = "skip",
        peso: float = 1.0,
        elapsed: float = 0.0,
        duration: float = 0.0
    ) -> int:
        """Registra um evento de feedback gradual (skip, parcial ou completa) no banco de dados."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback_events (track_id, caminho, genero, cluster_id, tipo, peso, elapsed, duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (track_id, caminho, genero, cluster_id, tipo, float(peso), elapsed, duration))
            conn.commit()
            event_id = cursor.lastrowid
            logger.info(
                f"[FEEDBACK REGISTRADO] Tipo='{tipo.upper()}' (peso={peso:+.2f}) | "
                f"Faixa='{os.path.basename(caminho or track_id)}' | Gênero='{genero}' | "
                f"Tempo={elapsed:.1f}s/{duration:.1f}s"
            )
            return event_id

    def record_decision(
        self,
        seed_track_id: str,
        seed_caminho: str,
        seed_genero: str,
        escolhida_track_id: str,
        escolhida_caminho: str,
        escolhida_genero: str,
        escolhida_cluster: str,
        origem: str,
        rank_escolhido: int,
        score_escolhido: float,
        penalidades: Dict[str, float],
        candidatas_detalhes: List[Dict[str, Any]]
    ) -> int:
        """Registra a auditoria completa de uma decisão de recomendação tomada."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO decisoes (
                    seed_track_id, seed_caminho, seed_genero,
                    escolhida_track_id, escolhida_caminho, escolhida_genero, escolhida_cluster,
                    origem, rank_escolhido, score_escolhido,
                    penalidades_json, candidatas_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                seed_track_id, seed_caminho, seed_genero,
                escolhida_track_id, escolhida_caminho, escolhida_genero, escolhida_cluster,
                origem, rank_escolhido, float(score_escolhido),
                json.dumps(penalidades), json.dumps(candidatas_detalhes)
            ))
            conn.commit()
            return cursor.lastrowid

    def update_queue_state(self, items: List[Dict[str, Any]]) -> None:
        """Atualiza a visão espelhada da fila do MPD no banco de dados."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM mpd_queue_state")
            for it in items:
                cursor.execute("""
                    INSERT INTO mpd_queue_state (pos, caminho, track_id, genero, cluster, origem, is_current)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    it.get("pos", 0),
                    it.get("caminho", ""),
                    it.get("track_id", ""),
                    it.get("genero", ""),
                    it.get("cluster", ""),
                    it.get("origem", "manual"),
                    1 if it.get("is_current") else 0
                ))
            conn.commit()

    def get_queue_state(self) -> List[Dict[str, Any]]:
        """Retorna a fila do MPD salva mais recentemente."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM mpd_queue_state ORDER BY pos ASC")
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retorna as decisões mais recentes tomadas pelo recomendador."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM decisoes ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            decisoes = []
            for r in rows:
                d = dict(r)
                try:
                    d["penalidades"] = json.loads(d.get("penalidades_json") or "{}")
                except Exception:
                    d["penalidades"] = {}
                try:
                    d["candidatas"] = json.loads(d.get("candidatas_json") or "[]")
                except Exception:
                    d["candidatas"] = []
                decisoes.append(d)
            return decisoes

    def get_last_decision(self) -> Optional[Dict[str, Any]]:
        """Retorna a última decisão tomada."""
        decs = self.get_recent_decisions(limit=1)
        return decs[0] if decs else None

    def get_recent_events(
        self,
        tipo: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retorna os eventos mais recentes de feedback."""
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

    def get_recent_positives(self, limit: int = fcfg.ROCCHIO_MAX_ITEMS) -> List[Dict[str, Any]]:
        """Retorna as faixas com feedback positivo (completas ou parciais)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM feedback_events
                WHERE tipo IN ('completa', 'parcial')
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_genre_penalties(self) -> Dict[str, float]:
        """
        Calcula a penalidade atual de cada gênero com decaimento exponencial contínuo por meia-vida:
        - skip: +1.0 * decay;
        - completa: -0.5 * decay;
        - parcial: -(0.5 * f) * decay (onde f = elapsed/duration).
        Mínimo: 0.0.
        """
        penalties: Dict[str, float] = {}
        now = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT genero, tipo, peso, elapsed, duration, timestamp
                FROM feedback_events
                WHERE genero IS NOT NULL AND genero != '' AND genero != 'desconhecido'
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()

        for row in rows:
            g = row["genero"]
            tipo = row["tipo"]
            peso = float(row["peso"] if row["peso"] is not None else 1.0)
            elapsed = float(row["elapsed"] or 0.0)
            duration = float(row["duration"] or 180.0)
            ts_str = row["timestamp"]

            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                dt_seconds = max(0.0, (now - ts).total_seconds())
            except Exception:
                dt_seconds = 0.0

            half_life = fcfg.GENRE_PENALTY_HALF_LIFE_SECONDS
            decay_factor = 2.0 ** (-dt_seconds / half_life) if half_life > 0 else 1.0

            curr = penalties.get(g, 0.0)
            if tipo == "skip":
                curr += fcfg.PENALTY_SKIP_GENRE * decay_factor
            elif tipo == "completa":
                curr = max(fcfg.GENRE_MIN_PENALTY, curr - fcfg.BONUS_COMPLETE_GENRE * decay_factor)
            elif tipo == "parcial":
                f = min(1.0, max(0.0, elapsed / duration)) if duration > 0 else 0.5
                curr = max(fcfg.GENRE_MIN_PENALTY, curr - (fcfg.BONUS_COMPLETE_GENRE * f) * decay_factor)

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
        Verifica se um determinado gênero está em cooldown ativo devido a 3+ skips consecutivos
        dentro de uma janela temporal recente.
        """
        if not genero or genero.lower() in ["", "desconhecido", "none"]:
            return False

        with self._get_connection() as conn:
            cursor = conn.cursor()
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
                elif tipo in ["completa", "parcial"]:
                    # Feedback positivo no mesmo gênero encerra a sequência de bloqueio
                    break

        if skips_in_window >= threshold and most_recent_skip_time is not None:
            time_since_last_skip = (now - most_recent_skip_time).total_seconds()
            if time_since_last_skip <= cooldown_duration:
                return True

        return False

    def get_cooldown_genres(self) -> List[str]:
        """Retorna a lista de todos os gêneros atualmente em cooldown."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT genero FROM feedback_events WHERE genero IS NOT NULL")
            genres = [r[0] for r in cursor.fetchall() if r[0]]

        return [g for g in genres if self.is_genre_in_cooldown(g)]

    def get_acceptance_stats(self, window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Calcula as métricas de aceitação:
        taxa = aceitas / (aceitas + skips)
        Retorna estatísticas gerais e por gênero.
        """
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tipo, genero, timestamp FROM feedback_events")
            rows = cursor.fetchall()

        total_aceitas = 0
        total_parciais = 0
        total_skips = 0
        por_genero = {}

        for r in rows:
            tipo = r["tipo"]
            g = r["genero"] or "Outros"
            ts_str = r["timestamp"]
            if window_seconds is not None:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                    if (now - ts).total_seconds() > window_seconds:
                        continue
                except Exception:
                    pass

            if g not in por_genero:
                por_genero[g] = {"aceitas": 0, "parciais": 0, "skips": 0}

            if tipo == "completa":
                total_aceitas += 1
                por_genero[g]["aceitas"] += 1
            elif tipo == "parcial":
                total_parciais += 1
                por_genero[g]["parciais"] += 1
            elif tipo == "skip":
                total_skips += 1
                por_genero[g]["skips"] += 1

        denom = total_aceitas + total_skips
        taxa_geral = (total_aceitas / denom) if denom > 0 else 1.0

        stats_por_genero = {}
        for g, counts in por_genero.items():
            g_denom = counts["aceitas"] + counts["skips"]
            taxa_g = (counts["aceitas"] / g_denom) if g_denom > 0 else 1.0
            stats_por_genero[g] = {
                "taxa": taxa_g,
                "aceitas": counts["aceitas"],
                "parciais": counts["parciais"],
                "skips": counts["skips"]
            }

        return {
            "taxa_aceitacao": taxa_geral,
            "total_aceitas": total_aceitas,
            "total_parciais": total_parciais,
            "total_skips": total_skips,
            "por_genero": stats_por_genero
        }

    def clear_history(self) -> None:
        """Limpa todo o histórico de feedback, decisões e estado espelhado."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM feedback_events")
            cursor.execute("DELETE FROM decisoes")
            cursor.execute("DELETE FROM mpd_queue_state")
            conn.commit()
        logger.info("[FEEDBACK ZERADO] Banco de dados de feedback foi totalmente limpo.")

    def close(self):
        """Método de compatibilidade."""
        pass


# ── 2. DETECTOR DE FEEDBACK GRADUAL (HEURÍSTICA MPD) ──────────────────────────

class FeedbackDetector:
    """
    Analisa a transição entre duas faixas consecutivas no MPD e classifica o feedback gradual:
    - skip: troca manual com elapsed < 30 s (peso negativo forte);
    - parcial: troca manual com 30 s <= elapsed < 80% (positivo fraco proporcional a f);
    - completa: ouvida até >= 80% da duração ou fim natural (positivo forte).
    """

    def __init__(self, storage: Optional[FeedbackStorage] = None):
        self.storage = storage or FeedbackStorage()

    def evaluate_transition(
        self,
        prev_song: Dict[str, Any],
        prev_status: Dict[str, Any],
        current_song: Optional[Dict[str, Any]],
        current_status: Optional[Dict[str, Any]],
        is_automated_trigger: bool = False,
        meta_resolver: Optional[callable] = None
    ) -> Optional[str]:
        """
        Avalia o comportamento de reprodução da faixa anterior e registra no SQLite.
        """
        if not prev_song or not prev_status:
            return None

        if is_automated_trigger:
            logger.debug("Troca automática ignorada pelo detector de feedback.")
            return None

        # 1. Posições para detectar se foi navegação para trás (voltou faixa)
        old_pos = int(prev_status.get("song", prev_song.get("pos", 0)))
        new_pos = int(current_status.get("song", current_song.get("pos", 0))) if (current_status and current_song) else old_pos + 1

        if new_pos < old_pos:
            logger.info(f"Navegação para faixa anterior detectada ({old_pos} → {new_pos}). Ignorando.")
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

        # Metadados da faixa
        subgenero = ""
        cluster_id = ""
        track_id = prev_file
        if meta_resolver:
            meta = meta_resolver(prev_file)
            if meta:
                subgenero = meta.get("subgenero", "")
                cluster_id = str(meta.get("cluster_id", ""))
                track_id = meta.get("track_id", prev_file)

        # Regra 1: Faixa completa (>= 80% ou fim natural) -> Positivo Forte
        if ratio >= fcfg.COMPLETE_MIN_DURATION_RATIO or (duration - elapsed) <= fcfg.NATURAL_END_TOLERANCE_SECONDS:
            self.storage.record_event(
                track_id=track_id,
                caminho=prev_file,
                genero=subgenero,
                cluster_id=cluster_id,
                tipo="completa",
                peso=fcfg.COMPLETE_WEIGHT_BASE,
                elapsed=elapsed,
                duration=duration
            )
            return "completa"

        # Regra 2: Skip precoce (< 30s) -> Negativo Forte
        if elapsed < fcfg.SKIP_MAX_ELAPSED_SECONDS:
            self.storage.record_event(
                track_id=track_id,
                caminho=prev_file,
                genero=subgenero,
                cluster_id=cluster_id,
                tipo="skip",
                peso=fcfg.SKIP_WEIGHT_BASE,
                elapsed=elapsed,
                duration=duration
            )
            return "skip"

        # Regra 3: Feedback Gradual Parcial (>= 30s até 80%) -> Positivo Fraco proporcional a f
        f = min(1.0, max(0.1, ratio))
        self.storage.record_event(
            track_id=track_id,
            caminho=prev_file,
            genero=subgenero,
            cluster_id=cluster_id,
            tipo="parcial",
            peso=float(f),
            elapsed=elapsed,
            duration=duration
        )
        return "parcial"


# ── 3. MOTOR DE ALGORITMO ROCCHIO & RE-RANKING ────────────────────────────────

class FeedbackEngine:
    """
    Executa a adaptação da consulta via Rocchio gradual e o re-ranking penalizado das recomendações.
    """

    def __init__(self, storage: Optional[FeedbackStorage] = None):
        self.storage = storage or FeedbackStorage()

    def compute_rocchio_query(
        self,
        query_emb: np.ndarray,
        collection
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aplica o algoritmo de Rocchio gradual ao vetor de consulta:
        q' = α·q + β·média_ponderada(positivas) − γ·média_ponderada(puladas), normalizado em L2.
        As faixas positivas incluem 'completa' (peso 1.0) e 'parcial' (peso f).
        """
        recent_pos = self.storage.get_recent_positives(limit=fcfg.ROCCHIO_MAX_ITEMS)
        recent_neg = self.storage.get_recent_skips(limit=fcfg.ROCCHIO_MAX_ITEMS)

        stats = {
            "n_pos": len(recent_pos),
            "n_neg": len(recent_neg),
            "applied": False,
            "pos_events": recent_pos,
            "neg_events": recent_neg
        }

        if not recent_pos and not recent_neg:
            norm_val = np.linalg.norm(query_emb) + 1e-9
            return (query_emb / norm_val).astype(np.float32), stats

        # Carregar embeddings das faixas recentes do ChromaDB
        needed_ids = list({e["track_id"] for e in recent_pos + recent_neg})
        emb_map = {}
        if needed_ids:
            try:
                db_res = collection.get(ids=needed_ids, include=["embeddings"])
                found_ids = db_res.get("ids", [])
                found_embs = db_res.get("embeddings", [])
                for fid, femb in zip(found_ids, found_embs):
                    emb_arr = np.array(femb, dtype=np.float32)
                    norm_e = np.linalg.norm(emb_arr) + 1e-9
                    emb_map[fid] = emb_arr / norm_e
            except Exception as e:
                logger.warning(f"Erro ao buscar embeddings do Rocchio no ChromaDB: {e}")

        def _fetch_weighted_average(events: List[Dict[str, Any]], is_positive: bool = True) -> Optional[np.ndarray]:
            if not events:
                return None
            weighted_sum = np.zeros_like(query_emb, dtype=np.float32)
            weights_total = 0.0

            for i, ev in enumerate(events):
                tid = ev["track_id"]
                if tid in emb_map:
                    # Decaimento temporal por ordem do evento
                    w_time = np.exp(-fcfg.ROCCHIO_DECAY_RATE * i)
                    # Peso gradual da faixa (para parciais, peso proporcional a f)
                    w_mag = abs(float(ev.get("peso", 1.0)))
                    w = w_time * w_mag
                    weighted_sum += w * emb_map[tid]
                    weights_total += w

            if weights_total > 0.0:
                return weighted_sum / weights_total
            return None

        mean_pos = _fetch_weighted_average(recent_pos, is_positive=True)
        mean_neg = _fetch_weighted_average(recent_neg, is_positive=False)

        q_prime = fcfg.ROCCHIO_ALPHA * query_emb.copy()
        if mean_pos is not None:
            q_prime += fcfg.ROCCHIO_BETA * mean_pos
        if mean_neg is not None:
            q_prime -= fcfg.ROCCHIO_GAMMA * mean_neg

        # Normalização L2
        norm_val = np.linalg.norm(q_prime) + 1e-9
        q_prime_normalized = (q_prime / norm_val).astype(np.float32)
        stats["applied"] = True

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
        Aplica o re-ranking penalizado:
        score = sim(q, c) − λ · Σ w_i·sim(c, skip_i) − μ · pen_genero(c)
        Registra detalhes de penalidade e motivo em cada candidata.
        """
        if not candidates:
            return []

        recent_skips = self.storage.get_recent_skips(limit=fcfg.EXCLUDE_RECENT_SKIPS_COUNT)
        recent_skip_ids = {e["track_id"] for e in recent_skips}
        genre_penalties = self.storage.get_genre_penalties()

        # Embeddings das faixas puladas recentes
        skip_embs: List[Tuple[float, np.ndarray]] = []
        if recent_skips:
            try:
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
            except Exception as e:
                logger.warning(f"Erro ao carregar embeddings de skips: {e}")

        # Embeddings dos candidatos
        cand_ids = [c["track_id"] for c in candidates]
        c_emb_map = {}
        try:
            db_cands = collection.get(ids=cand_ids, include=["embeddings", "metadatas"])
            for tid, emb in zip(db_cands.get("ids", []), db_cands.get("embeddings", [])):
                c_emb_map[tid] = np.array(emb, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Erro ao carregar embeddings de candidatos: {e}")

        norm_q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        scored_candidates = []

        for cand in candidates:
            cid = cand["track_id"]
            c_file = id_to_file_map.get(cid, "")
            c_gen = cand.get("subgenero", "")

            # Motivo padrão
            motivo = "Avaliando"
            rejeitada = False

            # 1. Filtro de exclusão: faixa pulada recentemente
            if cid in recent_skip_ids:
                motivo = "Rejeitada (pulada recentemente)"
                cand["motivo"] = motivo
                cand["rejeitada"] = True
                continue

            # 2. Filtro de exclusão: faixa já na fila
            if c_file and (c_file in queued_files or os.path.basename(c_file) in queued_files):
                motivo = "Rejeitada (já na fila)"
                cand["motivo"] = motivo
                cand["rejeitada"] = True
                continue

            # 3. Filtro de cooldown: 3+ skips no mesmo gênero
            if c_gen and self.storage.is_genre_in_cooldown(c_gen):
                motivo = f"Rejeitada (cooldown do gênero '{c_gen}')"
                cand["motivo"] = motivo
                cand["rejeitada"] = True
                continue

            # Similaridade base
            cand_emb = c_emb_map.get(cid)
            if cand_emb is not None:
                norm_c = cand_emb / (np.linalg.norm(cand_emb) + 1e-9)
                base_sim = float(np.dot(norm_q, norm_c))
            else:
                base_sim = float(1.0 - cand.get("distance", 1.0))

            # Penalidade de faixas específicas puladas
            track_penalty = 0.0
            if cand_emb is not None and skip_embs:
                norm_c = cand_emb / (np.linalg.norm(cand_emb) + 1e-9)
                for w_i, s_emb in skip_embs:
                    sim_skip = float(np.dot(norm_c, s_emb))
                    if sim_skip > 0:
                        track_penalty += w_i * sim_skip

            # Penalidade do gênero
            genre_pen_val = genre_penalties.get(c_gen, 0.0)

            # Score final com pesos configurados
            final_score = base_sim - (fcfg.LAMBDA_PENALTY_TRACK * track_penalty) - (fcfg.MU_PENALTY_GENRE * genre_pen_val)

            cand_copy = dict(cand)
            cand_copy["sim_query"] = round(base_sim, 4)
            cand_copy["track_penalty"] = round(track_penalty, 4)
            cand_copy["genre_penalty"] = round(genre_pen_val, 4)
            cand_copy["final_score"] = round(final_score, 4)
            cand_copy["rejeitada"] = False
            cand_copy["motivo"] = "Aprovada para seleção"
            scored_candidates.append(cand_copy)

        # Ordena candidatos pelo score final decrescente
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_candidates


# ── 4. CLI UTILITÁRIO ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Utilitário do Sistema de Feedback.")
    parser.add_argument("--zerar", action="store_true", help="Limpa todo o histórico de feedback e decisões.")
    parser.add_argument("--status", action="store_true", help="Exibe resumo do banco de feedback.")
    args = parser.parse_args()

    storage = FeedbackStorage()
    if args.zerar:
        storage.clear_history()
        print("[OK] Banco de dados de feedback zerado com sucesso.")
    elif args.status:
        events = storage.get_recent_events(limit=5)
        penalties = storage.get_genre_penalties()
        cooldowns = storage.get_cooldown_genres()
        stats = storage.get_acceptance_stats()
        print(f"Total eventos: {stats['total_aceitas'] + stats['total_parciais'] + stats['total_skips']}")
        print(f"Taxa aceitação: {stats['taxa_aceitacao']:.1%}")
        print(f"Penalidades vigentes: {penalties}")
        print(f"Gêneros em cooldown: {cooldowns}")
