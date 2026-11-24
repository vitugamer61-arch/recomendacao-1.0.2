"""
_cliente_servico.py
-------------------
Módulo auxiliar interno: cliente HTTP para o servico_consulta.py.
Usado por monitor.py e carga.py — nunca abrem PersistentClient diretamente.
"""

import json
import urllib.request
import urllib.error

TIMEOUT_PADRAO = 5  # segundos


class ErroServico(Exception):
    """Falha de conexão ou resposta HTTP inesperada."""


def health(base_url: str, timeout: float = TIMEOUT_PADRAO) -> dict:
    """
    GET /health → {"status": "ok", "count": N}
    Lança ErroServico em caso de falha.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        raise ErroServico(str(exc)) from exc


def random_track(base_url: str, timeout: float = TIMEOUT_PADRAO) -> str:
    """
    GET /random_track → track_id (str)
    Lança ErroServico em caso de falha.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/random_track", timeout=timeout) as r:
            return json.loads(r.read())["track_id"]
    except Exception as exc:
        raise ErroServico(str(exc)) from exc


def query(
    base_url: str,
    track_id: str,
    k: int = 5,
    timeout: float = TIMEOUT_PADRAO,
) -> dict:
    """
    POST /query → {"results": [...], "latencia_ms": float}
    Lança ErroServico em caso de falha ou timeout.
    """
    payload = json.dumps({"track_id": track_id, "k": k}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        raise ErroServico(str(exc)) from exc
