"""
carga.py
--------
Teste de carga: envia Y consultas ao servico_consulta.py (via HTTP) e mede:
  - Taxa de sucesso (%)
  - Latência: média, p50, p95, p99, máxima

Suporta modo concorrente via --threads.
Grava resultados em carga.csv.

Uso:
  python carga.py --url http://127.0.0.1:7700
                  --consultas 1000
                  --threads 1
                  --timeout-consulta 5
                  --semente 42
                  --saida resultados/YYYYMMDD_HHMMSS
"""

import argparse
import csv
import logging
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone
from statistics import mean, median, quantiles, stdev

sys.path.insert(0, os.path.dirname(__file__))
from _cliente_servico import ErroServico, query, random_track

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CARGA] %(levelname)s %(message)s",
)

CABECALHO_CSV = ["timestamp", "thread_id", "track_id", "resultado", "latencia_ms", "detalhe"]


def _agora_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _buscar_tracks(base_url: str, n: int, timeout: float, semente: int) -> list[str]:
    """Busca até N track_ids para usar como entrada das consultas."""
    rng = random.Random(semente)
    tracks: set[str] = set()
    tentativas = 0
    while len(tracks) < min(n, 50) and tentativas < 200:
        tentativas += 1
        try:
            tid = random_track(base_url, timeout=timeout)
            if tid:
                tracks.add(tid)
        except ErroServico:
            time.sleep(0.2)
    if not tracks:
        raise RuntimeError("Não foi possível obter track_ids do serviço.")
    tracks_list = list(tracks)
    # Repete para atingir n consultas
    resultado = [rng.choice(tracks_list) for _ in range(n)]
    return resultado


def _executar_consultas(
    base_url: str,
    track_ids: list[str],
    timeout: float,
    thread_id: int,
    registros: list,
    lock: threading.Lock,
) -> None:
    """Executa todas as consultas desta thread e acumula registros."""
    for track_id in track_ids:
        ts = _agora_iso()
        t0 = time.perf_counter()
        try:
            resp = query(base_url, track_id, k=5, timeout=timeout)
            lat = round((time.perf_counter() - t0) * 1000, 3)
            reg = {
                "timestamp": ts,
                "thread_id": thread_id,
                "track_id": track_id,
                "resultado": "sucesso",
                "latencia_ms": lat,
                "detalhe": f"k={len(resp.get('results', []))}",
            }
        except ErroServico as exc:
            lat = round((time.perf_counter() - t0) * 1000, 3)
            reg = {
                "timestamp": ts,
                "thread_id": thread_id,
                "track_id": track_id,
                "resultado": "falha",
                "latencia_ms": lat,
                "detalhe": str(exc)[:120],
            }
        with lock:
            registros.append(reg)


def _percentil(dados: list[float], p: float) -> float:
    """Calcula um percentil simples sem numpy."""
    if not dados:
        return 0.0
    dados_ord = sorted(dados)
    idx = (len(dados_ord) - 1) * p / 100
    baixo = int(idx)
    alto = min(baixo + 1, len(dados_ord) - 1)
    return dados_ord[baixo] + (dados_ord[alto] - dados_ord[baixo]) * (idx - baixo)


def main():
    parser = argparse.ArgumentParser(description="Teste de carga do servico_consulta.py")
    parser.add_argument("--url", default="http://127.0.0.1:7700", help="URL base do serviço")
    parser.add_argument("--consultas", type=int, default=1000, help="Total de consultas")
    parser.add_argument("--threads", type=int, default=1, help="Threads concorrentes")
    parser.add_argument("--timeout-consulta", type=float, default=5.0, help="Timeout por consulta (s)")
    parser.add_argument("--semente", type=int, default=42, help="Semente aleatória")
    parser.add_argument(
        "--saida",
        type=str,
        default=None,
        help="Pasta de saída para carga.csv",
    )
    args = parser.parse_args()

    if args.saida is None:
        ts_pasta = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.saida = os.path.join(
            os.path.dirname(__file__), "resultados", ts_pasta
        )
    os.makedirs(args.saida, exist_ok=True)

    caminho_csv = os.path.join(args.saida, "carga.csv")
    logging.info(
        f"Teste de carga | consultas={args.consultas} | threads={args.threads}"
        f" | url={args.url} | saída={caminho_csv}"
    )

    # Pré-carrega track_ids
    logging.info("Obtendo track_ids de referência...")
    track_ids = _buscar_tracks(
        args.url, args.consultas, args.timeout_consulta, args.semente
    )
    logging.info(f"{len(track_ids)} consultas preparadas. Iniciando carga...")

    # Divide as consultas entre as threads
    fatias: list[list[str]] = [[] for _ in range(args.threads)]
    for i, tid in enumerate(track_ids):
        fatias[i % args.threads].append(tid)

    registros: list[dict] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    t_inicio = time.perf_counter()

    for tid_idx, fatia in enumerate(fatias):
        t = threading.Thread(
            target=_executar_consultas,
            args=(args.url, fatia, args.timeout_consulta, tid_idx, registros, lock),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    duracao_total = time.perf_counter() - t_inicio

    # Salva CSV
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CABECALHO_CSV)
        writer.writeheader()
        writer.writerows(registros)

    # Estatísticas
    sucessos = [r for r in registros if r["resultado"] == "sucesso"]
    falhas = [r for r in registros if r["resultado"] == "falha"]
    lats = [r["latencia_ms"] for r in sucessos if isinstance(r["latencia_ms"], (int, float))]

    total = len(registros)
    taxa_sucesso = (len(sucessos) / total * 100) if total > 0 else 0

    p50 = _percentil(lats, 50) if lats else 0
    p95 = _percentil(lats, 95) if lats else 0
    p99 = _percentil(lats, 99) if lats else 0
    lat_max = max(lats) if lats else 0
    lat_media = mean(lats) if lats else 0
    lat_std = stdev(lats) if len(lats) > 1 else 0

    resumo = {
        "threads": args.threads,
        "total_consultas": total,
        "sucessos": len(sucessos),
        "falhas": len(falhas),
        "taxa_sucesso_pct": round(taxa_sucesso, 4),
        "duracao_total_s": round(duracao_total, 3),
        "throughput_qps": round(total / duracao_total, 2) if duracao_total > 0 else 0,
        "latencia_media_ms": round(lat_media, 3),
        "latencia_p50_ms": round(p50, 3),
        "latencia_p95_ms": round(p95, 3),
        "latencia_p99_ms": round(p99, 3),
        "latencia_max_ms": round(lat_max, 3),
        "latencia_std_ms": round(lat_std, 3),
    }

    # Salva resumo JSON e imprime
    import json
    caminho_resumo = os.path.join(args.saida, "carga_resumo.json")
    with open(caminho_resumo, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    logging.info("=" * 55)
    logging.info(f"  Total consultas : {total}")
    logging.info(f"  Sucessos        : {len(sucessos)} ({taxa_sucesso:.2f}%)")
    logging.info(f"  Falhas          : {len(falhas)}")
    logging.info(f"  Duração total   : {duracao_total:.2f}s")
    logging.info(f"  Throughput      : {resumo['throughput_qps']} qps")
    logging.info(f"  Lat média       : {lat_media:.2f}ms")
    logging.info(f"  Lat p50/p95/p99 : {p50:.2f}/{p95:.2f}/{p99:.2f}ms")
    logging.info(f"  Lat máxima      : {lat_max:.2f}ms")
    logging.info("=" * 55)
    logging.info(f"Resultados → {caminho_csv}")


if __name__ == "__main__":
    main()
