"""
monitor.py
----------
Monitoramento contínuo de disponibilidade do servico_consulta.py.

Características:
  - Comunica exclusivamente via 127.0.0.1 (evita overhead e DNS de 'localhost')
  - Timeout curto de conexão e leitura (padrão 0.3 s, configurável por --timeout)
  - Cadência fixa estrita: cada sondagem é agendada para t0 + k * N
    (sem dormir N segundos após cada resposta, garantindo amostragem contínua durante quedas)
  - Grava timestamp de início e de fim de cada sondagem no monitor.csv

Uso:
  python monitor.py --url http://127.0.0.1:7700
                    --intervalo 1.0
                    --duracao 30.0
                    --timeout 0.3
                    --saida resultados/YYYYMMDD_HHMMSS
"""

import argparse
import csv
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone

# Importa cliente HTTP do próprio pacote de testes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cliente_servico import ErroServico, health, query, random_track

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MONITOR] %(levelname)s %(message)s",
    stream=sys.stdout,
)

CABECALHO_CSV = ["timestamp_inicio", "timestamp_fim", "resultado", "latencia_ms", "detalhe"]


def _agora_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _sanitizar_url(url: str) -> str:
    """Garante o uso de 127.0.0.1 em vez de localhost."""
    return url.replace("localhost", "127.0.0.1")


def _popular_pool_faixas(base_url: str, tamanho: int = 20, timeout: float = 2.0) -> list[str]:
    """Obtém uma lista inicial de track_ids válidos para usar durante o monitoramento."""
    pool: list[str] = []
    for _ in range(tamanho):
        try:
            tid = random_track(base_url, timeout=timeout)
            if tid and tid not in pool:
                pool.append(tid)
        except Exception:
            break
    if not pool:
        pool.append("0001")
    return pool


def executar_sondagem(
    writer: csv.DictWriter,
    base_url: str,
    timeout: float,
    cache_tracks: list[str],
) -> dict:
    """Executa exatamente uma consulta de similaridade cronometrada e grava no CSV."""
    track_id = random.choice(cache_tracks) if cache_tracks else "0001"

    ts_inicio = _agora_iso()
    t0 = time.perf_counter()
    try:
        resp = query(base_url, track_id, k=5, timeout=timeout)
        t1 = time.perf_counter()
        ts_fim = _agora_iso()
        latencia_ms = round((t1 - t0) * 1000, 3)
        registro = {
            "timestamp_inicio": ts_inicio,
            "timestamp_fim": ts_fim,
            "resultado": "sucesso",
            "latencia_ms": latencia_ms,
            "detalhe": f"k={len(resp.get('results', []))}",
        }
    except ErroServico as exc:
        t1 = time.perf_counter()
        ts_fim = _agora_iso()
        latencia_ms = round((t1 - t0) * 1000, 3)
        registro = {
            "timestamp_inicio": ts_inicio,
            "timestamp_fim": ts_fim,
            "resultado": "falha",
            "latencia_ms": latencia_ms,
            "detalhe": str(exc)[:120],
        }

    writer.writerow(registro)
    return registro


def main():
    parser = argparse.ArgumentParser(description="Monitor de disponibilidade contínuo")
    parser.add_argument(
        "--url", default="http://127.0.0.1:7700", help="URL base do servico_consulta (padrão 127.0.0.1:7700)"
    )
    parser.add_argument(
        "--intervalo", type=float, default=1.0, help="Intervalo N da cadência de sondagens em segundos (padrão: 1.0)"
    )
    parser.add_argument(
        "--duracao", type=float, default=30.0, help="Duração total do monitoramento em minutos (padrão: 30.0)"
    )
    parser.add_argument(
        "--timeout",
        "--timeout-consulta",
        type=float,
        default=0.3,
        dest="timeout",
        help="Timeout curto de conexão/leitura em segundos (padrão: 0.3 s)",
    )
    parser.add_argument(
        "--saida",
        type=str,
        default=None,
        help="Pasta de saída para monitor.csv",
    )
    args = parser.parse_args()

    args.url = _sanitizar_url(args.url)

    if args.saida is None:
        ts_pasta = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.saida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", ts_pasta)
    os.makedirs(args.saida, exist_ok=True)

    caminho_csv = os.path.join(args.saida, "monitor.csv")
    duracao_seg = args.duracao * 60.0

    logging.info(
        f"Iniciando monitor | URL={args.url} | intervalo={args.intervalo}s"
        f" | timeout={args.timeout}s | duração={args.duracao}min | saída={caminho_csv}"
    )

    # Aguarda o serviço estar disponível antes de fixar t0
    logging.info("Aguardando o serviço ficar disponível...")
    prazo_espera = time.monotonic() + 60.0
    while time.monotonic() < prazo_espera:
        try:
            health(args.url, timeout=args.timeout if args.timeout > 1.0 else 1.0)
            logging.info("Serviço disponível. Carregando pool de faixas para teste...")
            break
        except ErroServico:
            time.sleep(0.5)
    else:
        logging.warning("Serviço não respondeu no prazo; prosseguindo.")

    cache_tracks = _popular_pool_faixas(args.url, tamanho=20, timeout=1.0)
    logging.info(f"Pool de teste pronto: {len(cache_tracks)} faixa(s) em cache. Iniciando cadência fixa.")

    t0 = time.monotonic()
    t_limite = t0 + duracao_seg
    k = 0
    sucesso = falha = 0

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CABECALHO_CSV)
        writer.writeheader()
        f.flush()

        while True:
            alvo = t0 + (k * args.intervalo)
            agora = time.monotonic()

            if agora >= t_limite:
                break

            # Se o instante agendado ainda está no futuro, dorme a diferença
            dormir = alvo - agora
            if dormir > 0:
                time.sleep(dormir)

            try:
                reg = executar_sondagem(writer, args.url, args.timeout, cache_tracks)
                f.flush()

                if reg["resultado"] == "sucesso":
                    sucesso += 1
                else:
                    falha += 1
                    logging.warning(f"FALHA | {reg['detalhe']}")
            except Exception as exc:
                falha += 1
                logging.error(f"Erro inesperado no monitor: {exc}")

            k += 1

            # Caso a sondagem tenha demorado mais que o intervalo N, avança k para não acumular atrasos
            agora_pos = time.monotonic()
            while (t0 + k * args.intervalo) < agora_pos and (t0 + k * args.intervalo) < t_limite:
                k += 1

    total = sucesso + falha
    pct = (sucesso / total * 100) if total > 0 else 0
    logging.info(f"Monitoramento concluído: {sucesso}/{total} sucessos ({pct:.2f}%) -> {caminho_csv}")


if __name__ == "__main__":
    main()
