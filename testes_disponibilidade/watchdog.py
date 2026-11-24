"""
watchdog.py
-----------
Supervisor simples para o servico_consulta.py no Windows (sem systemd).

Comportamento:
  1. Sobe o servico_consulta.py como subprocesso
  2. A cada --intervalo-watchdog segundos faz GET /health
  3. Se o health check falhar N vezes consecutivas → reinicia o processo
  4. Registra cada reinício em watchdog.csv (para cálculo de MTTR)
  5. Nunca encerra automaticamente (Ctrl+C para parar)

Uso:
  python watchdog.py --porta 7700
                     --chroma-dir ../audio_pipeline/chroma_db
                     --colecao musicas_embeddings
                     --intervalo-watchdog 1
                     --falhas-consecutivas 2
                     --saida resultados/YYYYMMDD_HHMMSS
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from _cliente_servico import ErroServico, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(levelname)s %(message)s",
)

CABECALHO_CSV = [
    "timestamp",
    "evento",           # "start" | "restart" | "health_ok" | "health_fail"
    "pid",
    "falhas_consecutivas",
    "detalhe",
]

PYTHON_EXE = sys.executable  # mesmo Python do venv


def _agora_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _caminho_servico() -> str:
    return os.path.join(os.path.dirname(__file__), "servico_consulta.py")


def _subir_processo(porta: int, chroma_dir: str, colecao: str) -> subprocess.Popen:
    """Inicia o servico_consulta.py como subprocesso."""
    cmd = [
        PYTHON_EXE,
        _caminho_servico(),
        "--porta", str(porta),
        "--chroma-dir", chroma_dir,
        "--colecao", colecao,
    ]
    logging.info(f"Subindo: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # Garante que o processo filho não compartilha console no Windows
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )


def _processo_vivo(proc: subprocess.Popen) -> bool:
    return proc is not None and proc.poll() is None


def loop_watchdog(
    writer: csv.DictWriter,
    saida_f,
    porta: int,
    chroma_dir: str,
    colecao: str,
    intervalo: float,
    max_falhas: int,
    duracao: float | None,
) -> None:
    """Loop principal do watchdog."""
    base_url = f"http://127.0.0.1:{porta}"
    proc = None
    falhas_consecutivas = 0
    t_inicio = time.monotonic()

    def registrar(evento: str, pid: str | int, falhas: int, detalhe: str = ""):
        writer.writerow(
            {
                "timestamp": _agora_iso(),
                "evento": evento,
                "pid": pid,
                "falhas_consecutivas": falhas,
                "detalhe": detalhe,
            }
        )
        saida_f.flush()

    # Subida inicial
    proc = _subir_processo(porta, chroma_dir, colecao)
    logging.info(f"Processo iniciado PID={proc.pid}")
    registrar("start", proc.pid, 0, "inicio_inicial")

    # Aguarda o serviço estar pronto
    time.sleep(3)

    while True:
        # Verifica duração total (opcional)
        if duracao is not None and (time.monotonic() - t_inicio) >= duracao:
            logging.info("Duração máxima do watchdog atingida. Encerrando.")
            if _processo_vivo(proc):
                proc.terminate()
            break

        # Health check
        try:
            health(base_url, timeout=3)
            falhas_consecutivas = 0
            registrar("health_ok", proc.pid if proc else "", 0)
            logging.debug("Health OK")
        except ErroServico as exc:
            falhas_consecutivas += 1
            logging.warning(
                f"Health FAIL ({falhas_consecutivas}/{max_falhas}): {exc}"
            )
            registrar(
                "health_fail",
                proc.pid if proc else "",
                falhas_consecutivas,
                str(exc)[:120],
            )

            if falhas_consecutivas >= max_falhas:
                logging.error(
                    f"Serviço falhou {falhas_consecutivas} vezes consecutivas. Reiniciando..."
                )
                # Garante que o processo antigo está morto
                if _processo_vivo(proc):
                    try:
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception:
                        pass

                proc = _subir_processo(porta, chroma_dir, colecao)
                logging.info(f"Processo reiniciado PID={proc.pid}")
                registrar("restart", proc.pid, falhas_consecutivas)
                falhas_consecutivas = 0
                # Dá tempo para o processo subir antes do próximo health check
                time.sleep(3)

        time.sleep(intervalo)


def main():
    parser = argparse.ArgumentParser(description="Watchdog para o servico_consulta.py")
    parser.add_argument("--porta", type=int, default=7700, help="Porta do serviço")
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "audio_pipeline", "chroma_db"
        ),
        help="Diretório do ChromaDB",
    )
    parser.add_argument(
        "--colecao", type=str, default="musicas_embeddings", help="Nome da coleção"
    )
    parser.add_argument(
        "--intervalo-watchdog",
        type=float,
        default=1.0,
        help="Intervalo entre health checks (s)",
    )
    parser.add_argument(
        "--falhas-consecutivas",
        type=int,
        default=2,
        help="Falhas consecutivas antes de reiniciar",
    )
    parser.add_argument(
        "--saida",
        type=str,
        default=None,
        help="Pasta de saída para watchdog.csv",
    )
    parser.add_argument(
        "--duracao",
        type=float,
        default=None,
        help="Duração máxima do watchdog em minutos (None = infinito)",
    )
    args = parser.parse_args()

    if args.saida is None:
        ts_pasta = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.saida = os.path.join(
            os.path.dirname(__file__), "resultados", ts_pasta
        )
    os.makedirs(args.saida, exist_ok=True)

    caminho_csv = os.path.join(args.saida, "watchdog.csv")
    duracao_seg = args.duracao * 60 if args.duracao else None

    logging.info(
        f"Watchdog iniciando | porta={args.porta} | intervalo={args.intervalo_watchdog}s"
        f" | saída={caminho_csv}"
    )

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CABECALHO_CSV)
        writer.writeheader()
        f.flush()

        try:
            loop_watchdog(
                writer=writer,
                saida_f=f,
                porta=args.porta,
                chroma_dir=args.chroma_dir,
                colecao=args.colecao,
                intervalo=args.intervalo_watchdog,
                max_falhas=args.falhas_consecutivas,
                duracao=duracao_seg,
            )
        except KeyboardInterrupt:
            logging.info("Watchdog interrompido pelo usuário.")

    logging.info(f"Watchdog encerrado → {caminho_csv}")


if __name__ == "__main__":
    main()
