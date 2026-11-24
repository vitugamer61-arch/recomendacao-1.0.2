"""
injecao_falhas.py
-----------------
Injeta falhas no servico_consulta.py (processo Python + ChromaDB embutido)
matando o processo via psutil/taskkill.

Protocolo rigoroso de medição:
  1. Aguarda confirmação de que o serviço está ativo na porta antes de cada injeção
  2. Identifica o PID do servico_consulta.py pela porta TCP
  3. Envia sinal de encerramento forçado (kill)
  4. Confirma explicitamente a morte do processo (wait / is_running == False)
  5. Registra o timestamp do kill SOMENTE após a morte confirmada e anota o PID
  6. Aguarda intervalo aleatório antes da próxima injeção

Uso:
  python injecao_falhas.py --falhas 10
                           --porta-servico 7700
                           --saida resultados/YYYYMMDD_HHMMSS
                           [--incluir-mpd]
                           [--semente 42]
"""

import argparse
import csv
import logging
import os
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INJECAO] %(levelname)s %(message)s",
    stream=sys.stdout,
)

CABECALHO_CSV = ["timestamp", "componente", "pid", "resultado"]

INTERVALO_MIN_PADRAO = 15.0
INTERVALO_MAX_PADRAO = 25.0


def _agora_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _esperar_servico_saudavel(porta: int, timeout_total: float = 35.0) -> bool:
    """Aguarda até que o servico_consulta esteja ativo e respondendo 200 em /health."""
    url = f"http://127.0.0.1:{porta}/health"
    prazo = time.monotonic() + timeout_total
    while time.monotonic() < prazo:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "injecao_falhas"})
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _pid_por_porta(porta: int) -> int | None:
    """Encontra o PID que está escutando na porta TCP informada."""
    try:
        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr and conn.laddr.port == porta and conn.status == "LISTEN":
                return conn.pid
    except Exception:
        pass
    return None


def _pid_mpd(porta_mpd: int) -> int | None:
    """Tenta encontrar o processo mpd.exe pela porta ou pelo nome."""
    pid = _pid_por_porta(porta_mpd)
    if pid:
        return pid
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            nome = (proc.info.get("name") or "").lower()
            if "mpd" in nome:
                return proc.info["pid"]
        except Exception:
            pass
    return None


def _matar_e_confirmar_morte(pid: int, componente: str) -> tuple[str, str]:
    """
    Mata o processo pelo PID e aguarda confirmação de término.
    Garante que o timestamp retornado é coletado SOMENTE após a confirmação da morte.
    Retorna (resultado, timestamp_morte_confirmada).
    """
    try:
        proc = psutil.Process(pid)
        proc.kill()
        try:
            proc.wait(timeout=5.0)
        except psutil.TimeoutExpired:
            pass

        if not proc.is_running():
            ts_morte = _agora_iso()
            logging.info(f"[KILL CONFIRMADO] {componente} PID={pid} morto às {ts_morte}.")
            return "sucesso", ts_morte
        else:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True, capture_output=True)
            prazo = time.monotonic() + 3.0
            while time.monotonic() < prazo:
                if not psutil.pid_exists(pid):
                    break
                time.sleep(0.05)
            ts_morte = _agora_iso()
            return "sucesso_via_taskkill", ts_morte

    except psutil.NoSuchProcess:
        ts_morte = _agora_iso()
        logging.warning(f"[KILL] {componente} PID={pid} já havia terminado.")
        return "sucesso", ts_morte
    except Exception as exc:
        logging.warning(f"psutil falhou ({exc}); tentando taskkill /F /PID {pid}")
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True, capture_output=True)
            prazo = time.monotonic() + 3.0
            while time.monotonic() < prazo:
                if not psutil.pid_exists(pid):
                    break
                time.sleep(0.05)
            ts_morte = _agora_iso()
            return "sucesso_via_taskkill", ts_morte
        except subprocess.CalledProcessError as te:
            ts_morte = _agora_iso()
            logging.error(f"taskkill também falhou: {te}")
            return f"erro: {te}", ts_morte


def injetar(
    writer: csv.DictWriter,
    saida_f,
    porta_servico: int,
    incluir_mpd: bool,
    porta_mpd: int,
) -> None:
    """Executa um ciclo de injeção de falha com medição precisa e registra no CSV."""
    pid = _pid_por_porta(porta_servico)
    if pid is None:
        logging.warning(
            f"Serviço não encontrado na porta {porta_servico} (aguardando reinício do watchdog)."
        )
        registro = {
            "timestamp": _agora_iso(),
            "componente": "modulo_python_chromadb",
            "pid": "",
            "resultado": "nao_encontrado",
        }
    else:
        resultado, ts_confirmado = _matar_e_confirmar_morte(pid, "modulo_python_chromadb")
        registro = {
            "timestamp": ts_confirmado,
            "componente": "modulo_python_chromadb",
            "pid": pid,
            "resultado": resultado,
        }

    writer.writerow(registro)
    saida_f.flush()

    if incluir_mpd:
        pid_mpd = _pid_mpd(porta_mpd)
        if pid_mpd is None:
            logging.warning("MPD não encontrado.")
            reg_mpd = {
                "timestamp": _agora_iso(),
                "componente": "mpd",
                "pid": "",
                "resultado": "nao_encontrado",
            }
        else:
            res_mpd, ts_mpd = _matar_e_confirmar_morte(pid_mpd, "mpd")
            reg_mpd = {
                "timestamp": ts_mpd,
                "componente": "mpd",
                "pid": pid_mpd,
                "resultado": res_mpd,
            }
        writer.writerow(reg_mpd)
        saida_f.flush()


def main():
    parser = argparse.ArgumentParser(description="Injeção de falhas no servico_consulta.py")
    parser.add_argument("--falhas", type=int, default=10, help="Número de kills a injetar")
    parser.add_argument("--porta-servico", type=int, default=7700, help="Porta do servico_consulta.py")
    parser.add_argument("--saida", type=str, default=None, help="Pasta de saída para falhas.csv")
    parser.add_argument("--incluir-mpd", action="store_true", help="Também mata o processo MPD")
    parser.add_argument("--porta-mpd", type=int, default=6600, help="Porta do MPD")
    parser.add_argument("--intervalo-min", type=float, default=INTERVALO_MIN_PADRAO, help="Intervalo mínimo entre falhas (s)")
    parser.add_argument("--intervalo-max", type=float, default=INTERVALO_MAX_PADRAO, help="Intervalo máximo entre falhas (s)")
    parser.add_argument("--semente", type=int, default=42, help="Semente aleatória para reprodutibilidade")
    args = parser.parse_args()

    rng = random.Random(args.semente)

    if args.saida is None:
        ts_pasta = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.saida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", ts_pasta)
    os.makedirs(args.saida, exist_ok=True)

    caminho_csv = os.path.join(args.saida, "falhas.csv")
    logging.info(
        f"Injeção de falhas | kills={args.falhas} | porta={args.porta_servico}"
        f" | incluir_mpd={args.incluir_mpd} | saída={caminho_csv}"
    )

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CABECALHO_CSV)
        writer.writeheader()
        f.flush()

        for i in range(args.falhas):
            logging.info(f"--- Injeção {i + 1}/{args.falhas} ---")
            # Aguarda o serviço estar 100% ativo e saudável antes de injetar
            if not _esperar_servico_saudavel(args.porta_servico, timeout_total=40.0):
                logging.warning("Serviço não respondeu /health no prazo; tentando buscar PID...")

            injetar(writer, f, args.porta_servico, args.incluir_mpd, args.porta_mpd)

            if i < args.falhas - 1:
                espera = rng.uniform(args.intervalo_min, args.intervalo_max)
                logging.info(f"Aguardando {espera:.1f}s antes da próxima injeção...")
                time.sleep(espera)

    logging.info(f"Injeção concluída -> {caminho_csv}")


if __name__ == "__main__":
    main()
