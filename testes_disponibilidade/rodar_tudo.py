"""
rodar_tudo.py
-------------
Runner da suíte de testes de disponibilidade.

Modos de Execução:
  1. Estresse (padrão):
     - Etapa 1: Restauração do índice ChromaDB
     - Etapa 2: Teste de carga com threads configuráveis
     - Etapa 3: Monitoramento + injeção de falhas (com Watchdog)
     - Etapa 4: Análise de métricas
     Salva em: resultados/<timestamp>/estresse/

  2. Linha de Base (--sem-falhas):
     - Somente monitoramento contínuo sem injeção de falhas
     - Etapa de Análise
     Salva em: resultados/<timestamp>/baseline/

  3. Ambos (--ambos):
     - Executa a Linha de Base seguida do Estresse
     - Gera relatório consolidado comparativo lado a lado
     Salva em: resultados/<timestamp>/

Parâmetros de Duração:
  --duracao          Duração do teste de estresse em minutos (padrão: 30)
  --duracao-baseline Duração da linha de base em minutos (padrão: 60)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RUNNER] %(levelname)s %(message)s",
    stream=sys.stdout,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

PIPELINE_DIR = os.path.abspath(os.path.join(HERE, "..", "audio_pipeline"))
CHROMA_DIR   = os.path.join(PIPELINE_DIR, "chroma_db")
COLECAO      = "musicas_embeddings"

NPZ_PATH     = os.path.join(PIPELINE_DIR, "results", "embeddings_1288.npz")
if not os.path.exists(NPZ_PATH):
    NPZ_PATH = os.path.join(PIPELINE_DIR, "results", "embeddings.npz")

META_PATH    = os.path.join(PIPELINE_DIR, "results", "metadata_1288.json")
if not os.path.exists(META_PATH):
    META_PATH = os.path.join(PIPELINE_DIR, "results", "metadata.json")


def _script(nome: str) -> str:
    return os.path.join(HERE, nome)


def _rodar(cmd: list[str], descricao: str, timeout: int | None = None) -> int:
    """Roda um subprocesso e espera sua conclusão."""
    logging.info(f">> {descricao}")
    logging.info(f"  CMD: {' '.join(cmd)}")
    proc = subprocess.run(cmd, timeout=timeout)
    if proc.returncode != 0:
        logging.warning(f"[AVISO] '{descricao}' terminou com código {proc.returncode}")
    else:
        logging.info(f"[OK] '{descricao}' concluído.")
    return proc.returncode


def _aguardar_servico(url: str, timeout: float = 60) -> bool:
    """Aguarda o servico_consulta.py responder ao /health."""
    import urllib.request
    prazo = time.monotonic() + timeout
    while time.monotonic() < prazo:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
                if r.status == 200:
                    logging.info("Serviço disponível.")
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    logging.error(f"Serviço não respondeu em {timeout}s.")
    return False


def _subir_servico(porta: int) -> subprocess.Popen:
    cmd = [
        PYTHON, _script("servico_consulta.py"),
        "--porta", str(porta),
        "--chroma-dir", CHROMA_DIR,
        "--colecao", COLECAO,
    ]
    logging.info(f"Subindo serviço: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )


def _derrubar_servico(proc: subprocess.Popen, porta: int = 7700) -> None:
    if proc:
        try:
            import psutil
            p = psutil.Process(proc.pid)
            for child in p.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            p.kill()
        except Exception:
            pass
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

    # Garante liberação da porta no Windows
    try:
        import psutil
        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr and conn.laddr.port == porta:
                if conn.pid and psutil.pid_exists(conn.pid):
                    try:
                        psutil.Process(conn.pid).kill()
                    except Exception:
                        pass
    except Exception:
        pass
    logging.info("Serviço encerrado.")


def _salvar_config(pasta: str, args: argparse.Namespace, modo: str):
    os.makedirs(pasta, exist_ok=True)
    duracao = args.duracao_baseline if modo == "baseline" else args.duracao
    cfg = {
        "modo": modo,
        "intervalo": args.intervalo,
        "falhas": args.falhas if modo == "estresse" else 0,
        "consultas": args.consultas if modo == "estresse" else 0,
        "duracao_min": duracao,
        "threads": args.threads if modo == "estresse" else 1,
        "porta": args.porta,
        "intervalo_watchdog": args.intervalo_watchdog,
        "semente": args.semente,
        "incluir_mpd": args.incluir_mpd if modo == "estresse" else False,
        "sem_falhas": modo == "baseline",
    }
    with open(os.path.join(pasta, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _executar_baseline(saida_baseline: str, args: argparse.Namespace, url_servico: str) -> None:
    """Executa a linha de base: apenas monitoramento contínuo sem injeção de falhas."""
    duracao_min = args.duracao_baseline
    logging.info("\n" + "=" * 60)
    logging.info(f"EXECUÇÃO: LINHA DE BASE (SEM FALHAS) — {duracao_min} min")
    logging.info(f"Saída: {saida_baseline}")
    logging.info("=" * 60)

    _salvar_config(saida_baseline, args, "baseline")
    duracao_mon_s = int(duracao_min * 60.0)

    cmd_watchdog = [
        PYTHON, _script("watchdog.py"),
        "--porta", str(args.porta),
        "--chroma-dir", CHROMA_DIR,
        "--colecao", COLECAO,
        "--intervalo-watchdog", str(args.intervalo_watchdog),
        "--falhas-consecutivas", "2",
        "--saida", saida_baseline,
        "--duracao", str(duracao_min + 0.5),
    ]
    logging.info(f"Subindo watchdog: {' '.join(cmd_watchdog)}")
    proc_watchdog = subprocess.Popen(
        cmd_watchdog,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    time.sleep(4)
    _aguardar_servico(url_servico, timeout=60)

    cmd_monitor = [
        PYTHON, _script("monitor.py"),
        "--url", url_servico,
        "--intervalo", str(args.intervalo),
        "--duracao", str(duracao_min),
        "--timeout", "0.3",
        "--saida", saida_baseline,
    ]
    logging.info(f"Iniciando monitoramento da linha de base: {' '.join(cmd_monitor)}")
    proc_monitor = subprocess.Popen(
        cmd_monitor,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    logging.info(f"Monitorando continuamente por {duracao_min} min...")
    try:
        proc_monitor.wait(timeout=duracao_mon_s + 30)
    except subprocess.TimeoutExpired:
        logging.warning("Monitor demorou mais que o esperado; encerrando.")
        proc_monitor.terminate()

    try:
        proc_watchdog.terminate()
        proc_watchdog.wait(timeout=10)
    except Exception:
        proc_watchdog.kill()

    time.sleep(2)
    logging.info("[OK] Linha de base concluída com sucesso.")


def _executar_estresse(saida_estresse: str, args: argparse.Namespace, url_servico: str) -> None:
    """Executa o teste de estresse: restauração, carga, monitoramento + falhas injetadas."""
    logging.info("\n" + "=" * 60)
    logging.info(f"EXECUÇÃO: ESTRESSE (FALHAS INJETADAS) — {args.duracao} min")
    logging.info(f"Saída: {saida_estresse}")
    logging.info("=" * 60)

    _salvar_config(saida_estresse, args, "estresse")
    duracao_mon_s = int(args.duracao * 60.0)

    # ----------------------------------------------------------------
    # ETAPA 1: Restauração
    # ----------------------------------------------------------------
    logging.info("\n[ETAPA 1/3] Teste de Restauração do Índice")
    _rodar(
        [
            PYTHON, _script("restauracao.py"),
            "--npz", NPZ_PATH,
            "--metadata", META_PATH,
            "--chroma-dir-original", CHROMA_DIR,
            "--colecao", COLECAO,
            "--saida", saida_estresse,
        ],
        "restauracao",
        timeout=600,
    )

    # ----------------------------------------------------------------
    # ETAPA 2: Teste de Carga
    # ----------------------------------------------------------------
    logging.info(f"\n[ETAPA 2/3] Teste de Carga ({args.consultas} consultas, {args.threads} thread(s))")
    proc_servico = _subir_servico(args.porta)
    try:
        if _aguardar_servico(url_servico, timeout=60):
            _rodar(
                [
                    PYTHON, _script("carga.py"),
                    "--url", url_servico,
                    "--consultas", str(args.consultas),
                    "--threads", str(args.threads),
                    "--timeout-consulta", "5",
                    "--semente", str(args.semente),
                    "--saida", saida_estresse,
                ],
                "carga",
                timeout=600,
            )
        else:
            logging.error("Serviço não disponível; etapa de carga pulada.")
    finally:
        _derrubar_servico(proc_servico)

    time.sleep(2)

    # ----------------------------------------------------------------
    # ETAPA 3: Monitoramento + Injeção de Falhas (com Watchdog)
    # ----------------------------------------------------------------
    logging.info("\n[ETAPA 3/3] Monitoramento + Injeção de Falhas")

    cmd_watchdog = [
        PYTHON, _script("watchdog.py"),
        "--porta", str(args.porta),
        "--chroma-dir", CHROMA_DIR,
        "--colecao", COLECAO,
        "--intervalo-watchdog", str(args.intervalo_watchdog),
        "--falhas-consecutivas", "2",
        "--saida", saida_estresse,
        "--duracao", str(args.duracao + 0.5),
    ]
    logging.info(f"Subindo watchdog: {' '.join(cmd_watchdog)}")
    proc_watchdog = subprocess.Popen(
        cmd_watchdog,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    time.sleep(4)
    _aguardar_servico(url_servico, timeout=60)

    cmd_monitor = [
        PYTHON, _script("monitor.py"),
        "--url", url_servico,
        "--intervalo", str(args.intervalo),
        "--duracao", str(args.duracao),
        "--timeout", "0.3",
        "--saida", saida_estresse,
    ]
    logging.info(f"Subindo monitor: {' '.join(cmd_monitor)}")
    proc_monitor = subprocess.Popen(
        cmd_monitor,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    time.sleep(10)
    cmd_injecao = [
        PYTHON, _script("injecao_falhas.py"),
        "--falhas", str(args.falhas),
        "--porta-servico", str(args.porta),
        "--saida", saida_estresse,
        "--semente", str(args.semente),
    ]
    if args.incluir_mpd:
        cmd_injecao.append("--incluir-mpd")

    _rodar(cmd_injecao, "injecao_falhas", timeout=duracao_mon_s)

    logging.info(f"Aguardando monitor completar {args.duracao} min...")
    try:
        proc_monitor.wait(timeout=duracao_mon_s + 30)
    except subprocess.TimeoutExpired:
        logging.warning("Monitor demorou mais do que o esperado; encerrando.")
        proc_monitor.terminate()

    try:
        proc_watchdog.terminate()
        proc_watchdog.wait(timeout=10)
    except Exception:
        proc_watchdog.kill()

    time.sleep(2)
    logging.info("[OK] Estresse concluído com sucesso.")


def main():
    parser = argparse.ArgumentParser(description="Runner da suíte de testes de disponibilidade")
    parser.add_argument("--intervalo", type=float, default=1.0, help="Intervalo do monitor (s)")
    parser.add_argument("--falhas", type=int, default=10, help="Falhas a injetar no modo estresse [padrão: 10]")
    parser.add_argument("--consultas", type=int, default=1000, help="Consultas no teste de carga [padrão: 1000]")
    parser.add_argument("--duracao", type=float, default=30.0, help="Duração do monitoramento de estresse (min) [padrão: 30]")
    parser.add_argument("--duracao-baseline", type=float, default=60.0, help="Duração do monitoramento da linha de base (min) [padrão: 60]")
    parser.add_argument("--threads", type=int, default=1, help="Threads do teste de carga [padrão: 1]")
    parser.add_argument("--porta", type=int, default=7700, help="Porta do servico_consulta.py")
    parser.add_argument("--intervalo-watchdog", type=float, default=1.0, help="Health check do watchdog (s)")
    parser.add_argument("--slo-disp", type=float, default=99.9)
    parser.add_argument("--slo-mttr", type=float, default=10.0)
    parser.add_argument("--slo-sucesso", type=float, default=99.9)
    parser.add_argument("--slo-restauracao", type=float, default=60.0)
    parser.add_argument("--semente", type=int, default=42, help="Semente aleatória")
    parser.add_argument("--incluir-mpd", action="store_true")
    parser.add_argument("--sem-falhas", action="store_true", help="Executa apenas linha de base (sem falhas)")
    parser.add_argument("--ambos", action="store_true", help="Executa linha de base e estresse em sequência")
    parser.add_argument("--saida", type=str, default=None, help="Pasta de saída personalizada")
    args = parser.parse_args()

    url_servico = f"http://127.0.0.1:{args.porta}"

    ts_pasta = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_raiz = args.saida if args.saida else os.path.join(HERE, "resultados", ts_pasta)
    os.makedirs(pasta_raiz, exist_ok=True)

    logging.info("=" * 60)
    logging.info(f"SUÍTE DE DISPONIBILIDADE — início: {ts_pasta}")
    logging.info(f"Destino raiz: {pasta_raiz}")
    logging.info(f"Modo: {'Linha de Base' if args.sem_falhas else ('Ambos (Baseline + Estresse)' if args.ambos else 'Estresse')}")
    logging.info(f"Duração Estresse: {args.duracao} min | Duração Baseline: {args.duracao_baseline} min")
    logging.info("=" * 60)

    cmd_analise_base = [
        PYTHON, _script("analise.py"),
        "--slo-disp", str(args.slo_disp),
        "--slo-mttr", str(args.slo_mttr),
        "--slo-sucesso", str(args.slo_sucesso),
        "--slo-restauracao", str(args.slo_restauracao),
    ]

    if args.sem_falhas:
        pasta_baseline = os.path.join(pasta_raiz, "baseline")
        _executar_baseline(pasta_baseline, args, url_servico)
        _rodar(cmd_analise_base + ["--saida", pasta_baseline], "analise_baseline")
        saida_final = pasta_baseline

    elif args.ambos:
        pasta_baseline = os.path.join(pasta_raiz, "baseline")
        pasta_estresse = os.path.join(pasta_raiz, "estresse")
        _executar_baseline(pasta_baseline, args, url_servico)
        _executar_estresse(pasta_estresse, args, url_servico)
        _rodar(cmd_analise_base + ["--saida", pasta_raiz], "analise_consolidada")
        saida_final = pasta_raiz

    else:
        pasta_estresse = os.path.join(pasta_raiz, "estresse")
        _executar_estresse(pasta_estresse, args, url_servico)
        _rodar(cmd_analise_base + ["--saida", pasta_estresse], "analise_estresse")
        saida_final = pasta_estresse

    logging.info("=" * 60)
    logging.info(f"[OK] Execução concluída. Resultados disponíveis em: {saida_final}")
    logging.info(f"  relatorio.md -> {os.path.join(saida_final, 'relatorio.md')}")
    logging.info(f"  resumo.csv   -> {os.path.join(saida_final, 'resumo.csv')}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
