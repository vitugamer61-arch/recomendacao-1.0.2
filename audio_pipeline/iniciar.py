"""
iniciar.py — Ponto de entrada único para o sistema de recomendação e clusterização musical.

Orquestra todo o fluxo em 6 etapas consecutivas:
1. Verificação e inicialização automática do MPD (porta 6600).
2. Verificação do ChromaDB e indexação automática de novas faixas da pasta de músicas.
3. Clusterização: execução do K-Means automático ou atribuição incremental de novas faixas.
4. Sincronização de playlists por cluster no MPD e em arquivos .m3u.
5. Inicialização do Monitor de Precisão (Dashboard Streamlit) em processo paralelo.
6. Inicialização do recomendador em primeiro plano no modo "mais faixas do mesmo cluster".
"""

import os
import sys
import time
import json
import socket
import logging
import argparse
import subprocess
import webbrowser
from pathlib import Path

# Garante path de importação local
_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("iniciar")

# Módulos pesados (chromadb, scikit-learn, pandas, matplotlib...) são carregados por _carregar_modulos(),
# DEPOIS de o banner aparecer. Importá-los no topo deixava o console mudo por vários segundos na partida,
# dando a impressão de que o sistema não estava funcionando.
vector_store = None
index_library = None
clustering = None
mpd_integration = None


def _carregar_modulos() -> None:
    """Importa os módulos pesados e informa o tempo gasto."""
    global vector_store, index_library, clustering, mpd_integration
    t0 = time.time()
    logger.info("Carregando módulos (ChromaDB, scikit-learn, pandas...). Pode levar alguns segundos...")
    import vector_store as _vector_store
    import index_library as _index_library
    import clustering as _clustering
    import mpd_integration as _mpd_integration
    vector_store, index_library, clustering, mpd_integration = _vector_store, _index_library, _clustering, _mpd_integration
    logger.info(f"Módulos carregados em {time.time() - t0:.1f} s.")


class _etapa:
    """Mede e registra a duração de cada etapa da inicialização."""

    def __init__(self, rotulo: str):
        self.rotulo = rotulo

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        logger.info(f"{self.rotulo} concluída em {time.time() - self.t0:.1f} s.")
        return False


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Verifica se um determinado host:porta está respondendo conexões TCP."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def liberar_porta(porta: int) -> None:
    """Encerra processos órfãos que possam estar prendendo a porta especificada no Windows."""
    if os.name != "nt":
        return
    try:
        cmd = f"netstat -ano | findstr :{porta}"
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        pids = set()
        for line in out.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pids.add(parts[-1])
        for pid in pids:
            if pid and pid != "0":
                logger.info(f"Liberando porta {porta}: encerrando processo órfão anterior (PID {pid})...")
                subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
    except Exception:
        pass


def encerrar_processo(proc: subprocess.Popen) -> None:
    """Encerra com segurança um processo e toda a sua árvore de processos filhos."""
    if proc is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                f"taskkill /F /T /PID {proc.pid}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            proc.terminate()
            proc.wait(timeout=3)
    except Exception as e:
        logger.warning(f"Erro ao encerrar processo: {e}")


def etapa_1_mpd(
    host: str = config.MPD_HOST,
    port: int = config.MPD_PORT,
    exe_path: str = config.MPD_EXE_PATH,
    conf_path: str = config.MPD_CONF_PATH,
    timeout: int = config.MPD_START_TIMEOUT
) -> None:
    """
    Etapa 1: Verifica se o servidor MPD está ativo na porta configurada.
    Se não estiver, inicia o executável e aguarda ficar pronto.
    """
    logger.info(f"[1/6] Verificando servidor MPD em {host}:{port}...")

    if is_port_open(host, port):
        logger.info(f"[1/6] [OK] Servidor MPD já está ativo e respondendo na porta {port}.")
        return

    logger.warning(f"[1/6] MPD não detectado na porta {port}. Iniciando servidor local...")

    if not os.path.exists(exe_path):
        raise FileNotFoundError(
            f"Executável do MPD não encontrado em '{exe_path}'. "
            f"Verifique a instalação do MPD ou ajuste MPD_EXE_PATH em config.py."
        )

    if not os.path.exists(conf_path):
        raise FileNotFoundError(
            f"Arquivo de configuração do MPD não encontrado em '{conf_path}'. "
            f"Verifique o caminho ou ajuste MPD_CONF_PATH em config.py."
        )

    # Inicia o processo do MPD no Windows
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NO_WINDOW

    try:
        subprocess.Popen(
            [exe_path, conf_path],
            creationflags=creationflags
        )
    except Exception as e:
        raise RuntimeError(f"Falha ao iniciar processo do MPD: {e}")

    # Polling até ficar disponível
    t_start = time.time()
    while time.time() - t_start < timeout:
        if is_port_open(host, port, timeout=0.5):
            logger.info(f"[1/6] [OK] Servidor MPD iniciado e pronto para conexões na porta {port}.")
            return
        time.sleep(0.5)

    raise TimeoutError(
        f"Tempo limite ({timeout}s) esgotado aguardando o MPD responder na porta {port}. "
        f"Verifique se outro serviço está usando a porta ou consulte os logs em C:\\mpd\\log."
    )


def etapa_2_indexacao(music_root: Path, collection) -> bool:
    """
    Etapa 2: Compara os arquivos da pasta de músicas com o ChromaDB
    e indexa novas faixas automaticamente.
    Retorna True se houve faixas novas indexadas, False caso contrário.
    """
    logger.info(f"[2/6] Verificando integridade da biblioteca em '{music_root}' contra o ChromaDB...")

    if not music_root.exists():
        raise FileNotFoundError(
            f"A pasta de músicas '{music_root}' não existe ou não está acessível. "
            f"Se for Google Drive, certifique-se de que a unidade está montada "
            f"ou use a opção --pasta para apontar para o local correto."
        )

    arquivos = index_library.varrer_biblioteca(music_root)
    ids_existentes = index_library.ids_existentes_no_chroma(collection)

    pendentes = [f for f in arquivos if index_library._rel_id(f, music_root) not in ids_existentes]

    if not pendentes:
        logger.info(f"[2/6] [OK] ChromaDB atualizado: {len(ids_existentes)} faixas catalogadas, nenhuma nova pendente.")
        return False

    logger.info(f"[2/6] Encontradas {len(pendentes)} faixas novas na biblioteca. Indexando...")
    resultado = index_library.indexar(
        root=music_root,
        collection=collection,
        ids_existentes=ids_existentes,
        limit=None,
        reindex=False
    )
    logger.info(f"[2/6] [OK] Indexação concluída: {resultado['processadas']} faixas adicionadas ao ChromaDB.")
    return resultado["processadas"] > 0


def etapa_3_clusterizacao(
    novas_faixas: bool,
    recalcular: bool,
    modo: str = "refinado",
    pureza_minima: float = 0.80,
    tamanho_minimo: int = 15,
    profundidade_maxima: int = 3,
    seed: int = 42
) -> None:
    """
    Etapa 3: Gerencia a clusterização.
    - Se recalcular ou modelo ausente: roda clusterização completa K-Means no modo escolhido.
    - Se modelo já existe e novas faixas: os clusters já foram atribuídos incrementalmente.
    - Se já calibrado e sem novas faixas: reaproveita sem recalcular.
    """
    logger.info("[3/6] Verificando modelos de clusterização...")

    model_path = Path(config.CLUSTER_MODEL_PATH)
    clusters_json_path = Path(config.CLUSTERS_JSON_PATH)

    deve_recalcular = recalcular or (not model_path.exists()) or (not clusters_json_path.exists())

    if deve_recalcular:
        motivo = "Solicitação explícita (--recalcular)" if recalcular else "Modelo não encontrado"
        logger.info(f"[3/6] Executando clusterização completa ({motivo}) no modo '{modo}'...")
        res = clustering.run_clustering_pipeline(
            k_arg="auto",
            k_min=4,
            k_max=30,
            algorithm="kmeans",
            modo=modo,
            pureza_minima=pureza_minima,
            tamanho_minimo=tamanho_minimo,
            profundidade_maxima=profundidade_maxima,
            seed=seed,
            export_playlists=False,
            generate_graph=True,
            recalculate=True,
            skip_backup=False
        )
        logger.info(f"[3/6] [OK] Clusterização finalizada com sucesso: {res['n_clusters']} clusters calibrados (Modo: '{modo}').")
    else:
        if novas_faixas:
            logger.info("[3/6] [OK] Novas faixas atribuídas incrementalmente aos clusters existentes.")
        else:
            logger.info("[3/6] [OK] Clusters já calibrados e persistidos no ChromaDB. Reagrupamento dispensado.")


def etapa_4_playlists(music_root: Path) -> None:
    """
    Etapa 4: Atualiza as playlists por cluster no MPD e grava arquivos .m3u.
    """
    logger.info("[4/6] Sincronizando playlists por cluster no MPD...")

    clusters_json = Path(config.CLUSTERS_JSON_PATH)
    if not clusters_json.exists():
        logger.warning(f"[4/6] {clusters_json} não encontrado. Pulando geração de playlists.")
        return

    with open(clusters_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Constrói mapeamento de IDs para caminhos aceitos pelo MPD
    _, id_to_file = mpd_integration.build_file_to_id_map(config.METADATA_PATH, str(music_root))

    playlists_dir = Path(config.PLAYLISTS_DIR)
    mpd_count, m3u_count = clustering.export_playlists_to_mpd(
        clusters_summary=summary,
        id_to_file_map=id_to_file,
        host=config.MPD_HOST,
        port=config.MPD_PORT,
        m3u_dir=playlists_dir
    )
    logger.info(f"[4/6] [OK] Playlists sincronizadas ({mpd_count} no MPD, {m3u_count} arquivos .m3u em {playlists_dir}).")


def etapa_6_recomendador(
    music_root: Path,
    mode: str = "cluster",
    gui: bool = True,
    use_feedback: bool = True,
    zerar_feedback: bool = False
) -> None:
    """
    Etapa 6: Inicia o sistema de recomendação em primeiro plano.
    """
    fb_txt = "ativo" if use_feedback else "desativado (--sem-feedback)"
    if zerar_feedback:
        fb_txt += " (ZERADO)"
    logger.info(f"[6/6] Iniciando Recomendador Musical (Modo: '{mode}' | Feedback por Skip: {fb_txt})...")
    mpd_integration.run_recommender(
        metadata_path=config.METADATA_PATH,
        music_root=str(music_root),
        mode=mode,
        gui=gui,
        use_feedback=use_feedback,
        zerar_feedback=zerar_feedback
    )


def main():
    parser = argparse.ArgumentParser(description="Ponto de entrada único do Sistema de Recomendação Musical")
    parser.add_argument("--pasta", type=str, default=None, help="Caminho absoluto da pasta de músicas (Google Drive / local)")
    parser.add_argument("--recalcular", action="store_true", help="Força recálculo completo da clusterização K-Means do zero")
    parser.add_argument("--modo", type=str, choices=["refinado", "nao-supervisionado", "por-genero"], default="refinado", help="Modo de clusterização: refinado (padrão), nao-supervisionado, por-genero")
    parser.add_argument("--sem-feedback", action="store_true", help="Desativa feedback implícito por skip (baseline sem reforço)")
    parser.add_argument("--zerar-feedback", action="store_true", help="Limpa todo o histórico de feedback.db antes de iniciar")
    parser.add_argument("--pureza-minima", type=float, default=0.80, help="Pureza mínima para parada de refinamento (padrão: 0.80)")
    parser.add_argument("--tamanho-minimo", type=int, default=15, help="Tamanho mínimo de cluster para refinamento (padrão: 15)")
    parser.add_argument("--profundidade-maxima", type=int, default=3, help="Profundidade máxima de refinamento recursivo (padrão: 3)")
    parser.add_argument("--sem-cluster", action="store_true", help="Inicia o recomendador no modo clássico de similaridade K-NN (sem priorizar cluster)")
    parser.add_argument("--no-gui", action="store_true", help="Executa o recomendador no console sem abrir janela gráfica Tkinter")
    parser.add_argument("--sem-monitor", action="store_true", help="Inicia sem o Monitor de Precisão (Dashboard Streamlit)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodução dos cálculos de clustering")

    args = parser.parse_args()

    # Determinar a raiz da biblioteca
    if args.pasta:
        music_root = Path(args.pasta).resolve()
    else:
        music_root = Path(config.get_music_library_root(config.MPD_CONF_PATH)).resolve()

    logger.info("============================================================")
    logger.info(" SISTEMA DE RECOMENDAÇÃO MUSICAL — INICIALIZAÇÃO UNIFICADA")
    logger.info(f" Pasta de músicas : {music_root}")
    logger.info(f" Servidor MPD     : {config.MPD_HOST}:{config.MPD_PORT}")
    logger.info(f" Banco ChromaDB   : {config.CHROMA_PERSIST_DIR}")
    logger.info(f" Modo recomendador: {'similarity (sem-cluster)' if args.sem_cluster else 'cluster (mais faixas do mesmo cluster)'}")
    logger.info(f" Monitor Precisão : {'desativado (--sem-monitor)' if args.sem_monitor else f'Streamlit na porta {config.STREAMLIT_PORT}'}")
    logger.info("============================================================\n")

    t_inicio_total = time.time()
    monitor_proc = None

    try:
        _carregar_modulos()

        # Etapa 1: MPD
        with _etapa("[1/6] MPD"):
            etapa_1_mpd(
                host=config.MPD_HOST,
                port=config.MPD_PORT,
                exe_path=config.MPD_EXE_PATH,
                conf_path=config.MPD_CONF_PATH
            )

        # Conectar ChromaDB
        with _etapa("Conexão com o ChromaDB"):
            chroma_client = vector_store.get_client(config.CHROMA_PERSIST_DIR)
            collection = vector_store.get_or_create_collection(chroma_client, config.COLLECTION_NAME)

        # Etapa 2: Indexação automática
        with _etapa("[2/6] Verificação da biblioteca"):
            novas_faixas = etapa_2_indexacao(music_root, collection)

        # Etapa 3: Clusterização (completa se --recalcular ou ausente, incremental se novas faixas)
        with _etapa("[3/6] Clusterização"):
            etapa_3_clusterizacao(
                novas_faixas=novas_faixas,
                recalcular=args.recalcular,
                modo=args.modo,
                pureza_minima=args.pureza_minima,
                tamanho_minimo=args.tamanho_minimo,
                profundidade_maxima=args.profundidade_maxima,
                seed=args.seed
            )

        # Etapa 4: Atualização de playlists no MPD (a menos que explicitamente sem cluster)
        if not args.sem_cluster:
            with _etapa("[4/6] Playlists por cluster"):
                etapa_4_playlists(music_root)

        # Etapa 5: Iniciar Monitor de Precisão (Dashboard Streamlit) em paralelo
        monitor_log_file = None
        if not args.sem_monitor:
            dashboard_path = Path(config.DASHBOARD_PATH)
            if dashboard_path.exists():
                try:
                    # Garante que a pasta de logs existe e prepara o arquivo de saída
                    logs_dir = Path(config.PROJECT_ROOT) / "logs"
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    monitor_log_path = logs_dir / "monitor.log"
                    monitor_log_file = open(monitor_log_path, "w", encoding="utf-8")

                    # Libera a porta 8501 se houver processo órfão de execução anterior
                    liberar_porta(config.STREAMLIT_PORT)

                    monitor_cmd = [
                        sys.executable, "-m", "streamlit", "run",
                        str(dashboard_path),
                        "--server.port", str(config.STREAMLIT_PORT),
                        "--server.headless", "true",
                        "--browser.gatherUsageStats", "false"
                    ]
                    monitor_proc = subprocess.Popen(
                        monitor_cmd,
                        stdout=monitor_log_file,
                        stderr=subprocess.STDOUT,
                        cwd=str(config.PROJECT_ROOT)
                    )

                    # Aguarda até 5 segundos para confirmar que o Streamlit iniciou e está ouvindo
                    t_inicio = time.time()
                    monitor_online = False
                    while time.time() - t_inicio < 5.0:
                        if monitor_proc.poll() is not None:
                            break
                        if is_port_open("localhost", config.STREAMLIT_PORT, timeout=0.5):
                            monitor_online = True
                            break
                        time.sleep(0.5)

                    if monitor_online:
                        monitor_url = f"http://localhost:{config.STREAMLIT_PORT}"
                        logger.info(
                            f"[5/6] [OK] Monitor de Precisão (Streamlit) ativo em {monitor_url} "
                            f"(logs em: {monitor_log_path})"
                        )
                        # Abre automaticamente a interface no navegador do usuário
                        try:
                            webbrowser.open(monitor_url)
                        except Exception:
                            pass
                    else:
                        ret_code = monitor_proc.poll()
                        logger.warning(
                            f"[5/6] [AVISO] Monitor de Precisão não respondeu na porta {config.STREAMLIT_PORT} "
                            f"(código de saída: {ret_code}). Consulte {monitor_log_path} para detalhes. "
                            f"O recomendador continuará normalmente."
                        )
                except Exception as e:
                    logger.warning(
                        f"[5/6] [AVISO] Falha ao iniciar Monitor de Precisão: {e}. "
                        f"O recomendador continuará normalmente sem o monitor."
                    )
            else:
                logger.warning(
                    f"[5/6] [AVISO] Dashboard não encontrado em '{dashboard_path}'. "
                    f"Monitor de Precisão não será iniciado."
                )
        else:
            logger.info("[5/6] Monitor de Precisão desativado (--sem-monitor).")

        logger.info(f"Preparação concluída em {time.time() - t_inicio_total:.1f} s. Iniciando o recomendador...")

        # Etapa 6: Iniciar recomendador em primeiro plano
        rec_mode = "similarity" if args.sem_cluster else "cluster"
        etapa_6_recomendador(
            music_root=music_root,
            mode=rec_mode,
            gui=not args.no_gui,
            use_feedback=not args.sem_feedback,
            zerar_feedback=args.zerar_feedback
        )

    except KeyboardInterrupt:
        logger.info("\nEncerramento solicitado pelo usuário (Ctrl+C). Sistema finalizado com sucesso.")
    except Exception as e:
        logger.error(f"\n[FALHA NA EXECUÇÃO]: {e}")
        logger.error("O processo foi interrompido para evitar inconsistências. Verifique a mensagem acima.")
        sys.exit(1)
    finally:
        # Encerramento limpo do Monitor de Precisão (sem processos órfãos)
        if monitor_proc is not None:
            logger.info("Encerrando Monitor de Precisão (Streamlit)...")
            encerrar_processo(monitor_proc)
            logger.info("Monitor de Precisão encerrado com sucesso.")
        if monitor_log_file is not None and not monitor_log_file.closed:
            try:
                monitor_log_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
