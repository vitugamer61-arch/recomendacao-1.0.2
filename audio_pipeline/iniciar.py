"""
iniciar.py — Ponto de entrada único para o sistema de recomendação e clusterização musical.

Orquestra todo o fluxo em 5 etapas consecutivas:
1. Verificação e inicialização automática do MPD (porta 6600).
2. Verificação do ChromaDB e indexação automática de novas faixas da pasta de músicas.
3. Clusterização: execução do K-Means automático ou atribuição incremental de novas faixas.
4. Sincronização de playlists por cluster no MPD e em arquivos .m3u.
5. Inicialização do recomendador em primeiro plano no modo "mais faixas do mesmo cluster".
"""

import os
import sys
import time
import json
import socket
import logging
import argparse
import subprocess
from pathlib import Path

# Garante path de importação local
_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

import config
import vector_store
import index_library
import clustering
import mpd_integration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("iniciar")


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Verifica se um determinado host:porta está respondendo conexões TCP."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


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
    logger.info(f"[1/5] Verificando servidor MPD em {host}:{port}...")

    if is_port_open(host, port):
        logger.info(f"[1/5] [OK] Servidor MPD já está ativo e respondendo na porta {port}.")
        return

    logger.warning(f"[1/5] MPD não detectado na porta {port}. Iniciando servidor local...")

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
            logger.info(f"[1/5] [OK] Servidor MPD iniciado e pronto para conexões na porta {port}.")
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
    logger.info(f"[2/5] Verificando integridade da biblioteca em '{music_root}' contra o ChromaDB...")

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
        logger.info(f"[2/5] [OK] ChromaDB atualizado: {len(ids_existentes)} faixas catalogadas, nenhuma nova pendente.")
        return False

    logger.info(f"[2/5] Encontradas {len(pendentes)} faixas novas na biblioteca. Indexando...")
    resultado = index_library.indexar(
        root=music_root,
        collection=collection,
        ids_existentes=ids_existentes,
        limit=None,
        reindex=False
    )
    logger.info(f"[2/5] [OK] Indexação concluída: {resultado['processadas']} faixas adicionadas ao ChromaDB.")
    return resultado["processadas"] > 0


def etapa_3_clusterizacao(
    novas_faixas: bool,
    recalcular: bool,
    seed: int = 42
) -> None:
    """
    Etapa 3: Gerencia a clusterização.
    - Se recalcular ou modelo ausente: roda clusterização completa K-Means.
    - Se modelo já existe e novas faixas: os clusters já foram atribuídos incrementalmente.
    - Se já calibrado e sem novas faixas: reaproveita sem recalcular.
    """
    logger.info("[3/5] Verificando modelos de clusterização...")

    model_path = Path(config.CLUSTER_MODEL_PATH)
    clusters_json_path = Path(config.CLUSTERS_JSON_PATH)

    deve_recalcular = recalcular or (not model_path.exists()) or (not clusters_json_path.exists())

    if deve_recalcular:
        motivo = "Solicitação explícita (--recalcular)" if recalcular else "Modelo não encontrado"
        logger.info(f"[3/5] Executando clusterização completa ({motivo})...")
        res = clustering.run_clustering_pipeline(
            k_arg="auto",
            k_min=4,
            k_max=30,
            algorithm="kmeans",
            seed=seed,
            export_playlists=False,
            generate_graph=True,
            recalculate=True,
            skip_backup=False
        )
        logger.info(f"[3/5] [OK] Clusterização finalizada com sucesso: {res['n_clusters']} clusters calibrados.")
    else:
        if novas_faixas:
            logger.info("[3/5] [OK] Novas faixas atribuídas incrementalmente aos clusters existentes.")
        else:
            logger.info("[3/5] [OK] Clusters já calibrados e persistidos no ChromaDB. Reagrupamento dispensado.")


def etapa_4_playlists(music_root: Path) -> None:
    """
    Etapa 4: Atualiza as playlists por cluster no MPD e grava arquivos .m3u.
    """
    logger.info("[4/5] Sincronizando playlists por cluster no MPD...")

    clusters_json = Path(config.CLUSTERS_JSON_PATH)
    if not clusters_json.exists():
        logger.warning(f"[4/5] {clusters_json} não encontrado. Pulando geração de playlists.")
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
    logger.info(f"[4/5] [OK] Playlists sincronizadas ({mpd_count} no MPD, {m3u_count} arquivos .m3u em {playlists_dir}).")


def etapa_5_recomendador(
    music_root: Path,
    mode: str = "cluster",
    gui: bool = True
) -> None:
    """
    Etapa 5: Inicia o sistema de recomendação em primeiro plano.
    """
    logger.info(f"[5/5] Iniciando Recomendador Musical ativo (Modo: '{mode}')...")
    mpd_integration.run_recommender(
        metadata_path=config.METADATA_PATH,
        music_root=str(music_root),
        mode=mode,
        gui=gui
    )


def main():
    parser = argparse.ArgumentParser(description="Ponto de entrada único do Sistema de Recomendação Musical")
    parser.add_argument("--pasta", type=str, default=None, help="Caminho absoluto da pasta de músicas (Google Drive / local)")
    parser.add_argument("--recalcular", action="store_true", help="Força recálculo completo da clusterização K-Means do zero")
    parser.add_argument("--sem-cluster", action="store_true", help="Inicia o recomendador no modo clássico de similaridade K-NN (sem priorizar cluster)")
    parser.add_argument("--no-gui", action="store_true", help="Executa o recomendador no console sem abrir janela gráfica Tkinter")
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
    logger.info("============================================================\n")

    try:
        # Etapa 1: MPD
        etapa_1_mpd(
            host=config.MPD_HOST,
            port=config.MPD_PORT,
            exe_path=config.MPD_EXE_PATH,
            conf_path=config.MPD_CONF_PATH
        )

        # Conectar ChromaDB
        chroma_client = vector_store.get_client(config.CHROMA_PERSIST_DIR)
        collection = vector_store.get_or_create_collection(chroma_client, config.COLLECTION_NAME)

        # Etapa 2: Indexação automática
        novas_faixas = etapa_2_indexacao(music_root, collection)

        # Etapa 3: Clusterização (completa se --recalcular ou ausente, incremental se novas faixas)
        etapa_3_clusterizacao(
            novas_faixas=novas_faixas,
            recalcular=args.recalcular,
            seed=args.seed
        )

        # Etapa 4: Atualização de playlists no MPD (a menos que explicitamente sem cluster)
        if not args.sem_cluster:
            etapa_4_playlists(music_root)

        # Etapa 5: Iniciar recomendador em primeiro plano
        rec_mode = "similarity" if args.sem_cluster else "cluster"
        etapa_5_recomendador(
            music_root=music_root,
            mode=rec_mode,
            gui=not args.no_gui
        )

    except KeyboardInterrupt:
        logger.info("\nEncerramento solicitado pelo usuário (Ctrl+C). Sistema finalizado com sucesso.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n[FALHA NA EXECUÇÃO]: {e}")
        logger.error("O processo foi interrompido para evitar inconsistências. Verifique a mensagem acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
