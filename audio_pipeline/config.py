"""
config.py — Centralização de caminhos e parâmetros do sistema de recomendação.

Permite configuração via variáveis de ambiente para facilitar integração
com o instalador automatizado (Inno Setup) e execução portátil.
"""

import os
from pathlib import Path

# Diretórios base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# MPD (Music Player Daemon)
MPD_EXE_PATH = os.environ.get("MPD_EXE_PATH", r"C:\mpd\mpd.exe")
MPD_CONF_PATH = os.environ.get("MPD_CONF_PATH", r"C:\mpd\mpd.conf")
MPD_HOST = os.environ.get("MPD_HOST", "localhost")
MPD_PORT = int(os.environ.get("MPD_PORT", 6600))
MPD_START_TIMEOUT = int(os.environ.get("MPD_START_TIMEOUT", 15))

# ChromaDB
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(BASE_DIR, "chroma_db"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "musicas_embeddings")

# Biblioteca Musical
DEFAULT_MUSIC_ROOT = os.environ.get("MUSIC_LIBRARY_ROOT", r"G:\Meu Drive\Musicas")
LAST_MUSIC_ROOT_FILE = os.path.join(BASE_DIR, "last_music_root.txt")

# Resultados, Modelos e Playlists
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METADATA_PATH = os.path.join(RESULTS_DIR, "metadata.json")
CLUSTERS_JSON_PATH = os.path.join(RESULTS_DIR, "clusters.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLUSTER_MODEL_PATH = os.path.join(MODELS_DIR, "cluster_model.joblib")
PLAYLISTS_DIR = os.path.join(BASE_DIR, "playlists")
SESSION_HISTORY_FILE = os.path.join(BASE_DIR, "session_history.json")

# Monitor de Precisão (Dashboard Streamlit)
DASHBOARD_PATH = os.path.join(RESULTS_DIR, "dashboard.py")
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", 8501))


def get_music_library_root(mpd_conf_path: str = None) -> str:
    """
    Determina o caminho da pasta de músicas com a seguinte prioridade:
    1. music_directory configurado no arquivo mpd.conf informado ou padrão.
    2. last_music_root.txt (último caminho salvo pelo usuário).
    3. DEFAULT_MUSIC_ROOT (G:\\Meu Drive\\Musicas ou variável de ambiente).
    """
    conf_candidates = [mpd_conf_path, MPD_CONF_PATH, os.path.join(BASE_DIR, "mpd.conf")]
    for candidate in conf_candidates:
        if candidate and os.path.exists(candidate):
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("music_directory") and not line.startswith("#"):
                            parts = line.split(maxsplit=1)
                            if len(parts) > 1:
                                path = parts[1].strip('"\'')
                                if os.path.exists(path):
                                    return os.path.normpath(os.path.abspath(path))
            except Exception:
                pass

    if os.path.exists(LAST_MUSIC_ROOT_FILE):
        try:
            with open(LAST_MUSIC_ROOT_FILE, "r", encoding="utf-8") as f:
                saved = f.read().strip()
                if saved and os.path.exists(saved):
                    return os.path.normpath(os.path.abspath(saved))
        except Exception:
            pass

    return os.path.normpath(os.path.abspath(DEFAULT_MUSIC_ROOT))
