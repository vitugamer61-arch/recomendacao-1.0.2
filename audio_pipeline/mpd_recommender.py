"""
mpd_recommender.py — Ponto de entrada compatível e wrapper para o recomendador MPD.
Encaminha as chamadas para mpd_integration.py mantendo compatibilidade direta de importação e CLI.
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

from mpd_integration import *

if __name__ == "__main__":
    import argparse
    import config

    parser = argparse.ArgumentParser(description="Recomendador Musical MPD")
    parser.add_argument("--metadata", type=str, default=config.METADATA_PATH, help="Caminho do metadata.json")
    parser.add_argument("--k", type=int, default=1, help="Quantidade de recomendações a consultar")
    parser.add_argument("--trigger_mode", type=str, choices=["on_change", "near_end"], default="on_change")
    parser.add_argument("--music_root", type=str, help="Caminho absoluto da biblioteca (pula prompt)")
    parser.add_argument("--mode", type=str, choices=["similarity", "cluster"], default="cluster", help="Modo de recomendação")
    parser.add_argument("--sem-feedback", action="store_true", help="Desativa feedback implícito por skip")
    parser.add_argument("--zerar-feedback", action="store_true", help="Limpa o banco de dados feedback.db antes de iniciar")
    parser.add_argument("--no-gui", action="store_true", help="Executa apenas no console sem abrir janela gráfica")
    
    args = parser.parse_args()

    m_root = args.music_root or config.get_music_library_root(config.MPD_CONF_PATH)
    m_root = str(Path(m_root).resolve())

    run_recommender(
        metadata_path=args.metadata,
        music_root=m_root,
        mode=args.mode,
        trigger_mode=args.trigger_mode,
        gui=not args.no_gui,
        use_feedback=not args.sem_feedback,
        zerar_feedback=args.zerar_feedback
    )

