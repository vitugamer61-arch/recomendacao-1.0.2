"""
servico_consulta.py
-------------------
Serviço HTTP local (stdlib) que encapsula o ChromaDB (PersistentClient embutido).
É o único processo que abre o banco; monitor.py e carga.py consultam via HTTP.

Endpoints:
  GET  /health           → {"status": "ok", "count": N}
  POST /query            → body: {"track_id": "...", "k": 5}
                         → {"results": [...], "latencia_ms": ...}
  GET  /random_track     → {"track_id": "..."}

Uso:
  python servico_consulta.py [--porta 7700] [--chroma-dir ./chroma_db]
                             [--colecao musicas_embeddings]
"""

import argparse
import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Caminho para importar os módulos do projeto principal
# ---------------------------------------------------------------------------
PIPELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "audio_pipeline")
sys.path.insert(0, os.path.abspath(PIPELINE_DIR))

import numpy as np
import vector_store  # do audio_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVICO] %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Estado global (inicializado no main)
# ---------------------------------------------------------------------------
_collection = None


def _inicializar_colecao(chroma_dir: str, nome_colecao: str) -> None:
    global _collection
    logging.info(f"Conectando ao ChromaDB em '{chroma_dir}'...")
    cliente = vector_store.get_client(chroma_dir)
    _collection = vector_store.get_or_create_collection(cliente, nome_colecao)
    contagem = _collection.count()
    logging.info(f"Coleção '{nome_colecao}' pronta — {contagem} vetores.")


# ---------------------------------------------------------------------------
# Handler HTTP
# ---------------------------------------------------------------------------
class ConsultaHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # silencia log por requisição
        pass

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        if self.path == "/health":
            self._responder_health()
        elif self.path == "/random_track":
            self._responder_random_track()
        else:
            self._erro(404, "Endpoint não encontrado")

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        if self.path == "/query":
            self._responder_query()
        else:
            self._erro(404, "Endpoint não encontrado")

    # ----------------------------------------------------------- Handlers
    def _responder_health(self):
        try:
            contagem = _collection.count()
            corpo = json.dumps({"status": "ok", "count": contagem})
            self._enviar(200, corpo)
        except Exception as exc:
            self._erro(500, str(exc))

    def _responder_random_track(self):
        try:
            # Pega até 10 itens e devolve um aleatório
            resultado = _collection.get(limit=10, include=[])
            ids = resultado.get("ids", [])
            if not ids:
                self._erro(503, "Coleção vazia")
                return
            import random
            track_id = random.choice(ids)
            self._enviar(200, json.dumps({"track_id": track_id}))
        except Exception as exc:
            self._erro(500, str(exc))

    def _responder_query(self):
        try:
            tamanho = int(self.headers.get("Content-Length", 0))
            corpo_raw = self.rfile.read(tamanho)
            payload = json.loads(corpo_raw)
        except Exception:
            self._erro(400, "JSON inválido")
            return

        track_id = payload.get("track_id")
        k = int(payload.get("k", 5))

        if not track_id:
            self._erro(400, "Campo 'track_id' obrigatório")
            return

        try:
            t0 = time.perf_counter()

            # Busca embedding da faixa de referência
            dados = _collection.get(ids=[track_id], include=["embeddings"])
            embs = dados.get("embeddings")
            if embs is None or len(embs) == 0:
                self._erro(404, f"track_id '{track_id}' não encontrado")
                return

            query_emb = np.array(embs[0])
            similares = vector_store.query_similar(
                _collection, query_emb, k=k, exclude_id=track_id
            )
            latencia_ms = (time.perf_counter() - t0) * 1000

            resposta = {
                "results": similares,
                "latencia_ms": round(latencia_ms, 3),
            }
            self._enviar(200, json.dumps(resposta))

        except Exception as exc:
            logging.error(f"Erro na consulta: {exc}")
            self._erro(500, str(exc))

    # ------------------------------------------------- helpers de resposta
    def _enviar(self, status: int, corpo: str):
        dados = corpo.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(dados))
        self.end_headers()
        self.wfile.write(dados)

    def _erro(self, status: int, mensagem: str):
        self._enviar(status, json.dumps({"erro": mensagem}))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Serviço HTTP local para consultas ao ChromaDB"
    )
    parser.add_argument("--porta", type=int, default=7700, help="Porta HTTP (padrão 7700)")
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=os.path.join(PIPELINE_DIR, "chroma_db"),
        help="Diretório persistido do ChromaDB",
    )
    parser.add_argument(
        "--colecao",
        type=str,
        default="musicas_embeddings",
        help="Nome da coleção ChromaDB",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    _inicializar_colecao(args.chroma_dir, args.colecao)

    servidor = HTTPServer(("127.0.0.1", args.porta), ConsultaHandler)
    logging.info(f"Serviço pronto em http://127.0.0.1:{args.porta}")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        logging.info("Serviço encerrado.")


if __name__ == "__main__":
    main()
