"""
restauracao.py
--------------
Teste de restauração do índice ChromaDB.

Protocolo:
  1. Copia o chroma_db original para uma pasta temporária (nunca toca o original)
  2. Coleta o top-K de uma faixa de referência ANTES da destruição (baseline)
     → via subprocesso isolado, garantindo que o lock do SQLite é liberado ao fim
  3. Apaga o conteúdo do índice da cópia (simula corrupção/falha)
  4. Reconstrói o índice a partir do .npz + metadata.json
  5. Mede o tempo até a primeira consulta correta na cópia restaurada
     → também via subprocesso isolado
  6. Compara o top-K DEPOIS com o baseline (valida fidelidade)
  7. Salva restauracao.csv com os tempos e resultado da comparação

Nota Windows: ChromaDB/SQLite mantém handles de arquivo mesmo após del/gc.
A solução é rodar cada consulta em um subprocesso separado que encerra sozinho.

Uso:
  python restauracao.py
         --npz      ../audio_pipeline/results/embeddings.npz
         --metadata ../audio_pipeline/results/metadata.json
         --chroma-dir-original ../audio_pipeline/chroma_db
         --colecao musicas_embeddings
         --k 5
         --saida resultados/YYYYMMDD_HHMMSS
"""

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RESTAURACAO] %(levelname)s %(message)s",
    stream=sys.stdout,
)

# ---------------------------------------------------------------------------
# Caminhos padrão
# ---------------------------------------------------------------------------
PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audio_pipeline")
PYTHON = sys.executable

CABECALHO_CSV = [
    "timestamp",
    "etapa",
    "duracao_s",
    "detalhe",
]

# ---------------------------------------------------------------------------
# Script auxiliar embutido — roda em subprocesso para liberar locks
# ---------------------------------------------------------------------------
_SCRIPT_QUERY = r"""
import sys, json, os
sys.path.insert(0, sys.argv[1])   # pipeline_dir
import numpy as np
import vector_store

chroma_dir   = sys.argv[2]
nome_colecao = sys.argv[3]
track_id     = sys.argv[4]        # "" = escolher o primeiro
k            = int(sys.argv[5])

cliente = vector_store.get_client(chroma_dir)
colecao = vector_store.get_or_create_collection(cliente, nome_colecao)

if not track_id:
    resultado = colecao.get(limit=1, include=[])
    ids = resultado.get("ids", [])
    if not ids:
        print(json.dumps({"erro": "colecao_vazia"}))
        sys.exit(1)
    track_id = ids[0]

dados = colecao.get(ids=[track_id], include=["embeddings"])
embs = dados.get("embeddings")
if embs is None or len(embs) == 0:
    print(json.dumps({"erro": f"track_id nao encontrado: {track_id}"}))
    sys.exit(1)

import numpy as np
query_emb = np.array(embs[0])
similares = vector_store.query_similar(colecao, query_emb, k=k, exclude_id=track_id)
ids_result = [r["track_id"] for r in similares]
print(json.dumps({"track_id": track_id, "top_k": ids_result}))
"""


def _query_em_subprocesso(
    chroma_dir: str,
    nome_colecao: str,
    track_id: str,
    k: int,
) -> dict:
    """
    Executa uma consulta ChromaDB em subprocesso isolado.
    O processo filho encerra sozinho → libera todos os handles de arquivo.
    Retorna {"track_id": str, "top_k": [str, ...]}.
    """
    resultado = subprocess.run(
        [
            PYTHON, "-c", _SCRIPT_QUERY,
            os.path.abspath(PIPELINE_DIR),
            chroma_dir,
            nome_colecao,
            track_id,
            str(k),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if resultado.returncode != 0:
        raise RuntimeError(
            f"Subprocesso de consulta falhou (code {resultado.returncode}):\n"
            f"  stdout: {resultado.stdout.strip()}\n"
            f"  stderr: {resultado.stderr.strip()}"
        )
    return json.loads(resultado.stdout.strip())


def _agora_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _copiar_chroma(origem: str, destino: str) -> None:
    if os.path.exists(destino):
        shutil.rmtree(destino)
    shutil.copytree(origem, destino)
    logging.info(f"ChromaDB copiado: '{origem}' → '{destino}'")


def _apagar_indice(chroma_dir_copia: str) -> None:
    """
    Remove os arquivos do índice dentro da cópia para simular corrupção.
    Aguarda até que todos os handles do subprocesso anterior sejam liberados.
    """
    max_tentativas = 8
    for tentativa in range(max_tentativas):
        try:
            for item in os.listdir(chroma_dir_copia):
                caminho = os.path.join(chroma_dir_copia, item)
                if os.path.isdir(caminho):
                    shutil.rmtree(caminho)
                else:
                    os.remove(caminho)
            logging.info(f"Conteúdo de '{chroma_dir_copia}' removido (simulação de falha).")
            return
        except PermissionError as exc:
            if tentativa < max_tentativas - 1:
                espera = 1.5 * (tentativa + 1)
                logging.warning(
                    f"Lock residual (tentativa {tentativa + 1}/{max_tentativas}): {exc}. "
                    f"Aguardando {espera:.1f}s..."
                )
                time.sleep(espera)
            else:
                raise


def _reconstruir_indice(
    chroma_dir_copia: str,
    nome_colecao: str,
    npz_path: str,
    metadata_path: str,
) -> None:
    """Reconstrói o índice na cópia via subprocesso (também libera locks ao fim)."""
    script = f"""
import sys, os
sys.path.insert(0, sys.argv[1])
import vector_store
chroma_dir   = sys.argv[2]
nome_colecao = sys.argv[3]
npz_path     = sys.argv[4]
meta_path    = sys.argv[5]
cliente = vector_store.get_client(chroma_dir)
colecao = vector_store.get_or_create_collection(cliente, nome_colecao)
vector_store.index_embeddings(colecao, npz_path, meta_path)
print("ok")
"""
    resultado = subprocess.run(
        [
            PYTHON, "-c", script,
            os.path.abspath(PIPELINE_DIR),
            chroma_dir_copia,
            nome_colecao,
            npz_path,
            metadata_path,
        ],
        capture_output=False,   # exibe saída do indexador para o usuário
        timeout=600,
    )
    if resultado.returncode != 0:
        raise RuntimeError(f"Reconstrução falhou com código {resultado.returncode}")
    logging.info("Índice reconstruído com sucesso na cópia.")


def _comparar_top_k(antes: list[str], depois: list[str]) -> dict:
    set_antes = set(antes)
    set_depois = set(depois)
    intersecao = set_antes & set_depois
    return {
        "identico": antes == depois,
        "sobreposicao": len(intersecao),
        "total_k": len(antes),
        "precisao_set": round(len(intersecao) / len(antes), 4) if antes else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Teste de restauração do índice ChromaDB")
    default_npz = os.path.join(PIPELINE_DIR, "results", "embeddings_1288.npz")
    if not os.path.exists(default_npz):
        default_npz = os.path.join(PIPELINE_DIR, "results", "embeddings.npz")
    default_meta = os.path.join(PIPELINE_DIR, "results", "metadata_1288.json")
    if not os.path.exists(default_meta):
        default_meta = os.path.join(PIPELINE_DIR, "results", "metadata.json")

    parser.add_argument(
        "--npz",
        type=str,
        default=default_npz,
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=default_meta,
    )
    parser.add_argument(
        "--chroma-dir-original",
        type=str,
        default=os.path.join(PIPELINE_DIR, "chroma_db"),
    )
    parser.add_argument("--colecao", type=str, default="musicas_embeddings")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--saida", type=str, default=None)
    args = parser.parse_args()

    if args.saida is None:
        ts_pasta = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.saida = os.path.join(
            os.path.dirname(__file__), "resultados", ts_pasta
        )
    os.makedirs(args.saida, exist_ok=True)

    chroma_copia = os.path.join(os.path.abspath(args.saida), "chroma_copia")
    caminho_csv = os.path.join(args.saida, "restauracao.csv")

    logging.info(f"Teste de restauração | original={args.chroma_dir_original}")
    logging.info(f"Cópia de trabalho: {chroma_copia}")

    registros: list[dict] = []

    def reg(etapa: str, duracao: float, detalhe: str = ""):
        r = {
            "timestamp": _agora_iso(),
            "etapa": etapa,
            "duracao_s": round(duracao, 4),
            "detalhe": detalhe,
        }
        registros.append(r)
        logging.info(f"[{etapa}] {duracao:.3f}s {detalhe}")

    # ---------------------------------------------------------------- ETAPA 1: Cópia
    t0 = time.perf_counter()
    _copiar_chroma(os.path.abspath(args.chroma_dir_original), chroma_copia)
    reg("copia_chroma", time.perf_counter() - t0)

    # ---------------------------------------------------------------- ETAPA 2: Baseline (subprocesso)
    t0 = time.perf_counter()
    dados_antes = _query_em_subprocesso(chroma_copia, args.colecao, "", args.k)
    track_ref = dados_antes["track_id"]
    top_k_antes = dados_antes["top_k"]
    reg("top_k_antes", time.perf_counter() - t0,
        f"track_ref={track_ref} top_k={top_k_antes}")

    # ---------------------------------------------------------------- ETAPA 3: Destruição
    t0 = time.perf_counter()
    _apagar_indice(chroma_copia)
    reg("destruicao_indice", time.perf_counter() - t0)

    # ---------------------------------------------------------------- ETAPA 4: Reconstrução
    t0 = time.perf_counter()
    _reconstruir_indice(chroma_copia, args.colecao, args.npz, args.metadata)
    duracao_reconstrucao = time.perf_counter() - t0
    reg("reconstrucao_indice", duracao_reconstrucao)

    # ---------------------------------------------------------------- ETAPA 5: Primeira consulta
    t0 = time.perf_counter()
    dados_depois = _query_em_subprocesso(chroma_copia, args.colecao, track_ref, args.k)
    top_k_depois = dados_depois["top_k"]
    duracao_primeira_consulta = time.perf_counter() - t0
    reg("primeira_consulta_pos_restauracao", duracao_primeira_consulta)

    tempo_total_restauracao = duracao_reconstrucao + duracao_primeira_consulta
    reg("tempo_total_restauracao", tempo_total_restauracao)

    # ---------------------------------------------------------------- ETAPA 6: Comparação
    comparacao = _comparar_top_k(top_k_antes, top_k_depois)
    reg("comparacao_top_k", 0.0, json.dumps(comparacao, ensure_ascii=False))

    logging.info("=" * 55)
    logging.info(f"  Faixa de referência : {track_ref}")
    logging.info(f"  Top-K ANTES         : {top_k_antes}")
    logging.info(f"  Top-K DEPOIS        : {top_k_depois}")
    logging.info(f"  Índice idêntico?    : {comparacao['identico']}")
    logging.info(f"  Sobreposição        : {comparacao['sobreposicao']}/{comparacao['total_k']}")
    logging.info(f"  Precisão (set)      : {comparacao['precisao_set']:.2%}")
    logging.info(f"  Tempo reconstrução  : {duracao_reconstrucao:.3f}s")
    logging.info(f"  Tempo total restore : {tempo_total_restauracao:.3f}s")
    logging.info("=" * 55)

    # Salva CSV
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CABECALHO_CSV)
        writer.writeheader()
        writer.writerows(registros)
    logging.info(f"Resultados → {caminho_csv}")

    # Coleta metadados da biblioteca para o relatório
    total_embeddings = None
    total_faixas = None
    try:
        import numpy as np
        d_npz = np.load(args.npz)
        for k in d_npz.keys():
            total_embeddings = int(d_npz[k].shape[0])
            break
        with open(args.metadata, encoding="utf-8") as f_meta:
            m_data = json.load(f_meta)
            total_faixas = len(m_data)
    except Exception as e_meta:
        logging.warning(f"Não foi possível obter metadados da biblioteca: {e_meta}")

    # Salva resumo JSON para analise.py
    resumo = {
        "track_referencia": track_ref,
        "total_embeddings": total_embeddings,
        "total_faixas": total_faixas,
        "top_k_antes": top_k_antes,
        "top_k_depois": top_k_depois,
        "comparacao": comparacao,
        "duracao_reconstrucao_s": round(duracao_reconstrucao, 4),
        "duracao_primeira_consulta_s": round(duracao_primeira_consulta, 4),
        "tempo_total_restauracao_s": round(tempo_total_restauracao, 4),
    }
    caminho_resumo = os.path.join(args.saida, "restauracao_resumo.json")
    with open(caminho_resumo, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    logging.info(f"Resumo → {caminho_resumo}")


if __name__ == "__main__":
    main()
