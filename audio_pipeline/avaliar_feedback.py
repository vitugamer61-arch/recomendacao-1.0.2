"""
avaliar_feedback.py — Avaliação e simulação do impacto do feedback implícito (skips).

Simula um usuário pulando consecutivamente músicas de um determinado gênero musical
e mede a proporção desse gênero nas próximas K recomendações, comparando:
1. Baseline: recomendação padrão por similaridade de cosseno (sem feedback).
2. Com feedback: reformulação de consulta por Rocchio, re-ranking com penalidade
   de faixas e gênero, e cooldown de gênero.

Gera:
- Tabela com métricas impressa no terminal.
- Arquivo CSV: audio_pipeline/results/avaliacao_feedback.csv
- Gráfico PNG: audio_pipeline/results/avaliacao_feedback.png
"""

import os
import sys
import argparse
import random
from typing import List, Dict, Any, Tuple
import numpy as np

# Garante inclusão do diretório base no path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import chromadb
import config
from feedback import FeedbackStorage, FeedbackEngine
import feedback_config


def carregar_dados_chromadb():
    """
    Carrega o client, collection, IDs, embeddings e metadados da coleção ChromaDB.
    """
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    col = client.get_collection(config.COLLECTION_NAME)
    
    dados = col.get(include=["embeddings", "metadatas"])
    ids = dados["ids"]
    embeddings = np.array(dados["embeddings"], dtype=np.float32)
    metadatas = dados["metadatas"]
    
    return client, col, ids, embeddings, metadatas


def simular_avaliacoes(
    col,
    ids: List[str],
    embeddings: np.ndarray,
    metadatas: List[Dict[str, Any]],
    genero_alvo: str,
    skips_steps: List[int] = [0, 1, 3, 5],
    k: int = 10,
    num_amostras_semente: int = 20,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Executa a simulação de skips para o gênero alvo em múltiplas faixas sementes.
    
    Retorna lista de registros com médias de fração do gênero alvo para cada passo e modo.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    total_faixas = len(ids)
    id_to_file = {
        ids[i]: metadatas[i].get("caminho", f"/musicas/{ids[i]}.mp3")
        for i in range(total_faixas)
    }
    
    # Identifica faixas pertencentes ao gênero alvo
    faixas_genero_alvo = [
        (ids[i], embeddings[i], metadatas[i])
        for i in range(total_faixas)
        if metadatas[i].get("subgenero") == genero_alvo
    ]
    
    if not faixas_genero_alvo:
        raise ValueError(f"Nenhuma faixa encontrada com subgenero='{genero_alvo}' no ChromaDB.")
    
    # Seleciona sementes de teste pertencentes ao gênero alvo
    num_amostras = min(num_amostras_semente, len(faixas_genero_alvo))
    amostras_sementes = random.sample(faixas_genero_alvo, num_amostras)
    
    # Faixas disponíveis para simular skips
    pool_skips = [f for f in faixas_genero_alvo]
    
    # Pré-normaliza embeddings para busca de similaridade rápida
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embs = embeddings / norms
    
    resultados = []
    
    for s in skips_steps:
        fracoes_sem_fb = []
        fracoes_com_fb = []
        
        for seed_id, seed_emb, seed_meta in amostras_sementes:
            # 1. Configuração do motor com SQLite em memória (para isolamento)
            storage_teste = FeedbackStorage(db_path=":memory:")
            engine_teste = FeedbackEngine(storage=storage_teste)
            
            # Simula s skips consecutivos de faixas do gênero alvo
            if s > 0:
                skips_amostrados = random.sample(pool_skips, min(s, len(pool_skips)))
                for skip_id, _, skip_meta in skips_amostrados:
                    caminho = skip_meta.get("caminho", f"/musicas/{skip_id}.mp3")
                    gen = skip_meta.get("subgenero", genero_alvo)
                    cluster = str(skip_meta.get("cluster_id", 0))
                    storage_teste.record_event(
                        track_id=skip_id,
                        caminho=caminho,
                        genero=gen,
                        cluster_id=cluster,
                        tipo="skip",
                        elapsed=5.0,
                        duration=180.0
                    )
            
            # --- Modo SEM Feedback (Baseline K-NN) ---
            q_norm = seed_emb / (np.linalg.norm(seed_emb) or 1.0)
            sims = np.dot(norm_embs, q_norm)
            ordem = np.argsort(-sims)
            
            candidatos_sem = []
            for idx in ordem:
                if ids[idx] == seed_id:
                    continue
                candidatos_sem.append(metadatas[idx])
                if len(candidatos_sem) >= k:
                    break
            
            n_genero_sem = sum(1 for m in candidatos_sem if m.get("subgenero") == genero_alvo)
            fracoes_sem_fb.append(n_genero_sem / float(k))
            
            # --- Modo COM Feedback (Rocchio + Re-ranking + Cooldown) ---
            q_rocchio, stats_rocchio = engine_teste.compute_rocchio_query(seed_emb, col)
            
            # Busca top N=50 candidatos pelo vetor ajustado q'
            sims_rocchio = np.dot(norm_embs, q_rocchio)
            ordem_rocchio = np.argsort(-sims_rocchio)
            
            n_candidatos = min(50, total_faixas)
            candidatos_raw = []
            for idx in ordem_rocchio[:n_candidatos]:
                candidatos_raw.append({
                    "track_id": ids[idx],
                    "score": float(sims_rocchio[idx]),
                    "cluster_nome": metadatas[idx].get("cluster_nome", ""),
                    "subgenero": metadatas[idx].get("subgenero", ""),
                    "cluster_id": str(metadatas[idx].get("cluster_id", "0"))
                })
            
            # Aplica o re-ranking penalizado
            candidatos_reranked = engine_teste.rerank_candidates(
                query_emb=seed_emb,
                candidates=candidatos_raw,
                collection=col,
                id_to_file_map=id_to_file,
                queued_files={id_to_file[seed_id]}
            )
            
            # Seleciona o top-K após o re-ranking
            candidatos_com = candidatos_reranked[:k]
            
            # Conta presença do gênero alvo
            n_genero_com = sum(1 for c in candidatos_com if c.get("subgenero") == genero_alvo)
            # Se candidatos_com for menor que k por causa de filtros, calcula a fração sobre as recomendadas
            total_rec = max(len(candidatos_com), 1)
            fracoes_com_fb.append(n_genero_com / float(total_rec))
            
            # Fecha a conexão do banco em memória
            storage_teste.close()
        
        media_sem = float(np.mean(fracoes_sem_fb))
        std_sem = float(np.std(fracoes_sem_fb))
        media_com = float(np.mean(fracoes_com_fb))
        std_com = float(np.std(fracoes_com_fb))
        
        resultados.append({
            "genero_alvo": genero_alvo,
            "num_skips": s,
            "modo": "sem_feedback",
            "fracao_genero_medio": media_sem,
            "fracao_genero_std": std_sem,
            "total_amostras": len(amostras_sementes)
        })
        resultados.append({
            "genero_alvo": genero_alvo,
            "num_skips": s,
            "modo": "com_feedback",
            "fracao_genero_medio": media_com,
            "fracao_genero_std": std_com,
            "total_amostras": len(amostras_sementes)
        })
    
    return resultados


def exportar_csv(resultados: List[Dict[str, Any]], caminho_csv: str):
    """
    Exporta os resultados para um arquivo CSV estruturado.
    """
    os.makedirs(os.path.dirname(caminho_csv), exist_ok=True)
    with open(caminho_csv, "w", encoding="utf-8") as f:
        f.write("genero_alvo,num_skips,modo,fracao_genero_medio,fracao_genero_std,total_amostras\n")
        for r in resultados:
            f.write(
                f"{r['genero_alvo']},{r['num_skips']},{r['modo']},"
                f"{r['fracao_genero_medio']:.4f},{r['fracao_genero_std']:.4f},{r['total_amostras']}\n"
            )
    print(f"[OK] CSV de avaliacao exportado para: {caminho_csv}")


def gerar_grafico(resultados: List[Dict[str, Any]], caminho_png: str, genero_alvo: str, k: int):
    """
    Gera gráfico de linha comparativo (Sem Feedback vs Com Feedback) e salva em PNG.
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(os.path.dirname(caminho_png), exist_ok=True)
    
    skips = sorted(list(set(r["num_skips"] for r in resultados)))
    
    sem_fb = [
        next(r["fracao_genero_medio"] * 100 for r in resultados if r["num_skips"] == s and r["modo"] == "sem_feedback")
        for s in skips
    ]
    com_fb = [
        next(r["fracao_genero_medio"] * 100 for r in resultados if r["num_skips"] == s and r["modo"] == "com_feedback")
        for s in skips
    ]
    
    plt.figure(figsize=(9, 5.5))
    
    # Curvas
    plt.plot(skips, sem_fb, marker="s", color="#e74c3c", linestyle="--", linewidth=2.2, markersize=8, label="Baseline (Sem Feedback)")
    plt.plot(skips, com_fb, marker="o", color="#2980b9", linestyle="-", linewidth=2.5, markersize=8, label="Com Feedback (Rocchio + Re-ranking + Cooldown)")
    
    # Linha vertical indicativa de Cooldown
    plt.axvline(x=3, color="#8e44ad", linestyle=":", alpha=0.75, linewidth=1.8, label="Gatilho de Cooldown (>= 3 skips)")
    
    plt.title(f"Impacto do Feedback Implícito de Skips na Presença de '{genero_alvo}' (K={k})", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Número de Skips Consecutivos do Gênero", fontsize=11)
    plt.ylabel(f"Fração de '{genero_alvo}' nas Próximas {k} Faixas (%)", fontsize=11)
    plt.xticks(skips)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    
    plt.savefig(caminho_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Grafico comparativo gerado em: {caminho_png}")


def main():
    parser = argparse.ArgumentParser(description="Avaliação de feedback implícito (skips).")
    parser.add_argument("--genero", type=str, default="Funk", help="Gênero alvo dos skips a simular.")
    parser.add_argument("--k", type=int, default=10, help="Número de faixas recomendadas por consulta.")
    parser.add_argument("--amostras", type=int, default=20, help="Quantidade de faixas sementes na simulação.")
    parser.add_argument("--csv", type=str, default=os.path.join(config.RESULTS_DIR, "avaliacao_feedback.csv"), help="Caminho do CSV de saída.")
    parser.add_argument("--png", type=str, default=os.path.join(config.RESULTS_DIR, "avaliacao_feedback.png"), help="Caminho do gráfico PNG.")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("   AVALIAÇÃO DE FEEDBACK IMPLÍCITO (ROCCHIO + RE-RANKING)")
    print("=" * 70)
    print(f"Gênero alvo dos skips: {args.genero}")
    print(f"Top-K recomendações:   {args.k}")
    print(f"Amostras de sementes:  {args.amostras}")
    print(f"Carregando biblioteca do ChromaDB...")
    
    client, col, ids, embeddings, metadatas = carregar_dados_chromadb()
    print(f"Total de faixas carregadas: {len(ids)}")
    
    print("\nExecutando simulações para 0, 1, 3 e 5 skips...")
    resultados = simular_avaliacoes(
        col=col,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        genero_alvo=args.genero,
        skips_steps=[0, 1, 3, 5],
        k=args.k,
        num_amostras_semente=args.amostras
    )
    
    # Exibe tabela no terminal
    print("\n" + "-" * 70)
    print(f"{'Skips':<8} | {'Modo':<15} | {'Presença Média (%)':<20} | {'Desvio Padrão (%)':<18}")
    print("-" * 70)
    for r in resultados:
        print(
            f"{r['num_skips']:<8} | {r['modo']:<15} | "
            f"{r['fracao_genero_medio']*100:>17.2f}% | "
            f"{r['fracao_genero_std']*100:>15.2f}%"
        )
    print("-" * 70)
    
    # Exporta CSV e PNG
    exportar_csv(resultados, args.csv)
    gerar_grafico(resultados, args.png, args.genero, args.k)
    print("\nAvaliação concluída com sucesso!")


if __name__ == "__main__":
    main()
