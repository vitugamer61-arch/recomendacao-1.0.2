# Clusterização das Faixas

Agrupa as faixas por similaridade sonora, usando os embeddings já gravados no ChromaDB (um vetor de 512 dimensões por faixa, agregado por média em `pooling.py`). Os clusters alimentam as playlists por cluster no MPD e o modo de recomendação "mais faixas do mesmo cluster".

- **Código:** `audio_pipeline/clustering.py`.
- **Orquestração:** `audio_pipeline/iniciar.py` (etapa 3 de 6).

---

## 1. Pipeline

1. **Carga:** `load_embeddings_from_chromadb` lê embeddings e metadados em lotes.
2. **Pré-processamento:** normalização L2 e redução opcional com PCA (o modelo é salvo para inferência incremental).
3. **Agrupamento:** K-Means com escolha automática de `k` pelo Coeficiente de Silhueta (`run_kmeans`), ou HDBSCAN (`run_hdbscan`).
4. **Nomeação e pureza** (`determine_cluster_names_and_purity`): internamente os clusters nascem como `cluster_01`, `cluster_02`, ...; o nome final é o do **subgênero predominante** (voto majoritário das faixas do cluster). Se vários clusters têm o mesmo gênero predominante, recebem sufixo numérico (`Rock 1`, `Rock 2`). Não são gerados nomes compostos (ex.: "Rock / MPB"); a mistura residual fica registrada em `clusters.json` (pureza e distribuição de gêneros). Clusters sem nenhuma faixa com gênero válido chamam-se `Sem gênero`. Faixas de ruído do HDBSCAN (rótulo −1) recebem o id `ruido` (nome `Ruído`).
5. **Refinamento recursivo** (`refine_impure_clusters`): clusters com pureza abaixo do limiar são subdivididos.
6. **Persistência:** grava o cluster em cada faixa do ChromaDB, com **backup prévio** (`backup_chromadb`) e validação estrita dos metadados.
7. **Saídas:** `clusters.json` (estatísticas e medoides), visualização 2D (UMAP ou t-SNE), gráficos de avaliação, playlists no MPD e arquivos `.m3u`.

---

## 2. Modos

| Modo (`--modo`) | Comportamento |
| :-- | :-- |
| `refinado` (padrão) | K-Means + refinamento recursivo dos clusters impuros, para que gêneros distintos não fiquem misturados |
| `nao-supervisionado` | K-Means puro, sem usar o gênero |
| `por-genero` | agrupa separadamente dentro de cada subgênero, gerando sub-clusters (`Rock 1`, `Rock 2`, `Eletrônica 1`, ...) |

Parâmetros do refinamento (valores padrão): pureza mínima **0,80** (`--pureza-minima`), tamanho mínimo de cluster **15** (`--tamanho-minimo`), profundidade máxima **3** (`--profundidade-maxima`). A semente é `--seed 42`.

---

## 3. Faixas Novas (atribuição incremental)

Faixas indexadas depois da clusterização não exigem recalcular tudo: `assign_cluster_incremental` atribui a faixa ao centroide mais próximo, usando o modelo salvo (`models/cluster_model.joblib`). Para recalcular do zero, use `--recalcular`.

---

## 4. Uso na Recomendação

Com `recommendation_mode = "cluster"`, `select_next_recommendations` prioriza vizinhas do mesmo cluster da faixa atual. O modo não é usado quando o cluster é `ruido` ou quando o gênero está em cooldown (ver [feedback_implicito.md](feedback_implicito.md)). Se todas as candidatas do cluster são rejeitadas, a busca é expandida globalmente no ChromaDB. `--sem-cluster` volta ao K-NN clássico.

---

## 5. Observação sobre Resultados

Este documento descreve o mecanismo. Métricas de qualidade (pureza, silhueta, tamanho dos clusters) são geradas a cada execução em `results/` (`clusters.json` e gráficos). Valores numéricos só devem ser citados no TCC a partir do arquivo gerado pela execução usada no texto.
