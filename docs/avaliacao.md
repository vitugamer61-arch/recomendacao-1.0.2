# Avaliação Experimental: Precision@K e Baseline Aleatória

Este documento descreve a metodologia e implementação da avaliação quantitativa do sistema de recomendação presente em `audio_pipeline/evaluation.py` e `audio_pipeline/results/generate_precision_k.py`.

---

## 1. Métrica Precision@K e Baseline Aleatória

A validação offline da qualidade semântico-musical dos embeddings CLAP baseia-se na capacidade dos vizinhos mais próximos preservarem a pertinência ao mesmo subgênero da faixa consultada (*seed*).

### 1.1. Definição Formal da Precision@K por Faixa
Para uma dada faixa semente $i$, com vetor de consulta $\mathbf{q}_i$ e subgênero conhecido $g_i \in \text{GroundTruth}$, o sistema consulta os $K$ vizinhos mais próximos no ChromaDB:
$$\mathcal{R}_K(i) = \{r_1, r_2, \dots, r_K\}, \quad r_j \neq i$$

A métrica para a faixa $i$ é a proporção de vizinhos recuperados cujo subgênero $g(r_j)$ coincide com $g_i$:
$$P@K(i) = \frac{1}{K} \sum_{j=1}^{K} \mathbb{I}[g(r_j) = g_i]$$
onde $\mathbb{I}[\cdot]$ é a função indicadora.

- **Arquivo:** [`audio_pipeline/evaluation.py`](../audio_pipeline/evaluation.py)
- **Linhas:** 200 a 218

```python
# audio_pipeline/evaluation.py:200-218
def precision_at_k_for_track(collection, track_id: str, ground_truth: Dict[str, str], k: int = 5) -> float:
    result = collection.get(ids=[track_id], include=["embeddings"])
    embeddings = result.get('embeddings')
    if embeddings is None or len(embeddings) == 0:
        return 0.0
        
    query_emb = np.array(result['embeddings'][0])
    similares = query_similar(collection, query_emb, k=k, exclude_id=track_id)
    
    if not similares:
        return 0.0
        
    seed_subgenero = ground_truth.get(track_id)
    if not seed_subgenero:
        return 0.0
        
    matches = sum(1 for res in similares if res.get('subgenero') == seed_subgenero)
    return matches / len(similares)
```

A média global é dada pela média aritmética sobre todas as faixas avaliadas:
$$\overline{P@K} = \frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} P@K(i)$$

### 1.2. Baseline Aleatória de Monte Carlo
Para contextualizar se a precisão obtida é superior ao acaso, calcula-se uma baseline de amostragem uniforme:
- **Arquivo:** [`audio_pipeline/evaluation.py`](../audio_pipeline/evaluation.py)
- **Linhas:** 230 a 250

```python
# audio_pipeline/evaluation.py:230-250
def random_baseline_precision_at_k(ground_truth: Dict[str, str], k: int = 5, n_trials: int = 100) -> float:
    track_ids = list(ground_truth.keys())
    if len(track_ids) <= k:
        return 0.0
        
    total_score = 0.0
    
    for seed_track_id in track_ids:
        seed_subgenero = ground_truth[seed_track_id]
        candidates = [t_id for t_id in track_ids if t_id != seed_track_id]
        
        track_score = 0.0
        for _ in range(n_trials):
            sampled_ids = random.sample(candidates, min(k, len(candidates)))
            matches = sum(1 for t_id in sampled_ids if ground_truth[t_id] == seed_subgenero)
            track_score += matches / len(sampled_ids)
            
        total_score += track_score / n_trials
        
    return total_score / len(track_ids)
```

Para cada faixa da coleção, sorteiam-se aleatoriamente $K$ faixas concorrentes durante $N_{\text{trials}} = 100$ repetições, computando a proporção média esperada sob hipótese nula.

---

## 2. Construção do Ground Truth

O conjunto de referência (gabarito) é extraído do arquivo `metadata.json` por `load_ground_truth`:

- **Arquivo:** [`audio_pipeline/evaluation.py`](../audio_pipeline/evaluation.py)
- **Linhas:** 167 a 198

```python
# audio_pipeline/evaluation.py:167-198
def load_ground_truth(metadata_path: str) -> Dict[str, str]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Arquivo metadata não encontrado em: {metadata_path}")
        
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    ground_truth = {}
    missing_subgenre = []
    
    if isinstance(metadata, dict):
        iterator = metadata.items()
    elif isinstance(metadata, list):
        iterator = []
        for i, m in enumerate(metadata):
            track_id = str(m.get('track_id', str(i)))
            iterator.append((track_id, m))
    else:
        raise ValueError("Formato de metadata não suportado. Esperado dict ou list.")
    
    for track_id, info in iterator:
        subgenero = info.get("subgenero", "").strip()
        if not subgenero:
            filename = info.get("filename", "desconhecido")
            missing_subgenre.append(f"ID: {track_id} | Arquivo: {filename}")
        else:
            ground_truth[track_id] = subgenero
            
    if missing_subgenre:
        raise AssertionError(f"Erro de sanidade: {len(missing_subgenre)} faixas estão sem subgenero preenchido. Isso não deveria acontecer com a inferência automática.")
        
    return ground_truth
```

---

## 3. Inconsistência Crítica entre Ground Truth e Espaço de Vizinhos

Uma análise rigorosa do código expõe uma divergência estrutural no cálculo da avaliação:

### 3.1. Origem das Chaves do Ground Truth vs. Origem dos Metadados do Vizinho
1. **O Ground Truth** é montado exclusivamente a partir das chaves do arquivo JSON passado por parâmetro (`metadata_path`). No caso padrão:
   $$\text{GroundTruth} = \{ \text{track\_id} \to \text{subgenero} \}$$
2. **A Busca de Vizinhos**, no entanto, consulta diretamente a coleção persistente do ChromaDB:
   ```python
   # evaluation.py:53
   similares = query_similar(collection, query_emb, k=k, exclude_id=track_id)
   ```
3. A rotina `query_similar` em `vector_store.py:91` preenche o campo `"subgenero"` a partir do dicionário embutido na coleção do ChromaDB:
   ```python
   # vector_store.py:91
   "subgenero": metadatas[i].get("subgenero", "")
   ```
4. A comparação em `evaluation.py:62` é feita da seguinte forma:
   ```python
   matches = sum(1 for res in similares if res.get('subgenero') == seed_subgenero)
   ```

### 3.2. As Consequências Práticas
- **Base do Ground Truth vs. Universo do Banco:**
  - A baseline aleatória sorteia candidatos estritamente de dentro do dicionário `ground_truth` (`candidates = [t_id for t_id in track_ids if t_id != seed_track_id]`).
  - O sistema de recomendação busca vizinhos no universo da coleção do ChromaDB.
  - Se o ChromaDB contiver faixas antigas (ou faixas indexadas com outros scripts que não estão no arquivo `metadata.json` avaliado), o ChromaDB retornará vizinhos cujos metadados vêm do banco.
  - Se o ChromaDB foi populado sem metadado de subgênero preenchido (ou com subgêneros ligeiramente divergentes em caixa de texto), `res.get('subgenero')` virá vazio ou incompatível, gerando contagem de `matches = 0` mesmo que a música seja contextualmente semelhante.
- **Risco de Divisão por Zero:**
  Se `similares` retornar vazio (`collection.get` sem embeddings), a função trata retornando `0.0`. Mas se a coleção tiver menos que $K$ itens, `len(similares) < k`, o cálculo é normalizado pelo número de similares encontrados (`len(similares)`), o que inflaciona a precisão para coleções pequenas com apenas uma ou duas faixas.
