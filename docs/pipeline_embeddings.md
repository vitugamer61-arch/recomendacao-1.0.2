# Pipeline de Extração e Agregação de Embeddings

Este documento detalha o pipeline de processamento digital de sinais de áudio, segmentação temporal, inferência neural e pooling vetorial utilizado no sistema.

A cadeia de processamento é orquestrada pela função `process_track` localizada em `audio_pipeline/pipeline.py`.

---

## 1. Visão Geral das Etapas

O processamento de cada arquivo de áudio divide-se em 4 etapas lineares:

1. **Ingestão com Amostragem Inteligente:** O áudio é decodificado via `librosa`, convertido para taxa de amostragem fixa de 48.000 Hz, transformado em canal mono e restrito a um trecho de 45 segundos após pular os primeiros 15 segundos da faixa.
2. **Segmentação com Sobreposição:** O trecho resultante é particionado em janelas temporais deslizantes de 10 segundos, com passo (*hop*) de 5 segundos, descartando caudas residuais menores que 2 segundos.
3. **Inferência Neural em Lote (CLAP):** Cada janela de 10 segundos é processada pelo extrator de características do modelo multimodal CLAP (`laion/clap-htsat-unfused`), gerando vetores densos de 512 dimensões.
4. **Agregação Vetorial (Pooling):** Os múltiplos embeddings dos segmentos são agregados pelo cálculo da média aritmética (*mean pooling*), produzindo um único vetor representativo da faixa.

---

## 2. Implementação e Código Real de Cada Etapa

### 2.1. Ingestão de Áudio (`ingestion.py`)

A leitura física do arquivo é implementada na função `load_audio`:

- **Arquivo:** [`audio_pipeline/ingestion.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/ingestion.py)
- **Linhas:** 7 a 30

```python
# audio_pipeline/ingestion.py:7-30
def load_audio(filepath: str, sr: int = 48000, offset: float = 0.0, duration: float = None) -> np.ndarray:
    """
    Carrega arquivo de áudio e converte para a taxa de amostragem especificada.
    Suporta wav, mp3, flac via librosa e soundfile.
    Se offset e duration forem passados, carrega apenas o trecho especificado, 
    com fallback seguro para faixas curtas.
    Retorna o array de áudio e não derruba o pipeline em caso de falha.
    """
    try:
        # Pega a duração total do áudio rapidamente sem decodificar tudo
        total_duration = librosa.get_duration(path=filepath)
        
        # Fallback de segurança para faixas muito curtas
        if duration is not None:
            if total_duration < offset + duration:
                logger.info(f"Faixa curta detectada ({total_duration:.1f}s). Ignorando offset inicial para '{filepath}'.")
                offset = 0.0
                duration = min(duration, total_duration)
                
        audio, _ = librosa.load(filepath, sr=sr, mono=True, offset=offset, duration=duration)
        return audio
    except Exception as e:
        logger.warning(f"Erro ao carregar áudio {filepath}: {e}")
        return None
```

A chamada concreta em `pipeline.py` define os parâmetros fixos:
- **Arquivo:** [`audio_pipeline/pipeline.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/pipeline.py)
- **Linhas:** 23 a 28

```python
# audio_pipeline/pipeline.py:23-28
    sr = 48000
    
    # 1. Ingestão (Amostragem Inteligente: pula 15s, lê 45s)
    audio = load_audio(filepath, sr=sr, offset=15.0, duration=45.0)
    if audio is None:
        return None
```

**Propósito dos parâmetros:**
- `sr = 48000`: Taxa de amostragem requerida nativamente pela arquitetura HTSAT do CLAP.
- `offset = 15.0`: Pula introduções silenciosas, ruídos de início de gravação ou vinhetas.
- `duration = 45.0`: Limita o consumo de memória RAM e CPU, selecionando o corpo central da faixa. Caso a música tenha menos de 60 segundos totais (`total_duration < 15.0 + 45.0`), o `offset` é resetado para `0.0`.

---

### 2.2. Segmentação Temporal (`segmentation.py`)

A divisão do vetor contínuo em janelas sobrepostas é tratada na função `segment_audio`:

- **Arquivo:** [`audio_pipeline/segmentation.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/segmentation.py)
- **Linhas:** 3 a 23

```python
# audio_pipeline/segmentation.py:3-23
def segment_audio(audio: np.ndarray, sr: int, window_sec: float = 10.0, hop_sec: float = 5.0) -> list[np.ndarray]:
    """
    Segmenta o áudio em janelas com sobreposição.
    Descartar segmentos residuais menores que 2 segundos.
    """
    window_length = int(window_sec * sr)
    hop_length = int(hop_sec * sr)
    min_length = int(2.0 * sr)
    
    segments = []
    total_length = len(audio)
    
    for start in range(0, total_length, hop_length):
        end = start + window_length
        segment = audio[start:end]
        
        # Descartar se for menor que 2s
        if len(segment) >= min_length:
            segments.append(segment)
            
    return segments
```

Em um sinal típico de 45 segundos a 48 kHz:
- `window_length = 480.000` amostras (10 s)
- `hop_length = 240.000` amostras (5 s)
- `min_length = 96.000` amostras (2 s)
- Quantidade gerada de segmentos: tipicamente entre 8 e 9 fatias sobrepostas.

---

### 2.3. Extração de Embeddings Neurais CLAP (`embedding.py`)

A transformação espectro-temporal dos segmentos em vetores latentes utiliza a arquitetura HTSAT (*Hierarchical Token-Semantic Audio Transformer*) do projeto LAION-AI implementada na biblioteca `transformers`.

- **Arquivo:** [`audio_pipeline/embedding.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/embedding.py)
- **Linhas:** 8 a 21 (Carregamento Singleton do Modelo)

```python
# audio_pipeline/embedding.py:8-21
# Padrão Singleton para modelo e processor
_MODEL = None
_PROCESSOR = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        logger.info(f"Carregando modelo CLAP em dispositivo: {_DEVICE}")
        model_name = "laion/clap-htsat-unfused"
        _PROCESSOR = ClapProcessor.from_pretrained(model_name)
        _MODEL = ClapModel.from_pretrained(model_name).to(_DEVICE)
        _MODEL.eval()
```

- **Arquivo:** [`audio_pipeline/embedding.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/embedding.py)
- **Linhas:** 44 a 63 (Extração em Lote)

```python
# audio_pipeline/embedding.py:44-63
def generate_embeddings_batch(audio_segments: list[np.ndarray], sr: int) -> list[np.ndarray]:
    """
    Gera embeddings para múltiplos segmentos em uma única passagem pelo modelo.
    Extremamente mais rápido que rodar o modelo um segmento por vez na CPU.
    """
    if not audio_segments:
        return []
        
    _load_model()
    
    with torch.no_grad():
        # O processor aceita uma lista de arrays numpy
        inputs = _PROCESSOR(audio=audio_segments, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        
        outputs = _MODEL.get_audio_features(**inputs)
        embeddings = outputs.pooler_output.cpu().numpy()
        
    # Retorna uma lista de embeddings (um para cada segmento)
    return [emb for emb in embeddings]
```

**Detalhes da extração:**
- `ClapProcessor`: Converte os sinais de áudio unidimensionais em espectrogramas log-mel compatíveis com o extrator HTSAT.
- `_MODEL.get_audio_features(**inputs)`: Extrai as características da torre de áudio do modelo multimodal.
- `outputs.pooler_output`: Vetor de saída denso com tamanho $D = 512$.

---

### 2.4. Agregação Vetorial / Pooling (`pooling.py`)

A redução da matriz temporal de segmentos para um vetor global fixo é executada por `aggregate_embeddings`:

- **Arquivo:** [`audio_pipeline/pooling.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/pooling.py)
- **Linhas:** 3 a 17

```python
# audio_pipeline/pooling.py:3-17
def aggregate_embeddings(embeddings: list[np.ndarray], method: str = "mean") -> np.ndarray:
    """
    Agrega embeddings de múltiplos segmentos em um único vetor global.
    """
    if not embeddings:
        return np.array([])
        
    if method == "mean":
        stacked = np.stack(embeddings)
        return np.mean(stacked, axis=0)
    elif method == "transformer":
        # Hook para implementação futura
        raise NotImplementedError("Transformer pooling ainda não implementado.")
    else:
        raise ValueError(f"Método de pooling '{method}' desconhecido.")
```

**Comportamento do Mean Pooling:**
Dado um conjunto de $N$ segmentos $\{\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_N\}$, onde cada $\mathbf{e}_i \in \mathbb{R}^{512}$, o vetor final da faixa $\mathbf{e}_{\text{global}}$ é dado por:
$$\mathbf{e}_{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{e}_i$$

O método `"transformer"` está explicitamente sinalizado com `NotImplementedError` e não é utilizado em nenhuma rotina da base.

---

### 2.5. Integração em `pipeline.process_track`

A unificação dos passos em uma estrutura de dados de retorno ocorre em:
- **Arquivo:** [`audio_pipeline/pipeline.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/pipeline.py)
- **Linhas:** 18 a 58

```python
# audio_pipeline/pipeline.py:18-58
def process_track(filepath: str, track_id: str, subgenero: str, rel_filepath: str) -> dict:
    """
    Processa uma única faixa: carrega, segmenta, extrai embeddings e agrega.
    """
    start_time = time()
    sr = 48000
    
    # 1. Ingestão (Amostragem Inteligente: pula 15s, lê 45s)
    audio = load_audio(filepath, sr=sr, offset=15.0, duration=45.0)
    if audio is None:
        return None
        
    # 2. Segmentação (janelas de 10s, hop de 5s)
    segments = segment_audio(audio, sr=sr, window_sec=10.0, hop_sec=5.0)
    if not segments:
        logger.warning(f"Faixa {filepath} não possui segmentos válidos (muito curta?).")
        return None
        
    # 3. Extração de Embeddings (Em Lote / Batch Inference)
    if len(segments) > 0:
        segment_embeddings = generate_embeddings_batch(segments, sr=sr)
    else:
        logger.warning(f"Faixa {filepath} não gerou segmentos.")
        return None
        
    # 4. Pooling
    global_embedding = aggregate_embeddings(segment_embeddings, method="mean")
    
    elapsed = time() - start_time
    logger.info(f"Processado {filepath} em {elapsed:.2f}s ({len(segments)} segmentos).")
    
    basename = os.path.basename(filepath)
    
    return {
        "track_id": track_id,
        "filename": basename,
        "filepath": rel_filepath,
        "subgenero": subgenero,
        "n_segments": len(segments),
        "embedding": global_embedding
    }
```
