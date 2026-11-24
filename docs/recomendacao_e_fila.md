# Seleção de Recomendações e Gerenciamento da Fila do MPD

Este documento descreve como o sistema captura os eventos de reprodução do MPD, gerencia o histórico de sessão em disco, aplica filtros sobre o espaço de candidatos e repõe faixas dinamicamente na fila.

---

## 1. Tratamento Real de Eventos do MPD

No loop principal de monitoramento (`mpd_integration.py:monitor_and_recommend`), a sincronização com o MPD é realizada por bloqueio através do comando `client.idle("player")`:

- **Arquivo:** [`audio_pipeline/mpd_integration.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/mpd_integration.py)
- **Linhas:** 455 a 478

```python
# audio_pipeline/mpd_integration.py:455-478
            # O client.idle("player") bloqueia até que ocorra um evento de player
            events = client.idle("player")
            
            if "player" in events:
                status = client.status()
                state = status.get("state")
                
                if state == "play":
                    current_song = client.currentsong()
                    raw_file = current_song.get("file")
                    if not raw_file:
                        continue
                        
                    current_file = normalize_mpd_filepath(raw_file, music_root)
                    
                    if current_file == last_processed_file:
                        continue
                        
                    logging.info(f"Usuário ouvindo: {current_file}")
                    ...
                    last_processed_file = current_file
```

### 1.1. Detecção por Arquivo vs. `songid` / Posição
Um ponto crítico de implementação:
- O protocolo MPD provê identificadores dinâmicos de fila: `song` (índice 0-based da posição na playlist) e `songid` (identificador único daquela ocorrência na fila).
- **Comportamento real do código:** A diferenciação de evento em `monitor_and_recommend` **não utiliza `songid`**. A lógica se baseia unicamente na comparação de string:
  ```python
  if current_file == last_processed_file:
      continue
  ```
- **Consequências para os eventos:**
  - **START / PLAY:** Detectado quando `state == "play"` e `current_file != last_processed_file`.
  - **NEXT / PREVIOUS / JUMP:** Todos são tratados indistintamente como "troca de faixa", desde que o arquivo seja diferente de `last_processed_file`.
  - **SEEK (Avanço/Retrocesso temporal):** Não gera recomendação adicional porque `current_file == last_processed_file`, caindo no `continue` de linha 470.
  - **Repetição da Mesma Faixa (Repeat 1):** Se o usuário colocar a mesma música para tocar repetidamente, o término da faixa e o reinício não disparam nova recomendação, pois `current_file` permanece igual a `last_processed_file`.
  - **Duas faixas idênticas seguidas na fila:** A segunda faixa não disparará o ciclo de recomendação pelo mesmo motivo.

---

## 2. Histórico de Sessão e Persistência Atômica

O controle de faixas tocadas evita repetições imediatas e garante diversidade ao longo do uso.

### 2.1. Tamanho Dinâmico do Histórico
Para evitar travamentos em coleções musicais pequenas, o tamanho máximo do histórico não é rígido:

- **Arquivo:** [`audio_pipeline/mpd_integration.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/mpd_integration.py)
- **Linhas:** 420 a 425

```python
# audio_pipeline/mpd_integration.py:420-425
    # Limitar HISTORY_SIZE a no máximo 50% das faixas indexadas
    total_indexed = max(1, len(file_to_id_map))
    effective_history_size = min(HISTORY_SIZE_DEFAULT, max(2, total_indexed // 2))
    logging.info(f"Tamanho do histórico de sessão configurado: {effective_history_size} faixas (total indexado: {total_indexed}).")

    play_history = load_session_history(SESSION_HISTORY_FILE, max_len=effective_history_size)
```

Por padrão, `HISTORY_SIZE_DEFAULT = 30`. Entretanto, se o acervo contiver apenas 20 faixas, o histórico é reduzido automaticamente para 10 faixas (50%), impedindo que todas as opções do catálogo sejam rejeitadas simultaneamente.

### 2.2. Gravação Atômica em Disco
Para resistir a quedas de energia ou interrupções abruptas do processo (`Ctrl+C`), a gravação do histórico utiliza escrita em arquivo temporário seguida de substituição atômica no sistema de arquivos:

- **Arquivo:** [`audio_pipeline/mpd_integration.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/mpd_integration.py)
- **Linhas:** 160 a 169

```python
# audio_pipeline/mpd_integration.py:160-169
def save_session_history(history: deque, filepath: str = SESSION_HISTORY_FILE) -> None:
    """Salva o histórico de faixas de forma atômica para evitar corrupção em caso de encerramento."""
    try:
        temp_file = f"{filepath}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(list(history), f, ensure_ascii=False, indent=2)
        os.replace(temp_file, filepath)
    except Exception as e:
        logging.error(f"Erro ao salvar histórico de sessão em '{filepath}': {e}")
```

---

## 3. Filtragem de Candidatas e Modos de Seleção

A escolha da próxima faixa ocorre em `select_next_recommendations` a partir de uma consulta de $K=20$ candidatos (`CANDIDATES_K = 20`) mais próximos no ChromaDB:

- **Arquivo:** [`audio_pipeline/mpd_integration.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/mpd_integration.py)
- **Linhas:** 262 a 283

```python
# audio_pipeline/mpd_integration.py:262-283
    def filter_candidates(effective_hist_set: set) -> list[tuple[int, dict]]:
        valid = []
        for rank, rec in enumerate(similares):
            rec_id = rec['track_id']
            rec_file = id_to_file_map.get(rec_id)

            if rec_id in failed_tracks:
                continue

            if rec_id in effective_hist_set:
                logging.debug(f"Candidata '{rec_id}' rejeitada (motivo: no histórico)")
                continue

            if rec_file and (rec_file in queued_files or normalize_mpd_filepath(rec_file) in queued_files):
                logging.info(f"Candidata '{rec_file}' rejeitada (motivo: na fila)")
                if msg_queue:
                    msg_queue.put({"type": "log", "text": f"Rejeitada {os.path.basename(rec_file)} (já na fila)"})
                continue

            if rec_file:
                valid.append((rank, rec))
        return valid
```

### 3.1. Três Filtros Cumulativos
1. **`failed_tracks`:** Registra faixas cujo comando `client.add()` resultou em erro (`CommandError`), evitando laços infinitos tentando adicionar arquivos corrompidos ou inacessíveis.
2. **`effective_hist_set`:** Faixas que já constam no histórico de sessão.
3. **`queued_files`:** Faixas que já constam na lista de reprodução atual do MPD (comparando tanto a string bruta quanto o caminho normalizado).

### 3.2. Mecanismo de Esgotamento do Histórico (*History Fallback*)
Se todas as 20 candidatas forem rejeitadas por já estarem no histórico, o sistema aplica um relaxamento progressivo:
- **Arquivo:** [`audio_pipeline/mpd_integration.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/mpd_integration.py)
- **Linhas:** 287 a 296

```python
# audio_pipeline/mpd_integration.py:287-296
    # Esgotamento: se todas forem rejeitadas por estarem no histórico, descartar entradas mais antigas
    if not valid_candidates and hist_set:
        logging.warning("Todas as candidatas estão no histórico. Descartando entradas mais antigas do histórico...")
        if msg_queue:
            msg_queue.put({"type": "log", "text": "Histórico esgotado: relaxando histórico antigo"})
        temp_hist = list(history)
        while temp_hist and not valid_candidates:
            temp_hist.pop(0)  # remove mais antiga
            valid_candidates = filter_candidates(set(temp_hist))
```
Ele remove sequencialmente as músicas mais antigas da memória temporária até liberar pelo menos um candidato viável.

### 3.3. Modos de Seleção (`SELECTION_MODE`)
Uma vez obtida a lista de candidatos válidos, o algoritmo decide qual faixa escolher:
- **`first`:** Escolhe estritamente o candidato de menor distância cosseno (`valid_candidates[0]`).
- **`random_top_n` (padrão):** Realiza sorteio pseudoaleatório uniforme (`rng.choice(pool)`) entre os primeiros `top_n = 3` candidatos válidos, garantindo que a execução não se torne determinística e repetitiva.
- **`mmr` (Maximal Marginal Relevance):** Aplica a fórmula:
  $$\text{Score}_{\text{MMR}} = \lambda \cdot \text{Sim}(\mathbf{q}, \mathbf{c}) - (1 - \lambda) \cdot \max_{\mathbf{h} \in \text{hist}} \text{Sim}(\mathbf{c}, \mathbf{h})$$
  onde $\lambda = 0.6$. Busca maximizar a proximidade com a música atual enquanto penaliza a proximidade vetorial com as últimas 5 faixas reproduzidas (`hist_ids = list(play_history)[-5:]`).

---

## 4. Reposição da Fila (`replenish_mpd_queue`)

A injeção de novas músicas na playlist do MPD é controlada pela função `replenish_mpd_queue`:

- **Arquivo:** [`audio_pipeline/mpd_integration.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/mpd_integration.py)
- **Linhas:** 334 a 348

```python
# audio_pipeline/mpd_integration.py:334-348
    playlist = client.playlistinfo()
    status = client.status()
    current_pos = int(status.get("song", -1))

    if current_pos == -1:
        ahead = len(playlist)
    else:
        ahead = len(playlist) - (current_pos + 1)

    added_count = 0
    attempts = 0

    while ahead < min_queue_ahead and attempts < max_tries:
        attempts += 1
        ...
```

- **Métrica `ahead`:** Indica quantas músicas restam a tocar antes do silêncio. Se a playlist tem 5 faixas e a faixa tocando é a posição 2 (0-indexed), `ahead = 5 - (2 + 1) = 2`.
- **Condição de disparo:** Enquanto `ahead < MIN_QUEUE_AHEAD` (padrão: 2 faixas à frente), o laço invoca `select_next_recommendations` e executa `client.add(candidate_file)`.
- **Proteção contra travamento (`MAX_REPLENISH_TRIES`):** O laço é limitado a no máximo 5 iterações (`MAX_REPLENISH_TRIES = 5`). Se o banco estiver sem candidatas ou houver sucessivos erros de `add`, o laço termina e não bloqueia a thread de execução.

---

## 5. Estado `stop` com Fila Esgotada e `AUTO_RESUME_PLAY_ON_STOP`

Quando as músicas da fila acabam por completo sem que novas tenham sido enfileiradas a tempo, ou quando o usuário pressiona "Stop" e a fila está vazia:

- **Arquivo:** [`audio_pipeline/mpd_integration.py`](file:///c:/Codigos/Recomendacao_musicas_c.1.01/audio_pipeline/mpd_integration.py)
- **Linhas:** 539 a 572

```python
# audio_pipeline/mpd_integration.py:539-572
                elif state == "stop":
                    # Fila esgotada ou usuário parou a reprodução
                    playlist = client.playlistinfo()
                    if len(playlist) == 0:
                        logging.warning("MPD em estado 'stop' com fila vazia. Repondo faixas na fila...")
                        if msg_queue:
                            msg_queue.put({"type": "log", "text": "MPD parado (fila vazia). Repondo fila..."})

                        # Usa a última faixa do histórico ou alguma faixa conhecida para repor
                        last_id = play_history[-1] if play_history else list(id_to_file_map.keys())[0] if id_to_file_map else None
                        if last_id:
                            replenish_mpd_queue(...)
                            # Se a constante permitir, retoma a reprodução automaticamente
                            if AUTO_RESUME_PLAY_ON_STOP:
                                try:
                                    client.play(0)
                                    logging.info("Reprodução retomada automaticamente após reposição da fila.")
                                except Exception as e:
                                    logging.error(f"Erro ao retomar reprodução: {e}")
```

**Comportamento constatado no código:**
1. A variável `AUTO_RESUME_PLAY_ON_STOP` está definida na linha 24 como `False`:
   ```python
   AUTO_RESUME_PLAY_ON_STOP = False  # Retomar play se o MPD parar com fila esgotada (padrão desligada)
   ```
2. Portanto, quando a fila zera, o sistema enfileira novas recomendações baseando-se no último item do histórico, mas **não dá play automaticamente**. O player permanece em `stop` aguardando que o usuário aperte o botão play na interface do Cantata.
