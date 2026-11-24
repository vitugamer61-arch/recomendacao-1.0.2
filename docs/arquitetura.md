# Arquitetura do Sistema de Recomendação Musical

Este documento descreve a arquitetura geral do sistema integrador de recomendação musical baseado em MPD (Music Player Daemon), interface cliente Cantata, banco vetorial ChromaDB e modelo de representação neural CLAP.

A descrição a seguir reflete a implementação física real presente na árvore do repositório (`audio_pipeline/`), registrando os nomes exatos de rotinas e o comportamento efetivo de execução.

---

## 1. Diagrama Textual do Fluxo de Dados e Controle

O ciclo operacional completo é orientado a eventos e polling de bloqueio longo (*idle*) provido pelo protocolo MPD. O fluxo entre interface, servidor, extrator de características e repositório vetorial ocorre na seguinte ordem:

```text
[Usuário / Interface Cantata]
        │
        ▼ (reproduz faixa, enfileira ou altera estado)
[Servidor MPD (porta 6600)]
        │
        ├─ Bloqueio / Notificação de evento: client.idle("player")
        ▼
[mpd_integration.py : monitor_and_recommend()]
        │
        ├─ 1. client.status() -> state == "play" ou "stop"
        ├─ 2. client.currentsong() -> raw_file (URL do Cantata ou path relativo)
        ├─ 3. normalize_mpd_filepath() -> caminho relativo padronizado
        │
        ├─ 4. Resolução de ID: file_to_id_map.get(current_file)
        │       ├── [Se não encontrado] ──► auto_indexer.index_unknown_track()
        │       │                                │
        │       │                                ├─ fingerprint.generate_fingerprint()
        │       │                                ├─ fingerprint.find_duplicate_by_fingerprint()
        │       │                                ├─ pipeline.process_track()
        │       │                                ├─ collection.upsert() (ChromaDB)
        │       │                                └─ atualiza metadata.json
        │       └── [Se encontrado] ─────► current_track_id
        │
        ├─ 5. Atualização de Histórico: play_history.append(current_track_id)
        │       └─ save_session_history() (gravação atômica em session_history.json)
        │
        ▼
[mpd_integration.py : replenish_mpd_queue()]
        │
        ├─ 6. Avaliação de fila: client.playlistinfo() -> ahead < min_queue_ahead?
        │
        ▼
[mpd_integration.py : select_next_recommendations()]
        │
        ├─ 7. Busca de embedding base: collection.get(ids=[current_track_id])
        │
        ▼
[vector_store.py : query_similar()]
        │
        ├─ 8. Consulta HNSW (distância cosseno): collection.query(query_embeddings, n_results=K+1)
        │
        ▼ [Retorna lista de K candidatos brutos]
[mpd_integration.py : select_next_recommendations()]
        │
        ├─ 9. Filtros de exclusão:
        │       - ID em failed_tracks (erros prévios de comando add)
        │       - ID no histórico da sessão (play_history)
        │       - Arquivo já presente na playlist ativa do MPD (queued_files)
        │
        ├─ 10. Política de seleção (SELECTION_MODE):
        │       - "first": primeiro candidato válido
        │       - "random_top_n": sorteio uniforme entre os top_n primeiros válidos
        │       - "mmr": Maximal Marginal Relevance via compute_mmr_selection()
        │
        ▼ [Retorna tupla: (chosen_file, chosen_rec, chosen_rank)]
[mpd_integration.py : replenish_mpd_queue()]
        │
        ├─ 11. Injeção de áudio: client.add(candidate_file)
        └─ 12. Repetição até preencher a margem mínima ou atingir MAX_REPLENISH_TRIES
        │
        ▼
[Servidor MPD]
        │
        ▼ (nova faixa adicionada ao final da fila)
[Cantata / Alto-falante]
```

---

## 2. Responsabilidades dos Módulos e Funções Principais

A divisão funcional do sistema é organizada nos seguintes arquivos dentro de `audio_pipeline/`:

### 2.1. `mpd_integration.py`
Módulo orquestrador em tempo real. Não existe um arquivo isolado denominado `mpd_recommender.py` na base; toda a lógica de recomendação e reposição de fila reside em `mpd_integration.py`. As referências abaixo são por função, pois os números de linha mudam.

- **`connect_mpd(host, port)`**: Estabelece conexão com o socket TCP do MPD aplicando retentativas automáticas (`max_retries = 3`) com recuo exponencial.
- **`reconnect_if_needed(client)`**: Executa `client.ping()`. Caso a conexão tenha caído (`ConnectionError`, `BrokenPipeError`, etc.), fecha o descritor e reconecta.
- **`build_file_to_id_map(metadata_path, music_library_root)`**: Lê o arquivo de metadados JSON e compila dicionários em memória para tradução bidirecional (`file_to_id` e `id_to_file`).
- **`normalize_mpd_filepath(raw_path, music_root)`**: Trata URLs geradas pelo servidor HTTP interno do Cantata (`?file=...`), aplica `urllib.parse.unquote()` e converte caminhos absolutos em caminhos relativos ao `music_root`.
- **`load_session_history` / `save_session_history`**: Histórico de faixas como `collections.deque(maxlen=...)`, persistido em `session_history.json` com gravação atômica (`.tmp` + `os.replace()`).
- **`compute_mmr_selection(...)`**: Score MMR ponderando similaridade com a faixa semente versus redundância em relação aos últimos 5 itens tocados.
- **`select_next_recommendations(...)`**: Escolhe a próxima faixa: vetor de consulta (com Rocchio se o feedback estiver ligado), busca de vizinhas no ChromaDB, re-ranking e rejeição por feedback, filtros (histórico, fila, falhas), relaxamento do histórico, **expansão progressiva da busca** quando todas são rejeitadas, e estratégia de escolha (`random_top_n`, `first` ou `mmr`). Detalhes em [recomendacao_e_fila.md](recomendacao_e_fila.md).
- **`replenish_mpd_queue(...)`**: Insere até `MIN_QUEUE_AHEAD` recomendações novas logo após a faixa atual (`addid` em posição explícita) e depois remove as pendências antigas do recomendador (swap), com descarte por corrida e limite `max_tries`.
- **`mpd_add_recommendation_unified(...)` / `mpd_purge_pending_recommendations(...)`**: Inserção e remoção de faixas do recomendador, com controle de propriedade por `songid` (a faixa atual e as anteriores nunca são removidas).
- **`sync_mpd_queue_state(...)`**: Espelha a fila do MPD no SQLite (`mpd_queue_state`) para o monitor.
- **`monitor_and_recommend(...)`**: Executa em thread secundária. Laço contínuo com `client.idle("player")`; no estado `play` identifica a faixa por `songid` e dispara o ciclo de recomendação; no estado `stop` distingue fim da fila de parada manual e retoma a reprodução quando cabe; aciona a indexação sob demanda quando a faixa não possui mapeamento e despacha mensagens para a interface de monitoramento.
- **`run_recommender(...)`**: Ponto de entrada que conecta ao MPD, monta os mapas e inicia o monitor.

### 2.2. `auto_indexer.py`
Módulo de contingência para catalogação imediata (*on-the-fly* / *lazy indexing*).

- **`load_known_fingerprints(metadata_path)`** (`auto_indexer.py:12`): Carrega a tabela `{track_id: fingerprint}` a partir do JSON de metadados.
- **`index_unknown_track(filepath, metadata_path, collection, known_fingerprints, music_library_root)`** (`auto_indexer.py:34`): Acionado quando uma faixa tocada no MPD não existe na base vetorial. Gera fingerprint acústico via Chromaprint, pesquisa duplicatas (`find_duplicate_by_fingerprint`), calcula novo ID sequencial `XXXX`, invoca o extrator de características (`pipeline.process_track`), insere no ChromaDB via `collection.upsert()` e salva a entrada em disco.

### 2.3. `index_library.py`
Script CLI para ingestão em lote (*batch indexing*) de diretórios locais ou mapeados.

- **`varrer_biblioteca(root)`** (`index_library.py:63`): Percorre recursivamente a árvore de diretórios buscando arquivos com extensões de áudio suportadas (`.mp3`, `.flac`, `.wav`, `.m4a`, `.ogg`).
- **`ids_existentes_no_chroma(collection)`** (`index_library.py:72`): Consulta todos os IDs já presentes no ChromaDB em lotes de 500 para evitar reprocessamento de arquivos inalterados.
- **`indexar(root, collection, ids_existentes, limit, reindex)`** (`index_library.py:186`): Itera sobre as faixas pendentes, invoca `pipeline.process_track()`, acumula tensores e persiste em lotes de 10 (`LOTE_SIZE`) no ChromaDB e no JSON de metadados (`results/metadata.json`).
- **`limpar_orfas(collection, root)`** (`index_library.py:140`): Localiza e remove vetores no banco cuja referência em disco não exista mais.

### 2.4. `vector_store.py`
Camada de abstração do cliente vetorial ChromaDB.

- **`get_client(persist_dir)`** (`vector_store.py:7`): Cria o cliente persistente `chromadb.PersistentClient`.
- **`get_or_create_collection(client, name)`** (`vector_store.py:10`): Obtém ou instancia a coleção configurada com métrica de espaço cosseno (`metadata={"hnsw:space": "cosine"}`).
- **`index_embeddings(collection, embeddings_path, metadata_path)`** (`vector_store.py:16`): Lê tensores de arquivos `.npz` legados e popula a coleção.
- **`query_similar(collection, query_embedding, k, exclude_id)`** (`vector_store.py:65`): Consulta os vizinhos mais próximos no espaço vetorial, filtrando opcionalmente o próprio ID semente.

### 2.5. `fingerprint.py`
Módulo de identificação acústica para evitar duplicatas de áudio idêntico com nomes de arquivo diferentes.

- **`run_fpcalc_raw(filepath)`** (`fingerprint.py:21`): Invoca o binário executável `fpcalc.exe` com argumento `-raw` via `subprocess.run()`.
- **`generate_fingerprint(filepath)`** (`fingerprint.py:42`): Gera a assinatura acústica usando a biblioteca C `pyacoustid`/`chromaprint` ou aplicando fallback para o binário `fpcalc`.
- **`compare_fingerprints(fp1, fp2)`** (`fingerprint.py:61`): Calcula a taxa de correspondência através de operação bit a bit XOR (Hamming distance) descompactada, buscando alinhamento de janelas deslizantes.
- **`find_duplicate_by_fingerprint(new_fp, known_fingerprints, threshold)`** (`fingerprint.py:119`): Varre o acervo de assinaturas conhecidas buscando similaridade acústica acima do limiar configurado (padrão: 0.95).

### 2.6. `evaluation.py` e `results/generate_precision_k.py`
Módulos de avaliação offline e acompanhamento estatístico da qualidade da representação vetorial.

- **`load_ground_truth(metadata_path)`** (`evaluation.py:167`): Mapeia `{track_id: subgenero}` para calcular correspondência temática.
- **`precision_at_k_for_track(collection, track_id, ground_truth, k)`** (`evaluation.py:200`): Calcula a métrica Precision@K para uma faixa individual inspecionando se os K vizinhos recuperados no ChromaDB pertencem ao mesmo subgênero do semente.
- **`random_baseline_precision_at_k(ground_truth, k, n_trials)`** (`evaluation.py:230`): Estima a precisão esperada por sorteio puramente aleatório uniforme através de simulações de Monte Carlo.
- **`run_full_evaluation(collection, ground_truth, k_values)`** (`evaluation.py:252`): Executa o benchmark para múltiplos valores de K (`[1, 3, 5, 10]`) e computa o ganho relativo em relação à baseline.

### 2.7. `status_window.py`
Interface gráfica em `tkinter` desacoplada via `queue.Queue`.
- **`StatusWindow`** (`status_window.py:49`): Renderiza janela flutuante com mostradores de estado operacional (Monitorando, Indexando, Erro), contadores de recomendações geradas, indexações sob demanda e log recente de eventos.

### 2.8. `feedback.py` e `feedback_config.py`
Feedback implícito por skip/reprodução parcial/completa, adaptação Rocchio do vetor de consulta, re-ranking com penalidade de faixa e de gênero, e cooldown de gênero. Persistência em SQLite (`results/feedback.db`). Ver [feedback_implicito.md](feedback_implicito.md).

### 2.9. `clustering.py`
Agrupamento das faixas por similaridade sonora (K-Means com refinamento por pureza, HDBSCAN opcional), nomeação por gênero predominante, atribuição incremental de faixas novas e exportação de playlists. Ver [clusterizacao.md](clusterizacao.md).

### 2.10. `iniciar.py`
Ponto de entrada único em 6 etapas: MPD, indexação, clusterização, sincronização de playlists por cluster, Monitor de Precisão (Streamlit) e recomendador. Opções: `--pasta`, `--recalcular`, `--modo`, `--sem-feedback`, `--zerar-feedback`, `--sem-cluster`, `--no-gui`, `--sem-monitor`, `--seed`.

### 2.11. `testes_disponibilidade/`
Suíte externa de disponibilidade do serviço de consulta ao ChromaDB (watchdog, injeção de falhas, carga e restauração). Ver [disponibilidade.md](disponibilidade.md).
