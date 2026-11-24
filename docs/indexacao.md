# Indexação de Biblioteca: Em Lote vs. Sob Demanda (On-The-Fly)

Este documento compara os dois mecanismos de indexação existentes no repositório: a indexação preventiva em lote (`index_library.py`) e a indexação automática sob demanda (*lazy indexing* em `auto_indexer.py`).

---

## 1. Tabela Comparativa dos Dois Mecanismos

A tabela abaixo sintetiza as discrepâncias comportamentais confirmadas na análise do código-fonte:

| Característica | Indexação em Lote (`index_library.py`) | Indexação Sob Demanda (`auto_indexer.py`) |
| :--- | :--- | :--- |
| **Gatilho de Execução** | Manual via linha de comando (`python index_library.py --root ...`) ou bat. | Automático durante a escuta, acionado por `mpd_integration.py` quando a faixa tocada não tem `track_id` mapeado. |
| **Formato de `track_id`** | Caminho relativo sanitizado: `"Subgenero/Musica.mp3"` (`index_library.py:53`). | Numérico sequencial com 4 dígitos: `"0000"`, `"0001"`, etc. (`auto_indexer.py:108`). |
| **Destino dos Metadados** | Exclusivamente em `audio_pipeline/results/metadata.json` (`index_library.py:41`). | Passado por argumento via CLI (`--metadata`), tipicamente `results/metadata.json`. |
| **Deduplicação Acústica (Fingerprint)** | **NÃO UTILIZA** (confirmado no código). Apenas verifica se o ID já existe no ChromaDB (`ids_existentes_no_chroma`). | **UTILIZA** via Chromaprint (`fingerprint.generate_fingerprint` e `find_duplicate_by_fingerprint`). |
| **Granularidade de Gravação no ChromaDB** | Em lotes de 10 faixas (`LOTE_SIZE = 10`) com `_flush()`. | Unitário imediato via `collection.upsert(ids=[new_track_id], ...)`. |
| **Tratamento de Arquivos Inacessíveis** | Tenta duas vezes (com pausa de 3s para streaming do Google Drive) e registra em `log/indexacao_falhas.csv`. | Aborta a faixa sem retry especializado e registra no log padrão de erro. |
| **Gestão de Aliases** | Não possui conceito de aliases; cada caminho de arquivo é tratado como um item único. | Adiciona caminhos duplicados ao campo `"aliases": [...]` no JSON de metadados se o fingerprint for idêntico. |

---

## 2. Deduplicação por Fingerprint Acústico

### 2.1. Confirmação sobre `index_library.py`
A análise completa das 425 linhas de [`audio_pipeline/index_library.py`](../audio_pipeline/index_library.py) confirma que o módulo:
1. **Não importa** o módulo `fingerprint.py`.
2. **Não invoca** `generate_fingerprint`, `run_fpcalc_raw` ou `acoustid`.
3. A verificação de existência baseia-se unicamente no conjunto retornado por `ids_existentes_no_chroma(collection)`:
   ```python
   # index_library.py:199
   pendentes = [f for f in arquivos if _rel_id(f, root) not in ids_existentes]
   ```
Portanto, se existirem dois arquivos binários exatamente idênticos na pasta (ex.: `Rock/musica.mp3` e `Favoritas/musica_copia.mp3`), o `index_library.py` calculará os embeddings CLAP para ambos e inserirá duas entradas separadas no ChromaDB com IDs diferentes.

### 2.2. Implementação em `auto_indexer.py`
Em contrapartida, a indexação sob demanda implementa deduplicação acústica estrita antes de calcular os embeddings neurais:

- **Arquivo:** [`audio_pipeline/auto_indexer.py`](../audio_pipeline/auto_indexer.py)
- **Linhas:** 41 a 87

```python
# auto_indexer.py:41-87
    # 1. Gerar fingerprint
    new_fp = generate_fingerprint(filepath)
    if not new_fp:
        logger.error(f"Falha ao gerar fingerprint para {filepath}. Indexação abortada.")
        return None
        
    # 2. Verificar duplicata
    duplicate_track_id = find_duplicate_by_fingerprint(new_fp, known_fingerprints, threshold=0.95)
    ...
    if duplicate_track_id:
        logger.info(f"Faixa '{filename}' identificada como duplicata de track_id={duplicate_track_id}")
        
        # Adicionar novo filepath aos aliases
        for entry in metadata_list:
            if str(entry.get("track_id")) == duplicate_track_id:
                if "aliases" not in entry:
                    entry["aliases"] = []
                if rel_filepath not in entry["aliases"] and entry.get("filepath") != rel_filepath:
                    entry["aliases"].append(rel_filepath)
                break
                
        # Salvar metadata atualizado
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            
        return duplicate_track_id
```

Se a similaridade de bits (XOR com deslocamento de janela) entre o fingerprint do arquivo tocado e qualquer uma das assinaturas conhecidas for superior a 95% (`threshold=0.95`):
- O áudio não é enviado para o modelo CLAP (economizando CPU).
- O ChromaDB não recebe novo vetor.
- O caminho é registrado no campo `aliases` da entrada já existente em `metadata.json`.
- O `track_id` original já existente é retornado para servir como semente de recomendação imediata.

---

## 3. Gestão de Falhas e Tolerância ao Google Drive

Como a biblioteca pode estar hospedada em uma unidade virtual montada do Google Drive (`G:\Meu Drive\Musicas`), arquivos em modo "Somente Online" podem ter 0 bytes no momento em que o sistema tenta acessá-los.

O `index_library.py` implementa tratamento específico para esse cenário:
- **Arquivo:** [`audio_pipeline/index_library.py`](../audio_pipeline/index_library.py)
- **Linhas:** 230 a 248

```python
# index_library.py:230-248
            # Verificar acesso (Drive ainda não baixado)
            try:
                tamanho = filepath.stat().st_size
                if tamanho == 0:
                    raise ValueError("Arquivo com 0 bytes (provavelmente não baixado do Drive)")
            except Exception as e:
                # Tentar 1 vez após 3s
                time.sleep(3)
                try:
                    tamanho = filepath.stat().st_size
                    if tamanho == 0:
                        raise ValueError("Ainda 0 bytes na segunda tentativa")
                except Exception as e2:
                    falhas.append({
                        "timestamp": datetime.now().isoformat(),
                        "filepath": str(filepath),
                        "motivo": f"Inacessível: {e2}",
                    })
                    continue
```

Se o arquivo continuar inacessível, a falha é registrada em `audio_pipeline/log/indexacao_falhas.csv` através de `salvar_falhas()` e a barra de progresso continua sem abortar o processo global.
