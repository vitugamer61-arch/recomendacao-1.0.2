# Identificação de Faixas e Mapeamento de Metadados

Este documento descreve como o sistema associa um arquivo em reprodução no servidor MPD ao identificador único (`track_id`) no banco de dados vetorial ChromaDB, detalhando as rotinas de normalização, coexistência de identificadores heterogêneos e inconsistências entre arquivos de metadados.

---

## 1. Normalização de Caminhos do MPD (`normalize_mpd_filepath`)

O protocolo MPD e seus clientes gráficos (em especial o Cantata) reportam nomes de arquivo com formatações distintas dependendo de como a biblioteca foi carregada.

### 1.1. URLs HTTP do Cantata e URL-Encoding
Quando o Cantata reproduz faixas através de seu recurso de servidor HTTP embutido, a resposta de `client.currentsong().get("file")` não vem no formato de caminho relativo comum, mas como uma URL no padrão:
`http://localhost:port/?file=/caminho/para/arquivo.mp3` ou contendo percent-encoding (`%20` para espaços, `%C3%A1` para acentos, etc.).

A função `normalize_mpd_filepath` trata essas variações:
- **Arquivo:** [`audio_pipeline/mpd_integration.py`](../audio_pipeline/mpd_integration.py)
- **Linhas:** 150 a 174

```python
# audio_pipeline/mpd_integration.py:150-174
def normalize_mpd_filepath(raw_path: str, music_root: str = "") -> str:
    """Normaliza caminhos retornados pelo MPD ou URLs do Cantata para o formato relativo padrão."""
    if not raw_path:
        return ""
    # Se for URL enviada pelo Cantata usando servidor HTTP interno (ex: http://.../?file=...)
    if raw_path.startswith("http") and "?file=" in raw_path:
        parsed_url = urllib.parse.urlparse(raw_path)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        if 'file' in query_params:
            raw_path = query_params['file'][0]
    
    # Decodificar URL encoding caso exista (ex: %20 -> espaço)
    decoded_path = urllib.parse.unquote(raw_path)
    norm_path = os.path.normpath(decoded_path)
    
    if music_root and os.path.isabs(norm_path):
        try:
            norm_root = os.path.normpath(music_root)
            if norm_path.lower().startswith(norm_root.lower()):
                rel_path = os.path.relpath(norm_path, norm_root)
                return rel_path.replace('\\', '/')
        except ValueError:
            pass

    return norm_path.replace('\\', '/')
```

### 1.2. Regra de Saída
O caminho retornado por `normalize_mpd_filepath` é sempre normalizado com barras para a direita (`/`) e relativo ao diretório raiz de músicas (`music_root`), correspondendo ao formato de diretório virtual aceito pelo comando `add` do MPD (ex.: `Eletrônica/Alok & Bhaskar - FUEGO (Original Mix).mp3`).

---

## 2. Construção do Mapa em Memória (`build_file_to_id_map`)

Na inicialização do integrador (`mpd_integration.py:618`), o sistema carrega o arquivo de metadados JSON indicado pelo parâmetro `--metadata` (por padrão `results/metadata.json` no script de inicialização):

- **Arquivo:** [`audio_pipeline/mpd_integration.py`](../audio_pipeline/mpd_integration.py)
- **Linhas:** 86 a 119

```python
# audio_pipeline/mpd_integration.py:86-119
def build_file_to_id_map(metadata_path: str, music_library_root: str) -> tuple[dict, dict]:
    """Cria mapeamentos entre os caminhos relativos ao root da biblioteca MPD e os IDs do ChromaDB."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
        
    file_to_id = {}
    id_to_file = {}
    
    for meta in metadata_list:
        track_id = str(meta['track_id'])
        filepath = meta['filepath']
        
        norm_filepath = os.path.normpath(filepath)
        norm_root = os.path.normpath(music_library_root)

        if os.path.isabs(norm_filepath):
            if norm_filepath.lower().startswith(norm_root.lower()):
                rel_path = os.path.relpath(norm_filepath, norm_root)
            else:
                rel_path = filepath
        else:
            abs_path = os.path.join(norm_root, norm_filepath)
            rel_path = os.path.relpath(abs_path, norm_root)
            
        rel_path = rel_path.replace('\\', '/')
        
        file_to_id[rel_path] = track_id
        id_to_file[track_id] = rel_path
        
    if file_to_id:
        example_file, example_id = list(file_to_id.items())[0]
        logging.info(f"Exemplo de mapeamento: MPD File '{example_file}' <-> Track ID '{example_id}'")
        
    return file_to_id, id_to_file
```

A estrutura produz dois dicionários:
- `file_to_id`: mapeia `caminho_relativo -> track_id` (usado quando o MPD notifica que uma música começou a tocar).
- `id_to_file`: mapeia `track_id -> caminho_relativo` (usado quando o ChromaDB retorna um ID recomendado e o sistema precisa emitir `client.add(caminho)`).

---

## 3. Coexistência de Formatos de ID e Risco de Colisão

No estado atual da base de código, existem **dois padrões mutuamente conflitantes** para a chave `track_id`:

### 3.1. Padrão 1: IDs Numéricos Sequenciais de 4 Dígitos (`"0000"`, `"0001"`, ...)
- **Origem:** Gerado pelo pipeline inicial em lote (`pipeline.py:114`) e pelo auto-indexador sob demanda (`auto_indexer.py:108`):
  ```python
  # auto_indexer.py:108
  new_id_int = max_id + 1 if max_id >= 0 else 0
  new_track_id = f"{new_id_int:04d}"
  ```
- **Formato:** String contendo inteiros formatados com zero à esquerda (`0000` a `9999`).

### 3.2. Padrão 2: IDs Baseados no Caminho Relativo (`"subgenero/arquivo.mp3"`)
- **Origem:** Gerado pela rotina de indexação em lote em `index_library.py`:
  ```python
  # index_library.py:53-60
  def _rel_id(filepath: Path, root: Path) -> str:
      rel = filepath.relative_to(root)
      return str(rel).replace("\\", "/")
  ```
  E aplicado diretamente como `track_id` no ChromaDB:
  ```python
  # index_library.py:276-277
  resultado["track_id"]  = track_id  # sobrescreve com o ID relativo
  resultado["filepath"]  = track_id
  ```

### 3.3. Risco Prático e Dados Reais no Repositório
A inspeção física do arquivo ativo [`audio_pipeline/results/metadata.json`](../audio_pipeline/results/metadata.json) comprova essa coexistência no mesmo banco:
- Total de faixas cadastradas: **1.288 faixas**.
- Registros com ID numérico (`"0000"` a `"0081"`): **67 faixas**.
- Registros com ID em caminho relativo (`"Eletrônica/..."`, `"Rock/..."`): **1.221 faixas**.

**Consequências dessa disparidade:**
1. **Quebra do cálculo de próximo ID:** Em `auto_indexer.py:98-107`, o algoritmo varre o `metadata_list` convertendo `int(entry.get("track_id"))`. Para os 1.221 itens com caminhos de texto, o bloco `except ValueError: pass` é disparado silenciosamente. O `max_id` obtido considera apenas as faixas numéricas residuais.
2. **Potencial de sobreposição de metadados:** Caso uma pasta musical contenha faixas numéricas puras no nome (ex: `0001.mp3`), pode ocorrer conflito semântico entre a string da chave e o número.
3. **Incompatibilidade em scripts legados:** Scripts como `migrate_ids.py` foram criados para forçar a renomeação de todos os IDs para numéricos (`migrate_ids.py:18`), mas seu uso exige limpeza manual completa do diretório `chroma_db/`, operação não executada automaticamente.

---

## 4. Inconsistência entre Arquivos de Metadados e Indexação

Outra divergência estrutural ocorre entre os arquivos de metadados utilizados por diferentes scripts:

### 4.1. `output/metadata.json` vs. `results/metadata.json`
- `pipeline.py:60` (`process_directory`) aceita um `--output_dir` configurável que, por padrão de projeto antigo, gerava `output/metadata.json` e `output/embeddings.npz`.
- `vector_store.py:103` tem como valor padrão `--metadata_path output/metadata.json`.
- `index_library.py:41` grava obrigatoriamente e exclusivamente em `audio_pipeline/results/metadata.json` (`RESULTS_METADATA = _SCRIPT_DIR / "results" / "metadata.json"`).
- `iniciar_sistema.bat:57` invoca `mpd_integration.py` apontando explicitamente para `--metadata results\metadata.json`.

### 4.2. Consequência Direta: Faixas Fantasma ou Ocultas
Se um usuário executar o script `pipeline.py` standalone salvando em `output/`, ou usar `vector_store.py --action index` com seus parâmetros padrão, o ChromaDB será populado com os vetores de `output/metadata.json`.

Entretanto, quando `mpd_integration.py` inicializa:
```bash
mpd_integration.py --metadata results\metadata.json
```
A função `build_file_to_id_map` compila apenas os IDs presentes em `results/metadata.json`.
Se uma música foi indexada em outro arquivo ou se houve defasagem entre `output/` e `results/`, ocorrem duas falhas:
1. **Falha de Resolução (Falso Positivo de Faixa Desconhecida):** O MPD toca a música, `file_to_id_map.get(current_file)` retorna `None` (pois ela não está em `results/metadata.json`), forçando o acionamento desnecessário do `auto_indexer.index_unknown_track()`.
2. **Recomendação Não Reproduzível:** O ChromaDB pode sugerir como vizinho mais próximo um `track_id` que existe na coleção do banco vetorial, mas cuja entrada inexiste em `id_to_file_map`. Ao tentar selecionar o arquivo para reprodução:
   ```python
   # mpd_integration.py:281
   if rec_file:
       valid.append((rank, rec))
   ```
   A recomendação é descartada porque `id_to_file_map.get(rec_id)` retorna `None`, reduzindo a quantidade efetiva de recomendações entregues ao usuário.
