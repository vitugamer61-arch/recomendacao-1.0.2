# Estado Atual, Inconsistências Conhecidas e Limitações

Este documento registra de forma técnica e objetiva as inconsistências, fragilidades e limitações arquiteturais constatadas diretamente no código-fonte do sistema. Para cada item, descreve-se o comportamento real do código, o comportamento esperado e o impacto prático (se afeta a experiência audível de reprodução ou apenas a camada de avaliação estatística).

---

## 1. Tabela Síntese de Limitações

| # | Problema Identificado | Arquivo e Ponto de Origem | Impacto Prático |
| :-: | :--- | :--- | :--- |
| **1** | Duplicidade e Conflito de Arquivos de Metadados | `pipeline.py:60`, `vector_store.py:103`, `index_library.py:41`, `iniciar_sistema.bat:57` | **Recomendação Audível e Avaliação** |
| **2** | Mistura de Padrões de Identificador (`track_id`) | `auto_indexer.py:108` vs. `index_library.py:53` | **Recomendação Audível e Indexação** |
| **3** | Deduplicação Acústica Inexistente na Indexação em Lote | `index_library.py:186` | **Recomendação Audível** (desperdício de espaço/cálculo) |
| **4** | Desacoplamento entre Base de Ground Truth e Base do ChromaDB | `evaluation.py:12` e `evaluation.py:53` | **Avaliação (Métricas P@K)** |
| **5** | Exceção `AssertionError` Não Capturada na Avaliação | `evaluation.py:41` | **Avaliação (Métricas P@K)** |
| **6** | Dependência Crítica de Ponto de Montagem do Google Drive | `iniciar_sistema.bat:26`, `clean_orphans2.py:6` | **Recomendação Audível** (bloqueio de inicialização) |
| **7** | Caminhos Absolutos Fixos no Código (*Hardcoded Paths*) | `iniciar_sistema.bat:34`, `clean_orphans2.py:6` | **Portabilidade e Execução** |
| **8** | Ausência de Tratamento de Erro na Conexão Inicial do MPD | `mpd_integration.py:652`, `mpd_integration.py:71` | **Recomendação Audível** (falha fatal na inicialização) |
| **9** | Custo Computacional do Modelo CLAP em CPU | `embedding.py:11`, `pipeline.py:23-45` | **Recomendação Audível e Indexação** (latência elevada) |
| **10** | Detecção de Mudança de Faixa sem Uso de `songid` | `mpd_integration.py:469` | **Recomendação Audível** (falha em faixas repetidas) |

---

## 2. Detalhamento Técnico de Cada Limitação

### 2.1. Metadados Duplicados e Descompassados
- **O que o código faz:**
  - O script histórico `pipeline.py` gera metadados em `output/metadata.json`.
  - O script de lote `index_library.py` grava forçosamente em `results/metadata.json`.
  - O inicializador `iniciar_sistema.bat` invoca `mpd_integration.py --metadata results\metadata.json`.
  - O script de limpeza `clean_orphans.py` tenta abrir `"results/metadata.json"` com caminhos relativos de trabalho que falham se o script for executado a partir da raiz do repositório.
- **O que deveria fazer:**
  Centralizar a configuração de caminhos em `config.py` ou ler a partir de um único arquivo de metadados unificado com caminho resolvido via `Path(__file__).parent`.
- **Impacto:** **Afeta a recomendação audível**. Faixas indexadas por ferramentas diferentes não aparecem no mapa do integrador, resultando em reindexações duplicadas sob demanda ou rejeição silenciosa de recomendações por falta de mapeamento de caminho.

---

### 2.2. IDs Heterogêneos Misturados no Mesmo Banco
- **O que o código faz:**
  - `pipeline.py` e `auto_indexer.py` usam strings de números inteiros (`"0000"`, `"0001"`).
  - `index_library.py` usa caminhos relativos sanitizados (`"Eletrônica/musica.mp3"`).
  - No arquivo atual `results/metadata.json`, existem 67 faixas com ID numérico e 1.221 faixas com ID em formato de caminho.
  - Ao tentar gerar um novo ID sob demanda, `auto_indexer.py:98-106` tenta converter todas as entradas para `int()`, ignorando as 1.221 faixas textuais via `except ValueError: pass`.
- **O que deveria fazer:**
  Adotar um padrão único e imutável de chave primária em todo o ecossistema (por exemplo, hash SHA-256 do áudio ou o caminho relativo normalizado de forma estrita em todos os módulos).
- **Impacto:** **Afeta a indexação e a recomendação audível**. Gera inconsistência no ChromaDB, impede a ordenação determinística e quebra o cálculo do sequenciador de novos IDs.

---

### 2.3. Ausência de Deduplicação na Indexação em Lote
- **O que o código faz:**
  - O script principal de catalogação em lote (`index_library.py`) não importa `fingerprint.py`. Ele verifica apenas se o caminho relativo do arquivo está no conjunto `ids_existentes_no_chroma`.
- **O que deveria fazer:**
  Consultar a assinatura acústica (Chromaprint) antes de calcular embeddings pesados, reaproveitando vetores existentes para faixas idênticas duplicadas em pastas distintas.
- **Impacto:** **Afeta a recomendação audível e o desempenho**. Se o acervo contiver faixas duplicadas (ex.: álbuns diferentes com a mesma música), o sistema gastará tempo processando tensores redundantes e poderá recomendar para o usuário uma música idêntica à que acabou de tocar, apenas com nome de arquivo diferente.

---

### 2.4. Desacoplamento entre Base de Ground Truth e Base do ChromaDB
- **O que o código faz:**
  - Em `evaluation.py`, o gabarito de avaliação é montado a partir de `metadata.json`.
  - A busca de vizinhos para a Precision@K, porém, consulta a coleção global do ChromaDB via `collection.query()`.
  - A baseline aleatória sorteia somente do `metadata.json`, enquanto o recomendador busca do ChromaDB.
- **O que deveria fazer:**
  Garantir sincronismo bidirecional estrito: ou avaliar apenas as faixas presentes na interseção comprovada entre o banco vetorial e o arquivo de metadados, ou reconstruir a coleção vetorial a partir do próprio conjunto avaliado.
- **Impacto:** **Afeta exclusivamente a avaliação experimental (Métricas P@K)**. Gera potenciais distorções na acurácia registrada, computando correspondências nulas quando vizinhos recuperados no banco não possuem subgênero preenchido nos metadados da coleção.

---

### 2.5. Exceção `AssertionError` Não Capturada na Avaliação
- **O que o código faz:**
  Em `evaluation.py:40-41`:
  ```python
  if missing_subgenre:
      raise AssertionError(f"Erro de sanidade: {len(missing_subgenre)} faixas estão sem subgenero preenchido...")
  ```
  No bloco principal `__main__` (`evaluation.py:136-139`), o código captura unicamente `except ValueError as e:`.
- **O que deveria fazer:**
  Capturar explicitamente `AssertionError` ou tratar entradas sem subgênero descartando-as com aviso no log em vez de abortar com traceback não tratado.
- **Impacto:** **Afeta exclusivamente a avaliação**. Se uma única faixa for salva na raiz da biblioteca (sem pasta mãe de subgênero), o script de avaliação é abortado com erro fatal antes de calcular qualquer métrica.

---

### 2.6. Dependência Rígida da Unidade `G:\` do Google Drive
- **O que o código faz:**
  - `iniciar_sistema.bat:26` entra em laço infinito de espera:
    ```bat
    :WAIT_DRIVE
    if not exist "G:\Meu Drive\Musicas\" (
        echo Aguardando o Google Drive montar a unidade G:...
        timeout /t 5 /nobreak >nul
        goto WAIT_DRIVE
    )
    ```
  - `clean_orphans2.py:6` define `MUSIC_ROOT = r"G:\Meu Drive\Musicas"`.
- **O que deveria fazer:**
  Permitir configuração de caminho via arquivo `.env`, parâmetro de linha de comando ou chave no `config.py`, com timeout e mensagem de erro amigável.
- **Impacto:** **Afeta a execução do sistema**. Em qualquer computador onde a biblioteca esteja em `D:\`, `C:\Musicas` ou no Google Drive montado sob outra letra de unidade, o inicializador trava em loop infinito sem iniciar o MPD ou o recomendador.

---

### 2.7. Caminhos Fixos (*Hardcoded*) do Binário do MPD
- **O que o código faz:**
  O arquivo `iniciar_sistema.bat:34` invoca diretamente:
  ```bat
  start "Servidor MPD" cmd /c "C:\mpd\mpd.exe C:\mpd\mpd.conf"
  ```
- **O que deveria fazer:**
  Detectar o binário `mpd.exe` no PATH do sistema, em uma pasta local do projeto ou através de variável de ambiente `MPD_PATH`.
- **Impacto:** **Afeta a execução do sistema**. Se o MPD estiver instalado em `Program Files`, em outra partição ou no Linux, o servidor falha silenciosamente ao abrir a janela separada.

---

### 2.8. Ausência de Tratamento de Erro Fatal na Conexão Inicial do MPD
- **O que o código faz:**
  - Na função `connect_mpd` (`mpd_integration.py:60-71`), há um laço de 3 tentativas com backoff exponencial.
  - Ao final das 3 tentativas, a função executa `raise e`.
  - No bloco `__main__` de `mpd_integration.py:652`, a chamada `connect_mpd` está dentro de um bloco `try` que trata apenas `KeyboardInterrupt` e `except:` genérico, mas que em seguida tenta manipular o processo e desconectar.
- **O que deveria fazer:**
  Validar explicitamente a disponibilidade da porta 6600 antes de iniciar as threads de monitoramento e a interface gráfica `StatusWindow`, emitindo diagnóstico claro ao usuário caso o daemon do MPD tenha caído na inicialização por erro de saída de áudio.
- **Impacto:** **Afeta a recomendação audível**. O integrador encerra o processo sem notificar de forma legível o usuário na interface gráfica caso o serviço de áudio do MPD não tenha subido a tempo.

---

### 2.9. Custo Computacional do Modelo CLAP em CPU
- **O que o código faz:**
  - `embedding.py:11` define:
    ```python
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ```
  - Em máquinas comuns sem GPU dedicada Nvidia com CUDA configurado, a inferência roda na CPU.
  - O processamento de uma faixa (`process_track`) exige carregar o arquivo, calcular espectrogramas e passar múltiplos segmentos de 10s pelo transformador HTSAT.
- **O que deveria fazer:**
  Utilizar quantização int8 via ONNX Runtime, implementar modelo destilado mais leve ou permitir execução assíncrona desacoplada da reprodução.
- **Impacto:** **Afeta a recomendação audível**. Quando uma música não indexada toca no MPD, a CPU fica em 100% de uso por vários segundos (geralmente entre 15 e 45 segundos por faixa na CPU). Durante esse período de indexação sob demanda, o sistema atrasa a reposição da fila.

---

### 2.10. Detecção de Mudança de Faixa sem Uso de `songid`
- **O que o código faz:**
  O integrador baseia-se unicamente em:
  ```python
  # mpd_integration.py:469
  if current_file == last_processed_file:
      continue
  ```
- **O que deveria fazer:**
  Comparar a tupla `(status.get("songid"), current_file)` para diferenciar instâncias temporais de reprodução.
- **Impacto:** **Afeta a recomendação audível**. Se o usuário repetir a mesma música em loop (*single track repeat*) ou tiver a mesma música duas vezes consecutivas na playlist, a transição é ignorada pelo integrador e novas recomendações deixam de ser adicionadas à fila.
