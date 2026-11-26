# Inicialização do Sistema (`iniciar.bat` / `iniciar.py`)

O ponto de entrada executa seis etapas em sequência. Esta página descreve o que cada uma faz, o que aparece no console e onde o tempo pode ser gasto.

## 1. Etapas

| Etapa | O que faz | Observação |
| :-: | :-- | :-- |
| — | Carrega os módulos (ChromaDB, scikit-learn, pandas, matplotlib) | O console mostra "Carregando módulos..." antes de começar |
| 1 | Verifica o MPD na porta 6600 e o inicia se necessário | |
| — | Conecta ao ChromaDB | |
| 2 | Varre a pasta de músicas e compara com o ChromaDB; indexa faixas novas | Varre a biblioteca inteira a cada partida |
| 3 | Clusterização | Completa só se o modelo/`clusters.json` não existir ou com `--recalcular`; caso contrário reaproveita |
| 4 | Sincroniza playlists por cluster no MPD e `.m3u` | Executa a cada partida (exceto com `--sem-cluster`) |
| 5 | Inicia o Monitor de Precisão (Streamlit) | Espera até 5 s para confirmar que subiu |
| 6 | Inicia o recomendador | |

## 2. O que aparece no console

Cada etapa registra sua duração (os números abaixo são ilustrativos):

```
Carregando módulos (ChromaDB, scikit-learn, pandas...). Pode levar alguns segundos...
Módulos carregados em 3.1 s.
[1/6] MPD concluída em 0.2 s.
...
Preparação concluída em 12.4 s. Iniciando o recomendador...
```

Esses tempos servem para localizar o que demora na sua máquina: se a partida parecer lenta, envie o trecho do log com as linhas `concluída em`.

## 3. O que mudou para reduzir a espera percebida

- **Banner antes dos imports pesados.** Antes, `iniciar.py` importava tudo no topo, e o console ficava **sem nenhuma saída** durante todo o carregamento.
- **`torch` e `transformers` não são mais carregados na partida.** O módulo `pipeline` (que importa os dois, para o CLAP) agora é importado só quando uma faixa realmente precisa ser indexada (`auto_indexer.process_track` e `index_library.process_track`). Com a biblioteca já indexada, ele nunca é carregado. O modelo CLAP em si já era carregado sob demanda (`embedding._load_model`).
- **Recomendação imediata se o MPD já estiver tocando.** Ver [recomendacao_e_fila.md](recomendacao_e_fila.md), seção 1.1.
- **Decisão mais rápida:** ver [recomendacao_e_fila.md](recomendacao_e_fila.md), seção 7.

## 4. O que não foi medido

O tempo real de cada etapa no Windows não foi medido. Os candidatos a gastar tempo, por leitura do código, são:

- **Etapa 2:** `varrer_biblioteca` percorre toda a pasta de músicas; em uma unidade de nuvem (Google Drive) a listagem pode ser lenta.
- **Etapa 3:** se `clusters.json` ou `models/cluster_model.joblib` não existirem, roda a clusterização completa (K-Means com `k` de 4 a 30 pela silhueta, refinamento e gráfico 2D).
- **Etapa 4:** reenvia as playlists de todos os clusters ao MPD a cada partida.
- **Etapa 5:** até 5 s de espera pelo Streamlit.

Com os tempos por etapa no log, dá para decidir o que otimizar com base em medição.
