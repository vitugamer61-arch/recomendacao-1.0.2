# Feedback Implícito: Skip, Reprodução Parcial e Completa

O sistema aprende com o comportamento de escuta, sem pedir avaliações. Cada troca de faixa é classificada e usada para (a) deslocar o vetor de consulta (Rocchio), (b) penalizar faixas e gêneros e (c) bloquear temporariamente gêneros muito pulados.

- **Código:** `audio_pipeline/feedback.py` (`FeedbackStorage`, `FeedbackDetector`, `FeedbackEngine`).
- **Parâmetros:** `audio_pipeline/feedback_config.py` (todos os pesos e limiares em um só lugar).
- **Desativar:** `python iniciar.py --sem-feedback` (baseline sem reforço). **Zerar histórico:** `--zerar-feedback`.

---

## 1. Classificação da Transição (`FeedbackDetector.evaluate_transition`)

Ao detectar a troca de faixa (ver [recomendacao_e_fila.md](recomendacao_e_fila.md), seção 1), o monitor avalia a faixa **anterior** com o tempo ouvido (`elapsed`) e a duração.

Não geram evento: gatilhos automáticos, navegação para uma faixa **anterior** na playlist (`nova posição < antiga`) e faixas sem duração conhecida. Nos demais casos, em ordem:

| Regra | Condição | Tipo | Peso |
| :-: | :-- | :-: | :-: |
| 1 | `elapsed/duração ≥ 0,80` **ou** faltam ≤ 2 s para o fim | `completa` | +1,0 |
| 2 | `elapsed < 30 s` | `skip` | −1,0 |
| 3 | demais casos (30 s até 80%) | `parcial` | `f = clamp(elapsed/duração, 0,1; 1,0)` |

Os eventos são gravados com caminho, gênero (subgênero), cluster, `elapsed` e `duration`.

---

## 2. Persistência (SQLite, `results/feedback.db`)

Modo WAL ativo, para permitir acesso concorrente entre recomendador e monitor.

| Tabela | Conteúdo |
| :-- | :-- |
| `feedback_events` | cada skip/parcial/completa, com peso, gênero e cluster |
| `decisoes` | auditoria de cada decisão do recomendador (semente, faixa escolhida, origem `recomendada`/`exploracao`/`manual`, gatilho), usada no acompanhamento em tempo real do monitor |
| `mpd_queue_state` | espelho da fila do MPD (posição, `songid`, origem, status `Tocada`/`Tocando`/`Na fila`) |
| `mpd_song_ownership` | registra quais `songid` foram inseridos pelo recomendador (origem `recomendada`/`exploracao`) e a decisão que os originou |

---

## 3. Adaptação do Vetor de Consulta (Rocchio)

$$\mathbf{q}' = \text{normalize}\big(\alpha\,\mathbf{q} + \beta\,\overline{\mathbf{p}} - \gamma\,\overline{\mathbf{s}}\big)$$

onde $\overline{\mathbf{p}}$ é a média ponderada das faixas positivas (completas e parciais) e $\overline{\mathbf{s}}$ a das puladas.

| Parâmetro | Valor | Significado |
| :-- | :-: | :-- |
| `ROCCHIO_ALPHA` | 1,0 | peso da consulta original |
| `ROCCHIO_BETA` | 0,5 | peso das positivas |
| `ROCCHIO_GAMMA` | 0,7 | peso das puladas |
| `ROCCHIO_MAX_ITEMS` | 10 | eventos recentes considerados |
| `ROCCHIO_DECAY_RATE` | 0,15 | peso do evento $i$: $e^{-0{,}15\,i}$ ($i=0$ é o mais recente) |

---

## 4. Re-ranking e Rejeição

Com `q'`, buscam-se `CANDIDATES_N = 50` vizinhas no ChromaDB e aplica-se:

$$\text{score}(c) = \text{sim}(\mathbf{q}, \mathbf{c}) - \lambda \sum_i w_i\,\text{sim}(\mathbf{c}, \mathbf{skip}_i) - \mu\,\text{pen}_{\text{gênero}}(c)$$

com $\lambda = 0{,}5$ (`LAMBDA_PENALTY_TRACK`) e $\mu = 0{,}15$ (`MU_PENALTY_GENRE`).

São **descartadas** (não apenas penalizadas):

- as últimas `EXCLUDE_RECENT_SKIPS_COUNT = 10` faixas puladas;
- faixas de um gênero em **cooldown** (seção 5).

### 4.1. Penalidade de gênero

Pontuação cumulativa por gênero: **+1,0** por skip, **−0,5** por faixa completa, **−0,5·f** por parcial, com mínimo 0. A penalidade decai continuamente com meia-vida de 1.800 s (30 min): $P(t) = P_0 \cdot 2^{-\Delta t/1800}$.

---

## 5. Cooldown de Gênero

Se o usuário pula **3 ou mais** faixas do mesmo gênero dentro de uma janela de **600 s**, o gênero é bloqueado por **300 s**.

Efeito colateral importante: com skips seguidos, a vizinhança inteira de uma faixa pode pertencer a um gênero em cooldown, e todas as 50 candidatas são rejeitadas. Para a música não parar, `select_next_recommendations` amplia a busca em níveis (ver [recomendacao_e_fila.md](recomendacao_e_fila.md), seção 3.4): os níveis 1 e 2 procuram faixas de outros gêneros respeitando o cooldown, e só o nível 3 (emergência) o ignora.

---

## 6. Testes

- `audio_pipeline/test_feedback.py`: detecção (skip real, fim natural, navegação para trás, gatilho automático), Rocchio, re-ranking, cooldown de gênero, sincronização do estado da fila, e regressões de fila/propriedade das faixas, skips rápidos e cenários A a D (skip durante o cálculo, fila esgotada com retomada automática, parada manual respeitada, rajada de 5 skips).
- `audio_pipeline/test_skip_sequencial.py`: fim da fila **sem o campo `song`** (comportamento real do MPD), parada manual, MPD ocioso, dois fins de fila seguidos e expansão da busca.
