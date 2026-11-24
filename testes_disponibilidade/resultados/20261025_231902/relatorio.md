# Relatório de Disponibilidade — Sistema de Recomendação de Músicas

> **Gerado em:** 2026-10-26 00:49:44  
> **Componente Supervisionado:** Módulo Python com ChromaDB Embutido (`servico_consulta.py`)  
> **Mecanismo de Recuperação:** Watchdog Automático com Health Check via HTTP (`watchdog.py`)

---

## 1. Configuração do Experimento

| Parâmetro | Símbolo / Campo | Valor Utilizado | Descrição |
|---|---|---|---|
| Intervalo de monitoramento | N | `1.0 s` | Cadência fixa estrita das sondagens |
| Falhas injetadas no estresse | X | `10` | Kills forçados com confirmação explícita de término |
| Consultas no teste de carga | Y | `1000` | Total de requisições de carga sintética |
| Duração configurada (Estresse) | D_estresse | `30.0 min` | Janela de observação do estresse |
| Duração da Linha de Base | D_baseline | `60.0 min` | Janela de observação sem injeção de falhas |
| Concorrência de carga | T | `4 thread(s) (concorrente)` | Nível de concorrência simultânea na carga |
| Semente aleatória | S | `42` | Semente pseudoaleatória para reprodutibilidade |
| Porta HTTP | P | `7700` | Endereço estrito 127.0.0.1 (evita overhead DNS) |
| Intervalo do Watchdog | W | `1.0 s` | Frequência do health check do supervisor |

---

## 2. Comparativo de Resultados: Linha de Base vs. Estresse

| Métrica | Meta (SLO) | Linha de Base (Sem Falhas) | Estresse (Falhas Injetadas) | Status Estresse |
|---|---|---|---|---|
| Disponibilidade ponderada pelo tempo | ≥ 99.9% | 100.00% | 95.56% | ❌ REPROVADO |
| Disponibilidade analítica (MTBF/(MTBF+MTTR)) | ≥ 99.9% | 100.00% | 95.31% | ❌ REPROVADO |
| Disponibilidade por consultas (amostragem simples) | ≥ 99.9% | 100.00% | 95.56% | ❌ REPROVADO |
| MTTR médio (falhas.csv — confirmação de morte) | ≤ 10.0 s | N/A (0 falhas) | 8.440 s | ✅ APROVADO |
| MTTR médio (verificação cruzada monitor.csv) | ≤ 10.0 s | N/A (0 falhas) | 8.000 s | ✅ APROVADO |
| MTTR mín / máx / desvio padrão (primário) | — | — | 6.207 s / 9.172 s / ±0.856 s | — |
| Taxa de sucesso — monitoramento | ≥ 99.9% | 100.00% | 95.56% | ❌ REPROVADO |
| Taxa de sucesso — teste de carga | ≥ 99.9% | — | 100.00% | ✅ APROVADO |
| Tempo de restauração do índice | ≤ 60.0 s | — | 7.245 s | ✅ APROVADO |

> ℹ️ **Linha de Base — Limite Inferior de Confiança (Regra dos Três):**  
> Como foram observadas zero falhas em $n=3601$ amostras consecutivas na linha de base, o limite inferior com **95% de confiança estatística** para a disponibilidade é:  
> $$\text{Disponibilidade}_{95\%} \ge 1 - \frac{3}{n} = 1 - \frac{3}{3601} = \mathbf{99.92\%}$$

---

## 3. Reconciliação Matemática das Disponibilidades

Foram calculadas três métricas de disponibilidade no teste de estresse:

1. **Disponibilidade Ponderada pelo Tempo ($A_{\text{ponderada}}$):**
   $$A_{\text{ponderada}} = \frac{\sum_{i} (w_i \cdot \mathbb{I}_{\text{sucesso}})}{\sum_{i} w_i} \times 100\%$$
   Cada sondagem tem peso igual ao intervalo de tempo até a sondagem seguinte ($w_i = t_{i+1} - t_i$).
   - Resultado: **95.56%**.

2. **Disponibilidade Analítica Temporal ($A_{\text{temporal}}$ via MTBF/MTTR):**
   $$A_{\text{temporal}} = \frac{\text{Uptime}}{\text{Uptime} + \text{Downtime}} \times 100\% = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}} \times 100\%$$
   - **Tempo Total Observado:** 30 min (1800.018 s)
   - **Downtime Acumulado (soma dos MTTRs primários):** 1 min 24.4 s (84.399 s)
   - **Uptime Efetivo de Funcionamento:** 28 min 35.6 s (1715.619 s)
   - **MTBF Operacional Médio (Uptime / n):** 2 min 51.6 s (171.562 s)
   - Resultado: **95.31%**.

3. **Diferença entre Disponibilidade Ponderada e Temporal:** **0.25%** (concordância dentro da margem estrita da cadência de amostragem).

4. **Disponibilidade por Consultas Simples (Não Ponderada):**
   - $N_{\text{sucessos}} / N_{\text{total}} = 1721 / 1801$ (**95.56%**).

---

## 4. Análise Detalhada de Cada Falha Injetada e Verificação Cruzada

| Falha # | PID Alvo | Horário do Kill (Morte Confirmada) | 1º Sucesso Válido | MTTR Primário (falhas.csv) | MTTR Cruzado (monitor.csv) | Queda Observada pelo Monitor? |
|---|---|---|---|---|---|---|
| 1 | `31028` | `03:19:48.606` | `03:19:54.813` | **6.207 s** | 6.000 s | Sim |
| 2 | `30200` | `03:20:10.065` | `03:20:18.812` | **8.748 s** | 8.000 s | Sim |
| 3 | `40132` | `03:20:25.396` | `03:20:33.812` | **8.416 s** | 8.000 s | Sim |
| 4 | `32388` | `03:20:43.204` | `03:20:51.812` | **8.608 s** | 8.000 s | Sim |
| 5 | `39020` | `03:21:00.491` | `03:21:08.812` | **8.322 s** | 8.000 s | Sim |
| 6 | `40644` | `03:21:22.931` | `03:21:31.812` | **8.881 s** | 8.000 s | Sim |
| 7 | `15216` | `03:21:44.749` | `03:21:53.812` | **9.063 s** | 9.000 s | Sim |
| 8 | `39920` | `03:22:08.735` | `03:22:16.812` | **8.078 s** | 8.000 s | Sim |
| 9 | `41084` | `03:22:24.640` | `03:22:33.812` | **9.172 s** | 9.000 s | Sim |
| 10 | `39320` | `03:22:43.908` | `03:22:52.812` | **8.904 s** | 8.000 s | Sim |

### 4.1. Confiabilidade Amostral do MTTR
- **Número de Falhas Injetadas ($n$):** 10
- **MTTR Médio Primário:** **8.440 s** (Mín: 6.207 s, Máx: 9.172 s)
- **MTTR Médio Cruzado (Monitor):** **8.000 s**
- **Desvio Padrão:** ±0.856 s

### 4.2. Integridade e Dimensão da Biblioteca (Teste de Restauração)
- **Faixas no Catálogo de Metadados:** 1288 faixas
- **Embeddings no Índice Vetorial:** 1288 vetores
- **Faixa de Referência:** `0015`
- **Fidelidade do Top-K:** Identidade estrita = **True** (Sobreposição: 5/5, Precisão: 100%)
- **Tempo de Reconstrução do Índice:** 3.6038 s
- **Latência da 1ª Consulta:** 3.641 s
- **Tempo Total de Restauração (RTO):** **7.2448 s** (Meta SLO: ≤ 60.0 s)

### 4.3. Teste de Carga
- **Consultas Realizadas:** 1000
- **Execução:** 4 thread(s) (concorrente)
- **Falhas Durante a Carga:** 0 falha(s) (Nenhuma interrupção durante a rajada de carga)
- **Taxa de Sucesso na Carga:** **100.00%** (Meta SLO: ≥ 99.9%)
- **Throughput Médio:** 163.48 req/s
- **Latência Média / P95 / P99:** 24.358 ms / 36.396 ms / 40.057 ms

---

## 5. Projeção de Disponibilidade em Longo Prazo

> ⚠️ **Rotulagem Obrigatória — Projeção por Fórmula Matemática, NÃO Medição Direta:**  
> As estimativas utilizam o MTTR médio real medido de **8.440 s** e projetam a disponibilidade contínua teórica $A = MTBF / (MTBF + MTTR)$ para diferentes cenários.

| Cenário Operacional | MTBF Teórico | Disponibilidade Projetada | Parada Anual Estimada |
|---|---|---|---|
| Falha a cada 1 hora | 1 h | **99.7661%** | ~20 h 30 min / ano |
| Falha a cada 1 dia | 1 dias | **99.9902%** | ~51 min 22.4 s / ano |
| Falha a cada 1 semana | 7 dias | **99.9986%** | ~7 min 20.4 s / ano |
| Falha a cada 30 dias | 30 dias | **99.9997%** | ~1 min 42.8 s / ano |

### Requisito Mínimo de MTBF para o SLO de 99,9%
- **MTBF mínimo necessário ($999 \times \text{MTTR}$):** 999 x 8.440 s = 8431.6 s (**~2 h 20 min**).
- **Interpretação:** Para que o sistema atinja o SLO de 99,9% em regime contínuo mantendo o MTTR de 8.440 s, o intervalo médio entre falhas não induzidas deve ser superior a **2 h 20 min**.

---

## 6. Conclusões Automáticas por SLO (Avaliação Honesta)

1. **Disponibilidade no Período (SLO ≥ 99.9%):** **NÃO ATINGIDA na execução de estresse** (resultado: 95.56%). **Causa identificada:** Foram injetadas forçadamente 10 falhas na janela de 30 min, produzindo 84.399 s de indisponibilidade induzida. Na **linha de base sem injeção de falhas**, a disponibilidade foi de **100.00%** no período observado.
2. **MTTR Médio (SLO ≤ 10.0 s):** **ATINGIDA**. O supervisor watchdog restabeleceu o serviço completamente em média de **8.440 s** (verificação cruzada no monitor: 8.000 s), cumprindo a meta máxima de 10.0 s.
3. **Taxa de Sucesso na Carga (SLO ≥ 99.9%):** **ATINGIDA**. Durante a execução com 4 thread(s), o serviço atendeu **100.00%** das consultas sem falhas.
4. **Tempo de Restauração do Índice (SLO ≤ 60.0 s):** **ATINGIDA**. A reconstrução completa do índice com 1288 faixas e embeddings e o atendimento da primeira consulta ocorreram em **7.245 s**, com 100% de precisão de ranking (Top-K idêntico ao original).

---
