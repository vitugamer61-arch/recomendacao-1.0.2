# Relatório de Disponibilidade — Sistema de Recomendação de Músicas

> **Gerado em:** 2026-10-25 22:48:19  
> **Componente Supervisionado:** Módulo Python com ChromaDB Embutido (`servico_consulta.py`)  
> **Mecanismo de Recuperação:** Watchdog Automático com Health Check via HTTP (`watchdog.py`)

---

## 1. Configuração do Experimento

| Parâmetro | Símbolo / Campo | Valor Utilizado | Descrição |
|---|---|---|---|
| Intervalo de monitoramento | N | `1.0 s` | Cadência fixa estrita das sondagens |
| Falhas injetadas no estresse | X | `2` | Kills forçados com confirmação explícita de término |
| Consultas no teste de carga | Y | `100` | Total de requisições de carga sintética |
| Duração configurada (Estresse) | D_estresse | `2.0 min` | Janela de observação do estresse |
| Duração da Linha de Base | D_baseline | `2.0 min` | Janela de observação sem injeção de falhas |
| Concorrência de carga | T | `1 thread(s) (1 thread (sequencial))` | Nível de concorrência simultânea na carga |
| Semente aleatória | S | `42` | Semente pseudoaleatória para reprodutibilidade |
| Porta HTTP | P | `7700` | Endereço estrito 127.0.0.1 (evita overhead DNS) |
| Intervalo do Watchdog | W | `1.0 s` | Frequência do health check do supervisor |

---

## 2. Comparativo de Resultados: Linha de Base vs. Estresse

| Métrica | Meta (SLO) | Linha de Base (Sem Falhas) | Estresse (Falhas Injetadas) | Status Estresse |
|---|---|---|---|---|
| Disponibilidade ponderada pelo tempo | ≥ 99.9% | 100.00% | 95.80% | ❌ REPROVADO |
| Disponibilidade analítica (MTBF/(MTBF+MTTR)) | ≥ 99.9% | 100.00% | 94.99% | ❌ REPROVADO |
| Disponibilidade por consultas (amostragem simples) | ≥ 99.9% | 100.00% | 99.10% | ❌ REPROVADO |
| MTTR médio (falhas.csv — confirmação de morte) | ≤ 10.0 s | N/A (0 falhas) | 3.001 s | ✅ APROVADO |
| MTTR médio (verificação cruzada monitor.csv) | ≤ 10.0 s | N/A (0 falhas) | 5.073 s | ✅ APROVADO |
| MTTR mín / máx / desvio padrão (primário) | — | — | 0.497 s / 5.505 s / ±3.541 s | — |
| Taxa de sucesso — monitoramento | ≥ 99.9% | 100.00% | 99.10% | ❌ REPROVADO |
| Taxa de sucesso — teste de carga | ≥ 99.9% | — | 100.00% | ✅ APROVADO |
| Tempo de restauração do índice | ≤ 60.0 s | — | 6.509 s | ✅ APROVADO |

> ℹ️ **Linha de Base — Limite Inferior de Confiança (Regra dos Três):**  
> Como foram observadas zero falhas em $n=116$ amostras consecutivas na linha de base, o limite inferior com **95% de confiança estatística** para a disponibilidade é:  
> $$\text{Disponibilidade}_{95\%} \ge 1 - \frac{3}{n} = 1 - \frac{3}{116} = \mathbf{97.41\%}$$

---

## 3. Reconciliação Matemática das Disponibilidades

Foram calculadas três métricas de disponibilidade no teste de estresse:

1. **Disponibilidade Ponderada pelo Tempo ($A_{\text{ponderada}}$):**
   $$A_{\text{ponderada}} = \frac{\sum_{i} (w_i \cdot \mathbb{I}_{\text{sucesso}})}{\sum_{i} w_i} \times 100\%$$
   Cada sondagem tem peso igual ao intervalo de tempo até a sondagem seguinte ($w_i = t_{i+1} - t_i$).
   - Resultado: **95.80%**.

2. **Disponibilidade Analítica Temporal ($A_{\text{temporal}}$ via MTBF/MTTR):**
   $$A_{\text{temporal}} = \frac{\text{Uptime}}{\text{Uptime} + \text{Downtime}} \times 100\% = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}} \times 100\%$$
   - **Tempo Total Observado:** 1 min 59.7 s (119.691 s)
   - **Downtime Acumulado (soma dos MTTRs primários):** 6.0 s (6.002 s)
   - **Uptime Efetivo de Funcionamento:** 1 min 53.7 s (113.689 s)
   - **MTBF Operacional Médio (Uptime / n):** 56.8 s (56.844 s)
   - Resultado: **94.99%**.

3. **Diferença entre Disponibilidade Ponderada e Temporal:** **0.81%** (concordância dentro da margem estrita da cadência de amostragem).

4. **Disponibilidade por Consultas Simples (Não Ponderada):**
   - $N_{\text{sucessos}} / N_{\text{total}} = 110 / 111$ (**99.10%**).

---

## 4. Análise Detalhada de Cada Falha Injetada e Verificação Cruzada

| Falha # | PID Alvo | Horário do Kill (Morte Confirmada) | 1º Sucesso Válido | MTTR Primário (falhas.csv) | MTTR Cruzado (monitor.csv) | Queda Observada pelo Monitor? |
|---|---|---|---|---|---|---|
| 1 | `28668` | `01:31:59.101` | `01:31:59.598` | **0.497 s** | 5.073 s | Sim |
| 2 | `35404` | `01:32:10.543` | `01:32:16.049` | **5.505 s** | 5.073 s | Sim |

### 4.1. Confiabilidade Amostral do MTTR
- **Número de Falhas Injetadas ($n$):** 2
- **MTTR Médio Primário:** **3.001 s** (Mín: 0.497 s, Máx: 5.505 s)
- **MTTR Médio Cruzado (Monitor):** **5.073 s**
- **Desvio Padrão:** ±3.541 s

> ⚠️ **Aviso de Confiabilidade Estatística:** Foram injetadas apenas $n=2$ falhas (menos que as 10 recomendadas para significância estatística). O desvio padrão amostral de ±3.541 s possui margem restrita nesta execução de validação.

### 4.2. Integridade e Dimensão da Biblioteca (Teste de Restauração)
- **Faixas no Catálogo de Metadados:** 1288 faixas
- **Embeddings no Índice Vetorial:** 1288 vetores
- **Faixa de Referência:** `0015`
- **Fidelidade do Top-K:** Identidade estrita = **True** (Sobreposição: 5/5, Precisão: 100%)
- **Tempo de Reconstrução do Índice:** 3.1652 s
- **Latência da 1ª Consulta:** 3.3434 s
- **Tempo Total de Restauração (RTO):** **6.5087 s** (Meta SLO: ≤ 60.0 s)

### 4.3. Teste de Carga
- **Consultas Realizadas:** 100
- **Execução:** 1 thread(s) (sequencial)
- **Falhas Durante a Carga:** 0 falha(s) (Nenhuma interrupção durante a rajada de carga)
- **Taxa de Sucesso na Carga:** **100.00%** (Meta SLO: ≥ 99.9%)
- **Throughput Médio:** 49.0 req/s
- **Latência Média / P95 / P99:** 20.353 ms / 33.895 ms / 35.145 ms

---

## 5. Projeção de Disponibilidade em Longo Prazo

> ⚠️ **Rotulagem Obrigatória — Projeção por Fórmula Matemática, NÃO Medição Direta:**  
> As estimativas utilizam o MTTR médio real medido de **3.001 s** e projetam a disponibilidade contínua teórica $A = MTBF / (MTBF + MTTR)$ para diferentes cenários.

| Cenário Operacional | MTBF Teórico | Disponibilidade Projetada | Parada Anual Estimada |
|---|---|---|---|
| Falha a cada 1 hora | 1 h | **99.9167%** | ~7 h 18 min / ano |
| Falha a cada 1 dia | 1 dias | **99.9965%** | ~18 min 16.1 s / ano |
| Falha a cada 1 semana | 7 dias | **99.9995%** | ~2 min 36.6 s / ano |
| Falha a cada 30 dias | 30 dias | **99.9999%** | ~36.5 s / ano |

### Requisito Mínimo de MTBF para o SLO de 99,9%
- **MTBF mínimo necessário ($999 \times \text{MTTR}$):** 999 x 3.001 s = 2998.0 s (**~49 min 58.0 s**).
- **Interpretação:** Para que o sistema atinja o SLO de 99,9% em regime contínuo mantendo o MTTR de 3.001 s, o intervalo médio entre falhas não induzidas deve ser superior a **49 min 58.0 s**.

---

## 6. Conclusões Automáticas por SLO (Avaliação Honesta)

1. **Disponibilidade no Período (SLO ≥ 99.9%):** **NÃO ATINGIDA na execução de estresse** (resultado: 95.80%). **Causa identificada:** Foram injetadas forçadamente 2 falhas na janela de 1 min 59.7 s, produzindo 6.002 s de indisponibilidade induzida. Na **linha de base sem injeção de falhas**, a disponibilidade foi de **100.00%** no período observado.
2. **MTTR Médio (SLO ≤ 10.0 s):** **ATINGIDA**. O supervisor watchdog restabeleceu o serviço completamente em média de **3.001 s** (verificação cruzada no monitor: 5.073 s), cumprindo a meta máxima de 10.0 s.
3. **Taxa de Sucesso na Carga (SLO ≥ 99.9%):** **ATINGIDA**. Durante a execução com 1 thread(s), o serviço atendeu **100.00%** das consultas sem falhas.
4. **Tempo de Restauração do Índice (SLO ≤ 60.0 s):** **ATINGIDA**. A reconstrução completa do índice com 1288 faixas e embeddings e o atendimento da primeira consulta ocorreram em **6.509 s**, com 100% de precisão de ranking (Top-K idêntico ao original).

---
