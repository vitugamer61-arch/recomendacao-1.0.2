# Relatório de Disponibilidade — Sistema de Recomendação de Músicas

> **Gerado em:** 2026-10-25 22:28:11  
> **Componente Supervisionado:** Módulo Python com ChromaDB Embutido (`servico_consulta.py`)  
> **Mecanismo de Recuperação:** Watchdog Automático com Health Check via HTTP (`watchdog.py`)

---

## 1. Configuração do Experimento

| Parâmetro | Símbolo / Campo | Valor Utilizado | Descrição |
|---|---|---|---|
| Intervalo de monitoramento | N | `1.0 s` | Frequência de sondagens do cliente de monitoramento |
| Falhas injetadas | X | `2` | Quantidade de kills forçados no processo do serviço |
| Consultas no teste de carga | Y | `100` | Total de requisições de carga sintética |
| Duração configurada | D | `2.0 min` | Janela de observação do monitoramento |
| Concorrência de carga | T | `1 thread(s)` | Nível de concorrência simultânea na carga |
| Semente aleatória | S | `42` | Semente pseudoaleatória para reprodutibilidade |
| Porta HTTP | P | `7700` | Porta TCP local de atendimento |
| Intervalo do Watchdog | W | `1.0 s` | Frequência do health check do supervisor |

---

## 2. Comparativo de Resultados: Linha de Base vs. Estresse

| Métrica | Meta (SLO) | Estresse (Falhas Injetadas) | Status |
|---|---|---|---|
| Disponibilidade (consultas) | ≥ 99.9% | 97.00% | ❌ REPROVADO |
| Disponibilidade (fórmula temporal MTBF/MTTR) | ≥ 99.9% | 90.50% | ❌ REPROVADO |
| MTTR médio (módulo Python + ChromaDB) | ≤ 10.0 s | 5.488 s | ✅ APROVADO |
| MTTR mín / máx / desvio padrão | — | 5.373 s / 5.603 s / ±0.163 s | — |
| Taxa de sucesso — monitoramento | ≥ 99.9% | 97.00% | ❌ REPROVADO |
| Taxa de sucesso — teste de carga | ≥ 99.9% | 100.00% | ✅ APROVADO |
| Tempo de restauração do índice | ≤ 60.0 s | 8.191 s | ✅ APROVADO |

---

## 3. Reconciliação e Metodologia das Duas Disponibilidades

Foram calculadas duas métricas de disponibilidade no período de teste:

1. **Disponibilidade por Consultas (Amostragem Discreta):**
   $$A_{\text{consultas}} = \frac{N_{\text{sucessos}}}{N_{\text{total}}} \times 100\%$$
   - Consultas com sucesso: **97** de **100** total.
   - Resultado: **97.00%**.

2. **Disponibilidade Temporal Efetiva (Fórmula Clássica MTBF / MTTR):**
   $$A_{\text{temporal}} = \frac{\text{Uptime}}{\text{Uptime} + \text{Downtime}} \times 100\% = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}} \times 100\%$$
   - **Tempo Total Observado:** 1 min 55.5 s (115.516 s)
   - **Downtime Acumulado (soma dos MTTRs reais):** 11.0 s (10.976 s)
   - **Uptime Efetivo de Funcionamento:** 1 min 44.5 s (104.54 s)
   - **MTBF Operacional Médio (Uptime / n):** 52.3 s (52.27 s)
   - **Tempo Médio Estrito entre Recuperação e Próxima Falha:** 5.8 s
   - Resultado: **90.50%**.

### Por que as duas medidas divergem durante o teste de estresse?
- **Granularidade e Latência de Timeout:** Quando o serviço está em execução normal, cada consulta leva ~15 ms de processamento + 1.0 s de pausa (`--intervalo`), completando 1 consulta a cada ~1.015 s.
- **Comportamento em Falha:** Durante a indisponibilidade provocada, o cliente HTTP tenta conectar e sofre bloqueio por recusa de conexão/timeout de socket pelo sistema operacional (~2.05 s no Windows) somado ao intervalo configurado (`1.0 s`). Assim, cada tentativa com falha consome mais de 3.0 s.
- **Efeito de Subamostragem:** Nos 10.976 s de indisponibilidade total, o monitor executou apenas 3 tentativas com falha. Enquanto no tempo ativo (~104.54 s), o monitor realizou 97 requisições. Como a taxa de sondagens diminui em até 3 vezes durante a queda, a disponibilidade baseada em consultas aparenta ser mais elevada (97.00%) do que a disponibilidade temporal real contínua (90.50%).
- **Na Linha de Base (Sem Falhas):** Como não há downtime, não ocorre distorção de amostragem e ambas as disponibilidades concordam perfeitamente em **100.00%**.

---

## 4. Metadados de Confiabilidade e Validação Técnica

### 4.1. Amostragem de Falhas e Confiabilidade Estatística
- **Número de Falhas Injetadas ($n$):** 2
- **MTTR Médio:** 5.488 s (Mínimo: 5.373 s, Máximo: 5.603 s)
- **Desvio Padrão do MTTR:** ±0.163 s
- **Valores Individuais Medidos (s):** `[5.603, 5.373]`

> ⚠️ **Aviso de Confiabilidade Estatística:** Foram injetadas apenas $n=2$ falhas (menos que as 10 recomendadas para significância estatística). O desvio padrão de ±0.163 s possui margem amostral restrita nesta execução de validação rápida.

### 4.2. Integridade e Dimensão da Biblioteca (Teste de Restauração)
- **Faixas de Metadados no Catálogo:** 1.288 faixas cadastradas
- **Embeddings no Índice Vetorial:** 1.288 vetores (512 dimensões)
- **Faixa de Referência do Teste:** `0015`
- **Fidelidade do Top-K Pós-Restauração:** Identidade estrita = **True** (Sobreposição: 5/5, Precisão: 100%)
- **Tempo de Reconstrução do Índice:** 4.0007 s
- **Latência da 1ª Consulta Pós-Restauração:** 4.1904 s
- **Tempo Total de Restauração (RTO):** **8.1911 s** (Meta SLO: ≤ 60.0 s)

### 4.3. Teste de Carga e Concorrência
- **Consultas Realizadas:** 100
- **Concorrência:** 1 thread(s) simultânea(s)
- **Falhas Durante a Carga:** 0 falha(s) (Nenhuma interrupção durante o estresse de requisições)
- **Taxa de Sucesso na Carga:** **100.00%** (Meta SLO: ≥ 99.9%)
- **Throughput Médio Atingido:** 133.42 req/s
- **Latência Média / P95 / P99:** 7.452 ms / 20.98 ms / 29.54 ms

---

## 5. Projeção de Disponibilidade em Longo Prazo

> ⚠️ **Rotulagem Obrigatória — Projeção por Fórmula Matemática, NÃO Medição Direta:**  
> As estimativas abaixo utilizam o **MTTR médio real medido de 5.488 s** e projetam a disponibilidade contínua teórica $A = MTBF / (MTBF + MTTR)$ para diferentes regimes operacionais de estabilidade.

| Cenário Operacional | MTBF Teórico | Disponibilidade Projetada | Parada Anual Estimada |
|---|---|---|---|
| Falha a cada 1 hora | 1 h | **99.8478%** | ~13 h 20 min / ano |
| Falha a cada 1 dia | 1 dias | **99.9936%** | ~33 min 24.4 s / ano |
| Falha a cada 1 semana | 7 dias | **99.9991%** | ~4 min 46.4 s / ano |
| Falha a cada 30 dias | 30 dias | **99.9998%** | ~1 min 6.8 s / ano |

### Requisito Mínimo de MTBF para o SLO de 99,9%
Com base na relação $A \ge 99{,}9\% \iff \text{MTBF} \ge 999 \times \text{MTTR}$:
- **MTBF mínimo necessário:** 999 x 5.488 s = 5482.5 s (**~1 h 31 min**).
- **Interpretação:** Para que o sistema atinja o SLO de 99,9% em produção mantendo o MTTR de 5.488 s, o intervalo médio entre falhas não induzidas deve ser superior a **1 h 31 min**.

---

## 6. Conclusões Automáticas por SLO (Avaliação Honesta)

1. **Disponibilidade no Período (SLO ≥ 99.9%):** **NÃO ATINGIDA na execução de estresse** (resultado: 97.0%). **Causa identificada:** Foram injetadas propositalmente 2 falhas em uma janela curta de 1 min 55.5 s, produzindo 10.976 s de indisponibilidade forçada. Em contrapartida, no teste de **linha de base sem falhas injetadas**, a disponibilidade foi de **100.00%**, demonstrando que o sistema é estável em condições normais.
2. **MTTR Médio (SLO ≤ 10.0 s):** **ATINGIDA**. O supervisor watchdog identificou a perda do processo e restabeleceu o serviço completamente em média de **5.488 s**, cumprindo com folga a meta máxima de 10.0 s.
3. **Taxa de Sucesso na Carga (SLO ≥ 99.9%):** **ATINGIDA**. Sob bombardeio concorrente de consultas, o serviço atendeu **100.00%** das requisições sem erros.
4. **Tempo de Restauração do Índice (SLO ≤ 60.0 s):** **ATINGIDA**. O índice com 1.288 faixas e embeddings foi totalmente reconstruído e a primeira consulta atendida em **8.191 s**, garantindo integridade idêntica do Top-K de recomendações.

---
