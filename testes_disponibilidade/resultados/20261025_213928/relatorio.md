# Relatório de Disponibilidade — Sistema de Recomendação de Músicas

> Gerado em: 2026-10-25 21:41:49


**MTTRs individuais (s):** [5.349, 5.643]


## Resultados por Métrica

| Métrica | Meta (SLO) | Resultado | Status |
|---|---|---|---|
| Disponibilidade no período (consultas) | ≥ 99.9% | 97.0% | ❌ REPROVADO |
| Disponibilidade (fórmula MTBF/MTTR) | ≥ 99.9% | 91.3651% | ❌ REPROVADO |
| MTTR médio (módulo Python + ChromaDB) | ≤ 10.0 s | 5.496 s | ✅ APROVADO |
| MTTR mín / máx / desvio | — | 5.349s / 5.643s / ±0.208s | — |
| Taxa de sucesso — monitoramento | ≥ 99.9% | 97.0% | ❌ REPROVADO |
| Taxa de sucesso — teste de carga | ≥ 99.9% | 100.0% | ✅ APROVADO |
| Tempo de restauração do índice | ≤ 60.0 s | N/A | N/A |

---

**Componente supervisionado:** Módulo Python + ChromaDB embutido (`servico_consulta.py`)

**Método de injeção de falha:** kill do processo Python (equivalente a falha do ChromaDB embutido)

**Recuperação:** automática via `watchdog.py`
