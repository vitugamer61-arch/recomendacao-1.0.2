# Testes de Disponibilidade e Recuperação de Falhas

Resultados medidos pela suíte `testes_disponibilidade/` (execução consolidada `resultados/20261025_231902/`, relatório gerado em 2026-10-26). Os valores abaixo vêm do `relatorio.md` e do `resumo.csv` dessa execução.

## 1. O que foi medido (e o que não foi)

O componente supervisionado é o **serviço de consulta ao ChromaDB** (`servico_consulta.py`, HTTP em `127.0.0.1:7700`), recuperado por um **watchdog** com health check. A suíte **não** testa a integração com o MPD nem a reposição da fila. Esse comportamento é coberto separadamente em [recomendacao_e_fila.md](recomendacao_e_fila.md), seção 6.

A suíte nunca altera a biblioteca de músicas nem o índice original (`audio_pipeline/chroma_db/`).

## 2. Configuração

| Parâmetro | Valor |
| :-- | :-: |
| Intervalo de monitoramento (N) | 1,0 s |
| Intervalo do watchdog (W) | 1,0 s |
| Linha de base (sem falhas) | 60 min |
| Estresse | 30 min, 10 falhas injetadas (kills forçados) |
| Teste de carga | 1.000 consultas, 4 threads |
| Semente aleatória | 42 |

## 3. Resultados

| Métrica | SLO | Linha de base | Estresse | Status |
| :-- | :-: | :-: | :-: | :-: |
| Disponibilidade ponderada pelo tempo | ≥ 99,9% | 100,00% | 95,56% | reprovado |
| Disponibilidade analítica MTBF/(MTBF+MTTR) | ≥ 99,9% | 100,00% | 95,31% | reprovado |
| Disponibilidade por consultas | ≥ 99,9% | 100,00% | 95,56% (1.721/1.801) | reprovado |
| MTTR médio (confirmação de morte) | ≤ 10,0 s | n/a | 8,440 s | aprovado |
| MTTR médio (verificação cruzada no monitor) | ≤ 10,0 s | n/a | 8,000 s | aprovado |
| MTTR mín. / máx. / desvio padrão | — | — | 6,207 s / 9,172 s / ±0,856 s | — |
| Taxa de sucesso na carga | ≥ 99,9% | — | 100,00% | aprovado |
| Tempo de restauração do índice | ≤ 60,0 s | — | 7,245 s | aprovado |

Leitura correta dos números:

- **A disponibilidade do estresse (95,56%) reprovou o SLO** porque as 10 quedas foram **forçadas** de propósito, cerca de uma a cada 3 minutos. Ela mede o custo de recuperação sob falha frequente, não a disponibilidade em operação normal.
- **Linha de base:** zero falhas em n = 3.601 amostras. Pela regra dos três, o limite inferior com 95% de confiança é 1 − 3/3601 = **99,92%**.
- **Restauração:** com 1.288 faixas no catálogo e 1.288 vetores no índice, a reconstrução levou 3,604 s e o tempo total (RTO) foi 7,245 s. O Top-K da faixa de referência (`0015`) foi idêntico ao original (5/5).
- **Carga:** 0 falhas, 163,48 req/s; latência média 24,358 ms, P95 36,396 ms, P99 40,057 ms.

## 4. Projeção (fórmula, não medição)

O próprio relatório rotula esta tabela como projeção por $A = \text{MTBF}/(\text{MTBF}+\text{MTTR})$ com o MTTR medido de 8,440 s:

| Falha a cada | Disponibilidade projetada |
| :-: | :-: |
| 1 hora | 99,7661% |
| 1 dia | 99,9902% |
| 1 semana | 99,9986% |
| 30 dias | 99,9997% |

No texto do TCC, esses valores devem aparecer como **projeção**, nunca como disponibilidade medida.

## 5. Como reexecutar

```powershell
# Execução completa usada no TCC (30 min, 10 falhas, 1000 consultas)
python rodar_tudo.py --ambos --duracao 30 --falhas 10 --consultas 1000
# Validação rápida (2 min)
python rodar_tudo.py --ambos --duracao 2 --falhas 2 --consultas 100
```

Detalhes da suíte e de cada script: `testes_disponibilidade/README.md`.
