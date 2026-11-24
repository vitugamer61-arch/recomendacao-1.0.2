# Testes de Disponibilidade — Sistema de Recomendação de Músicas

Suíte de testes para medir e reportar métricas de disponibilidade, resiliência e recuperação do sistema de recomendação, comparando cada resultado com um SLO definido para o TCC. Produz `relatorio.md`, `resumo.csv` e gráficos para apresentação.

> **Segurança:** Nunca altera a biblioteca de músicas nem o índice ChromaDB original em `audio_pipeline/chroma_db/`.

---

## Modos de Execução

A suíte opera em três modos:

1. **Linha de Base (`--sem-falhas`):**  
   Executa exclusivamente o monitoramento contínuo do serviço durante o período configurado (ex.: 30 min), medindo a disponibilidade natural e latência sob operação estável. Salva em `resultados/<timestamp>/baseline/`.

2. **Estresse com Falhas Injetadas (Padrão):**  
   Executa a restauração completa do índice vetorial, teste de carga concorrente e monitoramento com injeção forçada de falhas (kills repetidos no processo com ChromaDB embutido, recuperados automaticamente pelo Watchdog). Salva em `resultados/<timestamp>/estresse/`.

3. **Execução Consolidada (`--ambos`):**  
   Executa a Linha de Base e o Estresse em sequência, gerando um relatório único com tabela comparativa lado a lado. Salva em `resultados/<timestamp>/`.

---

## Como Rodar

### Validação rápida (2 minutos)
```powershell
# Com o venv ativado na pasta testes_disponibilidade/:

# 1. Somente Linha de Base (sem falhas):
python rodar_tudo.py --sem-falhas --duracao 2

# 2. Somente Estresse (2 falhas, 100 consultas):
python rodar_tudo.py --duracao 2 --falhas 2 --consultas 100

# 3. Ambos em sequência (relatório comparativo lado a lado):
python rodar_tudo.py --ambos --duracao 2 --falhas 2 --consultas 100
```

### Execução completa para o TCC (30 min, 10 falhas, 1000 consultas)
```powershell
python rodar_tudo.py --ambos --duracao 30 --falhas 10 --consultas 1000
```

---

## Arquitetura da Suíte

| Arquivo | Papel |
|---|---|
| `servico_consulta.py` | Serviço HTTP local (porta 7700) encapsulando o ChromaDB (`PersistentClient`). É o processo derrubado pela injeção de falhas. |
| `_cliente_servico.py` | Cliente HTTP auxiliar padronizado (com timeouts e parsing JSON) usado pelo monitor e pela carga. |
| `watchdog.py` | Supervisor independente com health checks periódicos via HTTP e reinício automático em caso de queda. |
| `monitor.py` | Cliente em processo separado que envia sondagens e consultas reais de similaridade, gravando série temporal. |
| `injecao_falhas.py` | Injeta falhas controladas matando o processo do serviço a intervalos pseudoaleatórios. |
| `carga.py` | Teste de carga com envio em rajada de consultas simultâneas sob concorrência configurável por threads. |
| `restauracao.py` | Simula destruição catastrófica em cópia do índice e mede tempo de reconstrução e fidelidade estrita do Top-K. |
| `analise.py` | Consolida os dados, reconcilia disponibilidades, projeta longo prazo e gera `relatorio.md`, `resumo.csv` e gráficos. |
| `rodar_tudo.py` | Orquestrador principal da suíte. |

---

## Parâmetros Configuráveis (`rodar_tudo.py`)

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--sem-falhas` | False | Executa apenas o modo Linha de Base (sem injeção) |
| `--ambos` | False | Executa Linha de Base e Estresse em sequência |
| `--duracao` | 30.0 min | Duração da janela de monitoramento |
| `--falhas` | 10 | Quantidade de falhas a injetar no modo estresse |
| `--consultas` | 1000 | Quantidade de consultas no teste de carga |
| `--threads` | 1 | Threads simultâneas no teste de carga |
| `--intervalo` | 1.0 s | Intervalo entre consultas do monitor |
| `--porta` | 7700 | Porta TCP do serviço de consulta |
| `--intervalo-watchdog` | 1.0 s | Intervalo de verificação de saúde do watchdog |
| `--slo-disp` | 99.9% | Meta de disponibilidade |
| `--slo-mttr` | 10.0 s | Meta máxima de tempo de recuperação |
| `--slo-sucesso` | 99.9% | Meta de taxa de sucesso |
| `--slo-restauracao` | 60.0 s | Meta de tempo de restauração do banco |
| `--semente` | 42 | Semente aleatória (reprodutibilidade) |
| `--incluir-mpd` | False | Também injeta falhas no daemon MPD |

---

## Métricas e Reconciliação Matemática

O `analise.py` calcula e compara duas visões fundamentais de disponibilidade:

1. **Disponibilidade por Consultas ($A_{\text{consultas}}$):**
   $$\frac{N_{\text{sucessos}}}{N_{\text{total}}} \times 100\%$$
   Reflete a experiência amostral discreta das requisições enviadas pelo monitor.

2. **Disponibilidade Temporal Efetiva ($A_{\text{temporal}}$):**
   $$\frac{\text{Uptime}}{\text{Uptime} + \text{Downtime}} \times 100\% = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}} \times 100\%$$
   Calculada a partir dos horários reais de injeção em `falhas.csv` e da detecção do primeiro sucesso em `monitor.csv`.

**Diferença explicada no relatório:** Em cenários com falha, o cliente HTTP sofre a latência de timeout do socket do SO (~2 s no Windows) além do intervalo configurado, reduzindo a frequência de sondagens durante a queda. Assim, a disponibilidade por consultas é menos sensível a janelas curtas de falha do que a disponibilidade temporal real. Na linha de base, ambas convergem para 100%.

3. **Projeção de Longo Prazo:**
   $$\text{MTBF}_{\min} = 999 \times \text{MTTR}$$
   Tabela projetando paradas anuais para MTBF de 1 hora, 1 dia, 1 semana e 30 dias com base no MTTR medido.
