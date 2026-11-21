"""
feedback_config.py — Hiperparâmetros e configurações do sistema de feedback implícito.
Centraliza todos os pesos, limiares temporais e opções de reforço negativo/positivo.
"""

from pathlib import Path

# Diretórios base
_BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_DB_PATH = _BASE_DIR / "results" / "feedback.db"

# ── 1. Detecção de Feedback (Heurística de Reprodução) ─────────────────────────
# Skip manual: troca de faixa antes de 30 segundos ou 30% da duração total
SKIP_MAX_ELAPSED_SECONDS = 30.0
SKIP_MAX_DURATION_RATIO = 0.30

# Faixa completa (feedback positivo): ouvida até pelo menos 80% da duração
COMPLETE_MIN_DURATION_RATIO = 0.80

# Tolerância para fim natural da faixa (segundos restantes para considerar fim da música)
NATURAL_END_TOLERANCE_SECONDS = 2.0


# ── 2. Algoritmo Rocchio no Vetor de Consulta (q') ────────────────────────────
# q' = normalize(α·q + β·média(positivas) − γ·média(puladas))
ROCCHIO_ALPHA = 1.0   # Peso da consulta original
ROCCHIO_BETA = 0.5    # Peso das faixas positivas (completadas)
ROCCHIO_GAMMA = 0.7   # Peso das faixas negativas (puladas)

# Quantidade de faixas recentes consideradas no Rocchio
ROCCHIO_MAX_ITEMS = 10

# Taxa de decaimento exponencial por ordem de antiguidade (mais recente tem peso 1.0)
# w_i = exp(-ROCCHIO_DECAY_RATE * i), onde i=0 é o evento mais recente
ROCCHIO_DECAY_RATE = 0.15


# ── 3. Busca e Re-ranking de Candidatos ───────────────────────────────────────
# Quantidade de candidatos preliminares buscados no ChromaDB com q'
CANDIDATES_N = 50

# Pesos do Re-ranking:
# score(c) = sim(q, c) − λ · Σ w_i·sim(c, skip_i) − μ · pen_genero(c)
LAMBDA_PENALTY_TRACK = 0.5   # λ: Peso da penalidade de faixas específicas puladas
MU_PENALTY_GENRE = 0.15       # μ: Peso da penalidade do gênero

# Exclusão de faixas recentemente puladas (últimas N faixas)
EXCLUDE_RECENT_SKIPS_COUNT = 10


# ── 4. Penalidade de Gênero e Cooldown ─────────────────────────────────────────
# Pontuação de penalidade cumulativa por gênero:
# +1.0 por skip, -0.5 por faixa completa (mínimo 0.0)
PENALTY_SKIP_GENRE = 1.0
BONUS_COMPLETE_GENRE = 0.5
GENRE_MIN_PENALTY = 0.0

# Decaimento temporal da penalidade de gênero (em segundos ou por evento)
GENRE_PENALTY_HALF_LIFE_SECONDS = 1800.0  # 30 minutos

# Regra de Cooldown por Skips Consecutivos:
# Se o usuário pular 3 ou mais faixas do mesmo gênero em uma janela de tempo,
# o gênero é temporariamente bloqueado/filtrado
GENRE_COOLDOWN_SKIP_THRESHOLD = 3
GENRE_COOLDOWN_WINDOW_SECONDS = 600.0     # Janela de 10 minutos
GENRE_COOLDOWN_DURATION_SECONDS = 300.0   # 5 minutos de cooldown

