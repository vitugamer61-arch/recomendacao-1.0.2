"""
Dashboard de acompanhamento das métricas de Precision@K e Agrupamento por Clusters.
Execute a partir da raiz do projeto:
    streamlit run audio_pipeline/results/dashboard.py
    ou via ponto de entrada único:
    iniciar.bat
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Caminhos e Configurações ──────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent.resolve()
_PIPELINE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _PIPELINE_DIR.parent

# Inclusão do diretório pai para importação do módulo de feedback
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

try:
    import feedback
    import feedback_config
    HAS_FEEDBACK = True
except ImportError:
    HAS_FEEDBACK = False

FEEDBACK_DB         = _THIS_DIR / "feedback.db"
METADATA_JSON       = _THIS_DIR / "metadata.json"
HISTORICO_CSV       = _THIS_DIR / "historico_execucoes.csv"
HISTOGRAMA_PNG      = _THIS_DIR / "precision_at_k_histograma.png"
GENERATOR_SCRIPT    = _THIS_DIR / "generate_precision_k.py"

CLUSTERS_JSON       = _THIS_DIR / "clusters.json"
CLUSTERING_EVAL_CSV = _THIS_DIR / "clustering_evaluation.csv"
CLUSTERS_2D_CSV     = _THIS_DIR / "clusters_2d.csv"
K_SELECTION_CSV     = _THIS_DIR / "k_selection_metrics.csv"
K_SELECTION_PNG     = _THIS_DIR / "k_selection_silhouette.png"

def _csv_por_k(k: int) -> Path:
    return _THIS_DIR / f"precision_at_{k}.csv"

def _json_por_k(k: int) -> Path:
    return _THIS_DIR / f"resumo_k_{k}.json"

# ── Funções de Cache de Dados ─────────────────────────────────────────────────
@st.cache_data
def load_clusters_summary(mtime: float) -> Optional[Dict[str, Any]]:
    if CLUSTERS_JSON.exists():
        try:
            with open(CLUSTERS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

@st.cache_data
def load_clustering_eval(mtime: float) -> Optional[pd.DataFrame]:
    if CLUSTERING_EVAL_CSV.exists():
        try:
            return pd.read_csv(CLUSTERING_EVAL_CSV)
        except Exception:
            return None
    return None

@st.cache_data
def load_clusters_2d(mtime: float) -> Optional[pd.DataFrame]:
    if CLUSTERS_2D_CSV.exists():
        try:
            return pd.read_csv(CLUSTERS_2D_CSV)
        except Exception:
            return None
    return None

@st.cache_data
def load_k_selection(mtime: float) -> Optional[pd.DataFrame]:
    if K_SELECTION_CSV.exists():
        try:
            return pd.read_csv(K_SELECTION_CSV)
        except Exception:
            return None
    return None

# ── Config da página ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monitor de Precisão & Clusters Musicais",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #4C72B0;
    }
    .metric-card.ganho { border-left-color: #2ecc71; }
    .metric-card.baseline { border-left-color: #DD8452; }
    .metric-card.cluster { border-left-color: #9b59b6; }
    .metric-card.coerencia { border-left-color: #f1c40f; }
    .metric-card.pureza { border-left-color: #1abc9c; }
    .metric-label { color: #aaaaaa; font-size: 0.85rem; margin-bottom: 4px; }
    .metric-value { color: #ffffff; font-size: 1.55rem; font-weight: bold; }
    .small-note { color: #888; font-size: 0.78rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎵 Monitor Musical")
    st.markdown("---")
    k_selecionado = st.selectbox("Valor de K (Recomendações)", options=[1, 3, 5, 10], index=2)
    st.markdown("---")

    st.markdown("### Atualizar Métricas")
    if st.button("▶ Executar geração", use_container_width=True):
        venv_python = _PIPELINE_DIR / "venv" / "Scripts" / "python.exe"
        python_bin = str(venv_python) if venv_python.exists() else sys.executable

        with st.spinner(f"Calculando Precision@{k_selecionado} (Global, Cluster e Coerência)..."):
            resultado = subprocess.run(
                [python_bin, str(GENERATOR_SCRIPT), "--k", str(k_selecionado)],
                capture_output=True, text=True, cwd=str(_PIPELINE_DIR)
            )

        if resultado.returncode == 0:
            st.success("Métricas atualizadas com sucesso!")
        else:
            st.error("Erro na geração de métricas!")
            st.code(resultado.stderr[-2000:] if resultado.stderr else "(sem stderr)", language="bash")

        st.rerun()

    st.markdown("---")
    st.markdown(
        "**Guia Rápido:**\n"
        "1. Escolha **K** acima para recalcular.\n"
        "2. Clique em **Executar geração**.\n"
        "3. Os gráficos e tabelas atualizam automaticamente com cache inteligente."
    )

# ── Cabeçalho Principal ───────────────────────────────────────────────────────
st.title("📊 Análise de Precision@K & Clusters — Recomendação Musical")
st.caption("Sistema CLAP + ChromaDB · Avaliação supervisionada por subgênero e clusterização sonora não-supervisionada")

# Carregamento de dados de cluster via cache
clusters_mtime = CLUSTERS_JSON.stat().st_mtime if CLUSTERS_JSON.exists() else 0.0
clusters_data = load_clusters_summary(clusters_mtime)

eval_mtime = CLUSTERING_EVAL_CSV.stat().st_mtime if CLUSTERING_EVAL_CSV.exists() else 0.0
eval_df = load_clustering_eval(eval_mtime)

# Leitura do resumo JSON do K selecionado
json_path = _json_por_k(k_selecionado)
resumo_disponivel = json_path.exists()
resumo = {}
if resumo_disponivel:
    try:
        with open(json_path, encoding="utf-8") as f:
            resumo = json.load(f)
    except Exception:
        resumo = {}

# Linha de status temporal
ts_pk = resumo.get("timestamp", "")
try:
    ts_pk_fmt = datetime.fromisoformat(ts_pk).strftime("%d/%m/%Y às %H:%M")
except Exception:
    ts_pk_fmt = ts_pk if ts_pk else "Não gerado"

ts_cl = clusters_data.get("generated_at", "") if clusters_data else ""
try:
    ts_cl_fmt = datetime.fromisoformat(ts_cl).strftime("%d/%m/%Y às %H:%M")
except Exception:
    ts_cl_fmt = ts_cl if ts_cl else "Não calibrado"

faixas_totais = resumo.get("faixas_avaliadas", clusters_data.get("total_tracks", "—") if clusters_data else "—")
st.markdown(
    f"🕒 **Última execução P@{k_selecionado}:** {ts_pk_fmt} &nbsp;|&nbsp; "
    f"🔮 **Última calibração de clusters:** {ts_cl_fmt} &nbsp;|&nbsp; "
    f"📁 **Faixas avaliadas:** {faixas_totais}"
)

st.markdown("---")

# ── Aba de Recomendação ao Vivo ───────────────────────────────────────────────
@st.fragment(run_every="2s")
def render_live_tab():
    st.markdown("### ⚡ Acompanhamento de Recomendação ao Vivo (MPD + Feedback)")

    if not FEEDBACK_DB.exists():
        st.info("ℹ️ Banco de dados `feedback.db` ainda não criado. Inicie o recomendador pelo `iniciar.bat` para iniciar o rastreamento em tempo real.")
        return

    storage = feedback.FeedbackStorage(db_path=FEEDBACK_DB)

    # Barra superior de ações
    col_hdr, col_btn = st.columns([4, 1.2])
    with col_hdr:
        st.caption("Atualização automática a cada ~2 segundos via SQLite WAL. Monitore decisões, histórico de skips e fila ativa.")
    with col_btn:
        if st.button("🗑️ Zerar Feedback", use_container_width=True, key="btn_zerar_fb_top"):
            storage.clear_history()
            st.success("Histórico de feedback limpo com sucesso!")
            st.rerun()

    # 1. Faixa atual e Fila do MPD
    queue_items = storage.get_queue_state()
    current_track = next((it for it in queue_items if it.get("is_current")), None)

    col_curr, col_queue = st.columns([1, 1.8])
    with col_curr:
        st.subheader("🎵 Faixa em Reprodução")
        if current_track:
            origem = current_track.get("origem", "manual")
            badge_color = "#2ecc71" if origem == "recomendada" else "#9b59b6" if origem == "exploracao" else "#95a5a6"
            origem_label = "Recomendada (#1)" if origem == "recomendada" else "Exploração (Top-N)" if origem == "exploracao" else "Manual (Usuário)"
            
            st.markdown(f"""
            <div style="background: #1e2130; border-radius: 10px; padding: 18px; border-left: 5px solid {badge_color};">
                <div style="font-size: 1.15rem; font-weight: bold; color: #ffffff;">{os.path.basename(current_track.get('caminho', ''))}</div>
                <div style="margin-top: 10px; color: #bbbbbb; font-size: 0.92rem; line-height: 1.6;">
                    <b>Subgênero:</b> <span style="color: #ffffff;">{current_track.get('genero') or '—'}</span><br>
                    <b>Cluster:</b> <span style="color: #ffffff;">{current_track.get('cluster') or '—'}</span><br>
                    <b>Origem:</b> <span style="background: {badge_color}; color: #ffffff; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">{origem_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #1e2130; border-radius: 10px; padding: 18px; border-left: 5px solid #7f8c8d;">
                <div style="color: #888888;">Nenhuma faixa em reprodução ativa no MPD.</div>
            </div>
            """, unsafe_allow_html=True)

    with col_queue:
        st.subheader("📜 Fila Atual do MPD")
        if queue_items:
            df_q = pd.DataFrame([
                {
                    "Pos": it.get("pos", 0) + 1,
                    "Status": "▶ Tocando" if it.get("is_current") else "Aguardando",
                    "Faixa": os.path.basename(it.get("caminho", "")),
                    "Gênero": it.get("genero") or "—",
                    "Cluster": it.get("cluster") or "—",
                    "Origem": "🟢 Recomendada" if it.get("origem") == "recomendada" else "🟣 Exploração" if it.get("origem") == "exploracao" else "⚪ Manual"
                }
                for it in queue_items
            ])
            st.dataframe(df_q, use_container_width=True, hide_index=True, height=210)
        else:
            st.info("Fila do MPD vazia ou aguardando leitura do estado.")

    st.markdown("---")

    # 2. Métricas de Aceitação e Cooldown (Cards)
    stats_geral = storage.get_acceptance_stats()
    stats_10m = storage.get_acceptance_stats(window_seconds=600.0)
    cooldown_genres = storage.get_cooldown_genres()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card ganho">
            <div class="metric-label">Taxa de Aceitação Global</div>
            <div class="metric-value">{stats_geral['taxa_aceitacao']:.1%}</div>
            <div class="small-note">{stats_geral['total_aceitas']} aceitas / {stats_geral['total_skips']} skips</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Taxa Recente (Últimos 10 min)</div>
            <div class="metric-value">{stats_10m['taxa_aceitacao']:.1%}</div>
            <div class="small-note">{stats_10m['total_aceitas']} aceitas / {stats_10m['total_skips']} skips</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card baseline">
            <div class="metric-label">Total de Skips (<30s)</div>
            <div class="metric-value">{stats_geral['total_skips']}</div>
            <div class="small-note">{stats_geral['total_parciais']} parciais (+fraco)</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        cd_count = len(cooldown_genres)
        cd_color = "#e74c3c" if cd_count > 0 else "#2ecc71"
        cd_text = ", ".join(cooldown_genres) if cooldown_genres else "Nenhum gênero bloqueado"
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {cd_color};">
            <div class="metric-label">Gêneros em Cooldown (Filtro Ativo)</div>
            <div class="metric-value" style="color: {cd_color}; font-size: 1.35rem;">{cd_count} ativo(s)</div>
            <div class="small-note">{cd_text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 3. Penalidades de Gênero & Feed de Eventos
    col_pen, col_feed = st.columns([1, 1.2])

    with col_pen:
        st.subheader("⚖️ Penalidades por Gênero (Tempo Real)")
        genre_penalties = storage.get_genre_penalties()
        if genre_penalties:
            df_pen = pd.DataFrame([
                {"Gênero": g, "Penalidade": pen}
                for g, pen in genre_penalties.items()
            ]).sort_values("Penalidade", ascending=True)

            fig_pen = go.Figure()
            fig_pen.add_trace(go.Bar(
                y=df_pen["Gênero"],
                x=df_pen["Penalidade"],
                orientation="h",
                marker=dict(
                    color=df_pen["Penalidade"],
                    colorscale="Reds",
                    cmin=0.0,
                    cmax=max(df_pen["Penalidade"].max(), 2.0)
                ),
                text=df_pen["Penalidade"].map(lambda x: f"{x:.2f}"),
                textposition="auto"
            ))
            fig_pen.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1a1a2e",
                xaxis_title="Penalidade (com decaimento temporal contínuo)",
                margin=dict(l=20, r=20, t=30, b=30),
                height=290
            )
            st.plotly_chart(fig_pen, use_container_width=True)
        else:
            st.info("Nenhuma penalidade de gênero registrada no momento.")

    with col_feed:
        st.subheader("⏱️ Feed Cronológico de Eventos")
        recent_events = storage.get_recent_events(limit=8)
        last_decision = storage.get_last_decision()

        feed_rows = []
        for ev in recent_events:
            tipo = ev.get("tipo", "")
            g = ev.get("genero") or "Sem gênero"
            musica = os.path.basename(ev.get("caminho") or ev.get("track_id") or "")
            t_str = str(ev.get("timestamp", ""))[-8:]
            elap = float(ev.get("elapsed") or 0.0)

            if tipo == "completa":
                feed_rows.append({
                    "Hora": t_str,
                    "Ícone": "🟢",
                    "Tipo": "ACEITA (≥80%)",
                    "Faixa": musica,
                    "Detalhes": f"{g} · {elap:.0f}s ouvida"
                })
            elif tipo == "parcial":
                peso = float(ev.get("peso") or 0.5)
                feed_rows.append({
                    "Hora": t_str,
                    "Ícone": "🟡",
                    "Tipo": f"PARCIAL (+{peso:.2f})",
                    "Faixa": musica,
                    "Detalhes": f"{g} · {elap:.0f}s ouvida"
                })
            elif tipo == "skip":
                feed_rows.append({
                    "Hora": t_str,
                    "Ícone": "🔴",
                    "Tipo": "SKIP (<30s)",
                    "Faixa": musica,
                    "Detalhes": f"{g} · {elap:.0f}s pulada"
                })

        if last_decision and last_decision.get("candidatas"):
            for c in last_decision["candidatas"]:
                if c.get("rejeitada"):
                    feed_rows.append({
                        "Hora": str(last_decision.get("timestamp", ""))[-8:],
                        "Ícone": "⚠️",
                        "Tipo": "FILTRADA",
                        "Faixa": c.get("filename", ""),
                        "Detalhes": c.get("motivo", "Filtro de exclusão")
                    })

        if feed_rows:
            df_feed = pd.DataFrame(feed_rows)
            st.dataframe(df_feed, use_container_width=True, hide_index=True, height=290)
        else:
            st.info("Nenhum evento registrado no histórico recente.")

    st.markdown("---")

    # 4. Tabela da Última Decisão do Recomendador
    st.subheader("🎯 Detalhamento da Última Decisão do Recomendador")
    last_dec = storage.get_last_decision()
    if last_dec:
        seed_f = last_dec.get("seed_caminho") or last_dec.get("seed_track_id", "")
        seed_nome = os.path.basename(seed_f) if seed_f else "—"
        escolhida_f = last_dec.get("escolhida_caminho") or ""
        escolhida_nome = os.path.basename(escolhida_f) if escolhida_f else "—"
        origem_dec = last_dec.get("origem", "recomendada")
        rank_dec = last_dec.get("rank_escolhido", 0) + 1

        st.markdown(
            f"📍 **Seed:** `{seed_nome}` ({last_dec.get('seed_genero')}) &nbsp;|&nbsp; "
            f"🏆 **Escolhida:** `{escolhida_nome}` ({last_dec.get('escolhida_genero')}) &nbsp;|&nbsp; "
            f"🏷️ **Origem:** `{origem_dec.upper()}` (Rank #{rank_dec}) &nbsp;|&nbsp; "
            f"⭐ **Score:** `{last_dec.get('score_escolhido', 0.0):.4f}` &nbsp;|&nbsp; "
            f"🕒 **Horário:** `{str(last_dec.get('timestamp', ''))}`"
        )

        cands = last_dec.get("candidatas", [])
        if cands:
            df_cands = pd.DataFrame([
                {
                    "Rank": c.get("rank", 0),
                    "Status": "🏆 Selecionada" if c.get("motivo") == "Selecionada" else "🚫 Rejeitada" if c.get("rejeitada") else "Alternativa",
                    "Faixa": c.get("filename", ""),
                    "Gênero": c.get("subgenero", ""),
                    "Cluster": c.get("cluster_nome", ""),
                    "Similaridade (q)": f"{c.get('sim_query', 0.0):.3f}",
                    "Pen. Faixa (λ)": f"{c.get('track_penalty', 0.0):.3f}",
                    "Pen. Gênero (μ)": f"{c.get('genre_penalty', 0.0):.3f}",
                    "Score Final": f"{c.get('final_score', 0.0):.3f}",
                    "Motivo / Decisão": c.get("motivo", "")
                }
                for c in cands
            ])
            st.dataframe(df_cands, use_container_width=True, hide_index=True)
    else:
        st.info("Aguardando primeira decisão do recomendador...")


def render_precision_tab():
    # ══════════════════════════════════════════════════════════════════════════════
    # SEÇÃO 1: PRECISÃO & RECOMENDAÇÃO (P@K GLOBAL VS RESTRICTED AO CLUSTER)
    # ══════════════════════════════════════════════════════════════════════════════
    st.header(f"🎯 Precisão@K (K = {k_selecionado})")

    if not FEEDBACK_DB.exists():
        st.info("ℹ️ Banco de dados `feedback.db` ainda não criado. Inicie o recomendador pelo `iniciar.bat` para iniciar o rastreamento em tempo real.")
        return

    storage = feedback.FeedbackStorage(db_path=FEEDBACK_DB)

    # Barra superior de ações
    col_hdr, col_btn = st.columns([4, 1.2])
    with col_hdr:
        st.caption("Atualização automática a cada ~2 segundos via SQLite WAL. Monitore decisões, histórico de skips e fila ativa.")
    with col_btn:
        if st.button("🗑️ Zerar Feedback", use_container_width=True, key="btn_zerar_fb_top"):
            storage.clear_history()
            st.success("Histórico de feedback limpo com sucesso!")
            st.rerun()

    # 1. Faixa atual e Fila do MPD
    queue_items = storage.get_queue_state()
    current_track = next((it for it in queue_items if it.get("is_current")), None)

    col_curr, col_queue = st.columns([1, 1.8])
    with col_curr:
        st.subheader("🎵 Faixa em Reprodução")
        if current_track:
            origem = current_track.get("origem", "manual")
            badge_color = "#2ecc71" if origem == "recomendada" else "#9b59b6" if origem == "exploracao" else "#95a5a6"
            origem_label = "Recomendada (#1)" if origem == "recomendada" else "Exploração (Top-N)" if origem == "exploracao" else "Manual (Usuário)"
            
            st.markdown(f"""
            <div style="background: #1e2130; border-radius: 10px; padding: 18px; border-left: 5px solid {badge_color};">
                <div style="font-size: 1.15rem; font-weight: bold; color: #ffffff;">{os.path.basename(current_track.get('caminho', ''))}</div>
                <div style="margin-top: 10px; color: #bbbbbb; font-size: 0.92rem; line-height: 1.6;">
                    <b>Subgênero:</b> <span style="color: #ffffff;">{current_track.get('genero') or '—'}</span><br>
                    <b>Cluster:</b> <span style="color: #ffffff;">{current_track.get('cluster') or '—'}</span><br>
                    <b>Origem:</b> <span style="background: {badge_color}; color: #ffffff; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">{origem_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #1e2130; border-radius: 10px; padding: 18px; border-left: 5px solid #7f8c8d;">
                <div style="color: #888888;">Nenhuma faixa em reprodução ativa no MPD.</div>
            </div>
            """, unsafe_allow_html=True)

    with col_queue:
        st.subheader("📜 Fila Atual do MPD")
        if queue_items:
            df_q = pd.DataFrame([
                {
                    "Pos": it.get("pos", 0) + 1,
                    "Status": "▶ Tocando" if it.get("is_current") else "Aguardando",
                    "Faixa": os.path.basename(it.get("caminho", "")),
                    "Gênero": it.get("genero") or "—",
                    "Cluster": it.get("cluster") or "—",
                    "Origem": "🟢 Recomendada" if it.get("origem") == "recomendada" else "🟣 Exploração" if it.get("origem") == "exploracao" else "⚪ Manual"
                }
                for it in queue_items
            ])
            st.dataframe(df_q, use_container_width=True, hide_index=True, height=210)
        else:
            st.info("Fila do MPD vazia ou aguardando leitura do estado.")

    st.markdown("---")

    # 2. Métricas de Aceitação e Cooldown (Cards)
    stats_geral = storage.get_acceptance_stats()
    stats_10m = storage.get_acceptance_stats(window_seconds=600.0)
    cooldown_genres = storage.get_cooldown_genres()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card ganho">
            <div class="metric-label">Taxa de Aceitação Global</div>
            <div class="metric-value">{stats_geral['taxa_aceitacao']:.1%}</div>
            <div class="small-note">{stats_geral['total_aceitas']} aceitas / {stats_geral['total_skips']} skips</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Taxa Recente (Últimos 10 min)</div>
            <div class="metric-value">{stats_10m['taxa_aceitacao']:.1%}</div>
            <div class="small-note">{stats_10m['total_aceitas']} aceitas / {stats_10m['total_skips']} skips</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card baseline">
            <div class="metric-label">Total de Skips (<30s)</div>
            <div class="metric-value">{stats_geral['total_skips']}</div>
            <div class="small-note">{stats_geral['total_parciais']} parciais (+fraco)</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        cd_count = len(cooldown_genres)
        cd_color = "#e74c3c" if cd_count > 0 else "#2ecc71"
        cd_text = ", ".join(cooldown_genres) if cooldown_genres else "Nenhum gênero bloqueado"
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {cd_color};">
            <div class="metric-label">Gêneros em Cooldown (Filtro Ativo)</div>
            <div class="metric-value" style="color: {cd_color}; font-size: 1.35rem;">{cd_count} ativo(s)</div>
            <div class="small-note">{cd_text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 3. Penalidades de Gênero & Feed de Eventos
    col_pen, col_feed = st.columns([1, 1.2])

    with col_pen:
        st.subheader("⚖️ Penalidades por Gênero (Tempo Real)")
        genre_penalties = storage.get_genre_penalties()
        if genre_penalties:
            df_pen = pd.DataFrame([
                {"Gênero": g, "Penalidade": pen}
                for g, pen in genre_penalties.items()
            ]).sort_values("Penalidade", ascending=True)

            fig_pen = go.Figure()
            fig_pen.add_trace(go.Bar(
                y=df_pen["Gênero"],
                x=df_pen["Penalidade"],
                orientation="h",
                marker=dict(
                    color=df_pen["Penalidade"],
                    colorscale="Reds",
                    cmin=0.0,
                    cmax=max(df_pen["Penalidade"].max(), 2.0)
                ),
                text=df_pen["Penalidade"].map(lambda x: f"{x:.2f}"),
                textposition="auto"
            ))
            fig_pen.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1a1a2e",
                xaxis_title="Penalidade (com decaimento temporal contínuo)",
                margin=dict(l=20, r=20, t=30, b=30),
                height=290
            )
            st.plotly_chart(fig_pen, use_container_width=True)
        else:
            st.info("Nenhuma penalidade de gênero registrada no momento.")

    with col_feed:
        st.subheader("⏱️ Feed Cronológico de Eventos")
        recent_events = storage.get_recent_events(limit=8)
        last_decision = storage.get_last_decision()

        feed_rows = []
        for ev in recent_events:
            tipo = ev.get("tipo", "")
            g = ev.get("genero") or "Sem gênero"
            musica = os.path.basename(ev.get("caminho") or ev.get("track_id") or "")
            t_str = str(ev.get("timestamp", ""))[-8:]
            elap = float(ev.get("elapsed") or 0.0)

            if tipo == "completa":
                feed_rows.append({
                    "Hora": t_str,
                    "Ícone": "🟢",
                    "Tipo": "ACEITA (≥80%)",
                    "Faixa": musica,
                    "Detalhes": f"{g} · {elap:.0f}s ouvida"
                })
            elif tipo == "parcial":
                peso = float(ev.get("peso") or 0.5)
                feed_rows.append({
                    "Hora": t_str,
                    "Ícone": "🟡",
                    "Tipo": f"PARCIAL (+{peso:.2f})",
                    "Faixa": musica,
                    "Detalhes": f"{g} · {elap:.0f}s ouvida"
                })
            elif tipo == "skip":
                feed_rows.append({
                    "Hora": t_str,
                    "Ícone": "🔴",
                    "Tipo": "SKIP (<30s)",
                    "Faixa": musica,
                    "Detalhes": f"{g} · {elap:.0f}s pulada"
                })

        if last_decision and last_decision.get("candidatas"):
            for c in last_decision["candidatas"]:
                if c.get("rejeitada"):
                    feed_rows.append({
                        "Hora": str(last_decision.get("timestamp", ""))[-8:],
                        "Ícone": "⚠️",
                        "Tipo": "FILTRADA",
                        "Faixa": c.get("filename", ""),
                        "Detalhes": c.get("motivo", "Filtro de exclusão")
                    })

        if feed_rows:
            df_feed = pd.DataFrame(feed_rows)
            st.dataframe(df_feed, use_container_width=True, hide_index=True, height=290)
        else:
            st.info("Nenhum evento registrado no histórico recente.")

    st.markdown("---")

    # 4. Tabela da Última Decisão do Recomendador
    st.subheader("🎯 Detalhamento da Última Decisão do Recomendador")
    last_dec = storage.get_last_decision()
    if last_dec:
        seed_f = last_dec.get("seed_caminho") or last_dec.get("seed_track_id", "")
        seed_nome = os.path.basename(seed_f) if seed_f else "—"
        escolhida_f = last_dec.get("escolhida_caminho") or ""
        escolhida_nome = os.path.basename(escolhida_f) if escolhida_f else "—"
        origem_dec = last_dec.get("origem", "recomendada")
        rank_dec = last_dec.get("rank_escolhido", 0) + 1

        st.markdown(
            f"📍 **Seed:** `{seed_nome}` ({last_dec.get('seed_genero')}) &nbsp;|&nbsp; "
            f"🏆 **Escolhida:** `{escolhida_nome}` ({last_dec.get('escolhida_genero')}) &nbsp;|&nbsp; "
            f"🏷️ **Origem:** `{origem_dec.upper()}` (Rank #{rank_dec}) &nbsp;|&nbsp; "
            f"⭐ **Score:** `{last_dec.get('score_escolhido', 0.0):.4f}` &nbsp;|&nbsp; "
            f"🕒 **Horário:** `{str(last_dec.get('timestamp', ''))}`"
        )

        cands = last_dec.get("candidatas", [])
        if cands:
            df_cands = pd.DataFrame([
                {
                    "Rank": c.get("rank", 0),
                    "Status": "🏆 Selecionada" if c.get("motivo") == "Selecionada" else "🚫 Rejeitada" if c.get("rejeitada") else "Alternativa",
                    "Faixa": c.get("filename", ""),
                    "Gênero": c.get("subgenero", ""),
                    "Cluster": c.get("cluster_nome", ""),
                    "Similaridade (q)": f"{c.get('sim_query', 0.0):.3f}",
                    "Pen. Faixa (λ)": f"{c.get('track_penalty', 0.0):.3f}",
                    "Pen. Gênero (μ)": f"{c.get('genre_penalty', 0.0):.3f}",
                    "Score Final": f"{c.get('final_score', 0.0):.3f}",
                    "Motivo / Decisão": c.get("motivo", "")
                }
                for c in cands
            ])
            st.dataframe(df_cands, use_container_width=True, hide_index=True)
    else:
        st.info("Aguardando primeira decisão do recomendador...")


def render_precision_tab():
    # ══════════════════════════════════════════════════════════════════════════════
    # SEÇÃO 1: PRECISÃO & RECOMENDAÇÃO (P@K GLOBAL VS RESTRICTED AO CLUSTER)
    # ══════════════════════════════════════════════════════════════════════════════
    st.header(f"🎯 Precisão@K (K = {k_selecionado})")

    if resumo_disponivel and resumo:
        # Captura dados dos modos global e restrito
        s_global = resumo.get("sistema_global", resumo.get("sistema", {}))
        s_cluster = resumo.get("sistema_cluster", {})
        coerencia = resumo.get("coerencia_cluster", {})
        base = resumo.get("baseline", {})

        media_glob = s_global.get("media", 0.0)
        desv_glob = s_global.get("desvio_padrao", 0.0)
        ganho_glob = s_global.get("ganho_absoluto", resumo.get("ganho_absoluto", 0.0))

        media_clust = s_cluster.get("media", media_glob)
        desv_clust = s_cluster.get("desvio_padrao", desv_glob)
        ganho_clust = s_cluster.get("ganho_absoluto", ganho_glob)

        media_coer = coerencia.get("media", 0.0)
        base_media = base.get("media", 0.0)
        base_desv = base.get("desvio_padrao", 0.0)

        # Cards comparativos
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">K-NN Global — Precision@{k_selecionado}</div>
                <div class="metric-value">{media_glob:.1%}</div>
                <div class="small-note">±{desv_glob:.1%} | Ganho: {ganho_glob:+.1%}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card cluster">
                <div class="metric-label">K-NN Restrito ao Cluster — P@{k_selecionado}</div>
                <div class="metric-value">{media_clust:.1%}</div>
                <div class="small-note">±{desv_clust:.1%} | Ganho: {ganho_clust:+.1%}</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card coerencia">
                <div class="metric-label">Coerência de Cluster no K-NN</div>
                <div class="metric-value">{media_coer:.1%}</div>
                <div class="small-note">Fração dos {k_selecionado} vizinhos no mesmo cluster</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card baseline">
                <div class="metric-label">Baseline Aleatória — P@{k_selecionado}</div>
                <div class="metric-value">{base_media:.1%}</div>
                <div class="small-note">±{base_desv:.1%} desvio padrão</div>
            </div>""", unsafe_allow_html=True)

        # Detalhamento de P@K por Cluster
        por_cluster_data = resumo.get("por_cluster", {})
        if por_cluster_data:
            st.subheader(f"Comparativo de Precision@{k_selecionado} por Cluster Sonoro")
            df_cluster_metrics = pd.DataFrame([
                {
                    "Cluster (Gênero)": cnome,
                    "Faixas": cinfo["tamanho"],
                    f"P@{k_selecionado} Global": cinfo["media_p_at_k"],
                    f"P@{k_selecionado} Restrito": cinfo["media_p_at_k_cluster"],
                    "Ganho Global": cinfo["ganho_global"],
                    "Ganho Restrito": cinfo["ganho_cluster"],
                    "Coerência": cinfo["coerencia_media"]
                }
                for cnome, cinfo in por_cluster_data.items()
            ]).sort_values("Faixas", ascending=False)

            col_bar_clust, col_tbl_clust = st.columns([1.2, 1])

            with col_bar_clust:
                fig_bar_comp = go.Figure()
                fig_bar_comp.add_trace(go.Bar(
                    x=df_cluster_metrics["Cluster (Gênero)"],
                    y=df_cluster_metrics[f"P@{k_selecionado} Global"],
                    name="K-NN Global",
                    marker_color="#4C72B0"
                ))
                fig_bar_comp.add_trace(go.Bar(
                    x=df_cluster_metrics["Cluster (Gênero)"],
                    y=df_cluster_metrics[f"P@{k_selecionado} Restrito"],
                    name="Restrito ao Cluster",
                    marker_color="#9b59b6"
                ))
                fig_bar_comp.add_hline(y=base_media, line_dash="dash", line_color="#DD8452", annotation_text=f"Baseline ({base_media:.1%})")
                fig_bar_comp.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#1a1a2e",
                    barmode="group",
                    yaxis_tickformat=".0%",
                    yaxis_title=f"Precision@{k_selecionado}",
                    margin=dict(l=20, r=20, t=30, b=30),
                    height=320
                )
                st.plotly_chart(fig_bar_comp, use_container_width=True)

            with col_tbl_clust:
                df_disp = df_cluster_metrics.copy()
                df_disp[f"P@{k_selecionado} Global"] = df_disp[f"P@{k_selecionado} Global"].map(lambda x: f"{x:.1%}")
                df_disp[f"P@{k_selecionado} Restrito"] = df_disp[f"P@{k_selecionado} Restrito"].map(lambda x: f"{x:.1%}")
                df_disp["Coerência"] = df_disp["Coerência"].map(lambda x: f"{x:.1%}")
                df_disp["Ganho Global"] = df_disp["Ganho Global"].map(lambda x: f"{x:+.1%}")
                df_disp["Ganho Restrito"] = df_disp["Ganho Restrito"].map(lambda x: f"{x:+.1%}")
                st.dataframe(df_disp, use_container_width=True, height=320, hide_index=True)

    else:
        st.info(f"Nenhum resultado encontrado para K={k_selecionado}. Clique em 'Executar geração' no painel lateral.")

    # ── Histograma e Detalhamento por Faixa com Filtro por Cluster ────────────────
    col_hist, col_tabela = st.columns([1.2, 1.2])

    csv_path = _csv_por_k(k_selecionado)
    df_faixas = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    with col_hist:
        st.subheader("Distribuição de P@K por Faixa")
        if not df_faixas.empty and f"precision_at_{k_selecionado}" in df_faixas.columns:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_faixas[f"precision_at_{k_selecionado}"],
                name="Sistema Global",
                nbinsx=15,
                marker_color="#4C72B0",
                opacity=0.75,
                histnorm="density"
            ))
            if f"precision_at_{k_selecionado}_cluster" in df_faixas.columns:
                fig_hist.add_trace(go.Histogram(
                    x=df_faixas[f"precision_at_{k_selecionado}_cluster"],
                    name="Restrito ao Cluster",
                    nbinsx=15,
                    marker_color="#9b59b6",
                    opacity=0.6,
                    histnorm="density"
                ))
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1a1a2e",
                legend=dict(font=dict(color="white")),
                xaxis_title=f"Precision@{k_selecionado}",
                yaxis_title="Densidade",
                margin=dict(l=20, r=20, t=30, b=30),
                height=340,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        elif HISTOGRAMA_PNG.exists():
            st.image(str(HISTOGRAMA_PNG), use_container_width=True)
        else:
            st.info("Histograma ainda não gerado.")

    with col_tabela:
        st.subheader("Detalhamento por Faixa")
        if not df_faixas.empty:
            colunas_disponiveis = df_faixas.columns.tolist()

            # Filtro de cluster
            clusters_opts = ["Todos"]
            if "cluster_nome" in colunas_disponiveis:
                clusters_opts += sorted([str(c) for c in df_faixas["cluster_nome"].dropna().unique()])
            elif "cluster_id" in colunas_disponiveis:
                clusters_opts += sorted([str(c) for c in df_faixas["cluster_id"].dropna().unique()])

            cluster_filtro = st.selectbox("Filtrar por cluster musical:", options=clusters_opts)

            df_show = df_faixas.copy()
            if cluster_filtro != "Todos":
                if "cluster_nome" in df_show.columns:
                    df_show = df_show[df_show["cluster_nome"] == cluster_filtro]
                elif "cluster_id" in df_show.columns:
                    df_show = df_show[df_show["cluster_id"] == cluster_filtro]

            col_score = f"precision_at_{k_selecionado}"
            if col_score in df_show.columns:
                df_show = df_show.sort_values(col_score, ascending=False).reset_index(drop=True)

            cols_para_exibir = [c for c in ["filename", "subgenero", "cluster_nome", f"precision_at_{k_selecionado}", f"precision_at_{k_selecionado}_cluster", f"cluster_coherence_at_{k_selecionado}"] if c in df_show.columns]
            st.dataframe(df_show[cols_para_exibir], use_container_width=True, height=290)
        else:
            st.info("Gere as métricas primeiro.")

    # ── Histórico de Execuções ────────────────────────────────────────────────────
    st.subheader("Histórico de Execuções")
    if HISTORICO_CSV.exists():
        df_hist = pd.read_csv(HISTORICO_CSV)
        if "data_hora" in df_hist.columns and "timestamp" not in df_hist.columns:
            df_hist = df_hist.rename(columns={"data_hora": "timestamp"})
        if "faixas_avaliadas" not in df_hist.columns and "faixas" in df_hist.columns:
            df_hist = df_hist.rename(columns={"faixas": "faixas_avaliadas"})

        if "timestamp" in df_hist.columns and "media_sistema" in df_hist.columns:
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            df_hist_k = df_hist[df_hist["k"] == k_selecionado].copy()

            if not df_hist_k.empty:
                fig_hist_line = go.Figure()
                fig_hist_line.add_trace(go.Scatter(
                    x=df_hist_k["timestamp"], y=df_hist_k["media_sistema"],
                    mode="lines+markers", name="Sistema", line=dict(color="#4C72B0", width=2)
                ))
                fig_hist_line.add_trace(go.Scatter(
                    x=df_hist_k["timestamp"], y=df_hist_k["media_baseline"],
                    mode="lines+markers", name="Baseline", line=dict(color="#DD8452", width=2, dash="dot")
                ))
                fig_hist_line.update_layout(
                    template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#1a1a2e",
                    xaxis_title="Execução", yaxis_title=f"Precision@{k_selecionado} Média",
                    height=260, margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_hist_line, use_container_width=True)

            with st.expander("Ver histórico acumulado completo"):
                st.dataframe(df_hist.sort_values("timestamp", ascending=False), use_container_width=True)
    else:
        st.info("Nenhuma execução registrada no histórico ainda.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # SEÇÃO 2: ANÁLISE DE CLUSTERS (AGRUPAMENTO SONORO NÃO-SUPERVISIONADO)
    # ══════════════════════════════════════════════════════════════════════════════
    st.header("🔮 Análise de Clusters Sonoros")

    if clusters_data and eval_df is not None:
        eval_row = eval_df.iloc[0] if not eval_df.empty else pd.Series()
        val_metrics = clusters_data.get("validation_metrics", {})

        # Cards de Qualidade Intrínseca e Validação Externa (TCC)
        st.subheader("Métricas de Qualidade de Agrupamento")
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6, m_col7 = st.columns(7)

        with m_col1:
            st.markdown(f"""
            <div class="metric-card cluster">
                <div class="metric-label">Nº de Clusters</div>
                <div class="metric-value">{clusters_data.get('n_clusters', eval_row.get('n_clusters', 0))}</div>
                <div class="small-note">Algoritmo: {clusters_data.get('algorithm', 'KMeans').upper()}</div>
            </div>""", unsafe_allow_html=True)

        with m_col2:
            sil_val = eval_row.get("silhouette", 0.0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Silhueta Média</div>
                <div class="metric-value">{sil_val:.4f}</div>
                <div class="small-note">[-1 a +1] Maior = melhor</div>
            </div>""", unsafe_allow_html=True)

        with m_col3:
            db_val = eval_row.get("davies_bouldin", 0.0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Davies-Bouldin</div>
                <div class="metric-value">{db_val:.4f}</div>
                <div class="small-note">[0 a +∞] Menor = melhor</div>
            </div>""", unsafe_allow_html=True)

        with m_col4:
            ch_val = eval_row.get("calinski_harabasz", 0.0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Calinski-Harabasz</div>
                <div class="metric-value">{ch_val:.1f}</div>
                <div class="small-note">Maior = separação melhor</div>
            </div>""", unsafe_allow_html=True)

        with m_col5:
            pureza_val = val_metrics.get("purity", eval_row.get("pureza_media", clusters_data.get("pureza_media", 0.0)))
            st.markdown(f"""
            <div class="metric-card pureza">
                <div class="metric-label">Pureza Média</div>
                <div class="metric-value">{pureza_val:.1%}</div>
                <div class="small-note">Concordância subgêneros</div>
            </div>""", unsafe_allow_html=True)

        with m_col6:
            ari_val = val_metrics.get("ari", eval_row.get("ari", 0.0))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Adjusted Rand (ARI)</div>
                <div class="metric-value">{ari_val:.4f}</div>
                <div class="small-note">Ajustado para o acaso</div>
            </div>""", unsafe_allow_html=True)

        with m_col7:
            nmi_val = val_metrics.get("nmi", eval_row.get("nmi", 0.0))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Info Mútua (NMI)</div>
                <div class="metric-value">{nmi_val:.4f}</div>
                <div class="small-note">Informação normalizada</div>
            </div>""", unsafe_allow_html=True)

        # ── Gráficos 2D e Tamanho de Clusters ─────────────────────────────────────
        col_scatter, col_bar = st.columns([1.3, 1])

        # Leitura das coordenadas 2D salvas
        df_2d_mtime = CLUSTERS_2D_CSV.stat().st_mtime if CLUSTERS_2D_CSV.exists() else 0.0
        df_2d = load_clusters_2d(df_2d_mtime)

        with col_scatter:
            st.subheader("Mapeamento Sonoro 2D Interativo (t-SNE / UMAP)")
            if df_2d is not None and not df_2d.empty:
                fig_2d = px.scatter(
                    df_2d,
                    x="x",
                    y="y",
                    color="cluster_nome",
                    hover_name="filename",
                    hover_data={
                        "subgenero": True,
                        "cluster_nome": True,
                        "cluster_id": True,
                        "x": False,
                        "y": False
                    },
                    template="plotly_dark",
                    labels={"cluster_nome": "Cluster / Gênero"}
                )
                fig_2d.update_layout(
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#1a1a2e",
                    xaxis_title="Dimensão 1",
                    yaxis_title="Dimensão 2",
                    margin=dict(l=20, r=20, t=30, b=30),
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_2d, use_container_width=True)
            else:
                st.info("Coordenadas 2D ainda não geradas em 'clusters_2d.csv'.")

        with col_bar:
            st.subheader("Distribuição de Faixas por Cluster")
            clusters_dict = clusters_data.get("clusters", {})
            if clusters_dict:
                df_sizes = pd.DataFrame([
                    {
                        "Cluster": c_info.get("nome", cid),
                        "Faixas": c_info.get("size", 0),
                        "Pureza": c_info.get("pureza", 0.0)
                    }
                    for cid, c_info in clusters_dict.items()
                ]).sort_values("Faixas", ascending=True)

                fig_sizes = go.Figure()
                fig_sizes.add_trace(go.Bar(
                    y=df_sizes["Cluster"],
                    x=df_sizes["Faixas"],
                    orientation="h",
                    marker_color="#9b59b6",
                    text=df_sizes["Faixas"].map(lambda x: f"{x} ({x/clusters_data.get('total_tracks', 1):.1%})"),
                    textposition="auto"
                ))
                fig_sizes.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#1a1a2e",
                    xaxis_title="Número de Faixas",
                    margin=dict(l=20, r=20, t=30, b=30),
                    height=380
                )
                st.plotly_chart(fig_sizes, use_container_width=True)

        # ── Tabela Detalhada de Clusters ──────────────────────────────────────────
        st.subheader("Detalhamento dos Clusters e Faixas Medóides")
        clusters_table_data = []
        for cid, c_info in clusters_dict.items():
            # Formata a distribuição de gêneros
            dist = c_info.get("distribuicao_generos", {})
            dist_str = ", ".join([f"{g}: {count}" for g, count in list(dist.items())[:4]])
            if len(dist) > 4:
                dist_str += f" (+{len(dist)-4} outros)"

            med = c_info.get("medoid") or {}
            medoid_name = med.get("filename", "—")
            medoid_dist = f" (dist: {med.get('distance_to_centroid', 0.0):.3f})" if "distance_to_centroid" in med else ""

            clusters_table_data.append({
                "Cluster ID": cid,
                "Nome (Gênero)": c_info.get("nome", cid),
                "Nº de Faixas": c_info.get("size", 0),
                "Pureza": f"{c_info.get('pureza', 0.0):.1%}",
                "Gênero Predominante": c_info.get("genero_predominante", "—"),
                "Distribuição de Subgêneros": dist_str,
                "Faixa Medóide (Centroide)": f"{medoid_name}{medoid_dist}"
            })

        df_clusters_tbl = pd.DataFrame(clusters_table_data).sort_values("Nº de Faixas", ascending=False)
        st.dataframe(df_clusters_tbl, use_container_width=True, hide_index=True)

        # ── Gráfico k x Silhueta (Critério de Escolha de K) ────────────────────────
        st.subheader("Escolha do Número de Clusters: Gráfico K × Silhueta")
        df_k_mtime = K_SELECTION_CSV.stat().st_mtime if K_SELECTION_CSV.exists() else 0.0
        df_k_metrics = load_k_selection(df_k_mtime)

        if df_k_metrics is not None and not df_k_metrics.empty:
            best_row = df_k_metrics.loc[df_k_metrics["silhouette"].idxmax()]
            best_k = int(best_row["k"])

            fig_k = go.Figure()
            fig_k.add_trace(go.Scatter(
                x=df_k_metrics["k"],
                y=df_k_metrics["silhouette"],
                mode="lines+markers",
                name="Coeficiente de Silhueta",
                line=dict(color="#4C72B0", width=2.5),
                marker=dict(size=7)
            ))
            fig_k.add_trace(go.Scatter(
                x=[best_k],
                y=[best_row["silhouette"]],
                mode="markers+text",
                name=f"Melhor k={best_k}",
                marker=dict(color="#d9534f", size=14, symbol="star"),
                text=[f"Pico: k={best_k} (Silhueta: {best_row['silhouette']:.4f})"],
                textposition="top center"
            ))
            fig_k.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1a1a2e",
                xaxis_title="Número de Clusters (k)",
                yaxis_title="Coeficiente de Silhueta Médio",
                margin=dict(l=20, r=20, t=30, b=30),
                height=320,
                xaxis=dict(tickmode="linear", dtick=1)
            )
            st.plotly_chart(fig_k, use_container_width=True)
        elif K_SELECTION_PNG.exists():
            st.image(str(K_SELECTION_PNG), use_container_width=True)
        else:
            st.info("Métricas de busca de K não encontradas.")

    else:
        st.warning("⚠️ Dados de clusterização não encontrados. Execute a clusterização pelo `iniciar.bat` ou `clustering.py`.")


# ── Divisão Principal em Abas ─────────────────────────────────────────────────
tab_precisao, tab_ao_vivo = st.tabs(["📊 Precisão P@K & Clusters", "⚡ Recomendação ao Vivo"])

with tab_precisao:
    render_precision_tab()

with tab_ao_vivo:
    render_live_tab()

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Projeto TCC · Sistema de Recomendação Musical & Agrupamento Sonoro · CLAP + ChromaDB + MPD + Streamlit")
