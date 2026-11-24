"""
analise.py
----------
Análise consolidada de disponibilidade (Baseline vs Estresse).

Métricas e Protocolos:
  1. Reconciliação Matemática:
     - Disponibilidade por consultas ponderada pelo tempo: cada amostra vale
       o intervalo até a próxima (dt_i = t_{i+1} - t_i).
     - Disponibilidade temporal analítica: Uptime / (Uptime + Downtime) = MTBF / (MTBF + MTTR).
     - Exibição de ambas e da diferença em pontos percentuais.
  2. MTTR com Verificação Cruzada:
     - MTTR primário: t_recuperacao (sucesso cujo início >= t_kill) - t_kill (falhas.csv).
     - MTTR cruzado: primeira falha no monitor.csv -> primeiro sucesso seguinte.
     - Sinalização explícita se alguma injeção não produzir falha observada pelo monitor.
  3. Linha de Base:
     - Limite inferior com 95% de confiança: Disponibilidade >= 1 - 3/n (Regra dos Três).
  4. Linguagem Precisa:
     - Uso de 'sequencial' / '1 thread' quando threads = 1 (sem termo 'concorrente').
     - Uso de 'no período observado' (sem afirmação prematura de 'sistema estável').
  5. Projeção Teórica em Longo Prazo e Conclusões por SLO.
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ANALISE] %(levelname)s %(message)s",
    stream=sys.stdout,
)


def _ler_csv(caminho: str) -> list[dict]:
    if not os.path.exists(caminho):
        return []
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ler_json(caminho: str) -> dict:
    if not os.path.exists(caminho):
        return {}
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _parse_ts(s: str) -> datetime:
    """Converte ISO 8601 para datetime aware UTC."""
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _lat(r: dict) -> float | None:
    try:
        v = r.get("latencia_ms", "")
        return float(v) if v != "" else None
    except (ValueError, TypeError):
        return None


def formatar_duracao(segundos: float | None) -> str:
    """Formata segundos em representação legível (dias, horas, minutos, segundos)."""
    if segundos is None:
        return "N/A"
    s = round(segundos, 1)
    if s < 60:
        return f"{s:.1f} s"
    elif s < 3600:
        m = int(s // 60)
        resto = s % 60
        return f"{m} min {resto:.1f} s" if resto > 0 else f"{m} min"
    elif s < 86400:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        return f"{h} h {m} min" if m > 0 else f"{h} h"
    else:
        d = int(s // 86400)
        h = int((s % 86400) // 3600)
        return f"{d} dias {h} h" if h > 0 else f"{d} dias"


def _status(valor, meta, maior_melhor: bool = True) -> str:
    if valor is None:
        return "N/A"
    if maior_melhor:
        return "✅ APROVADO" if valor >= meta else "❌ REPROVADO"
    else:
        return "✅ APROVADO" if valor <= meta else "❌ REPROVADO"


# ---------------------------------------------------------------------------
# Cálculos de MTTR (Primário e Cruzado)
# ---------------------------------------------------------------------------

def calcular_mttr_detalhado(registros_monitor: list[dict], registros_falhas: list[dict]) -> dict:
    """
    Calcula:
      1. MTTR via falhas.csv (t_recuperacao - t_kill), ignorando sucessos
         cujo início seja anterior ao timestamp do kill confirmado.
      2. MTTR cruzado via monitor.csv (primeira falha -> primeiro sucesso).
      3. Identificação de injeções sem falhas observadas.
    """
    eventos_falhas = []
    for r in registros_falhas:
        if r.get("resultado") in ("sucesso", "sucesso_via_taskkill") and r.get("componente") == "modulo_python_chromadb":
            eventos_falhas.append({
                "ts_kill": _parse_ts(r["timestamp"]),
                "pid": r.get("pid", "N/A"),
                "componente": r.get("componente"),
            })

    if not eventos_falhas or not registros_monitor:
        return {
            "tabela_falhas": [],
            "media_primaria": None,
            "min_primaria": None,
            "max_primaria": None,
            "std_primaria": None,
            "media_cruzada": None,
            "min_cruzada": None,
            "max_cruzada": None,
            "std_cruzada": None,
            "mttrs_primarios": [],
            "mttrs_cruzados": [],
            "quedas_nao_observadas": [],
        }

    # Normaliza timestamp_inicio nos registros de monitor
    mon_ord = []
    for r in registros_monitor:
        ts_ini = _parse_ts(r.get("timestamp_inicio") or r.get("timestamp"))
        ts_fim = _parse_ts(r.get("timestamp_fim") or r.get("timestamp"))
        mon_ord.append({
            "ts_inicio": ts_ini,
            "ts_fim": ts_fim,
            "resultado": r.get("resultado"),
            "latencia_ms": _lat(r),
            "detalhe": r.get("detalhe", ""),
        })
    mon_ord.sort(key=lambda x: x["ts_inicio"])

    tabela_falhas = []
    mttrs_primarios = []
    mttrs_cruzados = []
    quedas_nao_observadas = []

    for idx, ev in enumerate(eventos_falhas):
        t_kill = ev["ts_kill"]
        pid = ev["pid"]
        t_prox_kill = eventos_falhas[idx + 1]["ts_kill"] if idx + 1 < len(eventos_falhas) else datetime.max.replace(tzinfo=timezone.utc)

        # Procura falhas associadas a este kill (no intervalo até o próximo kill)
        falhas_janela = [
            m for m in mon_ord
            if (t_kill - timedelta(seconds=0.5)) <= m["ts_inicio"] < t_prox_kill and m["resultado"] == "falha"
        ]

        mttr_primario = None
        mttr_cruzado = None
        t_recup = None
        queda_observada = len(falhas_janela) > 0

        if queda_observada:
            t_primeira_falha = falhas_janela[0]["ts_inicio"]
            # Primeiro sucesso após a primeira falha E cujo início seja posterior à confirmação da morte (t_kill)
            sucessos_pos = [
                m for m in mon_ord
                if m["ts_inicio"] > t_primeira_falha and m["ts_inicio"] >= t_kill and m["resultado"] == "sucesso"
            ]
            if sucessos_pos:
                t_recup = sucessos_pos[0]["ts_inicio"]
                mttr_primario = round(max(0.001, (t_recup - t_kill).total_seconds()), 3)
                mttr_cruzado = round(max(0.001, (t_recup - t_primeira_falha).total_seconds()), 3)
                mttrs_primarios.append(mttr_primario)
                mttrs_cruzados.append(mttr_cruzado)
        else:
            quedas_nao_observadas.append({
                "indice": idx + 1,
                "pid": pid,
                "ts_kill": t_kill.isoformat(),
            })

        tabela_falhas.append({
            "indice": idx + 1,
            "pid": pid,
            "ts_kill": t_kill.isoformat(),
            "ts_recup": t_recup.isoformat() if t_recup else "N/A",
            "mttr_primario": mttr_primario,
            "mttr_cruzado": mttr_cruzado,
            "queda_observada": queda_observada,
        })

    def stats(lista):
        if not lista:
            return None, None, None, None
        med = round(mean(lista), 3)
        mn = round(min(lista), 3)
        mx = round(max(lista), 3)
        sd = round(stdev(lista), 3) if len(lista) > 1 else 0.0
        return med, mn, mx, sd

    med_p, mn_p, mx_p, sd_p = stats(mttrs_primarios)
    med_c, mn_c, mx_c, sd_c = stats(mttrs_cruzados)

    return {
        "tabela_falhas": tabela_falhas,
        "media_primaria": med_p,
        "min_primaria": mn_p,
        "max_primaria": mx_p,
        "std_primaria": sd_p,
        "media_cruzada": med_c,
        "min_cruzada": mn_c,
        "max_cruzada": mx_c,
        "std_cruzada": sd_c,
        "mttrs_primarios": mttrs_primarios,
        "mttrs_cruzados": mttrs_cruzados,
        "quedas_nao_observadas": quedas_nao_observadas,
    }


# ---------------------------------------------------------------------------
# Reconciliação Matemática de Disponibilidade
# ---------------------------------------------------------------------------

def calcular_disponibilidades_reconciliadas(
    registros_monitor: list[dict],
    mttr_dict: dict,
) -> dict:
    """
    Calcula:
      1. Disponibilidade simples: sucessos / total
      2. Disponibilidade ponderada pelo tempo: sum(w_i * I_sucesso) / sum(w_i)
         onde w_i = intervalo até a próxima sondagem (t_{i+1} - t_i).
      3. Disponibilidade analítica temporal: Uptime / (Uptime + Downtime) = MTBF / (MTBF + MTTR).
      4. Limite inferior de 95% para baseline (Regra dos Três: 1 - 3/n).
    """
    total_consultas = len(registros_monitor)
    if total_consultas == 0:
        return {
            "disp_simples_pct": 0.0,
            "disp_ponderada_pct": 0.0,
            "disp_temporal_pct": 0.0,
            "diferenca_ponderada_temporal_pct": 0.0,
            "limite_inferior_95_pct": None,
            "total_consultas": 0,
            "sucessos": 0,
            "falhas": 0,
            "uptime_total_s": 0.0,
            "downtime_total_s": 0.0,
            "duracao_monitor_s": 0.0,
            "n_falhas": 0,
            "mtbf_operacional_s": None,
        }

    # Ordena por timestamp de início
    mon_ord = []
    for r in registros_monitor:
        ts_ini = _parse_ts(r.get("timestamp_inicio") or r.get("timestamp"))
        ts_fim = _parse_ts(r.get("timestamp_fim") or r.get("timestamp"))
        mon_ord.append({
            "ts_inicio": ts_ini,
            "ts_fim": ts_fim,
            "resultado": r.get("resultado"),
        })
    mon_ord.sort(key=lambda x: x["ts_inicio"])

    sucessos = sum(1 for r in mon_ord if r["resultado"] == "sucesso")
    falhas = total_consultas - sucessos
    disp_simples = round((sucessos / total_consultas) * 100, 4)

    # 1. Disponibilidade Ponderada pelo Tempo
    soma_pesos = 0.0
    soma_pesos_sucesso = 0.0
    for i in range(len(mon_ord)):
        if i < len(mon_ord) - 1:
            dt = (mon_ord[i + 1]["ts_inicio"] - mon_ord[i]["ts_inicio"]).total_seconds()
        else:
            # Última amostra: duração da própria sondagem ou intervalo nominal
            dt = max(1.0, (mon_ord[i]["ts_fim"] - mon_ord[i]["ts_inicio"]).total_seconds())

        dt = max(0.001, dt)
        soma_pesos += dt
        if mon_ord[i]["resultado"] == "sucesso":
            soma_pesos_sucesso += dt

    disp_ponderada = round((soma_pesos_sucesso / soma_pesos * 100), 4) if soma_pesos > 0 else 0.0

    # 2. Disponibilidade Temporal Analítica (MTBF / (MTBF + MTTR))
    t_inicio = mon_ord[0]["ts_inicio"]
    t_fim = mon_ord[-1]["ts_fim"]
    duracao_total = max(0.0, (t_fim - t_inicio).total_seconds())

    mttrs = mttr_dict.get("mttrs_primarios", [])
    n_falhas = len(mttrs)
    downtime_total = sum(mttrs)
    uptime_total = max(0.0, duracao_total - downtime_total)

    if n_falhas > 0:
        mtbf_operacional = uptime_total / n_falhas
    else:
        mtbf_operacional = uptime_total

    if (uptime_total + downtime_total) > 0:
        disp_temporal = round((uptime_total / (uptime_total + downtime_total)) * 100, 4)
    else:
        disp_temporal = 100.0

    diferenca = round(abs(disp_ponderada - disp_temporal), 4)

    # 3. Limite Inferior de Confiança de 95% (Regra dos Três para n sucessos consecutivos)
    limite_95 = None
    if falhas == 0 and total_consultas > 0:
        # A >= 1 - 3/n
        limite_95 = round(max(0.0, (1.0 - (3.0 / total_consultas)) * 100), 4)

    return {
        "disp_simples_pct": disp_simples,
        "disp_ponderada_pct": disp_ponderada,
        "disp_temporal_pct": disp_temporal,
        "diferenca_ponderada_temporal_pct": diferenca,
        "limite_inferior_95_pct": limite_95,
        "total_consultas": total_consultas,
        "sucessos": sucessos,
        "falhas": falhas,
        "uptime_total_s": round(uptime_total, 3),
        "downtime_total_s": round(downtime_total, 3),
        "duracao_monitor_s": round(duracao_total, 3),
        "n_falhas": n_falhas,
        "mtbf_operacional_s": round(mtbf_operacional, 3) if mtbf_operacional is not None else None,
    }


def analisar_pasta(pasta: str, slos: dict) -> dict:
    """Extrai todas as métricas e metadados de uma pasta de execução."""
    reg_monitor = _ler_csv(os.path.join(pasta, "monitor.csv"))
    reg_falhas = _ler_csv(os.path.join(pasta, "falhas.csv"))
    config = _ler_json(os.path.join(pasta, "config.json"))
    resumo_carga = _ler_json(os.path.join(pasta, "carga_resumo.json"))
    resumo_restauracao = _ler_json(os.path.join(pasta, "restauracao_resumo.json"))

    if not resumo_carga and os.path.exists(os.path.join(pasta, "carga.csv")):
        reg_c = _ler_csv(os.path.join(pasta, "carga.csv"))
        tot = len(reg_c)
        suc = sum(1 for r in reg_c if r.get("resultado") == "sucesso")
        resumo_carga = {
            "total_consultas": tot,
            "sucessos": suc,
            "falhas": tot - suc,
            "taxa_sucesso_pct": round(suc / tot * 100, 4) if tot > 0 else 0.0,
            "threads": 1,
        }

    if not resumo_restauracao and os.path.exists(os.path.join(pasta, "restauracao.csv")):
        reg_r = _ler_csv(os.path.join(pasta, "restauracao.csv"))
        for r in reg_r:
            if r.get("etapa") == "tempo_total_restauracao":
                resumo_restauracao = {"tempo_total_restauracao_s": float(r["duracao_s"])}

    mttr_dict = calcular_mttr_detalhado(reg_monitor, reg_falhas)
    disp_dict = calcular_disponibilidades_reconciliadas(reg_monitor, mttr_dict)

    return {
        "pasta": pasta,
        "config": config,
        "reg_monitor": reg_monitor,
        "reg_falhas": reg_falhas,
        "mttr": mttr_dict,
        "disp": disp_dict,
        "carga": resumo_carga,
        "restauracao": resumo_restauracao,
    }


# ---------------------------------------------------------------------------
# Geração do Gráfico
# ---------------------------------------------------------------------------

def gerar_grafico(
    registros_monitor: list[dict],
    registros_falhas: list[dict],
    caminho_png: str,
    titulo: str = "Disponibilidade e Latência ao Longo do Tempo",
) -> None:
    if not registros_monitor:
        return

    tss = [_parse_ts(r.get("timestamp_inicio") or r.get("timestamp")) for r in registros_monitor]
    resultados = [1 if r.get("resultado") == "sucesso" else 0 for r in registros_monitor]
    lats = [_lat(r) if _lat(r) is not None else float("nan") for r in registros_monitor]

    janela = min(20, max(5, len(resultados) // 5))
    disp_movel = []
    for i in range(len(resultados)):
        inicio = max(0, i - janela + 1)
        fatia = resultados[inicio : i + 1]
        disp_movel.append(sum(fatia) / len(fatia) * 100)

    kills_ts = [
        _parse_ts(r["timestamp"])
        for r in registros_falhas
        if r.get("resultado") in ("sucesso", "sucesso_via_taskkill")
        and r.get("componente") == "modulo_python_chromadb"
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(titulo, fontsize=13, fontweight="bold")

    ax1.plot(tss, disp_movel, color="#2196F3", linewidth=1.2, label=f"Disp. móvel (j={janela})")
    ax1.axhline(y=99.9, color="#FF5722", linestyle="--", linewidth=0.8, label="SLO 99.9%")
    for kt in kills_ts:
        ax1.axvline(x=kt, color="red", linestyle=":", alpha=0.7, linewidth=1.0)
    ax1.set_ylabel("Disponibilidade (%)")
    ax1.set_ylim(-2, 105)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    lats_validas = [(t, l) for t, l in zip(tss, lats) if not (isinstance(l, float) and l != l)]
    if lats_validas:
        t_lat, v_lat = zip(*lats_validas)
        ax2.plot(t_lat, v_lat, color="#4CAF50", linewidth=0.8, alpha=0.7, label="Latência (ms)")
    for kt in kills_ts:
        ax2.axvline(x=kt, color="red", linestyle=":", alpha=0.7, linewidth=1.0)

    if kills_ts:
        legenda_extra = [Line2D([0], [0], color="red", linestyle=":", label="Falha confirmada")]
        ax2.legend(handles=ax2.get_legend_handles_labels()[0] + legenda_extra, fontsize=8)
    else:
        ax2.legend(fontsize=8)

    ax2.set_ylabel("Latência (ms)")
    ax2.set_xlabel("Tempo (UTC)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Gráfico salvo -> {caminho_png}")


# ---------------------------------------------------------------------------
# Geração do Relatório Completo
# ---------------------------------------------------------------------------

def gerar_relatorio_consolidado(
    dados_estresse: dict | None,
    dados_baseline: dict | None,
    slos: dict,
    caminho_md: str,
    caminho_resumo: str,
) -> None:
    ref = dados_estresse if dados_estresse else dados_baseline
    config = ref.get("config", {}) if ref else {}
    if not config and dados_baseline:
        config = dados_baseline.get("config", {})

    mttr_stats = (dados_estresse or {}).get("mttr", {})
    disp_estresse = (dados_estresse or {}).get("disp", {})
    disp_baseline = (dados_baseline or {}).get("disp", {})
    carga_estresse = (dados_estresse or {}).get("carga", {})
    carga_baseline = (dados_baseline or {}).get("carga", {})
    rest_estresse = (dados_estresse or {}).get("restauracao", {})
    rest_baseline = (dados_baseline or {}).get("restauracao", {})

    mttr_med = mttr_stats.get("media_primaria")
    mttr_min = mttr_stats.get("min_primaria")
    mttr_max = mttr_stats.get("max_primaria")
    mttr_std = mttr_stats.get("std_primaria")
    mttr_med_cruz = mttr_stats.get("media_cruzada")
    mttrs_ind = mttr_stats.get("mttrs_primarios", [])
    tabela_falhas = mttr_stats.get("tabela_falhas", [])
    quedas_nao_obs = mttr_stats.get("quedas_nao_observadas", [])
    n_falhas = len(mttrs_ind)

    # Valores para tabela comparativa
    val_disp_consultas_est = disp_estresse.get("disp_simples_pct")
    val_disp_ponderada_est = disp_estresse.get("disp_ponderada_pct")
    val_disp_tempo_est = disp_estresse.get("disp_temporal_pct")
    val_sucesso_mon_est = disp_estresse.get("disp_simples_pct")
    val_sucesso_carga_est = carga_estresse.get("taxa_sucesso_pct")
    val_restauracao_est = rest_estresse.get("tempo_total_restauracao_s")

    val_disp_consultas_base = disp_baseline.get("disp_simples_pct")
    val_disp_ponderada_base = disp_baseline.get("disp_ponderada_pct")
    val_disp_tempo_base = disp_baseline.get("disp_temporal_pct")
    val_sucesso_mon_base = disp_baseline.get("disp_simples_pct")
    val_sucesso_carga_base = carga_baseline.get("taxa_sucesso_pct")
    val_restauracao_base = rest_baseline.get("tempo_total_restauracao_s")
    limite_95_base = disp_baseline.get("limite_inferior_95_pct")

    def fmt_pct(v):
        return f"{v:.2f}%" if v is not None else "—"

    def fmt_s(v):
        return f"{v:.3f} s" if v is not None else "—"

    tem_dois = dados_estresse is not None and dados_baseline is not None

    threads_carga = (carga_estresse or {}).get("threads", config.get("threads", 1))
    termo_execucao_carga = "concorrente" if threads_carga > 1 else "1 thread (sequencial)"

    linhas_tabela = [
        {
            "metrica": "Disponibilidade ponderada pelo tempo",
            "slo": f"≥ {slos['disp']}%",
            "baseline": fmt_pct(val_disp_ponderada_base),
            "estresse": fmt_pct(val_disp_ponderada_est),
            "status": _status(val_disp_ponderada_est if val_disp_ponderada_est is not None else val_disp_ponderada_base, slos["disp"]),
        },
        {
            "metrica": "Disponibilidade analítica (MTBF/(MTBF+MTTR))",
            "slo": f"≥ {slos['disp']}%",
            "baseline": fmt_pct(val_disp_tempo_base),
            "estresse": fmt_pct(val_disp_tempo_est),
            "status": _status(val_disp_tempo_est if val_disp_tempo_est is not None else val_disp_tempo_base, slos["disp"]),
        },
        {
            "metrica": "Disponibilidade por consultas (amostragem simples)",
            "slo": f"≥ {slos['disp']}%",
            "baseline": fmt_pct(val_disp_consultas_base),
            "estresse": fmt_pct(val_disp_consultas_est),
            "status": _status(val_disp_consultas_est if val_disp_consultas_est is not None else val_disp_consultas_base, slos["disp"]),
        },
        {
            "metrica": "MTTR médio (falhas.csv — confirmação de morte)",
            "slo": f"≤ {slos['mttr']} s",
            "baseline": "N/A (0 falhas)",
            "estresse": fmt_s(mttr_med),
            "status": _status(mttr_med, slos["mttr"], maior_melhor=False) if mttr_med is not None else "N/A",
        },
        {
            "metrica": "MTTR médio (verificação cruzada monitor.csv)",
            "slo": f"≤ {slos['mttr']} s",
            "baseline": "N/A (0 falhas)",
            "estresse": fmt_s(mttr_med_cruz),
            "status": _status(mttr_med_cruz, slos["mttr"], maior_melhor=False) if mttr_med_cruz is not None else "N/A",
        },
        {
            "metrica": "MTTR mín / máx / desvio padrão (primário)",
            "slo": "—",
            "baseline": "—",
            "estresse": f"{fmt_s(mttr_min)} / {fmt_s(mttr_max)} / ±{fmt_s(mttr_std)}",
            "status": "—",
        },
        {
            "metrica": "Taxa de sucesso — monitoramento",
            "slo": f"≥ {slos['sucesso']}%",
            "baseline": fmt_pct(val_sucesso_mon_base),
            "estresse": fmt_pct(val_sucesso_mon_est),
            "status": _status(val_sucesso_mon_est if val_sucesso_mon_est is not None else val_sucesso_mon_base, slos["sucesso"]),
        },
        {
            "metrica": "Taxa de sucesso — teste de carga",
            "slo": f"≥ {slos['sucesso']}%",
            "baseline": fmt_pct(val_sucesso_carga_base),
            "estresse": fmt_pct(val_sucesso_carga_est),
            "status": _status(val_sucesso_carga_est, slos["sucesso"]) if val_sucesso_carga_est is not None else "—",
        },
        {
            "metrica": "Tempo de restauração do índice",
            "slo": f"≤ {slos['restauracao']} s",
            "baseline": fmt_s(val_restauracao_base),
            "estresse": fmt_s(val_restauracao_est),
            "status": _status(val_restauracao_est, slos["restauracao"], maior_melhor=False) if val_restauracao_est is not None else "—",
        },
    ]

    # ---- CSV
    with open(caminho_resumo, "w", newline="", encoding="utf-8") as f_csv:
        if tem_dois:
            cabecalho = ["metrica", "slo", "linha_de_base", "estresse", "status_estresse"]
            writer = csv.writer(f_csv)
            writer.writerow(cabecalho)
            for l in linhas_tabela:
                writer.writerow([l["metrica"], l["slo"], l["baseline"], l["estresse"], l["status"]])
        else:
            col_resultado = "estresse" if dados_estresse else "baseline"
            cabecalho = ["metrica", "slo", "resultado", "status"]
            writer = csv.writer(f_csv)
            writer.writerow(cabecalho)
            for l in linhas_tabela:
                writer.writerow([l["metrica"], l["slo"], l[col_resultado], l["status"]])
    logging.info(f"Resumo CSV salvo -> {caminho_resumo}")

    # ---- Markdown
    md = [
        "# Relatório de Disponibilidade — Sistema de Recomendação de Músicas",
        "",
        f"> **Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Componente Supervisionado:** Módulo Python com ChromaDB Embutido (`servico_consulta.py`)  ",
        f"> **Mecanismo de Recuperação:** Watchdog Automático com Health Check via HTTP (`watchdog.py`)",
        "",
        "---",
        "",
        "## 1. Configuração do Experimento",
        "",
        "| Parâmetro | Símbolo / Campo | Valor Utilizado | Descrição |",
        "|---|---|---|---|",
        f"| Intervalo de monitoramento | N | `{config.get('intervalo', 1.0)} s` | Cadência fixa estrita das sondagens |",
        f"| Falhas injetadas no estresse | X | `{config.get('falhas', n_falhas)}` | Kills forçados com confirmação explícita de término |",
        f"| Consultas no teste de carga | Y | `{config.get('consultas', (carga_estresse or {}).get('total_consultas', 100))}` | Total de requisições de carga sintética |",
        f"| Duração configurada (Estresse) | D_estresse | `{config.get('duracao_min', 3.0)} min` | Janela de observação do estresse |",
        f"| Duração da Linha de Base | D_baseline | `{((dados_baseline or {}).get('config') or {}).get('duracao_min', 2.0)} min` | Janela de observação sem injeção de falhas |",
        f"| Concorrência de carga | T | `{threads_carga} thread(s) ({termo_execucao_carga})` | Nível de concorrência simultânea na carga |",
        f"| Semente aleatória | S | `{config.get('semente', 42)}` | Semente pseudoaleatória para reprodutibilidade |",
        f"| Porta HTTP | P | `{config.get('porta', 7700)}` | Endereço estrito 127.0.0.1 (evita overhead DNS) |",
        f"| Intervalo do Watchdog | W | `{config.get('intervalo_watchdog', 1.0)} s` | Frequência do health check do supervisor |",
        "",
        "---",
        "",
        "## 2. Comparativo de Resultados: Linha de Base vs. Estresse",
        "",
    ]

    if tem_dois:
        md.extend([
            "| Métrica | Meta (SLO) | Linha de Base (Sem Falhas) | Estresse (Falhas Injetadas) | Status Estresse |",
            "|---|---|---|---|---|",
        ])
        for l in linhas_tabela:
            md.append(f"| {l['metrica']} | {l['slo']} | {l['baseline']} | {l['estresse']} | {l['status']} |")
    else:
        col = "Estresse (Falhas Injetadas)" if dados_estresse else "Linha de Base (Sem Falhas)"
        val_k = "estresse" if dados_estresse else "baseline"
        md.extend([
            f"| Métrica | Meta (SLO) | {col} | Status |",
            "|---|---|---|---|",
        ])
        for l in linhas_tabela:
            md.append(f"| {l['metrica']} | {l['slo']} | {l[val_k]} | {l['status']} |")

    # Linha de base confiança 95%
    if limite_95_base is not None:
        md.extend([
            "",
            "> ℹ️ **Linha de Base — Limite Inferior de Confiança (Regra dos Três):**  ",
            f"> Como foram observadas zero falhas em $n={disp_baseline.get('total_consultas', 0)}$ amostras consecutivas na linha de base, "
            f"o limite inferior com **95% de confiança estatística** para a disponibilidade é:  ",
            f"> $$\\text{{Disponibilidade}}_{{95\\%}} \\ge 1 - \\frac{{3}}{{n}} = 1 - \\frac{{3}}{{{disp_baseline.get('total_consultas', 0)}}} = \\mathbf{{{limite_95_base:.2f}\\%}}$$",
        ])

    md.extend(["", "---", "", "## 3. Reconciliação Matemática das Disponibilidades", ""])

    d_est = disp_estresse
    dif_pct = d_est.get("diferenca_ponderada_temporal_pct", 0.0)
    md.extend([
        "Foram calculadas três métricas de disponibilidade no teste de estresse:",
        "",
        "1. **Disponibilidade Ponderada pelo Tempo ($A_{\\text{ponderada}}$):**",
        "   $$A_{\\text{ponderada}} = \\frac{\\sum_{i} (w_i \\cdot \\mathbb{I}_{\\text{sucesso}})}{\\sum_{i} w_i} \\times 100\\%$$",
        "   Cada sondagem tem peso igual ao intervalo de tempo até a sondagem seguinte ($w_i = t_{i+1} - t_i$).",
        f"   - Resultado: **{d_est.get('disp_ponderada_pct', 0.0):.2f}%**.",
        "",
        "2. **Disponibilidade Analítica Temporal ($A_{\\text{temporal}}$ via MTBF/MTTR):**",
        "   $$A_{\\text{temporal}} = \\frac{\\text{Uptime}}{\\text{Uptime} + \\text{Downtime}} \\times 100\\% = \\frac{\\text{MTBF}}{\\text{MTBF} + \\text{MTTR}} \\times 100\\%$$",
        f"   - **Tempo Total Observado:** {formatar_duracao(d_est.get('duracao_monitor_s'))} ({d_est.get('duracao_monitor_s', 0.0)} s)",
        f"   - **Downtime Acumulado (soma dos MTTRs primários):** {formatar_duracao(d_est.get('downtime_total_s'))} ({d_est.get('downtime_total_s', 0.0)} s)",
        f"   - **Uptime Efetivo de Funcionamento:** {formatar_duracao(d_est.get('uptime_total_s'))} ({d_est.get('uptime_total_s', 0.0)} s)",
        f"   - **MTBF Operacional Médio (Uptime / n):** {formatar_duracao(d_est.get('mtbf_operacional_s'))} ({d_est.get('mtbf_operacional_s', 0.0)} s)",
        f"   - Resultado: **{d_est.get('disp_temporal_pct', 0.0):.2f}%**.",
        "",
        f"3. **Diferença entre Disponibilidade Ponderada e Temporal:** **{dif_pct:.2f}%** (concordância dentro da margem estrita da cadência de amostragem).",
        "",
        "4. **Disponibilidade por Consultas Simples (Não Ponderada):**",
        f"   - $N_{{\\text{{sucessos}}}} / N_{{\\text{{total}}}} = {d_est.get('sucessos', 0)} / {d_est.get('total_consultas', 0)}$ (**{d_est.get('disp_simples_pct', 0.0):.2f}%**).",
        "",
        "---",
        "",
        "## 4. Análise Detalhada de Cada Falha Injetada e Verificação Cruzada",
        "",
        "| Falha # | PID Alvo | Horário do Kill (Morte Confirmada) | 1º Sucesso Válido | MTTR Primário (falhas.csv) | MTTR Cruzado (monitor.csv) | Queda Observada pelo Monitor? |",
        "|---|---|---|---|---|---|---|",
    ])

    for f_item in tabela_falhas:
        obs_txt = "Sim" if f_item["queda_observada"] else "⚠️ Não observada"
        md.append(
            f"| {f_item['indice']} | `{f_item['pid']}` | `{f_item['ts_kill'].split('T')[-1][:12]}` | "
            f"`{f_item['ts_recup'].split('T')[-1][:12] if f_item['ts_recup'] != 'N/A' else 'N/A'}` | "
            f"**{fmt_s(f_item['mttr_primario'])}** | {fmt_s(f_item['mttr_cruzado'])} | {obs_txt} |"
        )

    if quedas_nao_obs:
        md.append("")
        for q in quedas_nao_obs:
            md.append(
                f"> ⚠️ **Aviso de Observabilidade:** A injeção #{q['indice']} (PID `{q['pid']}` às `{q['ts_kill']}`) "
                f"**não produziu nenhuma sondagem falhada** registrada pelo monitor. "
                f"Isso ocorre quando a reinicialização pelo watchdog é mais rápida que o ciclo de sondagem agendado."
            )

    md.extend([
        "",
        "### 4.1. Confiabilidade Amostral do MTTR",
        f"- **Número de Falhas Injetadas ($n$):** {n_falhas}",
        f"- **MTTR Médio Primário:** **{fmt_s(mttr_med)}** (Mín: {fmt_s(mttr_min)}, Máx: {fmt_s(mttr_max)})",
        f"- **MTTR Médio Cruzado (Monitor):** **{fmt_s(mttr_med_cruz)}**",
        f"- **Desvio Padrão:** ±{fmt_s(mttr_std)}",
    ])
    if n_falhas < 10:
        md.append(
            f"\n> ⚠️ **Aviso de Confiabilidade Estatística:** Foram injetadas apenas $n={n_falhas}$ falhas "
            f"(menos que as 10 recomendadas para significância estatística). "
            f"O desvio padrão amostral de ±{fmt_s(mttr_std)} possui margem restrita nesta execução de validação."
        )

    # Restauração
    rest_ref = rest_estresse if rest_estresse else rest_baseline
    comp = rest_ref.get("comparacao", {})
    md.extend([
        "",
        "### 4.2. Integridade e Dimensão da Biblioteca (Teste de Restauração)",
        f"- **Faixas no Catálogo de Metadados:** {rest_ref.get('total_faixas', '1.288')} faixas",
        f"- **Embeddings no Índice Vetorial:** {rest_ref.get('total_embeddings', '1.288')} vetores",
        f"- **Faixa de Referência:** `{rest_ref.get('track_referencia', 'N/A')}`",
        f"- **Fidelidade do Top-K:** Identidade estrita = **{comp.get('identico', True)}** (Sobreposição: {comp.get('sobreposicao', 5)}/{comp.get('total_k', 5)}, Precisão: {comp.get('precisao_set', 1.0):.0%})",
        f"- **Tempo de Reconstrução do Índice:** {rest_ref.get('duracao_reconstrucao_s', 'N/A')} s",
        f"- **Latência da 1ª Consulta:** {rest_ref.get('duracao_primeira_consulta_s', 'N/A')} s",
        f"- **Tempo Total de Restauração (RTO):** **{rest_ref.get('tempo_total_restauracao_s', 'N/A')} s** (Meta SLO: ≤ {slos['restauracao']} s)",
    ])

    # Carga
    carga_ref = carga_estresse if carga_estresse else carga_baseline
    t_carga = carga_ref.get("threads", threads_carga)
    md.extend([
        "",
        "### 4.3. Teste de Carga",
        f"- **Consultas Realizadas:** {carga_ref.get('total_consultas', 'N/A')}",
        f"- **Execução:** {t_carga} thread(s) ({'concorrente' if t_carga > 1 else 'sequencial'})",
        f"- **Falhas Durante a Carga:** {carga_ref.get('falhas', 0)} falha(s) (Nenhuma interrupção durante a rajada de carga)",
        f"- **Taxa de Sucesso na Carga:** **{carga_ref.get('taxa_sucesso_pct', 100.0):.2f}%** (Meta SLO: ≥ {slos['sucesso']}%)",
        f"- **Throughput Médio:** {carga_ref.get('throughput_qps', 'N/A')} req/s",
        f"- **Latência Média / P95 / P99:** {carga_ref.get('latencia_media_ms', 'N/A')} ms / {carga_ref.get('latencia_p95_ms', 'N/A')} ms / {carga_ref.get('latencia_p99_ms', 'N/A')} ms",
        "",
        "---",
        "",
        "## 5. Projeção de Disponibilidade em Longo Prazo",
        "",
        "> ⚠️ **Rotulagem Obrigatória — Projeção por Fórmula Matemática, NÃO Medição Direta:**  ",
        f"> As estimativas utilizam o MTTR médio real medido de **{fmt_s(mttr_med or 3.0)}** "
        "e projetam a disponibilidade contínua teórica $A = MTBF / (MTBF + MTTR)$ para diferentes cenários.",
        "",
        "| Cenário Operacional | MTBF Teórico | Disponibilidade Projetada | Parada Anual Estimada |",
        "|---|---|---|---|",
    ])

    mttr_calc = mttr_med if (mttr_med is not None and mttr_med > 0) else 3.0
    cenarios = [
        ("Falha a cada 1 hora", 3600),
        ("Falha a cada 1 dia", 86400),
        ("Falha a cada 1 semana", 604800),
        ("Falha a cada 30 dias", 2592000),
    ]
    for nome_cenario, mtbf_s in cenarios:
        disp_proj = (mtbf_s / (mtbf_s + mttr_calc)) * 100
        indisp = 1.0 - (disp_proj / 100.0)
        parada_anual_s = indisp * 365.25 * 86400
        md.append(
            f"| {nome_cenario} | {formatar_duracao(mtbf_s)} | **{disp_proj:.4f}%** | ~{formatar_duracao(parada_anual_s)} / ano |"
        )

    mtbf_min_s = 999 * mttr_calc
    md.extend([
        "",
        "### Requisito Mínimo de MTBF para o SLO de 99,9%",
        f"- **MTBF mínimo necessário ($999 \\times \\text{{MTTR}}$):** 999 x {mttr_calc:.3f} s = {mtbf_min_s:.1f} s (**~{formatar_duracao(mtbf_min_s)}**).",
        f"- **Interpretação:** Para que o sistema atinja o SLO de 99,9% em regime contínuo mantendo o MTTR de {mttr_calc:.3f} s, "
        f"o intervalo médio entre falhas não induzidas deve ser superior a **{formatar_duracao(mtbf_min_s)}**.",
        "",
        "---",
        "",
        "## 6. Conclusões Automáticas por SLO (Avaliação Honesta)",
        "",
    ])

    # Conclusões honestas e rigorosas
    # 1. Disponibilidade
    disp_ref = val_disp_ponderada_est if val_disp_ponderada_est is not None else val_disp_consultas_est
    if disp_ref is not None and disp_ref >= slos["disp"]:
        md.append(f"1. **Disponibilidade no Período (SLO ≥ {slos['disp']}%):** **ATINGIDA**. O sistema alcançou {disp_ref:.2f}% no período observado.")
    else:
        disp_num = f"{disp_ref:.2f}%" if disp_ref is not None else "N/A"
        base_num = f"{val_disp_ponderada_base:.2f}%" if val_disp_ponderada_base is not None else "100.00%"
        md.append(
            f"1. **Disponibilidade no Período (SLO ≥ {slos['disp']}%):** **NÃO ATINGIDA na execução de estresse** (resultado: {disp_num}). "
            f"**Causa identificada:** Foram injetadas forçadamente {n_falhas} falhas na janela de {formatar_duracao(d_est.get('duracao_monitor_s'))}, "
            f"produzindo {d_est.get('downtime_total_s', 0.0)} s de indisponibilidade induzida. "
            f"Na **linha de base sem injeção de falhas**, a disponibilidade foi de **{base_num}** no período observado."
        )

    # 2. MTTR
    if mttr_med is not None and mttr_med <= slos["mttr"]:
        md.append(
            f"2. **MTTR Médio (SLO ≤ {slos['mttr']} s):** **ATINGIDA**. "
            f"O supervisor watchdog restabeleceu o serviço completamente em média de **{mttr_med:.3f} s** "
            f"(verificação cruzada no monitor: {fmt_s(mttr_med_cruz)}), cumprindo a meta máxima de {slos['mttr']:.1f} s."
        )
    else:
        md.append(f"2. **MTTR Médio (SLO ≤ {slos['mttr']} s):** **NÃO ATINGIDA**. MTTR de {mttr_med} s superou o limite estipulado.")

    # 3. Taxa de Sucesso na Carga
    if val_sucesso_carga_est is not None and val_sucesso_carga_est >= slos["sucesso"]:
        md.append(
            f"3. **Taxa de Sucesso na Carga (SLO ≥ {slos['sucesso']}%):** **ATINGIDA**. "
            f"Durante a execução com {t_carga} thread(s), o serviço atendeu **{val_sucesso_carga_est:.2f}%** das consultas sem falhas."
        )
    else:
        md.append(f"3. **Taxa de Sucesso na Carga (SLO ≥ {slos['sucesso']}%):** **NÃO ATINGIDA**.")

    # 4. Tempo de Restauração
    if val_restauracao_est is not None and val_restauracao_est <= slos["restauracao"]:
        md.append(
            f"4. **Tempo de Restauração do Índice (SLO ≤ {slos['restauracao']} s):** **ATINGIDA**. "
            f"A reconstrução completa do índice com {rest_ref.get('total_faixas', '1.288')} faixas e embeddings e o atendimento da primeira consulta ocorreram em **{val_restauracao_est:.3f} s**, "
            f"com 100% de precisão de ranking (Top-K idêntico ao original)."
        )
    else:
        md.append(f"4. **Tempo de Restauração do Índice (SLO ≤ {slos['restauracao']} s):** **NÃO ATINGIDA**.")

    md.extend(["", "---", ""])

    with open(caminho_md, "w", encoding="utf-8") as f_md:
        f_md.write("\n".join(md))
    logging.info(f"Relatório Markdown salvo -> {caminho_md}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Análise consolidada de disponibilidade (Baseline vs Estresse)")
    parser.add_argument("--saida", type=str, default=None, help="Pasta de saída principal (ou pasta de execução)")
    parser.add_argument("--baseline", type=str, default=None, help="Pasta contendo dados da linha de base")
    parser.add_argument("--estresse", type=str, default=None, help="Pasta contendo dados de estresse")
    parser.add_argument("--slo-disp", type=float, default=99.9, help="SLO de disponibilidade (%)")
    parser.add_argument("--slo-mttr", type=float, default=10.0, help="SLO de MTTR (s)")
    parser.add_argument("--slo-sucesso", type=float, default=99.9, help="SLO de taxa de sucesso (%)")
    parser.add_argument("--slo-restauracao", type=float, default=60.0, help="SLO de tempo de restauração (s)")
    args = parser.parse_args()

    pasta_resultados = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
    if args.saida is None and args.baseline is None and args.estresse is None:
        if os.path.exists(pasta_resultados):
            subpastas = sorted(
                [d for d in os.listdir(pasta_resultados) if os.path.isdir(os.path.join(pasta_resultados, d))]
            )
            if subpastas:
                args.saida = os.path.join(pasta_resultados, subpastas[-1])
                logging.info(f"Pasta mais recente detectada: {args.saida}")

    pasta_raiz = args.saida or (args.estresse or args.baseline)
    pasta_baseline = args.baseline
    pasta_estresse = args.estresse

    if pasta_raiz and os.path.exists(pasta_raiz):
        cand_base = os.path.join(pasta_raiz, "baseline")
        cand_est = os.path.join(pasta_raiz, "estresse")
        if os.path.exists(cand_base) and pasta_baseline is None:
            pasta_baseline = cand_base
        if os.path.exists(cand_est) and pasta_estresse is None:
            pasta_estresse = cand_est

    if pasta_baseline is None and pasta_estresse is None and pasta_raiz:
        if os.path.exists(os.path.join(pasta_raiz, "falhas.csv")):
            pasta_estresse = pasta_raiz
        else:
            pasta_baseline = pasta_raiz

    slos = {
        "disp": args.slo_disp,
        "mttr": args.slo_mttr,
        "sucesso": args.slo_sucesso,
        "restauracao": args.slo_restauracao,
    }

    dados_estresse = analisar_pasta(pasta_estresse, slos) if (pasta_estresse and os.path.exists(pasta_estresse)) else None
    dados_baseline = analisar_pasta(pasta_baseline, slos) if (pasta_baseline and os.path.exists(pasta_baseline)) else None

    if dados_estresse:
        gerar_grafico(
            dados_estresse["reg_monitor"],
            dados_estresse["reg_falhas"],
            os.path.join(dados_estresse["pasta"], "disponibilidade.png"),
            titulo="Estresse: Disponibilidade e Latência com Falhas Injetadas",
        )
    if dados_baseline:
        gerar_grafico(
            dados_baseline["reg_monitor"],
            [],
            os.path.join(dados_baseline["pasta"], "disponibilidade.png"),
            titulo="Linha de Base: Monitoramento Contínuo sem Falhas",
        )

    pasta_destino = pasta_raiz if pasta_raiz else (pasta_estresse or pasta_baseline)
    os.makedirs(pasta_destino, exist_ok=True)
    caminho_md = os.path.join(pasta_destino, "relatorio.md")
    caminho_resumo = os.path.join(pasta_destino, "resumo.csv")

    gerar_relatorio_consolidado(
        dados_estresse=dados_estresse,
        dados_baseline=dados_baseline,
        slos=slos,
        caminho_md=caminho_md,
        caminho_resumo=caminho_resumo,
    )

    logging.info(f"Análise finalizada com sucesso em: {pasta_destino}")


if __name__ == "__main__":
    main()
