"""Avalia o modelo nas últimas semanas da padaria de exemplo e gera métricas e gráficos.

Uso:
    python -m demanda.avaliar
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from demanda import modelo as m

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
FIG = RAIZ / "reports" / "figuras"
AZUL, LARANJA, CINZA, TINTA, TINTA2, GRADE = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
SEMANAS = 12


def brl(v: float, casas: int = 0) -> str:
    return "R$ " + f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def estilo(ax, titulo: str, subtitulo: str | None = None):
    ax.set_title(titulo, loc="left", fontsize=12, fontweight="bold", color=TINTA, pad=18 if subtitulo else 8)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=9, color=TINTA2)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.tick_params(colors=TINTA2, labelsize=9)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    vendas = pd.read_csv(RAIZ / "api" / "exemplos" / "vendas_padaria.csv")
    r = m.backtest(vendas, semanas=SEMANAS)

    trad = m.resultado_financeiro(r, "tradicional_produzir")
    mod = m.resultado_financeiro(r, "modelo_produzir")
    por_produto = (r.assign(erro_trad=(r["quantidade"] - r["tradicional"]).abs(),
                            erro_mod=(r["quantidade"] - r["modelo"]).abs())
                   .groupby("produto")[["quantidade", "erro_trad", "erro_mod"]].sum())
    por_produto["wape_tradicional"] = por_produto["erro_trad"] / por_produto["quantidade"]
    por_produto["wape_modelo"] = por_produto["erro_mod"] / por_produto["quantidade"]

    metricas = {
        "semanas_avaliadas": SEMANAS,
        "periodo": [str(r["data"].min().date()), str(r["data"].max().date())],
        "wape": {"tradicional": m.wape(r["quantidade"], r["tradicional"]), "modelo": m.wape(r["quantidade"], r["modelo"])},
        "tradicional": trad,
        "modelo": mod,
        "ganho_lucro_periodo": mod["lucro"] - trad["lucro"],
        "ganho_lucro_anual_estimado": (mod["lucro"] - trad["lucro"]) * 52 / SEMANAS,
        "produtos_em_que_modelo_erra_menos": int((por_produto["wape_modelo"] < por_produto["wape_tradicional"]).sum()),
        "produtos": int(len(por_produto)),
    }
    (RAIZ / "reports").mkdir(exist_ok=True)
    (RAIZ / "reports" / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metricas, ensure_ascii=False, indent=2))

    # 1. Resultado financeiro
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    itens = [("Vendas perdidas\n(falta de produto)", trad["vendas_perdidas_rs"], mod["vendas_perdidas_rs"]),
             ("Desperdício\n(produto jogado fora)", trad["desperdicio_rs"], mod["desperdicio_rs"])]
    y = np.arange(len(itens))
    ax.barh(y - 0.18, [i[1] / 1000 for i in itens], 0.34, color=CINZA, label="Método tradicional")
    ax.barh(y + 0.18, [i[2] / 1000 for i in itens], 0.34, color=AZUL, label="Modelo")
    for i, (_, t, mo) in enumerate(itens):
        ax.text(t / 1000 + 1, i - 0.18, brl(t / 1000, 1) + " mil", va="center", fontsize=8.5, color=TINTA2)
        ax.text(mo / 1000 + 1, i + 0.18, brl(mo / 1000, 1) + " mil", va="center", fontsize=8.5, color=TINTA2)
    ax.set_yticks(y, [i[0] for i in itens]); ax.invert_yaxis()
    ax.set_xlim(0, max(max(i[1], i[2]) for i in itens) / 1000 * 1.3)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    estilo(ax, f"Perdas nas últimas {SEMANAS} semanas (mil R$)")
    ax.grid(axis="x", color=GRADE, linewidth=0.8); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "resultado_financeiro.png", dpi=160); plt.close(fig)

    # 2. Erro de previsão por produto
    pp = por_produto.sort_values("wape_modelo")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    y = np.arange(len(pp))
    ax.hlines(y, pp["wape_modelo"] * 100, pp["wape_tradicional"] * 100, color=GRADE, linewidth=2)
    ax.scatter(pp["wape_tradicional"] * 100, y, color=CINZA, s=36, label="Método tradicional", zorder=3)
    ax.scatter(pp["wape_modelo"] * 100, y, color=AZUL, s=36, label="Modelo", zorder=3)
    ax.set_yticks(y, pp.index); ax.invert_yaxis()
    ax.set_xlabel("Erro médio da previsão (%)", fontsize=9, color=TINTA2)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    estilo(ax, "Erro de previsão por produto (menor é melhor)")
    ax.set_title("Erro de previsão por produto (menor é melhor)", loc="left", fontsize=12, fontweight="bold", color=TINTA, pad=26)
    ax.grid(axis="x", color=GRADE, linewidth=0.8); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "erro_por_produto.png", dpi=160); plt.close(fig)

    # 3. Exemplo de previsão: pão francês nas últimas 4 semanas avaliadas
    ex = r[r["produto"] == "Pão francês"].sort_values("data").tail(28)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(ex["data"], ex["quantidade"], color=TINTA, linewidth=2, label="Venda real")
    ax.plot(ex["data"], ex["tradicional"], color=CINZA, linewidth=2, linestyle="--", label="Método tradicional")
    ax.plot(ex["data"], ex["modelo"], color=AZUL, linewidth=2, label="Modelo")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    ax.set_ylim(0, ex[["quantidade", "tradicional", "modelo"]].max().max() * 1.25)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m"))
    estilo(ax, "Pão francês: previsão feita 7 dias antes x venda real (unidades)")
    fig.tight_layout(); fig.savefig(FIG / "exemplo_pao_frances.png", dpi=160); plt.close(fig)

    # 4. O que o modelo aprendeu
    pl = m.planejar(vendas)
    imp = pd.Series(pl["modelo"].booster_.feature_importance("gain"), index=m.FEATURES)
    nomes = {"dia_semana": "Dia da semana", "mes": "Mês", "dia_mes": "Dia do mês", "semana_ano": "Semana do ano",
             "feriado": "Feriado", "vespera_feriado": "Véspera de feriado", "fim_de_ano": "Fim de ano",
             "promocao": "Promoção", "lag7_rel": "Venda há 7 dias", "lag14_rel": "Venda há 14 dias",
             "lag21_rel": "Venda há 21 dias", "lag28_rel": "Venda há 28 dias",
             "media_mesmo_dia_rel": "Média do mesmo dia da semana", "media7_rel": "Média da última semana",
             "tendencia": "Tendência recente", "produto_cod": "Produto"}
    imp = (imp / imp.sum()).rename(nomes).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(imp.index, imp * 100, color=AZUL, height=0.6)
    for i, v in enumerate(imp * 100):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8.5, color=TINTA2)
    estilo(ax, "O que mais pesa na previsão (importância no modelo)")
    ax.grid(axis="x", color=GRADE, linewidth=0.8); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "importancia.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
