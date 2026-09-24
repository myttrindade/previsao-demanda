"""Monta as respostas do app: plano de produção da semana e comparação com o método tradicional."""

from __future__ import annotations

import numpy as np
import pandas as pd

from demanda import modelo as m

DIAS_GRAFICO = 28
SEMANAS_COMPARACAO = 4


def _tem_precos(df: pd.DataFrame) -> bool:
    return bool({"preco", "custo"} <= set(df.columns) and df[["preco", "custo"]].notna().any().all())


def gerar_plano(vendas: pd.DataFrame, dias: int = m.HORIZONTE) -> dict:
    plano = m.planejar(vendas, dias)
    p, prev = plano["painel"], plano["previsao"]
    datas = sorted(prev["data"].unique())
    tem_precos = _tem_precos(vendas)
    categorias = vendas.groupby("produto")["categoria"].first() if "categoria" in vendas else pd.Series(dtype=str)
    inicio_grafico = p["data"].max() - pd.Timedelta(days=DIAS_GRAFICO - 1)

    produtos = []
    for produto, g in prev.groupby("produto"):
        g = g.sort_values("data")
        hist = p[(p["produto"] == produto) & (p["data"] >= inicio_grafico)].sort_values("data")
        item = {
            "produto": produto,
            "categoria": categorias.get(produto),
            "produzir": g["produzir"].astype(int).tolist(),
            "previsto": g["previsto"].round(1).tolist(),
            "total_semana": int(g["produzir"].sum()),
            "fator_seguranca": round(float(g["fator"].iloc[0]), 3),
            "historico": {"datas": hist["data"].dt.strftime("%Y-%m-%d").tolist(),
                          "quantidades": hist["quantidade"].astype(float).tolist()},
        }
        if tem_precos:
            preco = float(p.loc[p["produto"] == produto, "preco"].dropna().iloc[-1])
            custo = float(p.loc[p["produto"] == produto, "custo"].dropna().iloc[-1])
            item.update(preco=round(preco, 2), custo=round(custo, 2),
                        faturamento_previsto=round(float((np.minimum(g["previsto"], g["produzir"]) * preco).sum()), 2),
                        custo_producao=round(float(g["produzir"].sum() * custo), 2))
        produtos.append(item)

    chave = "faturamento_previsto" if tem_precos else "total_semana"
    produtos.sort(key=lambda x: x[chave], reverse=True)
    resumo = {
        "produtos": len(produtos),
        "historico_inicio": p["data"].min().strftime("%Y-%m-%d"),
        "historico_fim": p["data"].max().strftime("%Y-%m-%d"),
        "dias_historico": int(p["data"].nunique()),
        "total_produzir": int(sum(x["total_semana"] for x in produtos)),
        "tem_precos": tem_precos,
    }
    if tem_precos:
        resumo["faturamento_previsto"] = round(sum(x["faturamento_previsto"] for x in produtos), 2)
        resumo["custo_producao"] = round(sum(x["custo_producao"] for x in produtos), 2)
    return {"resumo": resumo, "dias": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in datas], "produtos": produtos}


def comparar(vendas: pd.DataFrame, semanas: int = SEMANAS_COMPARACAO) -> dict:
    """Testa o modelo nas últimas semanas do próprio histórico da empresa."""
    dias = vendas["data"].nunique()
    semanas = min(semanas, (dias - m.MIN_DIAS - 28) // 7)
    if semanas < 1:
        return {"disponivel": False,
                "motivo": "Com menos de 13 semanas de histórico não dá para testar o modelo no passado. "
                          "O plano de produção funciona normalmente."}
    r = m.backtest(vendas, semanas=semanas)
    saida = {
        "disponivel": True,
        "semanas": semanas,
        "periodo": [r["data"].min().strftime("%Y-%m-%d"), r["data"].max().strftime("%Y-%m-%d")],
        "erro_tradicional": round(m.wape(r["quantidade"], r["tradicional"]), 4),
        "erro_modelo": round(m.wape(r["quantidade"], r["modelo"]), 4),
        "tem_precos": _tem_precos(vendas),
    }
    if saida["tem_precos"]:
        trad = m.resultado_financeiro(r, "tradicional_produzir")
        mod = m.resultado_financeiro(r, "modelo_produzir")
        ganho = mod["lucro"] - trad["lucro"]
        saida.update(
            tradicional={k: round(v, 4) for k, v in trad.items()},
            modelo={k: round(v, 4) for k, v in mod.items()},
            ganho_periodo=round(ganho, 2),
            ganho_mensal_estimado=round(ganho * 30 / (7 * semanas), 2),
        )
    return saida
