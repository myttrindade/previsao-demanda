"""Previsão de vendas diárias por produto e recomendação de quanto produzir.

Abordagem:
- Um único modelo LightGBM para todos os produtos, que prevê a venda de cada dia
  como proporção do nível recente do produto. Assim ele funciona para itens que
  vendem 20 ou 2.000 unidades por dia e aprende padrões comuns (fim de semana,
  feriados, sazonalidade).
- Todas as variáveis usam só informação disponível 7 dias antes, então a mesma
  previsão serve para planejar a semana inteira.
- A quantidade recomendada vem do "problema do jornaleiro": produzir a mais custa
  o produto jogado fora; produzir a menos custa a margem da venda perdida. O ponto
  ótimo é um quantil da previsão que depende da margem de cada produto.
"""

from __future__ import annotations

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

HORIZONTE = 7
MIN_DIAS = 56
FEATURES = ["dia_semana", "mes", "dia_mes", "semana_ano", "feriado", "vespera_feriado", "fim_de_ano",
            "promocao", "lag7_rel", "lag14_rel", "lag21_rel", "lag28_rel", "media_mesmo_dia_rel",
            "media7_rel", "tendencia", "produto_cod"]
PARAMS = dict(objective="regression", n_estimators=400, learning_rate=0.03, num_leaves=31,
              min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, verbose=-1, random_state=42)


# ---------------------------------------------------------------- dados
def painel(vendas: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por produto e dia, com zero nos dias sem venda registrada."""
    v = vendas.copy()
    v["data"] = pd.to_datetime(v["data"])
    agg = {"quantidade": "sum"}
    for c in ("preco", "custo", "promocao"):
        if c in v:
            agg[c] = "max" if c == "promocao" else "median"
    v = v.groupby(["produto", "data"], as_index=False).agg(agg)
    dias = pd.date_range(v["data"].min(), v["data"].max(), freq="D")
    grade = pd.MultiIndex.from_product([sorted(v["produto"].unique()), dias], names=["produto", "data"])
    p = v.set_index(["produto", "data"]).reindex(grade).reset_index()
    p["quantidade"] = p["quantidade"].fillna(0)
    if "promocao" not in p:
        p["promocao"] = 0
    p["promocao"] = p["promocao"].fillna(0)
    for c in ("preco", "custo"):
        if c in p:
            p[c] = p.groupby("produto")[c].transform(lambda s: s.ffill().bfill())
    return p


def _calendario(datas: pd.Series) -> pd.DataFrame:
    anos = range(datas.min().year, datas.max().year + 2)
    fer = holidays.Brazil(years=anos)
    d = pd.DatetimeIndex(datas)
    feriado = np.array([x in fer for x in d.date])
    vespera = np.array([(x + pd.Timedelta(days=1)).date() in fer for x in d])
    return pd.DataFrame({
        "dia_semana": d.dayofweek, "mes": d.month, "dia_mes": d.day, "semana_ano": d.isocalendar().week.to_numpy(),
        "feriado": feriado.astype(int), "vespera_feriado": vespera.astype(int),
        "fim_de_ano": ((d.month == 12) & (d.day >= 20)).astype(int),
    }, index=datas.index)


def construir_features(p: pd.DataFrame, produtos: list[str]) -> pd.DataFrame:
    """Variáveis para cada produto e dia, usando só dados de 7 ou mais dias antes."""
    p = p.sort_values(["produto", "data"]).copy()
    g = p.groupby("produto")["quantidade"]
    for k in (7, 14, 21, 28):
        p[f"lag{k}"] = g.shift(k)
    # nível recente: média dos 28 dias que terminam 7 dias antes
    p["nivel"] = g.transform(lambda s: s.shift(7).rolling(28, min_periods=7).mean())
    p["media7"] = g.transform(lambda s: s.shift(7).rolling(7, min_periods=3).mean())
    p["media_mesmo_dia"] = p[["lag7", "lag14", "lag21", "lag28"]].mean(axis=1)
    base = p["nivel"].where(p["nivel"] > 0)
    for c in ("lag7", "lag14", "lag21", "lag28", "media_mesmo_dia", "media7"):
        p[f"{c}_rel"] = p[c] / base
    p["tendencia"] = p["media7"] / base
    p["produto_cod"] = p["produto"].map({n: i for i, n in enumerate(produtos)}).astype("category")
    p = p.join(_calendario(p["data"]))
    p["alvo_rel"] = p["quantidade"] / base
    return p


def _treinar(feat: pd.DataFrame) -> lgb.LGBMRegressor:
    treino = feat.dropna(subset=["alvo_rel", "nivel"])
    treino = treino[treino["nivel"] > 0]
    return lgb.LGBMRegressor(**PARAMS).fit(treino[FEATURES], treino["alvo_rel"].clip(upper=5))


def _prever(modelo, feat: pd.DataFrame) -> np.ndarray:
    rel = np.clip(modelo.predict(feat[FEATURES]), 0, None)
    return np.nan_to_num(rel * feat["nivel"].to_numpy(), nan=0.0)


# ---------------------------------------------------------------- decisão
def fator_seguranca(erros_rel: pd.DataFrame, margem_critica: pd.Series) -> pd.Series:
    """Multiplicador da previsão que maximiza o lucro esperado de cada produto.

    `erros_rel` tem colunas produto e razao (venda real / previsão) de um período
    de calibração. O quantil ótimo da razão é a margem crítica = margem / preço.
    """
    fatores = {}
    for produto, grupo in erros_rel.groupby("produto"):
        q = float(margem_critica.get(produto, 0.5))
        fatores[produto] = float(np.quantile(grupo["razao"], q)) if len(grupo) >= 14 else 1.0
    return pd.Series(fatores)


def margem_critica(p: pd.DataFrame) -> pd.Series:
    if not {"preco", "custo"} <= set(p.columns):
        return pd.Series(dtype=float)
    ult = p.dropna(subset=["preco", "custo"]).groupby("produto")[["preco", "custo"]].last()
    return ((ult["preco"] - ult["custo"]) / ult["preco"]).clip(0.05, 0.95)


def _calibrar(feat: pd.DataFrame, corte: pd.Timestamp, dias_calibracao: int = 56) -> pd.DataFrame:
    """Treina até `corte - dias_calibracao` e mede o erro relativo nos dias seguintes até `corte`."""
    inicio_cal = corte - pd.Timedelta(days=dias_calibracao)
    base = feat[feat["data"] < inicio_cal].dropna(subset=["alvo_rel", "nivel"])
    if len(base) < 200:  # histórico curto: sem calibração, usa a previsão pura (fator 1)
        return pd.DataFrame(columns=["produto", "razao"])
    modelo = _treinar(feat[feat["data"] < inicio_cal])
    cal = feat[(feat["data"] >= inicio_cal) & (feat["data"] < corte)].copy()
    cal["previsto"] = _prever(modelo, cal)
    cal = cal[cal["previsto"] > 0]
    return pd.DataFrame({"produto": cal["produto"], "razao": cal["quantidade"] / cal["previsto"]})


# ---------------------------------------------------------------- uso
def planejar(vendas: pd.DataFrame, dias: int = HORIZONTE) -> dict:
    """Treina com todo o histórico e devolve a previsão e a produção recomendada para os próximos dias."""
    p = painel(vendas)
    if p["data"].nunique() < MIN_DIAS:
        raise ValueError(f"O histórico precisa ter pelo menos {MIN_DIAS} dias; tem {p['data'].nunique()}.")
    produtos = sorted(p["produto"].unique())
    ultimo = p["data"].max()
    futuro = pd.DataFrame([(prod, ultimo + pd.Timedelta(days=i)) for prod in produtos for i in range(1, dias + 1)],
                          columns=["produto", "data"])
    futuro["quantidade"] = np.nan
    futuro["promocao"] = 0
    for c in ("preco", "custo"):
        if c in p:
            futuro[c] = futuro["produto"].map(p.groupby("produto")[c].last())
    tudo = pd.concat([p, futuro], ignore_index=True)
    feat = construir_features(tudo, produtos)

    historico = feat[feat["data"] <= ultimo]
    fatores = fator_seguranca(_calibrar(historico, ultimo + pd.Timedelta(days=1)), margem_critica(p))
    modelo = _treinar(historico)
    prox = feat[feat["data"] > ultimo].copy()
    prox["previsto"] = _prever(modelo, prox)
    prox["fator"] = prox["produto"].map(fatores).fillna(1.0)
    prox["produzir"] = np.ceil(prox["previsto"] * prox["fator"])
    return {"painel": p, "previsao": prox, "fatores": fatores, "modelo": modelo}


def backtest(vendas: pd.DataFrame, semanas: int = 12) -> pd.DataFrame:
    """Simula o uso real: a cada semana, treina com o passado e planeja os 7 dias seguintes.

    Compara o modelo com o método tradicional (média das últimas 4 semanas no mesmo dia da semana).
    """
    p = painel(vendas)
    produtos = sorted(p["produto"].unique())
    feat = construir_features(p, produtos)
    mc = margem_critica(p)
    fim = p["data"].max()
    resultados = []
    for s in range(semanas, 0, -1):
        inicio = fim - pd.Timedelta(days=7 * s - 1)
        treino = feat[feat["data"] < inicio]
        fatores = fator_seguranca(_calibrar(treino, inicio), mc)
        modelo = _treinar(treino)
        semana = feat[(feat["data"] >= inicio) & (feat["data"] < inicio + pd.Timedelta(days=7))].copy()
        semana["modelo"] = _prever(modelo, semana)
        semana["modelo_produzir"] = np.ceil(semana["modelo"] * semana["produto"].map(fatores).fillna(1.0))
        semana["tradicional"] = semana["media_mesmo_dia"].fillna(semana["media7"]).fillna(0)
        semana["tradicional_produzir"] = np.ceil(semana["tradicional"] * 1.10)  # prática comum: 10% a mais
        resultados.append(semana)
    r = pd.concat(resultados)
    colunas = ["produto", "data", "quantidade"] + [c for c in ("preco", "custo") if c in r] +         ["modelo", "modelo_produzir", "tradicional", "tradicional_produzir"]
    return r[colunas].reset_index(drop=True)


def resultado_financeiro(r: pd.DataFrame, coluna_produzir: str) -> dict:
    produzido = r[coluna_produzir]
    vendido = np.minimum(r["quantidade"], produzido)
    sobra = produzido - vendido
    falta = r["quantidade"] - vendido
    return {
        "lucro": float((vendido * r["preco"] - produzido * r["custo"]).sum()),
        "desperdicio_unid": float(sobra.sum()),
        "desperdicio_rs": float((sobra * r["custo"]).sum()),
        "vendas_perdidas_unid": float(falta.sum()),
        "vendas_perdidas_rs": float((falta * r["preco"]).sum()),
        "atendimento": float(vendido.sum() / r["quantidade"].sum()),
    }


def wape(real, previsto) -> float:
    real, previsto = np.asarray(real), np.asarray(previsto)
    return float(np.abs(real - previsto).sum() / real.sum())
