"""Gera o histórico de vendas de uma padaria fictícia em Santos (SP).

Os dados são sintéticos, mas seguem padrões reais do varejo de alimentos:
dia da semana diferente para cada tipo de produto, verão com turistas,
itens quentes no inverno, feriados brasileiros, Natal e Ano-Novo,
dias de pagamento, promoções, dias de chuva e crescimento ao longo do tempo.

Uso:
    python -m demanda.gerar_dados
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
INICIO, FIM = date(2024, 9, 1), date(2026, 9, 23)
SEMENTE = 2026

# nome, categoria, vendas médias por dia, preço (R$), custo (R$), perfil semanal
PRODUTOS = [
    ("Pão francês", "Pães", 900, 0.90, 0.30, "cafe"),
    ("Pão de queijo", "Pães", 180, 3.50, 1.10, "cafe"),
    ("Croissant", "Pães", 60, 7.00, 2.40, "cafe"),
    ("Baguete", "Pães", 40, 8.00, 2.60, "familia"),
    ("Pão de forma caseiro", "Pães", 25, 12.00, 4.50, "familia"),
    ("Pão na chapa", "Lanches", 110, 6.00, 1.50, "cafe"),
    ("Misto quente", "Lanches", 70, 9.00, 3.00, "lanche"),
    ("Sanduíche natural", "Lanches", 45, 14.00, 5.50, "lanche"),
    ("Coxinha", "Salgados", 120, 7.50, 2.30, "lanche"),
    ("Esfiha de carne", "Salgados", 80, 6.50, 2.00, "lanche"),
    ("Empada de frango", "Salgados", 60, 7.00, 2.20, "lanche"),
    ("Pão de batata", "Salgados", 70, 6.00, 1.80, "lanche"),
    ("Sonho", "Doces", 55, 6.00, 1.90, "doce"),
    ("Brigadeiro", "Doces", 90, 3.00, 0.90, "doce"),
    ("Bolo de cenoura (fatia)", "Doces", 50, 7.00, 2.00, "doce"),
    ("Torta de limão (fatia)", "Doces", 30, 11.00, 3.50, "doce"),
    ("Pudim (fatia)", "Doces", 25, 10.00, 3.20, "doce"),
    ("Suco de laranja natural", "Bebidas", 70, 9.00, 3.00, "cafe"),
]

# Multiplicador por dia da semana (segunda = 0 ... domingo = 6)
SEMANA = {
    "cafe":    [1.00, 0.97, 0.98, 1.00, 1.05, 1.30, 1.40],
    "familia": [0.90, 0.88, 0.90, 0.95, 1.05, 1.30, 1.40],
    "lanche":  [1.05, 1.03, 1.04, 1.06, 1.10, 0.85, 0.70],
    "doce":    [0.85, 0.85, 0.90, 0.95, 1.10, 1.30, 1.45],
}
QUENTES = {"Pão na chapa", "Misto quente", "Sonho", "Coxinha"}


def gerar() -> pd.DataFrame:
    rng = np.random.default_rng(SEMENTE)
    dias = pd.date_range(INICIO, FIM, freq="D")
    feriados = holidays.Brazil(years=range(INICIO.year, FIM.year + 1), subdiv="SP")
    chuva = rng.random(len(dias)) < np.where(dias.month.isin([12, 1, 2, 3]), 0.35, 0.20)
    anos = (dias - dias[0]).days / 365.25

    linhas = []
    for nome, categoria, base, preco, custo, perfil in PRODUTOS:
        semana = np.array(SEMANA[perfil])[dias.dayofweek]
        verao = np.where(dias.month.isin([12, 1, 2]), 1.15, 1.0)          # turistas no litoral
        inverno = np.where(dias.month.isin([6, 7, 8]), 1.0, 1.0)
        if nome in QUENTES:
            inverno = np.where(dias.month.isin([6, 7, 8]), 1.15, 1.0)
        if nome == "Suco de laranja natural":
            verao = np.where(dias.month.isin([12, 1, 2]), 1.45, np.where(dias.month.isin([6, 7, 8]), 0.75, 1.0))
        feriado = np.array([d in feriados for d in dias.date])
        efeito_feriado = np.where(feriado, np.array(SEMANA[perfil])[6] / semana, 1.0)  # feriado se comporta como domingo
        natal = np.ones(len(dias))
        fim_de_ano = (dias.month == 12) & (dias.day >= 20)
        if categoria == "Doces" or perfil == "familia":
            natal = np.where(fim_de_ano, 1.40, 1.0)
        vespera = np.where((dias.month == 12) & dias.day.isin([24, 31]), 1.5 if categoria == "Pães" else 1.2, 1.0)
        carnaval = np.array([("Carnaval" in feriados.get(d, "")) for d in dias.date])
        pagamento = np.where(dias.day.isin([5, 6, 7, 8, 20, 21]), 1.06, 1.0)
        tendencia = 1 + 0.07 * anos
        efeito_chuva = np.where(chuva, 1.05 if nome in QUENTES else 0.88, 1.0)
        promocao = rng.random(len(dias)) < 0.04
        efeito_promo = np.where(promocao, 1.35, 1.0)

        media = (base * semana * efeito_feriado * verao * inverno * natal * vespera * np.where(carnaval, 1.25, 1.0)
                 * pagamento * tendencia * efeito_chuva * efeito_promo)
        dispersao = 25 if base >= 100 else 12
        vendas = rng.poisson(media * rng.gamma(dispersao, 1 / dispersao, len(dias)))

        linhas.append(pd.DataFrame({
            "data": dias.date,
            "produto": nome,
            "categoria": categoria,
            "quantidade": vendas,
            "preco": np.round(np.where(promocao, preco * 0.85, preco), 2),
            "custo": custo,
            "promocao": promocao.astype(int),
        }))
    return pd.concat(linhas, ignore_index=True).sort_values(["data", "produto"]).reset_index(drop=True)


def main() -> None:
    df = gerar()
    df.to_csv(RAIZ / "api" / "exemplos" / "vendas_padaria.csv", index=False)
    print(f"{len(df):,} linhas | {df['data'].min()} a {df['data'].max()} | {df['produto'].nunique()} produtos")
    print(df.groupby("produto")["quantidade"].mean().round(0).sort_values(ascending=False).head(5))


if __name__ == "__main__":
    main()
