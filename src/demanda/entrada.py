"""Leitura flexível do histórico de vendas enviado pela empresa.

Aceita Excel ou CSV, uma linha por venda ou por produto e dia, com colunas em
português ou inglês. Obrigatórias: data, produto e quantidade. Opcionais:
preço, custo, promoção e categoria.
"""

from __future__ import annotations

import io
import re
import unicodedata

import pandas as pd

OBRIGATORIAS = ["data", "produto", "quantidade"]
OPCIONAIS = ["preco", "custo", "promocao", "categoria"]
ROTULOS = {"data": "Data da venda", "produto": "Produto", "quantidade": "Quantidade vendida",
           "preco": "Preço de venda", "custo": "Custo unitário", "promocao": "Promoção", "categoria": "Categoria"}
LIMITE_LINHAS = 500_000


def normaliza(texto) -> str:
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", texto.lower())


SINONIMOS = {
    "data": ["data", "date", "dia", "datadavenda", "datavenda", "datadopedido", "emissao", "dataemissao", "datamovimento"],
    "produto": ["produto", "product", "item", "descricao", "descricaodoproduto", "nomedoproduto", "nomeproduto",
                "mercadoria", "sku", "artigo"],
    "quantidade": ["quantidade", "quantity", "qtd", "qtde", "quant", "unidades", "quantidadevendida", "qtdvendida",
                   "vendas", "vendido", "volume"],
    "preco": ["preco", "price", "precodevenda", "precounitario", "valorunitario", "precovenda", "vlunitario", "valor"],
    "custo": ["custo", "cost", "custounitario", "custodeproducao", "custoproducao", "custoun", "cmv"],
    "promocao": ["promocao", "promo", "promotion", "empromocao", "promocional"],
    "categoria": ["categoria", "category", "grupo", "secao", "departamento", "familia"],
}


class ErroDeEntrada(ValueError):
    """Planilha fora do formato esperado, com mensagem para o usuário."""


class MapeamentoNecessario(ErroDeEntrada):
    def __init__(self, faltando: list[str], colunas: list[str], reconhecidas: dict[str, str]):
        super().__init__("Algumas colunas não foram reconhecidas.")
        self.faltando, self.colunas, self.reconhecidas = faltando, colunas, reconhecidas

    def detalhe(self) -> dict:
        return {
            "tipo": "mapeamento",
            "mensagem": "Não reconheci algumas colunas. Indique qual coluna da sua planilha corresponde a cada informação.",
            "faltando": [{"campo": c, "nome": ROTULOS[c]} for c in self.faltando],
            "colunas_arquivo": self.colunas,
            "reconhecidas": {ROTULOS[k]: v for k, v in self.reconhecidas.items()},
        }


def ler_arquivo(conteudo: bytes, nome: str) -> pd.DataFrame:
    nome = (nome or "").lower()
    try:
        if nome.endswith((".xlsx", ".xlsm", ".xls")) or conteudo[:2] == b"PK":
            return pd.read_excel(io.BytesIO(conteudo), dtype=object)
        try:
            texto = conteudo.decode("utf-8-sig")
        except UnicodeDecodeError:
            texto = conteudo.decode("latin-1")
        return pd.read_csv(io.StringIO(texto), sep=None, engine="python", dtype=str)
    except Exception:
        raise ErroDeEntrada("Não foi possível ler o arquivo. Envie uma planilha Excel (.xlsx) ou CSV.")


def reconhecer_colunas(colunas: list[str], mapeamento: dict[str, str] | None = None) -> dict[str, str]:
    por_nome = {normaliza(c): c for c in colunas}
    achadas: dict[str, str] = {}
    for campo in OBRIGATORIAS + OPCIONAIS:
        for candidato in SINONIMOS[campo]:
            coluna = por_nome.get(normaliza(candidato))
            if coluna and coluna not in achadas.values():
                achadas[campo] = coluna
                break
    for campo, coluna in (mapeamento or {}).items():
        if coluna in colunas:
            achadas[campo] = coluna
    return achadas


def _numero(serie: pd.Series) -> pd.Series:
    def conv(v):
        if pd.isna(v) or isinstance(v, (int, float)):
            return v
        t = re.sub(r"[^0-9,.\-]", "", str(v))
        if "," in t and "." in t:
            t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
        elif "," in t:
            t = t.replace(",", ".")
        return t or None
    return pd.to_numeric(serie.map(conv), errors="coerce")


def _data(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie).dt.normalize()
    texto = serie.astype(str).str.strip()
    iso = texto.str.match(r"^\d{4}-\d{2}-\d{2}")
    datas = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    datas[iso] = pd.to_datetime(texto[iso].str[:10], errors="coerce", format="%Y-%m-%d")
    datas[~iso] = pd.to_datetime(texto[~iso], errors="coerce", dayfirst=True)
    return datas.dt.normalize()


def padronizar(bruto: pd.DataFrame, mapeamento: dict[str, str] | None = None) -> pd.DataFrame:
    """Converte a planilha do usuário em: data, produto, quantidade [, preco, custo, promocao, categoria]."""
    bruto = bruto.dropna(how="all")
    bruto.columns = [str(c).strip() for c in bruto.columns]
    if bruto.empty:
        raise ErroDeEntrada("A planilha está vazia.")
    if len(bruto) > LIMITE_LINHAS:
        raise ErroDeEntrada(f"A planilha tem {len(bruto):,} linhas; o limite é {LIMITE_LINHAS:,}.")
    achadas = reconhecer_colunas(list(bruto.columns), mapeamento)
    faltando = [c for c in OBRIGATORIAS if c not in achadas]
    if faltando:
        raise MapeamentoNecessario(faltando, list(bruto.columns), achadas)

    df = pd.DataFrame({
        "data": _data(bruto[achadas["data"]]),
        "produto": bruto[achadas["produto"]].astype(str).str.strip(),
        "quantidade": _numero(bruto[achadas["quantidade"]]),
    })
    ruins = df["data"].isna()
    if ruins.mean() > 0.05:
        exemplos = bruto.loc[ruins, achadas["data"]].astype(str).head(3).tolist()
        raise ErroDeEntrada(f"Não entendi as datas da coluna '{achadas['data']}' (ex.: {', '.join(exemplos)}). "
                            "Use o formato dd/mm/aaaa.")
    if df["quantidade"].isna().mean() > 0.05:
        raise ErroDeEntrada(f"A coluna '{achadas['quantidade']}' tem valores vazios ou que não são números.")
    for campo in ("preco", "custo"):
        if campo in achadas:
            df[campo] = _numero(bruto[achadas[campo]])
    if "promocao" in achadas:
        v = bruto[achadas["promocao"]].map(normaliza)
        df["promocao"] = v.isin({"1", "sim", "s", "yes", "y", "true", "x", "verdadeiro"}).astype(int)
    if "categoria" in achadas:
        df["categoria"] = bruto[achadas["categoria"]].astype(str).str.strip()

    df = df.dropna(subset=["data", "quantidade"])
    df = df[(df["produto"] != "") & (df["produto"].str.lower() != "nan")]
    if df["data"].nunique() < 56:
        raise ErroDeEntrada(f"O histórico tem só {df['data'].nunique()} dias com venda. "
                            "Envie pelo menos 8 semanas (56 dias) para o modelo aprender o padrão da semana.")
    return df.reset_index(drop=True)
