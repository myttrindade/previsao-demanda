"""API de previsão de demanda e plano de produção.

Uso local:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demanda import servico  # noqa: E402
from demanda.entrada import ErroDeEntrada, MapeamentoNecessario, ler_arquivo, padronizar  # noqa: E402

AQUI = Path(__file__).resolve().parent
PAGINA = AQUI / "static" / "index.html"
BASE_EXEMPLO = AQUI / "exemplos" / "vendas_padaria.csv"
TAMANHO_MAXIMO = 20 * 1024 * 1024  # 20 MB

app = FastAPI(
    title="API de Previsão de Demanda",
    description="Prevê as vendas dos próximos dias por produto e recomenda quanto produzir, "
                "equilibrando o custo da sobra e o da falta.",
    version="1.0.0",
)


async def _ler(arquivo: UploadFile, mapeamento: str | None) -> pd.DataFrame:
    conteudo = await arquivo.read()
    if len(conteudo) > TAMANHO_MAXIMO:
        raise HTTPException(status_code=413, detail="Arquivo maior que 20 MB.")
    try:
        mapa = json.loads(mapeamento) if mapeamento else None
        return padronizar(ler_arquivo(conteudo, arquivo.filename), mapa)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Mapeamento de colunas inválido.")
    except MapeamentoNecessario as pendente:
        raise HTTPException(status_code=422, detail=pendente.detalhe())
    except ErroDeEntrada as erro:
        raise HTTPException(status_code=422, detail=str(erro))


@lru_cache(maxsize=1)
def _exemplo() -> pd.DataFrame:
    return padronizar(pd.read_csv(BASE_EXEMPLO, dtype=str))


@lru_cache(maxsize=1)
def _plano_exemplo() -> dict:
    return servico.gerar_plano(_exemplo())


@lru_cache(maxsize=1)
def _comparacao_exemplo() -> dict:
    return servico.comparar(_exemplo())


@app.get("/", include_in_schema=False)
def pagina():
    return FileResponse(PAGINA)


@app.get("/saude")
def saude():
    return {"status": "ok"}


@app.post("/planejar")
async def planejar(
    arquivo: UploadFile = File(description="Histórico de vendas em Excel (.xlsx) ou CSV"),
    mapeamento: str | None = Form(None, description='JSON opcional {"campo": "coluna da planilha"}'),
):
    """Treina com o histórico enviado e devolve o plano de produção dos próximos 7 dias."""
    return servico.gerar_plano(await _ler(arquivo, mapeamento))


@app.post("/comparar")
async def comparar(
    arquivo: UploadFile = File(description="Histórico de vendas em Excel (.xlsx) ou CSV"),
    mapeamento: str | None = Form(None),
):
    """Testa o modelo nas últimas 4 semanas do histórico e compara com o método tradicional."""
    return servico.comparar(await _ler(arquivo, mapeamento))


@app.get("/planejar/exemplo")
def planejar_exemplo():
    """Plano de produção da padaria de exemplo."""
    return _plano_exemplo()


@app.get("/comparar/exemplo")
def comparar_exemplo():
    return _comparacao_exemplo()


def _em_portugues(df: pd.DataFrame) -> pd.DataFrame:
    pt = df.copy()
    pt["data"] = pd.to_datetime(pt["data"]).dt.date
    pt["promocao"] = pt["promocao"].astype(int).map({0: "não", 1: "sim"})
    return pt.rename(columns={"data": "Data", "produto": "Produto", "categoria": "Categoria",
                              "quantidade": "Quantidade", "preco": "Preço", "custo": "Custo", "promocao": "Promoção"})


def _excel(df: pd.DataFrame, nome: str) -> Response:
    saida = io.BytesIO()
    with pd.ExcelWriter(saida, engine="openpyxl", date_format="DD/MM/YYYY") as escritor:
        df.to_excel(escritor, index=False, sheet_name="vendas")
        aba = escritor.sheets["vendas"]
        for coluna in aba.columns:
            aba.column_dimensions[coluna[0].column_letter].width = max(12, len(str(coluna[0].value)) + 4)
        for celula in aba["A"][1:]:
            celula.number_format = "DD/MM/YYYY"
    return Response(saida.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename={nome}"})


@app.get("/base-exemplo.xlsx", include_in_schema=False)
def baixar_base_exemplo():
    return _excel(_em_portugues(pd.read_csv(BASE_EXEMPLO)), "vendas_padaria_exemplo.xlsx")


@app.get("/modelo-planilha.xlsx", include_in_schema=False)
def baixar_modelo_planilha():
    base = pd.read_csv(BASE_EXEMPLO)
    return _excel(_em_portugues(base[base["data"] == base["data"].min()].head(4)), "modelo_planilha_vendas.xlsx")
