import io

import pandas as pd
from fastapi.testclient import TestClient

from api.main import BASE_EXEMPLO, app

cliente = TestClient(app)


def _envia(df: pd.DataFrame, nome="vendas.csv", sep=",", mapeamento=None):
    arquivo = io.BytesIO(df.to_csv(index=False, sep=sep).encode("utf-8"))
    dados = {"mapeamento": mapeamento} if mapeamento else None
    return cliente.post("/planejar", files={"arquivo": (nome, arquivo, "text/csv")}, data=dados)


def _recente(dias=120) -> pd.DataFrame:
    df = pd.read_csv(BASE_EXEMPLO)
    return df[pd.to_datetime(df["data"]) > pd.to_datetime(df["data"]).max() - pd.Timedelta(days=dias)]


def test_plano_do_exemplo_tem_7_dias_para_cada_produto():
    r = cliente.get("/planejar/exemplo").json()
    assert len(r["dias"]) == 7
    assert r["resumo"]["produtos"] == 18
    assert all(len(p["produzir"]) == 7 and min(p["produzir"]) >= 0 for p in r["produtos"])
    assert r["resumo"]["total_produzir"] == sum(p["total_semana"] for p in r["produtos"])


def test_modelo_erra_menos_que_o_metodo_tradicional_no_exemplo():
    r = cliente.get("/comparar/exemplo").json()
    assert r["disponivel"]
    assert r["erro_modelo"] < r["erro_tradicional"]
    assert r["ganho_periodo"] > 0


def test_excel_em_portugues_gera_o_mesmo_plano():
    xlsx = cliente.get("/base-exemplo.xlsx").content
    r = cliente.post("/planejar", files={"arquivo": ("vendas.xlsx", io.BytesIO(xlsx))})
    assert r.status_code == 200
    assert r.json()["resumo"] == cliente.get("/planejar/exemplo").json()["resumo"]


def test_so_data_produto_e_quantidade_com_datas_brasileiras():
    df = _recente()[["data", "produto", "quantidade"]]
    df = df.assign(data=pd.to_datetime(df["data"]).dt.strftime("%d/%m/%Y"))
    df = df.rename(columns={"data": "Data da venda", "produto": "Descrição", "quantidade": "Qtd"})
    r = _envia(df, sep=";")
    assert r.status_code == 200
    assert r.json()["resumo"]["tem_precos"] is False


def test_coluna_com_outro_nome_pede_correspondencia():
    df = _recente().rename(columns={"produto": "Mercadoria vendida"})
    r = _envia(df)
    assert r.status_code == 422
    assert r.json()["detail"]["faltando"] == [{"campo": "produto", "nome": "Produto"}]
    r2 = _envia(df, mapeamento='{"produto": "Mercadoria vendida"}')
    assert r2.status_code == 200


def test_historico_curto_demais_gera_mensagem_clara():
    r = _envia(_recente(dias=30))
    assert r.status_code == 422
    assert "8 semanas" in r.json()["detail"]


def test_comparacao_com_historico_curto_explica_o_motivo():
    df = _recente(dias=70)
    arquivo = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    r = cliente.post("/comparar", files={"arquivo": ("v.csv", arquivo, "text/csv")}).json()
    assert r["disponivel"] is False and "13 semanas" in r["motivo"]


def test_vendas_por_cupom_sao_somadas_por_dia():
    df = _recente()
    duplicado = pd.concat([df.assign(quantidade=df["quantidade"] // 2),
                           df.assign(quantidade=df["quantidade"] - df["quantidade"] // 2)])
    a = _envia(df).json()["produtos"]
    b = _envia(duplicado).json()["produtos"]
    assert {p["produto"]: p["total_semana"] for p in a} == {p["produto"]: p["total_semana"] for p in b}


def test_pagina_inicial():
    r = cliente.get("/")
    assert r.status_code == 200 and "Previsão de Demanda" in r.text
