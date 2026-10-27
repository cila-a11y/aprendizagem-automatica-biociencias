#!/usr/bin/env python3
"""Avalia previsões já calculadas, depois de ligar os ficheiros pelo identificador.

``classificacao`` usa as seis amostras fictícias do apêndice A.
``abalone`` usa as exportações reais da avaliação do capítulo 14.
O script não ajusta modelos nem escolhe limiares.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "biociencias_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    average_precision_score, confusion_matrix, roc_auc_score,
)

RAIZ = Path(__file__).resolve().parents[2]


def ler(caminho: Path, colunas: list[str]) -> pd.DataFrame:
    tabela = pd.read_csv(
        caminho, dtype={"id_registo": "string"}, float_precision="round_trip"
    )
    em_falta = set(colunas) - set(tabela.columns)
    if em_falta:
        raise ValueError(f"Faltam colunas em {caminho.name}: {sorted(em_falta)}")
    ids = tabela["id_registo"]
    if tabela.empty or ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError(f"Tabela vazia ou identificadores em falta: {caminho.name}")
    if ids.duplicated().any():
        raise ValueError(f"Identificadores repetidos: {caminho.name}")
    return tabela


def ligar(referencia: Path, previsoes: Path, resposta: str, previsao: str,
          ids_esperados: Path | None = None) -> pd.DataFrame:
    ref = ler(referencia, ["id_registo", resposta])[["id_registo", resposta]]
    prev = ler(previsoes, ["id_registo", previsao])
    colunas = ["id_registo", previsao]
    if "score_positivo" in prev.columns:
        colunas.append("score_positivo")
    prev = prev[colunas]
    if ids_esperados is not None:
        esperados = set(ler(ids_esperados, ["id_registo"])["id_registo"])
        for nome, tabela in [("referência", ref), ("previsões", prev)]:
            recebidos = set(tabela["id_registo"])
            if recebidos != esperados:
                raise ValueError(
                    f"Os identificadores de {nome} não coincidem com o plano: "
                    f"{len(esperados - recebidos)} em falta; "
                    f"{len(recebidos - esperados)} inesperados."
                )
    conjunto = ref.merge(
        prev, on="id_registo", how="outer", validate="one_to_one", indicator=True
    )
    if not conjunto["_merge"].eq("both").all():
        raise ValueError("Há amostras sem correspondência entre os dois ficheiros.")
    return conjunto.drop(columns="_merge").sort_values("id_registo").reset_index(drop=True)


def guardar_figura(fig, pasta: Path, nome: str) -> None:
    fig.tight_layout()
    for formato in ["pdf", "png"]:
        fig.savefig(pasta / f"{nome}.{formato}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def razao(numerador: float, denominador: float) -> float:
    return numerador / denominador if denominador else np.nan


def classificacao(tabela: pd.DataFrame, saida: Path) -> pd.Series:
    for coluna in ["maligno", "classe_prevista"]:
        if not tabela[coluna].isin([0, 1]).all():
            raise ValueError("As classes devem ser 0 ou 1, sem omissões.")
    y = tabela["maligno"].to_numpy(dtype=int)
    previsto = tabela["classe_prevista"].to_numpy(dtype=int)
    matriz = confusion_matrix(y, previsto, labels=[0, 1])
    vn, fp, fn, vp = matriz.ravel()
    ambas = set(y) == {0, 1}
    scores = None
    if "score_positivo" in tabela.columns:
        scores = pd.to_numeric(tabela["score_positivo"], errors="raise").to_numpy(float)
        if not np.isfinite(scores).all():
            raise ValueError("Há pontuações omissas ou não finitas.")
    # Só depois de validar tudo removemos figuras condicionais que deixaram
    # de ser aplicáveis. Outros ficheiros da pasta permanecem intocados.
    for nome, aplicavel in [
        ("matriz_normalizada", ambas),
        ("roc", ambas and scores is not None),
        ("vpp_sensibilidade", ambas and scores is not None),
    ]:
        if not aplicavel:
            for formato in ["png", "pdf"]:
                (saida / f"{nome}.{formato}").unlink(missing_ok=True)
    sensibilidade = razao(vp, vp + fn)
    especificidade = razao(vn, vn + fp)
    medidas = pd.Series({
        "Acerto": (vp + vn) / len(y),
        "Sensibilidade": sensibilidade,
        "Especificidade": especificidade,
        "VPP": razao(vp, vp + fp),
        "VPN": razao(vn, vn + fn),
        "F1": razao(2 * vp, 2 * vp + fp + fn),
        "Exatidao_equilibrada": (
            (sensibilidade + especificidade) / 2 if ambas else np.nan
        ),
    }, name="valor")
    tabela_matriz = pd.DataFrame(
        matriz, index=["Referencia benigna", "Referencia maligna"],
        columns=["Previsao benigna", "Previsao maligna"],
    )
    erros = tabela.loc[y != previsto].copy()
    erros["tipo_erro"] = np.where(erros["maligno"] == 1, "FN", "FP")

    fig, ax = plt.subplots(figsize=(5.1, 4.5))
    ConfusionMatrixDisplay(matriz, display_labels=["Benigno", "Maligno"]).plot(
        ax=ax, colorbar=False, values_format="d"
    )
    ax.set(xlabel="Classe prevista", ylabel="Resposta de referência")
    guardar_figura(fig, saida, "matriz_contagens")
    if ambas:
        normalizada = matriz / matriz.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(5.1, 4.5))
        ConfusionMatrixDisplay(normalizada, display_labels=["Benigno", "Maligno"]).plot(
            ax=ax, colorbar=False, values_format=".2f"
        )
        ax.set(xlabel="Classe prevista", ylabel="Resposta de referência")
        guardar_figura(fig, saida, "matriz_normalizada")

    if scores is not None:
        if ambas:
            medidas["AUC"] = roc_auc_score(y, scores)
            medidas["AP"] = average_precision_score(y, scores)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            RocCurveDisplay.from_predictions(y, scores, ax=ax)
            ax.set(xlabel="Taxa de falsos positivos", ylabel="Sensibilidade")
            guardar_figura(fig, saida, "roc")
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            PrecisionRecallDisplay.from_predictions(y, scores, ax=ax)
            ax.set(xlabel="Sensibilidade", ylabel="VPP")
            guardar_figura(fig, saida, "vpp_sensibilidade")
        else:
            print("ROC e VPP–sensibilidade requerem ambas as classes nesta atividade.")

    tabela.to_csv(saida / "avaliacao_por_amostra.csv", index=False)
    erros.to_csv(saida / "erros_por_amostra.csv", index=False)
    tabela_matriz.to_csv(saida / "matriz_contagens.csv")
    medidas.to_csv(saida / "medidas.csv", index_label="medida", na_rep="nao_definido")
    print(tabela_matriz.to_string())
    print(erros.to_string(index=False))
    return medidas


def regressao(tabela: pd.DataFrame, saida: Path) -> pd.Series:
    for coluna in ["Rings", "previsao"]:
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="raise")
        if not np.isfinite(tabela[coluna].to_numpy(float)).all():
            raise ValueError("Há respostas ou previsões omissas ou não finitas.")
    y = tabela["Rings"].to_numpy(float)
    if not ((y > 0) & (y == np.floor(y))).all():
        raise ValueError("Rings deve conter contagens inteiras positivas.")
    tabela["residuo"] = tabela["Rings"] - tabela["previsao"]
    tabela["erro_absoluto"] = tabela["residuo"].abs()
    medidas = pd.Series({
        "n": len(tabela),
        "MAE": tabela["erro_absoluto"].mean(),
        "RMSE": np.sqrt(np.mean(tabela["residuo"] ** 2)),
        "Erro_absoluto_maximo": tabela["erro_absoluto"].max(),
    }, name="valor")
    tabela.to_csv(saida / "avaliacao_por_registo.csv", index=False, float_format="%.17g")
    tabela.sort_values("erro_absoluto", ascending=False).to_csv(
        saida / "erros_ordenados.csv", index=False, float_format="%.17g"
    )
    medidas.to_csv(saida / "medidas.csv", index_label="medida")
    return medidas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tarefa", choices=["classificacao", "abalone"])
    parser.add_argument("--referencia", type=Path)
    parser.add_argument("--previsoes", type=Path)
    parser.add_argument("--ids-esperados", type=Path)
    parser.add_argument("--saida", type=Path)
    args = parser.parse_args()
    if args.tarefa == "classificacao":
        pasta = RAIZ / "exemplo_previsoes"
        resposta, previsao = "maligno", "classe_prevista"
        ids_padrao = pasta / "ids_esperados.csv"
        saida_padrao = pasta / "resultados"
    else:
        pasta = RAIZ / "reproducao" / "abalone" / "resultados"
        resposta, previsao = "Rings", "previsao"
        ids_padrao = RAIZ / "dados" / "abalone_teste_reservado.csv"
        saida_padrao = Path(__file__).resolve().parent / "resultados_abalone"
    entradas_personalizadas = args.referencia is not None or args.previsoes is not None
    if entradas_personalizadas and (args.referencia is None or args.previsoes is None):
        parser.error("Para avaliar outros ficheiros, indique --referencia e --previsoes.")
    referencia = args.referencia or pasta / "referencia_avaliacao.csv"
    previsoes = args.previsoes or pasta / "previsoes_avaliacao.csv"
    esperados = args.ids_esperados or (None if entradas_personalizadas else ids_padrao)
    saida = args.saida or saida_padrao
    if entradas_personalizadas and args.saida is None:
        parser.error("Indique também --saida para conservar os resultados do exemplo.")
    if esperados is None:
        print("Sem lista de IDs do plano: verifica-se apenas a correspondência entre ficheiros.")
    tabela = ligar(referencia, previsoes, resposta, previsao, esperados)
    saida.mkdir(parents=True, exist_ok=True)
    resultado = classificacao(tabela, saida) if args.tarefa == "classificacao" else regressao(tabela, saida)
    print(resultado.to_string())
    print(f"Resultados guardados em {saida}")


if __name__ == "__main__":
    main()
