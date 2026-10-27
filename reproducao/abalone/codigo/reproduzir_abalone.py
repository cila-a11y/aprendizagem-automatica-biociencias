#!/usr/bin/env python3
"""Reprodução do estudo Abalone, com desenvolvimento e teste separados.

Exemplos, a partir de qualquer pasta:
    python reproduzir_abalone.py preparar
    python reproduzir_abalone.py desenvolver
    python reproduzir_abalone.py avaliar

A opção ``tudo`` reproduz as três etapas. Os dados originais acompanham
o livro; só são descarregados se ainda não existirem na pasta ``originais``.
Os parâmetros são lidos de protocolo.json, fixado antes do ajustamento.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from datetime import datetime, timezone
from urllib.request import urlopen

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "abalone_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


ESTUDO = Path(__file__).resolve().parents[1]
RAIZ = ESTUDO.parents[1]
DADOS = RAIZ / "dados"
ORIGINAIS = ESTUDO / "originais"
RESULTADOS = ESTUDO / "resultados"
FIGURAS = RAIZ / "figuras"
PROTOCOLO = ESTUDO / "protocolo.json"
NOMES = [
    "Sex", "Length", "Diameter", "Height", "Whole_weight",
    "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings",
]
HASHES_ORIGINAIS = {
    "abalone.data": "de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f",
    "abalone.names": "542eee5a2a0ee72db563de010dc3b5055b03c5dc624049ae1f2208fc66a14aac",
}
ROTULOS = {
    "Referencia": "Referência pela mediana",
    "Regressao_linear": "Regressão linear",
    "Floresta": "Floresta",
}


def sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def guardar_json(caminho: Path, objeto: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def ler_json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def guardar_csv(tabela: pd.DataFrame, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(caminho, index=False, float_format="%.17g")


def ler_csv(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho, float_precision="round_trip")


def versoes() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def confirmar_tabela(tabela: pd.DataFrame, protocolo: dict) -> None:
    campos = protocolo["features"] + [protocolo["target"]]
    if tabela.empty or tabela["id_registo"].isna().any():
        raise ValueError("Tabela vazia ou identificadores em falta.")
    if not tabela["id_registo"].is_unique:
        raise ValueError("Há identificadores repetidos.")
    valores = tabela[campos].to_numpy(dtype=float)
    if not np.isfinite(valores).all() or not (valores > 0).all():
        raise ValueError("As entradas escolhidas e a resposta devem ser positivas e finitas.")
    resposta = tabela[protocolo["target"]].to_numpy(dtype=float)
    if not np.equal(resposta, np.floor(resposta)).all():
        raise ValueError("Rings deve conter contagens inteiras.")


def construir_modelos(protocolo: dict) -> dict:
    classes = {
        "DummyRegressor": DummyRegressor,
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
    }
    modelos = {}
    for nome, configuracao in protocolo["models"].items():
        argumentos = configuracao.copy()
        tipo = argumentos.pop("type")
        modelos[nome] = classes[tipo](**argumentos)
    return modelos


def medidas(y, previsto) -> dict:
    y = np.asarray(y, dtype=float)
    previsto = np.asarray(previsto, dtype=float)
    residuos = y - previsto
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, previsto)),
        "rmse": float(np.sqrt(mean_squared_error(y, previsto))),
        "r2": float(r2_score(y, previsto)),
        "median_absolute_error": float(np.median(np.abs(residuos))),
        "max_absolute_error": float(np.max(np.abs(residuos))),
        "mean_signed_residual": float(np.mean(residuos)),
        "prediction_min": float(np.min(previsto)),
        "prediction_max": float(np.max(previsto)),
    }


def preparar() -> None:
    protocolo = ler_json(PROTOCOLO)
    for pasta in [DADOS, ORIGINAIS, RESULTADOS, FIGURAS]:
        pasta.mkdir(parents=True, exist_ok=True)
    base_url = protocolo["source_url"].rsplit("/", 1)[0]
    for nome, hash_esperado in HASHES_ORIGINAIS.items():
        caminho = ORIGINAIS / nome
        if not caminho.exists():
            with urlopen(f"{base_url}/{nome}", timeout=60) as resposta:
                caminho.write_bytes(resposta.read())
        if sha256(caminho) != hash_esperado:
            raise ValueError(f"O ficheiro original {nome} não corresponde à cópia documentada.")

    original = pd.read_csv(
        ORIGINAIS / "abalone.data", header=None, names=NOMES,
        float_precision="round_trip",
    )
    original.insert(0, "id_registo", np.arange(1, len(original) + 1))
    confirmar_tabela(original, protocolo)
    divisao = protocolo["splitting"]
    desenvolvimento, teste = train_test_split(
        original,
        test_size=divisao["test_size"], random_state=divisao["random_state"],
        shuffle=divisao["shuffle"], stratify=divisao["stratify"],
    )
    desenvolvimento = desenvolvimento.sort_values("id_registo").reset_index(drop=True)
    teste = teste.sort_values("id_registo").reset_index(drop=True)
    if set(desenvolvimento.id_registo) & set(teste.id_registo):
        raise RuntimeError("A divisão contém sobreposição de identificadores.")
    guardar_csv(desenvolvimento, DADOS / "abalone_desenvolvimento.csv")
    guardar_csv(teste, DADOS / "abalone_teste_reservado.csv")
    guardar_json(RESULTADOS / "preparacao.json", {
        "utc": agora(), "source_url": protocolo["source_url"],
        "dataset_doi": protocolo["doi"], "protocol_sha256": sha256(PROTOCOLO),
        "source_sha256": HASHES_ORIGINAIS,
        "n_original": len(original), "n_development": len(desenvolvimento),
        "n_test": len(teste), "n_removed": 0,
        "development_sha256": sha256(DADOS / "abalone_desenvolvimento.csv"),
        "test_sha256": sha256(DADOS / "abalone_teste_reservado.csv"),
        "versions": versoes(),
        "note": "Preparação não usa valores de teste para ajustar transformações ou modelos.",
    })
    print(f"Preparação: {len(desenvolvimento)} registos de desenvolvimento; {len(teste)} de teste.")


def figuras_desenvolvimento(dados: pd.DataFrame) -> None:
    configurar_figuras()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    contagens = dados["Rings"].value_counts().sort_index()
    axes[0].bar(contagens.index, contagens.values, color="#21618c", width=0.85)
    axes[0].set(xlabel="Número de anéis observado", ylabel="Número de registos")
    axes[0].set_title("Contagem de anéis\nno desenvolvimento")
    axes[1].scatter(
        dados["Whole_weight"], dados["Rings"], s=13, alpha=0.20,
        color="#21618c", edgecolors="none", rasterized=True,
    )
    axes[1].set(
        xlabel="Massa total na escala publicada\n(valor em gramas dividido por 200)",
        ylabel="Número de anéis observado",
    )
    axes[1].set_title("Massa total e\ncontagem de anéis")
    for ax in axes:
        ax.grid(axis="y", color="#dce1e5", alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.4)
    guardar_figura(fig, "u14_abalone_desenvolvimento")


def desenvolver() -> None:
    protocolo = ler_json(PROTOCOLO)
    preparacao = ler_json(RESULTADOS / "preparacao.json")
    if sha256(PROTOCOLO) != preparacao["protocol_sha256"]:
        raise ValueError("O protocolo mudou depois da preparação.")
    caminho = DADOS / "abalone_desenvolvimento.csv"
    if sha256(caminho) != preparacao["development_sha256"]:
        raise ValueError("A cópia de desenvolvimento mudou.")
    dados = ler_csv(caminho)
    confirmar_tabela(dados, protocolo)
    X = dados[protocolo["features"]]
    y = dados[protocolo["target"]]
    config_cv = protocolo["cv"]
    particoes = list(KFold(
        n_splits=config_cv["n_splits"], shuffle=config_cv["shuffle"],
        random_state=config_cv["random_state"],
    ).split(X))
    modelos = construir_modelos(protocolo)
    linhas = []
    oof = dados[["id_registo", "Rings"]].copy()
    oof["parte_validacao"] = 0
    for parte, (_, validacao) in enumerate(particoes, 1):
        oof.loc[validacao, "parte_validacao"] = parte
    for nome, modelo in modelos.items():
        previsoes = np.full(len(dados), np.nan)
        for parte, (treino, validacao) in enumerate(particoes, 1):
            ajustado = clone(modelo).fit(X.iloc[treino], y.iloc[treino])
            previsto = ajustado.predict(X.iloc[validacao])
            previsoes[validacao] = previsto
            resultado = medidas(y.iloc[validacao], previsto)
            linhas.append({
                "procedimento": nome, "parte": parte,
                "n_treino": len(treino), "n_validacao": len(validacao),
                "mae": resultado["mae"], "rmse": resultado["rmse"],
            })
        if not np.isfinite(previsoes).all():
            raise RuntimeError("Faltam previsões fora do treino.")
        oof[nome] = previsoes
    resultados = pd.DataFrame(linhas)
    resumos = []
    for nome in modelos:
        parte = resultados.loc[resultados.procedimento == nome]
        resumos.append({
            "procedimento": nome, "mae_medio": float(parte.mae.mean()),
            "desvio_mae": float(parte.mae.std(ddof=1)),
            "rmse_medio": float(parte.rmse.mean()),
        })
    resumo = pd.DataFrame(resumos)
    # idxmin devolve a primeira posição em caso de empate exato.
    escolhido = resumo.loc[resumo.mae_medio.idxmin(), "procedimento"]
    guardar_csv(resultados, RESULTADOS / "validacao_por_parte.csv")
    guardar_csv(resumo, RESULTADOS / "validacao_resumo.csv")
    guardar_csv(oof, RESULTADOS / "previsoes_validacao.csv")
    guardar_json(RESULTADOS / "decisao.json", {
        "utc": agora(), "selected_model": escolhido,
        "selected_configuration": protocolo["models"][escolhido],
        "features": protocolo["features"], "target": protocolo["target"],
        "criterion": protocolo["model_selection"], "cv_results": resumos,
        "protocol_sha256": sha256(PROTOCOLO),
        "development_sha256": sha256(caminho),
        "test_values_used_for_selection": False,
        "predictions_rounded_or_clipped": False,
        "versions": versoes(),
    })
    # Esta etapa não abre o ficheiro de teste.
    figuras_desenvolvimento(dados)
    guardar_json(RESULTADOS / "descricao_desenvolvimento.json", {
        "n": len(dados),
        "rings": {
            "min": int(y.min()), "max": int(y.max()),
            "mean": float(y.mean()), "median": float(y.median()),
            "counts": {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()},
        },
        "features_summary": dados[protocolo["features"]].describe().to_dict(),
        "no_missing_chosen_features": bool(not X.isna().any().any()),
    })
    print(resumo.to_string(index=False))
    print(f"Decisão registada antes de abrir o teste: {escolhido}.")


def configurar_figuras() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "axes.titlesize": 13, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def guardar_figura(fig, nome: str) -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / f"{nome}.pdf", bbox_inches="tight")
    fig.savefig(FIGURAS / f"{nome}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def figuras_teste(avaliacao: pd.DataFrame, nome_modelo: str) -> None:
    configurar_figuras()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    cor = "#147d64"
    axes[0].scatter(
        avaliacao.Rings, avaliacao.previsao, s=16, alpha=0.40,
        color=cor, edgecolors="none", rasterized=True,
    )
    limites = [
        min(avaliacao.Rings.min(), avaliacao.previsao.min()) - 0.5,
        max(avaliacao.Rings.max(), avaliacao.previsao.max()) + 0.5,
    ]
    axes[0].plot(limites, limites, color="#4d4d4d", linestyle="--", linewidth=1.2)
    axes[0].set(
        xlabel="Número de anéis observado", ylabel="Número de anéis previsto",
        xlim=limites, ylim=limites,
    )
    axes[0].set_title("Observado e previsto\nno teste")
    axes[1].scatter(
        avaliacao.previsao, avaliacao.residuo, s=16, alpha=0.40,
        color=cor, edgecolors="none", rasterized=True,
    )
    axes[1].axhline(0, color="#4d4d4d", linestyle="--", linewidth=1.2)
    axes[1].set(
        xlabel="Número de anéis previsto",
        ylabel="Resíduo\n(observado menos previsto)",
    )
    axes[1].set_title(f"Resíduos da {ROTULOS[nome_modelo].lower()}")
    for ax in axes:
        ax.grid(color="#dce1e5", alpha=0.65)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=1.4)
    guardar_figura(fig, "u14_abalone_teste")


def avaliar() -> None:
    protocolo = ler_json(PROTOCOLO)
    preparacao = ler_json(RESULTADOS / "preparacao.json")
    caminho_decisao = RESULTADOS / "decisao.json"
    decisao = ler_json(caminho_decisao)
    if decisao["protocol_sha256"] != sha256(PROTOCOLO):
        raise ValueError("O protocolo mudou depois da seleção.")
    caminho_dev = DADOS / "abalone_desenvolvimento.csv"
    caminho_teste = DADOS / "abalone_teste_reservado.csv"
    if decisao["development_sha256"] != sha256(caminho_dev):
        raise ValueError("O desenvolvimento não corresponde à seleção registada.")
    if preparacao["test_sha256"] != sha256(caminho_teste):
        raise ValueError("O teste não corresponde à divisão preparada.")
    desenvolvimento = ler_csv(caminho_dev)
    teste = ler_csv(caminho_teste)
    confirmar_tabela(desenvolvimento, protocolo)
    confirmar_tabela(teste, protocolo)
    if set(desenvolvimento.id_registo) & set(teste.id_registo):
        raise ValueError("Há registos em ambas as partes.")
    modelos = construir_modelos(protocolo)
    escolhido = decisao["selected_model"]
    nomes_avaliados = list(dict.fromkeys(["Referencia", escolhido]))
    X_dev = desenvolvimento[protocolo["features"]]
    y_dev = desenvolvimento[protocolo["target"]]
    X_teste = teste[protocolo["features"]]
    y_teste = teste[protocolo["target"]]
    resultado = {}
    previsoes = {}
    for nome in nomes_avaliados:
        ajustado = clone(modelos[nome]).fit(X_dev, y_dev)
        previsto = ajustado.predict(X_teste)
        previsoes[nome] = previsto
        resultado[nome] = medidas(y_teste, previsto)

    avaliacao = teste[["id_registo", "Rings"] + protocolo["features"]].copy()
    avaliacao["previsao"] = previsoes[escolhido]
    avaliacao["previsao_referencia"] = previsoes["Referencia"]
    avaliacao["residuo"] = avaliacao.Rings - avaliacao.previsao
    avaliacao["erro_absoluto"] = avaliacao.residuo.abs()
    guardar_csv(avaliacao, RESULTADOS / "avaliacao_por_registo.csv")
    guardar_csv(
        avaliacao.sort_values(["erro_absoluto", "id_registo"], ascending=[False, True]),
        RESULTADOS / "erros_ordenados.csv",
    )
    guardar_csv(
        teste[["id_registo", "Rings"]], RESULTADOS / "referencia_avaliacao.csv",
    )
    # A ordem inversa torna necessário ligar os ficheiros pelo identificador.
    guardar_csv(
        avaliacao[["id_registo", "previsao"]].iloc[::-1],
        RESULTADOS / "previsoes_avaliacao.csv",
    )
    guardar_csv(
        pd.DataFrame([{"procedimento": k, **v} for k, v in resultado.items()]),
        RESULTADOS / "teste_resumo.csv",
    )
    maximo = avaliacao.loc[avaliacao.erro_absoluto.idxmax()].to_dict()
    guardar_json(RESULTADOS / "resultado_final.json", {
        "utc": agora(), "decision_utc": decisao["utc"],
        "decision_sha256": sha256(caminho_decisao),
        "protocol_sha256": sha256(PROTOCOLO),
        "selected_model": escolhido, "tested_procedures": nomes_avaliados,
        "n_development": len(desenvolvimento), "n_test": len(teste),
        "training_median": float(y_dev.median()), "cv_results": decisao["cv_results"],
        "test_results": resultado, "largest_error_record": maximo,
        "test_rings_range": [int(y_teste.min()), int(y_teste.max())],
        "versions": versoes(),
        "note": "Resultados descritivos desta divisão; o teste não orientou a escolha do procedimento.",
    })
    figuras_teste(avaliacao, escolhido)
    print(pd.DataFrame(resultado).T.to_string())
    print("Ficheiros por identificador, erros e figuras guardados.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=["preparar", "desenvolver", "avaliar", "tudo"])
    args = parser.parse_args()
    if args.etapa in ["preparar", "tudo"]:
        preparar()
    if args.etapa in ["desenvolver", "tudo"]:
        desenvolver()
    if args.etapa in ["avaliar", "tudo"]:
        avaliar()


if __name__ == "__main__":
    main()
