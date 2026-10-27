#!/usr/bin/env python3
"""Prepara as cópias de dados de Palmer Penguins e Wisconsin.

Este script não ajusta modelos nem calcula medidas de desempenho.
Reconstitui apenas as fontes e as divisões descritas no livro. Os dados
Abalone são preparados pelo script próprio e só são incluídos nos ZIP.
"""

from pathlib import Path
import hashlib
import json
from urllib.request import urlopen
import zipfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "dados"
ORIGENS = Path(__file__).resolve().parent / "origens.json"
PENGUINS_URL = (
    "https://raw.githubusercontent.com/allisonhorst/"
    "palmerpenguins/main/inst/extdata/penguins.csv"
)
PENGUINS_SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"


def sha256(caminho):
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def guardar_csv(tabela, caminho):
    tabela.sort_values("id_registo").to_csv(
        caminho, index=False, float_format="%.17g", na_rep=""
    )


def confirmar_divisao(original, desenvolvimento, teste):
    ids_dev = set(desenvolvimento.id_registo)
    ids_teste = set(teste.id_registo)
    if ids_dev & ids_teste:
        raise ValueError("Há identificadores em desenvolvimento e teste.")
    if ids_dev | ids_teste != set(original.id_registo):
        raise ValueError("A divisão não conserva todas as linhas.")
    if not original.id_registo.is_unique:
        raise ValueError("Há identificadores repetidos.")


def main():
    DADOS.mkdir(parents=True, exist_ok=True)
    origem_penguins = DADOS / "penguins_original.csv"
    if not origem_penguins.exists():
        with urlopen(PENGUINS_URL, timeout=60) as resposta:
            origem_penguins.write_bytes(resposta.read())
    if sha256(origem_penguins) != PENGUINS_SHA256:
        raise ValueError("A cópia original de Penguins difere da fonte documentada.")

    penguins = pd.read_csv(origem_penguins, float_precision="round_trip")
    penguins.insert(0, "id_registo", np.arange(1, len(penguins) + 1))
    penguins_dev, penguins_teste = train_test_split(
        penguins, test_size=0.2, shuffle=True, random_state=2026
    )
    confirmar_divisao(penguins, penguins_dev, penguins_teste)
    if (len(penguins), len(penguins_dev), len(penguins_teste)) != (344, 275, 69):
        raise ValueError("As dimensões de Penguins não correspondem às descritas.")
    if int(penguins_dev.body_mass_g.notna().sum()) != 273:
        raise ValueError("A contagem de massas observadas não corresponde à descrita.")
    campos_regressao = ["flipper_length_mm", "bill_length_mm", "body_mass_g"]
    if not penguins_teste[campos_regressao].notna().all().all():
        raise ValueError("O teste de Penguins tem omissões nas colunas usadas.")

    copia = load_breast_cancer(as_frame=True)
    wisconsin = copia.data.copy()
    wisconsin["maligno"] = 1 - copia.target
    wisconsin.insert(0, "id_registo", np.arange(1, len(wisconsin) + 1))
    wisconsin_dev, wisconsin_teste = train_test_split(
        wisconsin, test_size=0.2, shuffle=True, random_state=2026,
        stratify=wisconsin["maligno"],
    )
    confirmar_divisao(wisconsin, wisconsin_dev, wisconsin_teste)
    if (len(wisconsin), len(wisconsin_dev), len(wisconsin_teste)) != (569, 455, 114):
        raise ValueError("As dimensões de Wisconsin não correspondem às descritas.")
    if wisconsin_dev.maligno.value_counts().to_dict() != {0: 285, 1: 170}:
        raise ValueError("A composição do desenvolvimento de Wisconsin mudou.")
    if wisconsin_teste.maligno.value_counts().to_dict() != {0: 72, 1: 42}:
        raise ValueError("A composição do teste de Wisconsin mudou.")

    tabelas = {
        "penguins_desenvolvimento.csv": penguins_dev,
        "penguins_teste_reservado.csv": penguins_teste,
        "wisconsin_desenvolvimento.csv": wisconsin_dev,
        "wisconsin_teste_reservado.csv": wisconsin_teste,
    }
    for nome, tabela in tabelas.items():
        guardar_csv(tabela, DADOS / nome)

    # Os ficheiros Abalone são incluídos sem os modificar.
    for sufixo, nome_zip in [
        ("desenvolvimento", "dados_desenvolvimento.zip"),
        ("teste_reservado", "dados_teste_reservado.zip"),
    ]:
        ficheiros = [DADOS / f"{caso}_{sufixo}.csv" for caso in ["penguins", "wisconsin", "abalone"]]
        if not all(caminho.is_file() for caminho in ficheiros):
            raise FileNotFoundError("Prepare primeiro os dados Abalone com o script próprio.")
        with zipfile.ZipFile(RAIZ / nome_zip, "w", compression=zipfile.ZIP_DEFLATED) as arquivo:
            for caminho in ficheiros:
                arquivo.write(caminho, arcname=f"dados/{caminho.name}")

    metadados = {
        "scope": "Preparação das cópias, codificações e divisões dos dados. Os modelos são executados pelos percursos próprios.",
        "random_state": 2026,
        "test_size": 0.2,
        "id_registo": "Número da linha original, começando em 1; identificadores próprios de cada conjunto.",
        "penguins": {
            "url": PENGUINS_URL,
            "original_sha256": PENGUINS_SHA256,
            "source_license": "CC0",
            "n_original": 344,
            "n_development": 275,
            "n_development_mass_observed": 273,
            "n_test": 69,
            "stratified": False,
        },
        "wisconsin": {
            "source": "sklearn.datasets.load_breast_cancer(as_frame=True)",
            "doi": "10.24432/C5DW2B",
            "source_license": "CC BY 4.0, conforme o repositório UCI",
            "n_original": 569,
            "n_development": 455,
            "n_test": 114,
            "development_classes": {"0_benigno": 285, "1_maligno": 170},
            "test_classes": {"0_benigno": 72, "1_maligno": 42},
            "response_conversion": "maligno = 1 - target",
            "stratified_by": "maligno",
        },
        "prepared_sha256": {nome: sha256(DADOS / nome) for nome in tabelas},
        "versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    ORIGENS.write_text(json.dumps(metadados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Dados iniciais preparados; contagens e identificadores conferidos.")
    print("Arquivos de desenvolvimento e teste criados com três CSV cada.")
    print("Para ajustar e avaliar os procedimentos, execute o percurso de desenvolvimento.")


if __name__ == "__main__":
    main()
