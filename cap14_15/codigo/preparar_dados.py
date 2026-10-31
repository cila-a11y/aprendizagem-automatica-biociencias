"""Reconstituir as cópias fixas de desenvolvimento e teste dos casos finais.

Usa fontes já locais. Não escolhe modelos nem calcula desempenho. Os testes
são os complementos das divisões usadas nos capítulos anteriores. A abertura
estatística do teste cabe exclusivamente a avaliar_teste_final.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from modelos import DADOS, HASHES_DESENVOLVIMENTO, guardar_json, sha256_ficheiro, versoes


SHA_PINGUINS_ORIGINAL = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"


def guardar_copia(nome, tabela):
    conteudo = tabela.sort_values("id_registo").to_csv(index=False)
    sha = hashlib.sha256(conteudo.encode("utf-8")).hexdigest()
    if nome in HASHES_DESENVOLVIMENTO and sha != HASHES_DESENVOLVIMENTO[nome]:
        raise ValueError(f"O desenvolvimento reconstituído difere do já utilizado: {nome}")
    caminho = DADOS / nome
    if caminho.exists() and sha256_ficheiro(caminho) != sha:
        raise ValueError(f"O ficheiro existente difere; não foi substituído: {nome}")
    caminho.write_text(conteudo, encoding="utf-8")
    return {"ficheiro": nome, "n": len(tabela), "sha256": sha,
            "ids": sorted(tabela.id_registo.astype(int).tolist())}


def main():
    DADOS.mkdir(parents=True, exist_ok=True)
    fonte_p = DADOS / "penguins_original.csv"
    if sha256_ficheiro(fonte_p) != SHA_PINGUINS_ORIGINAL:
        raise ValueError("A fonte local dos pinguins difere da cópia verificada.")
    p = pd.read_csv(fonte_p)
    p.insert(0, "id_registo", np.arange(1, len(p) + 1))
    p_dev, p_test = train_test_split(p, test_size=.2, random_state=2026)

    fonte_w = load_breast_cancer(as_frame=True)
    assert list(fonte_w.target_names) == ["malignant", "benign"]
    w = fonte_w.data.copy()
    w.insert(0, "id_registo", np.arange(1, len(w) + 1))
    w["maligno"] = 1 - fonte_w.target
    w_dev, w_test = train_test_split(
        w, test_size=.2, stratify=w["maligno"], random_state=2026
    )
    assert len(w) == 569 and len(p) == 344
    assert set(w_dev.id_registo).isdisjoint(w_test.id_registo)
    assert set(p_dev.id_registo).isdisjoint(p_test.id_registo)
    ficheiros = [
        guardar_copia("wisconsin_desenvolvimento.csv", w_dev),
        guardar_copia("wisconsin_teste_reservado.csv", w_test),
        guardar_copia("penguins_desenvolvimento.csv", p_dev),
        guardar_copia("penguins_teste_reservado.csv", p_test),
    ]
    guardar_json("proveniencia_dados.json", {
        "estatuto": "Reprodução didática; os resultados dos testes já constam do livro.",
        "fonte_wisconsin": "sklearn.datasets.load_breast_cancer(as_frame=True)",
        "doi_wisconsin": "https://doi.org/10.24432/C5DW2B",
        "licenca_wisconsin": "CC BY 4.0, conforme a fonte UCI",
        "fonte_penguins": "cópia local de palmerpenguins/inst/extdata/penguins.csv",
        "url_penguins": "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv",
        "sha256_penguins_original": SHA_PINGUINS_ORIGINAL,
        "licenca_penguins": "CC0",
        "identificador": "id_registo é a posição na fonte original, começando em 1; não é um preditor",
        "regra_divisao": "train_test_split(test_size=.2, random_state=2026); estratificação por maligno apenas em Wisconsin",
        "ordem": "id_registo crescente em cada cópia",
        "ficheiros": ficheiros, "versoes": versoes(),
        "desempenho_calculado_nesta_etapa": False,
    })
    print("Cópias reconstituídas; os dois desenvolvimentos são idênticos aos capítulos anteriores.")
    print("Nenhuma medida de desempenho foi calculada na preparação.")


if __name__ == "__main__":
    main()
