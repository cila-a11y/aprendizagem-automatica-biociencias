"""Preparar a cópia de desenvolvimento usada no capítulo 8.

A fonte é a cópia de Breast Cancer Wisconsin (Diagnostic) incluída no
scikit-learn. Não é necessário acesso à rede. Este script guarda apenas
os 455 registos de desenvolvimento; não guarda nem avalia o teste reservado.

Executar a partir de qualquer pasta:
    python codigo/preparar_dados.py
"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


RAIZ = Path(__file__).resolve().parents[1]
HASH_ESPERADO = "997edc11984a788044ef3ceb17595f0b77b1aac91a7186e8afd3c2dde3b6153d"


def main():
    fonte = load_breast_cancer(as_frame=True)
    assert list(fonte.target_names) == ["malignant", "benign"]
    dados = fonte.data.copy()
    dados.insert(0, "id_registo", np.arange(1, len(dados) + 1))
    dados["maligno"] = 1 - fonte.target
    assert len(dados) == 569

    desenvolvimento, _ = train_test_split(
        dados, test_size=0.20, stratify=dados["maligno"], random_state=2026
    )
    desenvolvimento = desenvolvimento.sort_values("id_registo").reset_index(drop=True)
    assert desenvolvimento["maligno"].value_counts().to_dict() == {0: 285, 1: 170}
    conteudo = desenvolvimento.to_csv(index=False)
    hash_obtido = sha256(conteudo.encode("utf-8")).hexdigest()
    if hash_obtido != HASH_ESPERADO:
        raise ValueError(
            "A fonte ou a representação CSV difere da versão verificada. "
            "A cópia existente não foi substituída."
        )

    destino = RAIZ / "dados" / "wisconsin_desenvolvimento.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(conteudo, encoding="utf-8")
    metadados = {
        "fonte": "sklearn.datasets.load_breast_cancer(as_frame=True)",
        "conjunto": "Breast Cancer Wisconsin (Diagnostic)",
        "doi": "https://doi.org/10.24432/C5DW2B",
        "licenca_fonte": "CC BY 4.0, conforme a indicação do repositório UCI",
        "identificador": "id_registo: posição original, de 1 a 569; não é preditor",
        "classe_original": {"0": "maligno", "1": "benigno"},
        "recodificacao": "maligno = 1 - target",
        "divisao": "train_test_split(test_size=0.20, stratify=maligno, random_state=2026)",
        "ordem_guardada": "id_registo crescente",
        "registos_desenvolvimento": 455,
        "malignos": 170,
        "benignos": 285,
        "registos_reservados_excluidos_deste_pacote": 114,
        "teste_avaliado": False,
        "sha256_csv_desenvolvimento": hash_obtido,
        "versoes": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    resultados = RAIZ / "resultados"
    resultados.mkdir(parents=True, exist_ok=True)
    (resultados / "proveniencia_dados.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Cópia de desenvolvimento guardada: {destino}")
    print("455 registos: 285 benignos e 170 malignos. O teste não foi guardado nem avaliado.")


if __name__ == "__main__":
    main()
