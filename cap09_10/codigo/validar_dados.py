"""Verificar as duas cópias de desenvolvimento incluídas no pacote.

Executar de qualquer pasta: python codigo/validar_dados.py
Não descarrega dados, não substitui ficheiros e não lê o teste reservado.
"""

from hashlib import sha256
from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
HASHES = {
    "penguins_desenvolvimento.csv": "8f3a9c1a2acef4ce13f8ba13cd30927194b0c30b73d83eaa82d65d269abb03d0",
    "wisconsin_desenvolvimento.csv": "997edc11984a788044ef3ceb17595f0b77b1aac91a7186e8afd3c2dde3b6153d",
}


def validar():
    tabelas = {}
    for nome, esperado in HASHES.items():
        caminho = RAIZ / "dados" / nome
        obtido = sha256(caminho.read_bytes()).hexdigest()
        if obtido != esperado:
            raise ValueError(f"A cópia de {nome} não corresponde à versão verificada.")
        tabela = pd.read_csv(caminho)
        assert tabela["id_registo"].is_unique
        assert tabela["id_registo"].is_monotonic_increasing
        tabelas[nome] = tabela
    pinguins = tabelas["penguins_desenvolvimento.csv"]
    wisconsin = tabelas["wisconsin_desenvolvimento.csv"]
    assert len(pinguins) == 275
    assert pinguins.loc[pinguins["body_mass_g"].isna(), "id_registo"].tolist() == [4, 272]
    observados = pinguins.dropna(subset=["body_mass_g"]).reset_index(drop=True)
    assert len(observados) == 273
    assert not observados[["flipper_length_mm", "bill_length_mm"]].isna().any().any()
    assert len(wisconsin) == 455
    assert wisconsin["maligno"].value_counts().to_dict() == {0: 285, 1: 170}
    assert not wisconsin[["mean radius", "mean texture"]].isna().any().any()
    return wisconsin, observados


if __name__ == "__main__":
    validar()
    print("Dados verificados: 455 amostras Wisconsin e 273 pinguins com massa observada.")
    print("O teste reservado não foi utilizado.")
