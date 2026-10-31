"""Reconstituir apenas o desenvolvimento dos exemplos dos capítulos 5 e 6.

A fonte pública acompanha este pacote. Não se descarregam dados durante a
execução, não se analisam respostas de teste e não se cria um ficheiro de teste.
"""

from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
FONTE = RAIZ / "dados" / "penguins_original.csv"
SHA256_FONTE = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"


def main():
    if sha256(FONTE.read_bytes()).hexdigest() != SHA256_FONTE:
        raise ValueError("A cópia da fonte não coincide com a cópia verificada.")
    original = pd.read_csv(FONTE)
    original.insert(0, "id_registo", range(1, len(original) + 1))
    desenvolvimento, _ = train_test_split(
        original, test_size=0.2, random_state=2026
    )
    desenvolvimento = desenvolvimento.sort_values("id_registo")
    assert len(original) == 344 and len(desenvolvimento) == 275
    destino = RAIZ / "dados" / "penguins_desenvolvimento.csv"
    desenvolvimento.to_csv(destino, index=False)
    print("Desenvolvimento preparado:", destino)


if __name__ == "__main__":
    main()
