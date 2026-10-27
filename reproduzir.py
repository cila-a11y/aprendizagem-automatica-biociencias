#!/usr/bin/env python3
"""Executa os percursos de reprodução dos materiais de apoio ao livro."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

RAIZ = Path(__file__).resolve().parent


def executar(caminho: str, *argumentos: str) -> None:
    ambiente = os.environ.copy()
    ambiente.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "biociencias_mpl"))
    # Uma linha de execução por biblioteca evita multiplicar os recursos
    # quando vários modelos são ajustados na mesma sessão.
    for nome in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        ambiente[nome] = "1"
    print(f"\nExecutar {caminho} {' '.join(argumentos)}", flush=True)
    subprocess.run(
        [sys.executable, str(RAIZ / caminho), *argumentos],
        cwd=RAIZ, env=ambiente, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=[
        "dados", "fundamentos", "modelos", "abalone", "previsoes",
        "desenvolvimento", "teste", "tudo", "verificar",
    ])
    args = parser.parse_args()
    etapa = args.etapa
    if etapa in ["dados", "tudo"]:
        executar("reproducao/abalone/codigo/reproduzir_abalone.py", "preparar")
        executar("reproducao/dados/preparar_dados_iniciais.py")
    if etapa in ["fundamentos", "desenvolvimento", "tudo"]:
        executar("reproducao/fundamentos/reproduzir_fundamentos.py")
    if etapa in ["modelos", "desenvolvimento", "tudo"]:
        executar("reproducao/modelos/reproduzir_modelos.py", "desenvolvimento")
    if etapa in ["abalone", "desenvolvimento", "tudo"]:
        executar("reproducao/abalone/codigo/reproduzir_abalone.py", "desenvolver")
    if etapa in ["teste", "tudo"]:
        executar("reproducao/modelos/reproduzir_modelos.py", "teste")
        executar("reproducao/abalone/codigo/reproduzir_abalone.py", "avaliar")
    if etapa in ["previsoes", "tudo"]:
        executar("reproducao/previsoes/avaliar_previsoes.py", "classificacao")
        executar("reproducao/previsoes/avaliar_previsoes.py", "abalone")
    if etapa in ["verificar", "tudo"]:
        executar("verificar_materiais.py")


if __name__ == "__main__":
    main()
