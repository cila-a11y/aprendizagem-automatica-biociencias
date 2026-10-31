"""Conferir a integridade dos ficheiros distribuídos, sem ajustar modelos.

Usa apenas a biblioteca padrão de Python. Não altera os ficheiros nem lê
os dados para calcular desempenho. O manifesto não é uma assinatura de
autoria, uma prova de pré-registo ou uma validação científica.
"""

import hashlib
import json
from pathlib import Path
import sys
import zipfile


RAIZ = Path(__file__).resolve().parent
ZIP_ESPERADOS = {
    "dados_desenvolvimento.zip": ["penguins_desenvolvimento.csv", "wisconsin_desenvolvimento.csv"],
    "dados_teste_reservado.zip": ["penguins_teste_reservado.csv", "wisconsin_teste_reservado.csv"],
}


def main():
    manifesto = json.loads((RAIZ / "manifesto_sha256.json").read_text(encoding="utf-8"))
    erros = []
    nomes = set()
    for entrada in manifesto["ficheiros"]:
        nome = entrada["ficheiro"]
        caminho = (RAIZ / nome).resolve()
        if not caminho.is_relative_to(RAIZ) or nome in nomes:
            erros.append(f"Entrada inválida no manifesto: {nome}")
            continue
        nomes.add(nome)
        if not caminho.is_file():
            erros.append(f"Falta o ficheiro: {nome}")
            continue
        conteudo = caminho.read_bytes()
        if len(conteudo) != entrada["bytes"] or hashlib.sha256(conteudo).hexdigest() != entrada["sha256"]:
            erros.append(f"Ficheiro diferente da cópia distribuída: {nome}")
    for nome_zip, csvs in ZIP_ESPERADOS.items():
        caminho = RAIZ / nome_zip
        if not caminho.is_file():
            continue
        try:
            with zipfile.ZipFile(caminho) as arquivo:
                esperados = {f"dados/{nome}" for nome in csvs}
                if set(arquivo.namelist()) != esperados or len(arquivo.namelist()) != len(esperados):
                    erros.append(f"Conteúdo inesperado no arquivo: {nome_zip}")
                    continue
                if arquivo.testzip() is not None:
                    erros.append(f"Falha na integridade do ZIP: {nome_zip}")
                for nome in csvs:
                    referencia = RAIZ / "cap14_15/dados" / nome
                    if referencia.is_file() and arquivo.read(f"dados/{nome}") != referencia.read_bytes():
                        erros.append(f"Cópias divergentes de {nome} em {nome_zip}")
        except (OSError, zipfile.BadZipFile) as erro:
            erros.append(f"Não foi possível verificar {nome_zip}: {erro}")
    if erros:
        for erro in erros:
            print(erro)
        print("A cópia não coincide integralmente com a distribuição. Investigue as diferenças.")
        return 1
    print(f"Integridade confirmada: {len(nomes)} ficheiros e conteúdo dos dois ZIP de dados.")
    print("Não foram ajustados modelos. Esta verificação não demonstra validade científica.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
