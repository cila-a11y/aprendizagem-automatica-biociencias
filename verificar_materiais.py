#!/usr/bin/env python3
"""Confere os dados e os principais resultados da execução de referência.

Não reajusta modelos. Uma divergência assinala uma diferença em relação aos
materiais distribuídos; consulte as entradas e as versões antes de interpretar.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, roc_auc_score

RAIZ = Path(__file__).resolve().parent
FALHAS = []
N_VERIFICACOES = 0


def conferir(condicao, descricao):
    global N_VERIFICACOES
    N_VERIFICACOES += 1
    if not bool(condicao):
        FALHAS.append(descricao)


def perto(valor, esperado, casas, descricao):
    conferir(np.isfinite(valor) and abs(valor - esperado) < 0.5 * 10 ** (-casas), descricao)


def ler(nome):
    return pd.read_csv(RAIZ / nome, float_precision="round_trip")


def main():
    manifesto = json.loads((RAIZ / "reproducao/dados/manifesto.json").read_text())
    for nome, esperado in manifesto["sha256"].items():
        caminho = RAIZ / nome
        conferir(caminho.is_file(), f"Ficheiro em falta: {nome}")
        if caminho.is_file():
            conferir(hashlib.sha256(caminho.read_bytes()).hexdigest() == esperado,
                     f"Conteúdo dos dados diferente: {nome}")
    dados = {}
    for conjunto, quantidades in {"penguins": (275, 69), "wisconsin": (455, 114), "abalone": (3341, 836)}.items():
        partes = []
        for sufixo, n in zip(["desenvolvimento", "teste_reservado"], quantidades):
            t = ler(f"dados/{conjunto}_{sufixo}.csv")
            conferir(len(t) == n, f"Dimensão de {conjunto}/{sufixo}")
            conferir(t.id_registo.notna().all() and t.id_registo.is_unique,
                     f"Identificadores de {conjunto}/{sufixo}")
            conferir(t.id_registo.is_monotonic_increasing, f"Ordem de {conjunto}/{sufixo}")
            partes.append(t)
            dados[conjunto, sufixo] = t
        conferir(not (set(partes[0].id_registo) & set(partes[1].id_registo)),
                 f"Sobreposição entre desenvolvimento e teste de {conjunto}")
    for sufixo in ["desenvolvimento", "teste_reservado"]:
        with zipfile.ZipFile(RAIZ / f"dados_{sufixo}.zip") as arquivo:
            nomes = [f"dados/{d}_{sufixo}.csv" for d in ["penguins", "wisconsin", "abalone"]]
            conferir(set(arquivo.namelist()) == set(nomes), f"Conteúdo de dados_{sufixo}.zip")
            for nome in nomes:
                conferir(arquivo.read(nome) == (RAIZ / nome).read_bytes(), f"CSV no ZIP: {nome}")

    p = dados["penguins", "desenvolvimento"]
    conferir(p.body_mass_g.notna().sum() == 273, "Massas observadas dos pinguins")
    conferir(p.sex.isna().sum() == 11, "Omissões de sexo dos pinguins")
    perto(p.body_mass_g.mean(), 4208.6, 1, "Média de massa dos pinguins")
    grupos = json.loads((RAIZ / "reproducao/fundamentos/resultados/capitulo_05.json").read_text())
    conferir(grupos["contagens_K3"] == {"1": 104, "2": 98, "3": 71}, "Contagens K=3")
    for k, esperado in {"2": .538, "3": .434, "4": .388}.items():
        perto(grupos["silhuetas"][k], esperado, 3, f"Silhueta K={k}")
    acp = json.loads((RAIZ / "reproducao/fundamentos/resultados/capitulo_06.json").read_text())
    perto(acp["variabilidade_acumulada_percentagem"][1], 88.19, 2, "Variabilidade das duas primeiras componentes")
    conferir(acp["erro_maximo_reconstrucao_quatro_componentes"] < 1e-10, "Reconstrução ACP")

    wcv = ler("reproducao/modelos/resultados/wisconsin_validacao_resumo.csv").set_index("modelo")
    for modelo, valores in {
        "logistica": (.952, .010, .895), "arvore_2": (.912, .025, .888),
        "arvore_4": (.932, .025, .881), "floresta_1": (.928, .037, .873),
        "floresta_5": (.934, .036, .881), "rede_8": (.944, .032, .892),
    }.items():
        for campo, esperado in zip(["AUC_media_partes", "AUC_DP_partes", "acerto_medio_validacao"], valores):
            perto(wcv.loc[modelo, campo], esperado, 3, f"Wisconsin {modelo} {campo}")
    pcv = ler("reproducao/modelos/resultados/pinguins_validacao_resumo.csv").set_index("modelo")
    for modelo, valores in {
        "linear_multipla": (320.1, 23.4, 397.9), "arvore_2": (342.1, 25.1, 433.4),
        "floresta_1": (305.8, 10.7, 391.1), "floresta_5": (295.8, 12.3, 378.5),
    }.items():
        for campo, esperado in zip(["MAE_medio_partes_g", "MAE_DP_partes_g", "RMSE_medio_partes_g"], valores):
            perto(pcv.loc[modelo, campo], esperado, 1, f"Pinguins {modelo} {campo}")

    decisao = json.loads((RAIZ / "reproducao/modelos/resultados/decisao.json").read_text())
    conferir(decisao["wisconsin"]["modelo"] == "logistica" and decisao["wisconsin"]["limiar"] == .2,
             "Procedimento escolhido para Wisconsin")
    conferir(decisao["penguins"]["modelo"] == "floresta_5", "Procedimento escolhido para pinguins")
    wt = ler("reproducao/modelos/resultados/wisconsin_teste_previsoes.csv")
    conferir(set(wt.id_registo) == set(dados["wisconsin", "teste_reservado"].id_registo), "IDs do teste Wisconsin")
    conferir(wt.id_registo.is_unique, "Previsões Wisconsin sem duplicações")
    matriz = confusion_matrix(wt.maligno, wt.classe_prevista_limiar_020, labels=[0, 1])
    conferir(np.array_equal(matriz, [[54, 18], [2, 40]]), "Matriz do teste Wisconsin")
    perto(roc_auc_score(wt.maligno, wt.p_maligno), .946, 3, "AUC do teste Wisconsin")
    pt = ler("reproducao/modelos/resultados/pinguins_teste_previsoes.csv")
    conferir(pt.id_registo.is_unique, "Previsões dos pinguins sem duplicações")
    conferir(set(pt.id_registo) == set(dados["penguins", "teste_reservado"].id_registo), "IDs do teste dos pinguins")
    perto(mean_absolute_error(pt.body_mass_g, pt.previsao_floresta_5), 293.7, 1, "MAE do teste dos pinguins")
    perto((pt.body_mass_g - pt.previsao_floresta_5).abs().max(), 1027.2, 1, "Maior erro dos pinguins")

    acv = ler("reproducao/abalone/resultados/validacao_resumo.csv").set_index("procedimento")
    for modelo, esperado in {"Referencia": 2.371464, "Regressao_linear": 1.915101, "Floresta": 1.914672}.items():
        perto(acv.loc[modelo, "mae_medio"], esperado, 6, f"MAE de validação Abalone {modelo}")
    at = ler("reproducao/abalone/resultados/teste_resumo.csv").set_index("procedimento")
    perto(at.loc["Floresta", "mae"], 1.907, 3, "MAE do teste Abalone")
    perto(at.loc["Referencia", "mae"], 2.355, 3, "MAE da referência Abalone")
    perto(at.loc["Floresta", "max_absolute_error"], 14.853, 3, "Maior erro Abalone")
    ar = ler("reproducao/previsoes/resultados_abalone/avaliacao_por_registo.csv")
    conferir(ar.id_registo.is_unique and set(ar.id_registo) == set(dados["abalone", "teste_reservado"].id_registo),
             "Correspondência dos identificadores Abalone")
    conferir(np.isclose(mean_absolute_error(ar.Rings, ar.previsao), at.loc["Floresta", "mae"], atol=1e-12, rtol=0),
             "MAE reconstituído dos ficheiros Abalone")

    erros = ler("exemplo_previsoes/resultados/erros_por_amostra.csv")
    conferir(dict(zip(erros.id_registo, erros.tipo_erro)) == {"A03": "FP", "A04": "FN"},
             "Erros do exemplo fictício")
    for nome in ["fundamentos.ipynb", "modelos.ipynb", "estudo_abalone.ipynb", "previsoes_ficheiros.ipynb"]:
        nb = json.loads((RAIZ / nome).read_text())
        conferir(nb["nbformat"] == 4 and bool(nb["cells"]), f"Caderno {nome}")

    if FALHAS:
        print(f"{len(FALHAS)} divergência(s) em {N_VERIFICACOES} verificações:")
        for falha in FALHAS:
            print(f"- {falha}")
        raise SystemExit(1)
    print(f"{N_VERIFICACOES} verificações concluídas: dados, divisões, arquivos e resultados correspondem à referência.")


if __name__ == "__main__":
    main()
