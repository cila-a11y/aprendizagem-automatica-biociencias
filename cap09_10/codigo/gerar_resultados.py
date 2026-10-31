"""Reproduzir os cálculos dos capítulos 9 e 10.

Executar de qualquer pasta: python codigo/gerar_resultados.py
Usa exclusivamente as cópias de desenvolvimento, sem consultar o teste.
Os acertos na classificação aplicam explicitamente p >= 0.50, incluindo
os empates na classe positiva. As médias entre partes e as medidas das
previsões reunidas são guardadas separadamente.
"""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, root_mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from validar_dados import HASHES, validar


RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"
SEMENTE = 2026
CONFIGURACOES = [
    ("arvore_profundidade_2", "Árvore profundidade 2", 2, 5),
    ("arvore_profundidade_4", "Árvore profundidade 4", 4, 5),
    ("arvore_sem_limite", "Árvore sem limite", None, 1),
]
AMOSTRAS_BOOTSTRAP = [
    list("AACDDFHH"), list("BCCEFFGG"), list("ABDDEGHH"),
]


def guardar_json(nome, objeto):
    (RESULTADOS / nome).write_text(
        json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def procedimentos():
    classificadores = {
        "logistica": ("Logística", make_pipeline(
            StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=1000)
        )),
    }
    regressores = {
        "linear": ("Regressão linear", make_pipeline(
            SimpleImputer(strategy="median"), LinearRegression()
        )),
    }
    for chave, nome, profundidade, folha in CONFIGURACOES:
        classificadores[chave] = (nome, DecisionTreeClassifier(
            criterion="gini", max_depth=profundidade,
            min_samples_leaf=folha, random_state=SEMENTE,
        ))
        regressores[chave] = (nome, make_pipeline(
            SimpleImputer(strategy="median"), DecisionTreeRegressor(
                criterion="squared_error", max_depth=profundidade,
                min_samples_leaf=folha, random_state=SEMENTE,
            ),
        ))
    for folha in [1, 5]:
        chave, nome = f"floresta_folha_{folha}", f"Floresta folha mínima {folha}"
        opcoes = dict(n_estimators=200, max_features=1, bootstrap=True,
                      max_depth=None, min_samples_leaf=folha,
                      random_state=SEMENTE, n_jobs=1)
        classificadores[chave] = (nome, RandomForestClassifier(criterion="gini", **opcoes))
        regressores[chave] = (nome, make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(criterion="squared_error", **opcoes),
        ))
    regressores["mediana"] = ("Referência mediana", make_pipeline(
        SimpleImputer(strategy="median"), DummyRegressor(strategy="median")
    ))
    return classificadores, regressores


def medidas_classificacao(y, p):
    previstas = (p >= .50).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, previstas, labels=[0, 1]).ravel()
    return {"AUC": float(roc_auc_score(y, p)),
            "acerto": float(np.mean(previstas == np.asarray(y))),
            "VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp),
            "empates_050": int(np.count_nonzero(p == .50))}


def medidas_regressao(y, p):
    return {"MAE_g": float(mean_absolute_error(y, p)),
            "RMSE_g": float(root_mean_squared_error(y, p))}


def avaliar(tarefa, dados, colunas, resposta, modelos, particoes):
    X, y = dados[colunas], dados[resposta]
    n_parte = np.zeros(len(dados), dtype=int)
    partes = []
    for numero, (treino, validacao) in enumerate(particoes, start=1):
        n_parte[validacao] = numero
        assert not np.intersect1d(treino, validacao).size
        partes.append({
            "parte": numero, "n_treino": len(treino), "n_validacao": len(validacao),
            "ids_treino": dados.iloc[treino]["id_registo"].astype(int).tolist(),
            "ids_validacao": dados.iloc[validacao]["id_registo"].astype(int).tolist(),
        })
    assert np.all(n_parte > 0)
    guardar_json(f"particoes_{tarefa}.json", partes)
    previsoes = dados[["id_registo", *colunas, resposta]].copy()
    previsoes["parte_validacao"] = n_parte
    linhas, agregadas, resumos, agregacao = [], [], [], None
    indice_exemplo = int(np.flatnonzero(dados["id_registo"].to_numpy() == 1)[0])

    for chave, (nome, prototipo) in modelos.items():
        p_oof = np.zeros(len(X))
        linhas_modelo = []
        for numero, (treino, validacao) in enumerate(particoes, start=1):
            modelo = clone(prototipo).fit(X.iloc[treino], y.iloc[treino])
            if tarefa == "classificacao":
                p_treino = modelo.predict_proba(X.iloc[treino])[:, 1]
                p_val = modelo.predict_proba(X.iloc[validacao])[:, 1]
                m_treino = medidas_classificacao(y.iloc[treino], p_treino)
                m_val = medidas_classificacao(y.iloc[validacao], p_val)
            else:
                p_treino = modelo.predict(X.iloc[treino])
                p_val = modelo.predict(X.iloc[validacao])
                m_treino = medidas_regressao(y.iloc[treino], p_treino)
                m_val = medidas_regressao(y.iloc[validacao], p_val)
            p_oof[validacao] = p_val
            linha = {"chave": chave, "procedimento": nome, "parte": numero,
                     "n_treino": len(treino), "n_validacao": len(validacao)}
            linha.update({f"treino_{k}": v for k, v in m_treino.items()})
            linha.update({f"validacao_{k}": v for k, v in m_val.items()})
            linhas.append(linha)
            linhas_modelo.append(linha)

            if chave == "floresta_folha_5" and indice_exemplo in validacao:
                assert indice_exemplo not in treino
                if tarefa == "classificacao":
                    floresta = modelo
                    entrada = X.iloc[[indice_exemplo]].to_numpy()
                    individuais = np.array([a.predict_proba(entrada)[0, 1]
                                              for a in floresta.estimators_])
                    media_modelo = float(modelo.predict_proba(X.iloc[[indice_exemplo]])[0, 1])
                else:
                    floresta = modelo[-1]
                    entrada = modelo[0].transform(X.iloc[[indice_exemplo]])
                    individuais = np.array([a.predict(entrada)[0] for a in floresta.estimators_])
                    media_modelo = float(modelo.predict(X.iloc[[indice_exemplo]])[0])
                assert len(individuais) == 200
                assert np.isclose(individuais.mean(), media_modelo, atol=1e-11)
                n_distintos = [int(len(np.unique(a))) for a in floresta.estimators_samples_]
                pd.DataFrame({
                    "arvore": np.arange(1, 201), "saida_individual": individuais,
                    "sorteios_bootstrap": len(treino),
                    "registos_distintos_bootstrap": n_distintos,
                }).to_csv(RESULTADOS / f"agregacao_{tarefa}_arvores.csv", index=False)
                agregacao = {
                    "id_registo": 1, "parte_validacao": numero, "n_treino": len(treino),
                    "resposta_observada": float(y.iloc[indice_exemplo]),
                    "entradas": {c: float(X.iloc[indice_exemplo][c]) for c in colunas},
                    "arvores": 200, "previsao_floresta": media_modelo,
                    "media_saidas_individuais": float(individuais.mean()),
                    "minimo_saida_individual": float(individuais.min()),
                    "maximo_saida_individual": float(individuais.max()),
                    "id_excluido_de_todos_os_treinos": True,
                    "interpretacao": "Distribuição de saídas de árvores para a mesma observação; não é um intervalo de incerteza.",
                }

        previsoes[chave] = p_oof
        if tarefa == "classificacao":
            medidas_agregadas = medidas_classificacao(y, p_oof)
            metricas = ["AUC", "acerto"]
        else:
            medidas_agregadas = medidas_regressao(y, p_oof)
            metricas = ["MAE_g", "RMSE_g"]
        agregadas.append({"chave": chave, "procedimento": nome, **medidas_agregadas})
        tabela_modelo = pd.DataFrame(linhas_modelo)
        resumo = {"chave": chave, "procedimento": nome}
        for origem in ["treino", "validacao"]:
            for metrica in metricas:
                valores = tabela_modelo[f"{origem}_{metrica}"]
                resumo[f"{origem}_{metrica}_media"] = float(valores.mean())
                resumo[f"{origem}_{metrica}_dp"] = float(valores.std(ddof=1))
        resumos.append(resumo)

    tabela_partes, tabela_resumos = pd.DataFrame(linhas), pd.DataFrame(resumos)
    tabela_partes.to_csv(RESULTADOS / f"metricas_{tarefa}_por_parte.csv", index=False)
    tabela_resumos.to_csv(RESULTADOS / f"resumo_{tarefa}_medias_partes.csv", index=False)
    pd.DataFrame(agregadas).to_csv(RESULTADOS / f"metricas_{tarefa}_previsoes_reunidas.csv", index=False)
    previsoes.to_csv(RESULTADOS / f"previsoes_{tarefa}_validacao.csv", index=False)
    guardar_json(f"agregacao_{tarefa}.json", agregacao)
    return tabela_resumos, tabela_partes, agregadas, agregacao


def extrair_arvore(tarefa, modelo, dados, colunas, resposta):
    if tarefa == "classificacao":
        arvore, X = modelo, dados[colunas]
    else:
        arvore, X = modelo[-1], modelo[0].transform(dados[colunas])
    estrutura = arvore.tree_
    y = dados[resposta].to_numpy()
    folhas = arvore.apply(X)
    caminhos = arvore.decision_path(X).toarray().astype(bool)
    ids_folhas = np.flatnonzero(estrutura.children_left == -1).tolist()
    letras = {no: chr(ord("A") + i) for i, no in enumerate(ids_folhas)}
    regras, profundidades = {}, {}

    def visitar(no, caminho, profundidade):
        regras[no], profundidades[no] = caminho, profundidade
        if estrutura.children_left[no] == -1:
            return
        coluna = colunas[estrutura.feature[no]]
        limiar = float(estrutura.threshold[no])
        visitar(int(estrutura.children_left[no]), caminho + [
            {"variavel": coluna, "operador": "<=", "limiar": limiar}
        ], profundidade + 1)
        visitar(int(estrutura.children_right[no]), caminho + [
            {"variavel": coluna, "operador": ">", "limiar": limiar}
        ], profundidade + 1)

    visitar(0, [], 0)
    nos = []
    for no in range(estrutura.node_count):
        indices = np.flatnonzero(caminhos[:, no])
        terminal = no in letras
        registo = {
            "no": no, "profundidade": profundidades[no], "folha": letras.get(no),
            "n_amostras": int(len(indices)),
            "variavel": None if terminal else colunas[estrutura.feature[no]],
            "limiar": None if terminal else float(estrutura.threshold[no]),
            "filho_esquerdo": None if terminal else int(estrutura.children_left[no]),
            "filho_direito": None if terminal else int(estrutura.children_right[no]),
            "impureza": float(estrutura.impurity[no]),
            "caminho": regras[no],
            "ids_registos": dados.iloc[indices]["id_registo"].astype(int).tolist(),
        }
        assert len(indices) == estrutura.n_node_samples[no]
        if tarefa == "classificacao":
            registo["benignas"] = int(np.sum(y[indices] == 0))
            registo["malignas"] = int(np.sum(y[indices] == 1))
            registo["p_maligno"] = float(np.mean(y[indices]))
            registo["classe_limiar_050"] = int(registo["p_maligno"] >= .50)
        else:
            registo["media_massa_g"] = float(np.mean(y[indices]))
            assert np.isclose(registo["media_massa_g"], estrutura.value[no, 0, 0])
        nos.append(registo)
    assert len(nos) == 7 and len(ids_folhas) == 4
    guardar_json(f"arvore_{tarefa}_completa.json", nos)
    resumo = [{k: v for k, v in no.items() if k not in ["caminho", "ids_registos"]} for no in nos]
    pd.DataFrame(resumo).to_csv(RESULTADOS / f"arvore_{tarefa}_nos.csv", index=False)
    pd.DataFrame({"id_registo": dados["id_registo"], "no_folha": folhas,
                  "folha": [letras[n] for n in folhas]}).to_csv(
        RESULTADOS / f"arvore_{tarefa}_folhas_por_registo.csv", index=False
    )
    return nos


def bootstrap_ficticio():
    original = list("ABCDEFGH")
    posicoes, multiplicidades, resumo = [], [], []
    for numero, amostra in enumerate(AMOSTRAS_BOOTSTRAP, start=1):
        ausentes = [letra for letra in original if letra not in amostra]
        resumo.append({"arvore_ficticia": numero, "sorteios": amostra,
                       "n_sorteios": 8, "n_distintos": len(set(amostra)), "ausentes": ausentes})
        for posicao, individuo in enumerate(amostra, start=1):
            posicoes.append({"arvore_ficticia": numero, "posicao_sorteio": posicao,
                             "individuo": individuo})
        for individuo in original:
            multiplicidades.append({"arvore_ficticia": numero, "individuo": individuo,
                                    "multiplicidade": amostra.count(individuo)})
    pd.DataFrame(posicoes).to_csv(RESULTADOS / "bootstrap_ficticio_posicoes.csv", index=False)
    pd.DataFrame(multiplicidades).to_csv(RESULTADOS / "bootstrap_ficticio_multiplicidades.csv", index=False)
    guardar_json("bootstrap_ficticio.json", {"treino_original": original, "amostras": resumo})
    return resumo


def verificar_permutacao(wisconsin):
    """Reproduzir a demonstração interna antecipada no exercício 10.4."""
    colunas = ["mean radius", "mean texture"]
    treino, validacao = train_test_split(
        np.arange(len(wisconsin)), test_size=.30,
        stratify=wisconsin["maligno"], random_state=SEMENTE,
    )
    X, y = wisconsin[colunas], wisconsin["maligno"]
    modelo = RandomForestClassifier(
        n_estimators=200, max_features=1, bootstrap=True,
        min_samples_leaf=5, random_state=SEMENTE, n_jobs=1,
    ).fit(X.iloc[treino], y.iloc[treino])
    auc_original = roc_auc_score(y.iloc[validacao], modelo.predict_proba(X.iloc[validacao])[:, 1])
    resultado = permutation_importance(
        modelo, X.iloc[validacao], y.iloc[validacao], scoring="roc_auc",
        n_repeats=20, random_state=SEMENTE, n_jobs=1,
    )
    assert len(treino) == 318 and len(validacao) == 137
    assert np.isclose(auc_original, .964204286365709)
    np.testing.assert_allclose(resultado.importances_mean, [.3433937528499771, .051003191974464165])
    linhas = [{"variavel": coluna, "repeticao": i + 1, "diminuicao_AUC": float(v)}
              for coluna, valores in zip(colunas, resultado.importances)
              for i, v in enumerate(valores)]
    pd.DataFrame(linhas).to_csv(RESULTADOS / "permutacao_diminuicoes_AUC.csv", index=False)
    resumo = {
        "origem": "Demonstração do capítulo 13 antecipada no exercício 10.4",
        "divisao": "train_test_split(test_size=.30, stratify=maligno, random_state=2026)",
        "n_treino": len(treino), "n_validacao": len(validacao),
        "ids_treino": wisconsin.iloc[treino]["id_registo"].astype(int).tolist(),
        "ids_validacao": wisconsin.iloc[validacao]["id_registo"].astype(int).tolist(),
        "auc_original": float(auc_original), "repeticoes": 20,
        "diminuicoes_medias_AUC": dict(zip(colunas, resultado.importances_mean.tolist())),
        "desvios_baralhamentos_ddof0": dict(zip(colunas, resultado.importances_std.tolist())),
        "ajustamento_mantido_fixo_durante_as_permutacoes": True,
        "teste_utilizado": False,
    }
    guardar_json("permutacao_resumo.json", resumo)
    return resumo


def main():
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    wisconsin, pinguins = validar()
    colunas_c = ["mean radius", "mean texture"]
    colunas_r = ["flipper_length_mm", "bill_length_mm"]
    classificadores, regressores = procedimentos()
    partes_c = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE).split(
        wisconsin[colunas_c], wisconsin["maligno"]
    ))
    partes_r = list(KFold(n_splits=5, shuffle=True, random_state=SEMENTE).split(
        pinguins[colunas_r], pinguins["body_mass_g"]
    ))
    with threadpool_limits(limits=1):
        sc, pc, ac, gc = avaliar("classificacao", wisconsin, colunas_c, "maligno", classificadores, partes_c)
        sr, pr, ar, gr = avaliar("regressao", pinguins, colunas_r, "body_mass_g", regressores, partes_r)
        arvore_c = clone(classificadores["arvore_profundidade_2"][1]).fit(
            wisconsin[colunas_c], wisconsin["maligno"]
        )
        arvore_r = clone(regressores["arvore_profundidade_2"][1]).fit(
            pinguins[colunas_r], pinguins["body_mass_g"]
        )
        nos_c = extrair_arvore("classificacao", arvore_c, wisconsin, colunas_c, "maligno")
        nos_r = extrair_arvore("regressao", arvore_r, pinguins, colunas_r, "body_mass_g")
        permutacao = verificar_permutacao(wisconsin)
    np.testing.assert_allclose(sc["validacao_AUC_media"],
        [.952115583, .912280702, .932249742, .844427245, .927915377, .933642931], atol=1e-9)
    np.testing.assert_allclose(sr["validacao_MAE_g_media"],
        [320.08239507, 342.09197155, 313.02759855, 412.31425365,
         305.84276128, 295.84126677, 672.54208754], atol=1e-7)
    empates = pc.groupby("chave")["validacao_empates_050"].sum().to_dict()
    assert empates["arvore_profundidade_2"] == 2 and empates["arvore_profundidade_4"] == 11
    assert gc["parte_validacao"] == 2 and gr["parte_validacao"] == 3
    ficticio = bootstrap_ficticio()
    metadados = {
        "hashes_dados": HASHES,
        "proveniencia": {
            "penguins": {
                "fonte": "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv",
                "sha256_fonte": "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93",
                "licenca": "CC0", "registos_originais": 344,
                "preparacao": "Acrescentar id_registo 1..344; train_test_split(test_size=.20, random_state=2026); ordenar desenvolvimento por id_registo.",
                "registos_desenvolvimento": 275, "registos_com_massa_observada": 273,
                "ids_excluidos_sem_resposta": [4, 272],
            },
            "wisconsin": {
                "fonte": "sklearn.datasets.load_breast_cancer(as_frame=True)",
                "doi": "https://doi.org/10.24432/C5DW2B", "licenca": "CC BY 4.0 (UCI)",
                "registos_originais": 569,
                "preparacao": "Acrescentar id_registo 1..569; maligno=1-target; train_test_split(test_size=.20, stratify=maligno, random_state=2026); ordenar desenvolvimento por id_registo.",
                "registos_desenvolvimento": 455, "malignas": 170, "benignas": 285,
            },
            "copias_incluidas": "Cópias de desenvolvimento com os mesmos bytes já verificados nos capítulos 7 e 8.",
        },
        "semente": SEMENTE, "numero_partes": 5,
        "particoes_classificacao": "StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)",
        "particoes_regressao": "KFold(n_splits=5, shuffle=True, random_state=2026)",
        "convencao_decisao": "Probabilidade >=0.50 prevê a classe 1; não se usa predict() para contar acertos de classificação.",
        "convencao_desvio": "Desvio-padrão amostral entre cinco partes, ddof=1; não é um intervalo de confiança.",
        "configuracoes_arvores": [{"chave": c, "max_depth": d, "min_samples_leaf": f}
                                   for c, _, d, f in CONFIGURACOES],
        "configuracoes_florestas": {"n_estimators": 200, "max_features": 1,
                                    "bootstrap": True, "max_depth": None,
                                    "min_samples_leaf": [1, 5], "random_state": 2026, "n_jobs": 1},
        "referencia_classificacao": {"classe_prevista": 0, "AUC": .5, "acerto": 285/455, "sensibilidade": 0},
        "resumo_classificacao_medias_partes": sc.to_dict(orient="records"),
        "resumo_regressao_medias_partes": sr.to_dict(orient="records"),
        "classificacao_previsoes_reunidas": ac,
        "regressao_previsoes_reunidas": ar,
        "empates_050_classificacao_por_modelo": empates,
        "arvore_classificacao_completa": nos_c,
        "arvore_regressao_completa": nos_r,
        "agregacao_classificacao": gc, "agregacao_regressao": gr,
        "bootstrap_ficticio": ficticio,
        "demonstracao_permutacao": permutacao,
        "teste_utilizado": False,
        "versoes": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    guardar_json("resultados_verificados.json", metadados)
    print(sc.to_string(index=False))
    print(sr.to_string(index=False))
    print("Cálculos verificados; teste reservado não utilizado.")


if __name__ == "__main__":
    main()
