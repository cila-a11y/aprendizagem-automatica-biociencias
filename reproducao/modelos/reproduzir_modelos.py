"""Reproduzir os exemplos supervisionados dos capítulos 7 a 13 e os testes finais.

Executar a partir de qualquer pasta:
    python reproducao/modelos/reproduzir_modelos.py desenvolvimento
    python reproducao/modelos/reproduzir_modelos.py teste

A primeira etapa lê apenas o desenvolvimento. A segunda exige a decisão
gravada pela primeira e aplica os procedimentos escolhidos ao teste reservado.
Os ficheiros fornecidos permitem reproduzir uma análise já descrita no livro;
voltar a executá-la não cria uma nova avaliação independente.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree


PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]
DADOS = RAIZ / "dados"
RESULTADOS = PASTA / "resultados"
FIGURAS = PASTA / "figuras"
SEMENTE = 2026
ENTRADAS_W = ["mean radius", "mean texture"]
ENTRADAS_P = ["flipper_length_mm", "bill_length_mm"]

NOMES = {
    "referencia_benigno": "Sempre benigno",
    "logistica": "Logística",
    "arvore_2": "Árvore: profundidade 2",
    "arvore_4": "Árvore: profundidade 4",
    "arvore_sem_limite": "Árvore sem limite",
    "floresta_1": "Floresta: folha mínima 1",
    "floresta_5": "Floresta: folha mínima 5",
    "rede_8": "Rede: 8 unidades",
    "rede_32": "Rede: 32 unidades",
    "referencia_media": "Referência média",
    "referencia_mediana": "Referência mediana",
    "linear_simples": "Regressão simples",
    "linear_multipla": "Regressão múltipla",
}


def guardar_json(nome, objeto):
    """JSON legível, sem valores NaN ou infinitos."""
    (RESULTADOS / nome).write_text(
        json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def guardar_tabela(nome, tabela):
    tabela.to_csv(RESULTADOS / nome, index=False, float_format="%.17g")


def sha256(caminho):
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def guardar_figura(nome, figura):
    figura.savefig(FIGURAS / f"{nome}.png", dpi=170, bbox_inches="tight")
    figura.savefig(FIGURAS / f"{nome}.pdf", bbox_inches="tight")
    plt.close(figura)


def versoes():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def preparar_pastas():
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    })


def ler_dados(nome, parte):
    """Conserva a ordem publicada; não volta a ordenar nem divide novamente."""
    caminho = DADOS / f"{nome}_{parte}.csv"
    dados = pd.read_csv(caminho, float_precision="round_trip")
    if "id_registo" not in dados or dados["id_registo"].isna().any():
        raise ValueError(f"Identificadores ausentes em {caminho.name}.")
    if dados["id_registo"].duplicated().any():
        raise ValueError(f"Identificadores repetidos em {caminho.name}.")
    if not dados["id_registo"].is_monotonic_increasing:
        raise ValueError(f"A ordem de {caminho.name} difere da cópia preparada.")
    entradas = ENTRADAS_W if nome == "wisconsin" else ENTRADAS_P
    resposta = "maligno" if nome == "wisconsin" else "body_mass_g"
    for coluna in entradas + [resposta]:
        if coluna not in dados:
            raise ValueError(f"Falta a coluna {coluna} em {caminho.name}.")
        if not pd.api.types.is_numeric_dtype(dados[coluna]):
            raise ValueError(f"A coluna {coluna} deve ser numérica.")
        if np.isinf(dados[coluna].to_numpy(dtype=float)).any():
            raise ValueError(f"Há valores infinitos na coluna {coluna}.")
    if nome == "wisconsin":
        if dados[entradas + [resposta]].isna().any().any():
            raise ValueError("A cópia Wisconsin usada no livro não tem omissões.")
        if set(dados[resposta]) != {0, 1}:
            raise ValueError("A resposta Wisconsin deve usar 0=benigno e 1=maligno.")
    return dados


def criar_rede(unidades):
    return MLPClassifier(
        hidden_layer_sizes=(unidades,), activation="relu", solver="adam",
        alpha=0.001, learning_rate_init=0.01, batch_size=32, max_iter=800,
        n_iter_no_change=30, tol=1e-4, random_state=SEMENTE,
        early_stopping=False,
    )


def modelos_classificacao():
    return {
        "referencia_benigno": DummyClassifier(strategy="most_frequent"),
        "logistica": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, max_iter=1000, random_state=SEMENTE),
        ),
        "arvore_2": DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=5, random_state=SEMENTE),
        "arvore_4": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, random_state=SEMENTE),
        "arvore_sem_limite": DecisionTreeClassifier(
            max_depth=None, min_samples_leaf=1, random_state=SEMENTE),
        "floresta_1": RandomForestClassifier(
            n_estimators=200, max_features=1, bootstrap=True,
            min_samples_leaf=1, random_state=SEMENTE, n_jobs=1),
        "floresta_5": RandomForestClassifier(
            n_estimators=200, max_features=1, bootstrap=True,
            min_samples_leaf=5, random_state=SEMENTE, n_jobs=1),
        "rede_8": make_pipeline(StandardScaler(), criar_rede(8)),
        "rede_32": make_pipeline(StandardScaler(), criar_rede(32)),
    }


def modelos_regressao():
    modelos = {
        "referencia_media": DummyRegressor(strategy="mean"),
        "referencia_mediana": DummyRegressor(strategy="median"),
        "linear_simples": LinearRegression(),
        "linear_multipla": LinearRegression(),
        "arvore_2": DecisionTreeRegressor(
            criterion="squared_error", max_depth=2, min_samples_leaf=5,
            random_state=SEMENTE),
        "arvore_4": DecisionTreeRegressor(
            criterion="squared_error", max_depth=4, min_samples_leaf=5,
            random_state=SEMENTE),
        "arvore_sem_limite": DecisionTreeRegressor(
            criterion="squared_error", max_depth=None, min_samples_leaf=1,
            random_state=SEMENTE),
        "floresta_1": RandomForestRegressor(
            n_estimators=200, max_features=1, bootstrap=True,
            min_samples_leaf=1, random_state=SEMENTE, n_jobs=1),
        "floresta_5": RandomForestRegressor(
            n_estimators=200, max_features=1, bootstrap=True,
            min_samples_leaf=5, random_state=SEMENTE, n_jobs=1),
    }
    return {nome: make_pipeline(SimpleImputer(strategy="median"), modelo)
            for nome, modelo in modelos.items()}


def fracao(numerador, denominador):
    return float(numerador / denominador) if denominador else np.nan


def medidas_classes(y, previsao):
    vn, fp, fn, vp = confusion_matrix(y, previsao, labels=[0, 1]).ravel()
    return {
        "VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp),
        "acerto": fracao(vp + vn, len(y)),
        "sensibilidade": fracao(vp, vp + fn),
        "especificidade": fracao(vn, vn + fp),
        "VPP": fracao(vp, vp + fp), "VPN": fracao(vn, vn + fn),
        "F1": fracao(2 * vp, 2 * vp + fp + fn),
        "acerto_equilibrado": (fracao(vp, vp + fn) + fracao(vn, vn + fp)) / 2,
        "taxa_FP": fracao(fp, fp + vn),
    }


def medidas_regressao(y, previsao):
    erro = np.asarray(y) - np.asarray(previsao)
    return {
        "MAE_g": float(mean_absolute_error(y, previsao)),
        "RMSE_g": float(mean_squared_error(y, previsao) ** 0.5),
        "R2": float(r2_score(y, previsao)),
        "residuo_medio_g": float(np.mean(erro)),
        "erro_absoluto_maximo_g": float(np.max(np.abs(erro))),
    }


def probabilidade_positiva(modelo, X):
    indice = list(modelo.classes_).index(1)
    return modelo.predict_proba(X)[:, indice]


def validacao_classificacao(dados):
    X, y = dados[ENTRADAS_W], dados["maligno"]
    partes = list(StratifiedKFold(5, shuffle=True, random_state=SEMENTE).split(X, y))
    ids_partes = np.zeros(len(dados), dtype=int)
    por_parte, resumos = [], []
    previsoes = dados[["id_registo", "maligno"]].copy()
    for parte, (_, va) in enumerate(partes, start=1):
        ids_partes[va] = parte
    previsoes["parte_validacao"] = ids_partes
    for nome, candidato in modelos_classificacao().items():
        print(f"Classificação: {NOMES[nome]}", flush=True)
        fora_treino = np.empty(len(y))
        for parte, (tr, va) in enumerate(partes, start=1):
            ajuste = clone(candidato).fit(X.iloc[tr], y.iloc[tr])
            p_tr = probabilidade_positiva(ajuste, X.iloc[tr])
            p_va = probabilidade_positiva(ajuste, X.iloc[va])
            fora_treino[va] = p_va
            linha = {
                "modelo": nome, "parte": parte, "n_treino": len(tr),
                "n_validacao": len(va),
                "AUC_treino": roc_auc_score(y.iloc[tr], p_tr),
                "AUC_validacao": roc_auc_score(y.iloc[va], p_va),
                "acerto_treino": accuracy_score(y.iloc[tr], p_tr >= 0.5),
                "acerto_validacao": accuracy_score(y.iloc[va], p_va >= 0.5),
                "AP_validacao": average_precision_score(y.iloc[va], p_va),
                "log_loss_validacao": log_loss(y.iloc[va], p_va),
            }
            if nome.startswith("rede_"):
                linha["epocas_fit"] = ajuste[-1].n_iter_
            por_parte.append(linha)
        valores = pd.DataFrame(por_parte).query("modelo == @nome")
        resumos.append({
            "modelo": nome,
            "AUC_media_partes": valores["AUC_validacao"].mean(),
            "AUC_DP_partes": valores["AUC_validacao"].std(ddof=1),
            "acerto_medio_treino": valores["acerto_treino"].mean(),
            "acerto_medio_validacao": valores["acerto_validacao"].mean(),
            "acerto_DP_validacao": valores["acerto_validacao"].std(ddof=1),
            "AUC_agregada": roc_auc_score(y, fora_treino),
            "AP_agregada": average_precision_score(y, fora_treino),
            "log_loss_agregada": log_loss(y, fora_treino),
            **medidas_classes(y, fora_treino >= 0.5),
        })
        previsoes[f"p_{nome}"] = fora_treino
    tabela = pd.DataFrame(resumos)
    guardar_tabela("wisconsin_validacao_por_parte.csv", pd.DataFrame(por_parte))
    guardar_tabela("wisconsin_validacao_resumo.csv", tabela)
    guardar_tabela("wisconsin_previsoes_validacao.csv", previsoes)
    guardar_tabela("wisconsin_partes.csv", previsoes[["id_registo", "parte_validacao"]])
    return tabela, previsoes


def validacao_regressao(dados):
    sem_resposta = dados.loc[dados["body_mass_g"].isna(), ["id_registo"]]
    guardar_tabela("pinguins_sem_resposta.csv", sem_resposta)
    dados = dados.dropna(subset=["body_mass_g"]).copy()
    X, y = dados[ENTRADAS_P], dados["body_mass_g"]
    partes = list(KFold(5, shuffle=True, random_state=SEMENTE).split(X, y))
    ids_partes = np.zeros(len(dados), dtype=int)
    por_parte, resumos = [], []
    previsoes = dados[["id_registo", "body_mass_g"]].copy()
    for parte, (_, va) in enumerate(partes, start=1):
        ids_partes[va] = parte
    previsoes["parte_validacao"] = ids_partes
    for nome, candidato in modelos_regressao().items():
        print(f"Regressão: {NOMES[nome]}", flush=True)
        entradas = [ENTRADAS_P[0]] if nome == "linear_simples" else ENTRADAS_P
        atual = dados[entradas]
        fora_treino = np.empty(len(y))
        for parte, (tr, va) in enumerate(partes, start=1):
            ajuste = clone(candidato).fit(atual.iloc[tr], y.iloc[tr])
            p_tr = ajuste.predict(atual.iloc[tr])
            p_va = ajuste.predict(atual.iloc[va])
            fora_treino[va] = p_va
            por_parte.append({
                "modelo": nome, "parte": parte, "n_treino": len(tr),
                "n_validacao": len(va),
                "MAE_treino_g": mean_absolute_error(y.iloc[tr], p_tr),
                "MAE_validacao_g": mean_absolute_error(y.iloc[va], p_va),
                "RMSE_validacao_g": mean_squared_error(y.iloc[va], p_va) ** 0.5,
                "R2_validacao": r2_score(y.iloc[va], p_va),
            })
        valores = pd.DataFrame(por_parte).query("modelo == @nome")
        resumos.append({
            "modelo": nome,
            "MAE_medio_treino_g": valores["MAE_treino_g"].mean(),
            "MAE_medio_partes_g": valores["MAE_validacao_g"].mean(),
            "MAE_DP_partes_g": valores["MAE_validacao_g"].std(ddof=1),
            "RMSE_medio_partes_g": valores["RMSE_validacao_g"].mean(),
            "R2_medio_partes": valores["R2_validacao"].mean(),
            **{f"agregado_{k}": v for k, v in medidas_regressao(y, fora_treino).items()},
        })
        previsoes[f"previsao_{nome}"] = fora_treino
    tabela = pd.DataFrame(resumos)
    guardar_tabela("pinguins_validacao_por_parte.csv", pd.DataFrame(por_parte))
    guardar_tabela("pinguins_validacao_resumo.csv", tabela)
    guardar_tabela("pinguins_previsoes_validacao.csv", previsoes)
    guardar_tabela("pinguins_partes.csv", previsoes[["id_registo", "parte_validacao"]])
    return tabela, previsoes


def limiares_e_curvas(dados, previsoes):
    y = dados["maligno"]
    p = previsoes["p_logistica"].to_numpy()
    tabela = pd.DataFrame([
        {"limiar": t, **medidas_classes(y, p >= t)} for t in [0.2, 0.3, 0.5, 0.7]
    ])
    guardar_tabela("wisconsin_logistica_limiares.csv", tabela)
    for t in [0.2, 0.3, 0.5, 0.7]:
        matriz = confusion_matrix(y, p >= t, labels=[0, 1])
        normalizada = matriz / matriz.sum(axis=1, keepdims=True)
        guardar_tabela(f"wisconsin_matriz_{int(t*100):03d}.csv", pd.DataFrame({
            "classe_real": [0, 1], "previsto_0": matriz[:, 0], "previsto_1": matriz[:, 1],
            "proporcao_linha_0": normalizada[:, 0],
            "proporcao_linha_1": normalizada[:, 1],
        }))
    fpr, tpr, t = roc_curve(y, p, pos_label=1)
    guardar_tabela("wisconsin_logistica_roc.csv", pd.DataFrame({
        "limiar": t, "taxa_FP": fpr, "sensibilidade": tpr,
    }))
    precision, recall, t = precision_recall_curve(y, p, pos_label=1)
    guardar_tabela("wisconsin_logistica_vpp_sensibilidade.csv", pd.DataFrame({
        "limiar": np.r_[t, np.nan], "VPP": precision, "sensibilidade": recall,
        "ponto_convencional_sem_limiar": [False] * len(t) + [True],
    }))
    z = np.linspace(-7, 7, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, 1 / (1 + np.exp(-z)), color="#27647b")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.set(xlabel="Soma ponderada z", ylabel="Probabilidade", ylim=(0, 1))
    guardar_figura("u8_sigmoide", fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for classe, nome, cor in [(0, "Benigno", "#27647b"), (1, "Maligno", "#b95b36")]:
        ax.hist(p[y == classe], bins=np.linspace(0, 1, 21), alpha=0.65,
                color=cor, label=nome)
    ax.axvline(0.5, color="black", ls="--", label="Limiar 0,50")
    ax.set(xlabel="Probabilidade de malignidade fora do treino", ylabel="Amostras")
    ax.legend()
    guardar_figura("u8_probabilidades", fig)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.plot(fpr, tpr, color="#27647b", label=f"Logística: AUC agregada = {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="Taxa de falsos positivos", ylabel="Sensibilidade",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    guardar_figura("u8b_roc", fig)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.step(recall, precision, where="post", color="#27647b",
            label=f"Logística: AP = {average_precision_score(y, p):.3f}")
    ax.axhline(y.mean(), color="grey", ls="--", label="Proporção de positivos")
    ax.set(xlabel="Sensibilidade", ylabel="VPP", xlim=(0, 1), ylim=(0, 1.05))
    ax.legend(loc="lower left")
    guardar_figura("u8_curva_vpp_sensibilidade", fig)
    return tabela


def grafico_regressao(dados, previsto, nome, tipo):
    y = dados["body_mass_g"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if tipo == "residuos":
        ax.scatter(previsto, y - previsto, s=20, alpha=0.7, color="#27647b")
        ax.axhline(0, color="grey", ls="--")
        ax.set(xlabel="Massa prevista (g)", ylabel="Resíduo: observado − previsto (g)")
    else:
        ax.scatter(y, previsto, s=20, alpha=0.7, color="#27647b")
        limites = [min(y.min(), previsto.min()), max(y.max(), previsto.max())]
        ax.plot(limites, limites, "--", color="grey")
        ax.set(xlabel="Massa observada (g)", ylabel="Massa prevista (g)")
    guardar_figura(nome, fig)


def ajustes_descritivos(wisconsin, pinguins, previsoes_p):
    coeficientes = []
    d = pinguins.dropna(subset=["body_mass_g"])
    for nome in ["linear_simples", "linear_multipla"]:
        entradas = [ENTRADAS_P[0]] if nome == "linear_simples" else ENTRADAS_P
        modelo = modelos_regressao()[nome].fit(d[entradas], d["body_mass_g"])
        for termo, valor in zip(["constante"] + entradas,
                               [modelo[-1].intercept_] + list(modelo[-1].coef_)):
            coeficientes.append({"modelo": nome, "termo": termo, "coeficiente": valor})
        if nome == "linear_simples":
            valores = np.linspace(d[entradas[0]].min(), d[entradas[0]].max(), 200)
            fig, ax = plt.subplots(figsize=(7, 4.8))
            ax.scatter(d[entradas[0]], d["body_mass_g"], s=20, alpha=0.6, color="#27647b")
            ax.plot(valores, modelo.predict(pd.DataFrame({entradas[0]: valores})),
                    color="#b95b36", label="Ajuste ao desenvolvimento completo")
            ax.set(xlabel="Comprimento da barbatana (mm)", ylabel="Massa corporal (g)")
            ax.legend()
            guardar_figura("u7_ajuste_simples", fig)
    modelo = modelos_classificacao()["logistica"].fit(
        wisconsin[ENTRADAS_W], wisconsin["maligno"])
    for termo, valor in zip(["constante"] + ENTRADAS_W,
                           [modelo[-1].intercept_[0]] + list(modelo[-1].coef_[0])):
        coeficientes.append({"modelo": "logistica_padronizada", "termo": termo,
                             "coeficiente": valor})
    guardar_tabela("coeficientes_desenvolvimento_completo.csv", pd.DataFrame(coeficientes))
    guardar_tabela("wisconsin_padronizacao_completa.csv", pd.DataFrame({
        "entrada": ENTRADAS_W, "media": modelo[0].mean_, "escala": modelo[0].scale_,
    }))
    for nome, coluna, tipo in [
        ("u7_observado_previsto", "previsao_linear_simples", "observado"),
        ("u7_residuos", "previsao_linear_simples", "residuos"),
        ("u13_residuos", "previsao_linear_multipla", "residuos"),
    ]:
        grafico_regressao(d, previsoes_p[coluna].to_numpy(), nome, tipo)
    exemplos = previsoes_p[["id_registo", "body_mass_g", "previsao_linear_simples"]].copy()
    exemplos["residuo_g"] = exemplos["body_mass_g"] - exemplos["previsao_linear_simples"]
    guardar_tabela("pinguins_regressao_simples_por_registo.csv", exemplos)


def desenhar_arvores(wisconsin, pinguins):
    casos = [
        ("classificacao", wisconsin, ENTRADAS_W, "maligno", modelos_classificacao()["arvore_2"]),
        ("regressao", pinguins.dropna(subset=["body_mass_g"]), ENTRADAS_P,
         "body_mass_g", modelos_regressao()["arvore_2"]),
    ]
    for tarefa, d, entradas, resposta, procedimento in casos:
        procedimento.fit(d[entradas], d[resposta])
        arvore = procedimento if tarefa == "classificacao" else procedimento[-1]
        fig, ax = plt.subplots(figsize=(12, 5.8))
        nomes = ["Raio médio", "Textura média"] if tarefa == "classificacao" else ["Barbatana (mm)", "Bico (mm)"]
        anotacoes = plot_tree(
            arvore, feature_names=nomes,
            class_names=["Benigno", "Maligno"] if tarefa == "classificacao" else None,
            filled=True, impurity=False, precision=2, ax=ax, fontsize=9,
        )
        for texto in anotacoes:
            legenda = texto.get_text().replace("samples =", "amostras =")
            legenda = legenda.replace("class =", "classe =")
            legenda = legenda.replace("value =", "contagens =" if tarefa == "classificacao" else "massa média =")
            texto.set_text(legenda)
        for texto in ax.texts:
            if texto.get_text() in {"True", "False"}:
                texto.set_text({"True": "Sim", "False": "Não"}[texto.get_text()])
        guardar_figura(f"u9_arvore_{tarefa}", fig)
        t = arvore.tree_
        linhas = []
        for i in range(t.node_count):
            linha = {
                "no": i, "filho_esquerdo": int(t.children_left[i]),
                "filho_direito": int(t.children_right[i]),
                "entrada": entradas[t.feature[i]] if t.feature[i] >= 0 else "folha",
                "limite": float(t.threshold[i]) if t.feature[i] >= 0 else np.nan,
                "n_treino": int(t.n_node_samples[i]),
            }
            if tarefa == "classificacao":
                proporcoes = t.value[i, 0] / t.value[i, 0].sum()
                linha.update({"p_benigno": proporcoes[0], "p_maligno": proporcoes[1],
                              "n_benigno": int(round(proporcoes[0] * t.n_node_samples[i])),
                              "n_maligno": int(round(proporcoes[1] * t.n_node_samples[i]))})
            else:
                linha["massa_media_g"] = t.value[i, 0, 0]
            linhas.append(linha)
        guardar_tabela(f"arvore_{tarefa}_nos.csv", pd.DataFrame(linhas))


def comparacoes(resumo_w, resumo_p):
    nomes_w = ["logistica", "arvore_2", "floresta_1", "floresta_5"]
    nomes_p = ["linear_multipla", "arvore_2", "floresta_1", "floresta_5"]
    w = resumo_w.set_index("modelo").loc[nomes_w]
    p = resumo_p.set_index("modelo").loc[nomes_p]
    fig, eixos = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, tabela, valor, dp, rotulos, titulo, unidade in [
        (eixos[0], w, "AUC_media_partes", "AUC_DP_partes", nomes_w, "Classificação", "AUC média"),
        (eixos[1], p, "MAE_medio_partes_g", "MAE_DP_partes_g", nomes_p, "Regressão", "MAE médio (g)"),
    ]:
        ax.bar(np.arange(4), tabela[valor], yerr=tabela[dp], capsize=4, color="#27647b")
        ax.set_xticks(np.arange(4), ["Logística" if n == "logistica" else
                                   "Linear" if n == "linear_multipla" else
                                   "Árvore 2" if n == "arvore_2" else
                                   "Floresta 1" if n == "floresta_1" else "Floresta 5"
                                   for n in rotulos], rotation=15)
        ax.set(title=titulo, ylabel=unidade)
    fig.suptitle("Mesmas partes de validação; barras de variação = desvio entre partes")
    fig.tight_layout()
    guardar_figura("u10_comparacao", fig)


def curvas_redes(dados):
    X, y = dados[ENTRADAS_W], dados["maligno"]
    it, iv = train_test_split(np.arange(len(y)), test_size=0.30,
                             stratify=y, random_state=SEMENTE)
    escala = StandardScaler()
    treino = escala.fit_transform(X.iloc[it])
    validacao = escala.transform(X.iloc[iv])
    divisao = dados[["id_registo"]].copy()
    divisao["parte"] = "treino"
    divisao.iloc[iv, divisao.columns.get_loc("parte")] = "validacao"
    guardar_tabela("wisconsin_divisao_interna_70_30.csv", divisao)
    for tamanho in [8, 32]:
        print(f"Curva de treino: rede com {tamanho} unidades", flush=True)
        rede = criar_rede(tamanho)
        linhas = []
        for epoca in range(1, 201):
            rede.partial_fit(treino, y.iloc[it], classes=[0, 1])
            p_tr = probabilidade_positiva(rede, treino)
            p_va = probabilidade_positiva(rede, validacao)
            linhas.append({
                "epoca": epoca, "perda_treino": log_loss(y.iloc[it], p_tr),
                "perda_validacao": log_loss(y.iloc[iv], p_va),
                "acerto_validacao": accuracy_score(y.iloc[iv], p_va >= 0.5),
            })
        historico = pd.DataFrame(linhas)
        guardar_tabela(f"rede_{tamanho}_curva_200_epocas.csv", historico)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.plot(historico["epoca"], historico["perda_treino"], label="Treino", color="#27647b")
        ax.plot(historico["epoca"], historico["perda_validacao"], label="Validação interna", color="#b95b36")
        ax.set(xlabel="Época", ylabel="Perda logarítmica média", title=f"Rede com {tamanho} unidades")
        ax.legend()
        guardar_figura(f"u11_curvas{tamanho}", fig)


def interpretacao(dados, previsoes):
    X, y = dados[ENTRADAS_W], dados["maligno"]
    p = previsoes["p_logistica"].to_numpy()
    limites = np.linspace(0, 1, 6)
    grupos = np.searchsorted(limites[1:-1], p, side="left")
    linhas = []
    for i in range(5):
        sel = grupos == i
        linhas.append({
            "faixa": i + 1, "limite_inferior": limites[i], "limite_superior": limites[i + 1],
            "n": int(sel.sum()), "probabilidade_media": p[sel].mean(),
            "fracao_maligna": y.iloc[np.flatnonzero(sel)].mean(),
        })
    tabela = pd.DataFrame(linhas)
    guardar_tabela("wisconsin_calibracao.csv", tabela)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.plot(tabela["probabilidade_media"], tabela["fracao_maligna"], "o-", color="#27647b")
    for r in linhas:
        deslocamento = (6, 8) if r["fracao_maligna"] < 0.1 else (6, -15)
        ax.annotate(f"n={r['n']}", (r["probabilidade_media"], r["fracao_maligna"]),
                    xytext=deslocamento, textcoords="offset points", fontsize=9)
    ax.set(xlabel="Probabilidade média prevista", ylabel="Fração de amostras malignas",
           xlim=(0, 1.05), ylim=(0, 1.05))
    guardar_figura("u13_calibracao", fig)
    it, iv = train_test_split(np.arange(len(y)), test_size=0.30,
                             stratify=y, random_state=SEMENTE)
    modelo = modelos_classificacao()["floresta_5"].fit(X.iloc[it], y.iloc[it])
    auc = roc_auc_score(y.iloc[iv], probabilidade_positiva(modelo, X.iloc[iv]))
    resultado = permutation_importance(modelo, X.iloc[iv], y.iloc[iv],
                                       scoring="roc_auc", n_repeats=20,
                                       random_state=SEMENTE, n_jobs=1)
    tabela = pd.DataFrame({
        "entrada": ENTRADAS_W, "AUC_original": auc,
        "diminuicao_AUC_media": resultado.importances_mean,
        "DP_20_repeticoes_ddof0": resultado.importances_std,
    })
    guardar_tabela("wisconsin_permutacao_resumo.csv", tabela)
    guardar_tabela("wisconsin_permutacao_repeticoes.csv", pd.DataFrame([
        {"entrada": entrada, "repeticao": j + 1, "AUC_original": auc,
         "diminuicao_AUC": valor, "AUC_apos_permutacao": auc - valor}
        for entrada, valores in zip(ENTRADAS_W, resultado.importances)
        for j, valor in enumerate(valores)
    ]))
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.bar(["Raio médio", "Textura média"], resultado.importances_mean,
           yerr=resultado.importances_std, capsize=5, color="#27647b")
    ax.set(ylabel="Diminuição da AUC", title="20 permutações na validação interna")
    guardar_figura("u13_permutacao", fig)
    colunas = ["mean radius", "mean perimeter", "mean area", "mean texture"]
    correlacoes = dados[colunas].corr().rename_axis("entrada").reset_index()
    guardar_tabela("wisconsin_correlacoes.csv", correlacoes)


def demonstracoes_validacao(dados):
    """Exemplos didáticos; o contraste errado nunca seleciona o modelo final."""
    X, y = dados[ENTRADAS_W], dados["maligno"]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)
    correto = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=1, max_iter=1000, random_state=SEMENTE))
    auc_correto = cross_val_score(correto, X, y, cv=cv, scoring="roc_auc")
    # Preparação intencionalmente indevida, apenas para o contraste do capítulo 12.
    preparado = SimpleImputer(strategy="median").fit_transform(X)
    preparado = StandardScaler().fit_transform(preparado)
    auc_errado = cross_val_score(
        LogisticRegression(C=1, max_iter=1000, random_state=SEMENTE),
        preparado, y, cv=cv, scoring="roc_auc")
    guardar_tabela("demonstracao_preparacao_correta_e_indevida.csv", pd.DataFrame({
        "parte": range(1, 6), "AUC_preparacao_dentro_treino": auc_correto,
        "AUC_preparacao_global_INCORRETA": auc_errado,
    }))
    exemplo = pd.DataFrame({
        "medicao": range(1, 13),
        "doente": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F"],
        "medida": [2.1, 2.3, 5.2, 5.0, 2.8, 2.9, 4.8, 4.6, 3.0, 2.7, 5.5, 5.4],
        "classe": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    })
    guardar_tabela("exemplo_ficticio_seis_doentes.csv", exemplo)
    registos = []
    for parte, (it, iv) in enumerate(GroupKFold(3).split(
            exemplo[["medida"]], exemplo["classe"], groups=exemplo["doente"]), start=1):
        if not set(exemplo.iloc[it]["doente"]).isdisjoint(exemplo.iloc[iv]["doente"]):
            raise RuntimeError("Um doente aparece nos dois lados da divisão.")
        for papel, indices in [("treino", it), ("validacao", iv)]:
            for i in indices:
                registos.append({"parte": parte, "papel": papel, **exemplo.iloc[i].to_dict()})
    guardar_tabela("exemplo_ficticio_divisao_grupos.csv", pd.DataFrame(registos))


def desenvolvimento():
    preparar_pastas()
    w = ler_dados("wisconsin", "desenvolvimento")
    p = ler_dados("penguins", "desenvolvimento")
    rw, pw = validacao_classificacao(w)
    rp, pp = validacao_regressao(p)
    tabela_limiares = limiares_e_curvas(w, pw)
    ajustes_descritivos(w, p, pp)
    desenhar_arvores(w, p)
    comparacoes(rw, rp)
    curvas_redes(w)
    interpretacao(w, pw)
    demonstracoes_validacao(w)

    # Regra descrita no livro. O desempate conserva a ordem explícita dos candidatos.
    escolhido_w = rw.sort_values("AUC_media_partes", ascending=False, kind="stable").iloc[0]["modelo"]
    escolhido_p = rp.sort_values("MAE_medio_partes_g", kind="stable").iloc[0]["modelo"]
    if escolhido_w != "logistica" or escolhido_p != "floresta_5":
        raise RuntimeError(
            "A reprodução alterou a seleção descrita no livro. Consulte as tabelas "
            "e as versões das bibliotecas antes de abrir o teste.")
    elegiveis = tabela_limiares.query("sensibilidade >= 0.90")
    if elegiveis.empty:
        raise RuntimeError("Nenhum limiar cumpre a sensibilidade de validação de 0,90.")
    limiar = float(elegiveis.sort_values("especificidade", ascending=False, kind="stable").iloc[0]["limiar"])
    protocolo = PASTA / "protocolo.json"
    guardar_json("decisao.json", {
        "finalidade": "reproducao_da_analise_descrita_no_livro",
        "dados_utilizados_nesta_etapa": "apenas_desenvolvimento",
        "protocolo_sha256": sha256(protocolo),
        "ficheiros_desenvolvimento_sha256": {
            nome: sha256(DADOS / nome) for nome in ["wisconsin_desenvolvimento.csv", "penguins_desenvolvimento.csv"]
        },
        "versoes": versoes(),
        "wisconsin": {"modelo": escolhido_w, "entradas": ENTRADAS_W, "limiar": limiar,
                       "criterio_modelo": "maior_media_AUC_por_parte",
                       "criterio_limiar": "sensibilidade >= 0.90; maior especificidade entre elegiveis"},
        "penguins": {"modelo": escolhido_p, "entradas": ENTRADAS_P,
                      "criterio_modelo": "menor_media_MAE_por_parte"},
    })
    print("Desenvolvimento concluído. Decisão gravada; o teste não foi lido.", flush=True)


def teste():
    preparar_pastas()
    caminho = RESULTADOS / "decisao.json"
    if not caminho.exists():
        raise FileNotFoundError("Execute primeiro a etapa desenvolvimento para guardar a decisão.")
    decisao = json.loads(caminho.read_text(encoding="utf-8"))
    if decisao["protocolo_sha256"] != sha256(PASTA / "protocolo.json"):
        raise ValueError("O protocolo foi alterado depois de gravar a decisão.")
    for nome, esperado in decisao["ficheiros_desenvolvimento_sha256"].items():
        if sha256(DADOS / nome) != esperado:
            raise ValueError(f"O ficheiro {nome} foi alterado depois da decisão.")
    if decisao["wisconsin"]["modelo"] != "logistica" or decisao["wisconsin"]["limiar"] != 0.2:
        raise ValueError("Esta reprodução final exige a logística e o limiar 0,20 descritos no livro.")
    if decisao["penguins"]["modelo"] != "floresta_5":
        raise ValueError("Esta reprodução final exige a floresta escolhida no livro.")
    # A abertura dos dois ficheiros de teste ocorre apenas depois das verificações anteriores.
    wd = ler_dados("wisconsin", "desenvolvimento")
    wt = ler_dados("wisconsin", "teste_reservado")
    pdv = ler_dados("penguins", "desenvolvimento").dropna(subset=["body_mass_g"])
    pt = ler_dados("penguins", "teste_reservado")
    for dev, reservado in [(wd, wt), (pdv, pt)]:
        if not set(dev["id_registo"]).isdisjoint(reservado["id_registo"]):
            raise ValueError("Há registos partilhados entre desenvolvimento e teste.")
    if pt["body_mass_g"].isna().any():
        raise ValueError("O teste dos pinguins usado no livro tem todas as massas observadas.")
    modelo = modelos_classificacao()["logistica"].fit(wd[ENTRADAS_W], wd["maligno"])
    probs = probabilidade_positiva(modelo, wt[ENTRADAS_W])
    classes = (probs >= decisao["wisconsin"]["limiar"]).astype(int)
    y = wt["maligno"]
    tabela = pd.DataFrame([
        {"modelo": "referencia_benigno", "n": len(y), "AUC": 0.5,
         **medidas_classes(y, np.zeros(len(y), dtype=int))},
        {"modelo": "logistica", "n": len(y), "AUC": roc_auc_score(y, probs),
         **medidas_classes(y, classes)},
    ])
    guardar_tabela("wisconsin_teste_resumo.csv", tabela)
    por_registo = wt[["id_registo", "maligno"]].copy()
    por_registo["p_maligno"] = probs
    por_registo["classe_prevista_limiar_020"] = classes
    guardar_tabela("wisconsin_teste_previsoes.csv", por_registo)
    matriz = confusion_matrix(y, classes, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(matriz, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if matriz[i, j] > matriz.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Benigno", "Maligno"])
    ax.set_yticks([0, 1], ["Benigno", "Maligno"])
    ax.set(xlabel="Classe prevista", ylabel="Classe real", title="Teste; limiar 0,20")
    guardar_figura("u14_matriz", fig)
    fpr, tpr, limiares = roc_curve(y, probs, pos_label=1)
    guardar_tabela("wisconsin_teste_roc.csv", pd.DataFrame({
        "limiar": limiares, "taxa_FP": fpr, "sensibilidade": tpr,
    }))
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.plot(fpr, tpr, color="#27647b", label=f"AUC = {roc_auc_score(y, probs):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set(xlabel="Taxa de falsos positivos", ylabel="Sensibilidade", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    guardar_figura("u14_roc", fig)
    modelos = modelos_regressao()
    linhas = []
    prev_p = pt[["id_registo", "body_mass_g"]].copy()
    for nome in ["referencia_mediana", "floresta_5"]:
        modelo = modelos[nome].fit(pdv[ENTRADAS_P], pdv["body_mass_g"])
        previstas = modelo.predict(pt[ENTRADAS_P])
        linhas.append({"modelo": nome, "n": len(pt), **medidas_regressao(pt["body_mass_g"], previstas)})
        prev_p[f"previsao_{nome}"] = previstas
    prev_p["residuo_floresta_g"] = prev_p["body_mass_g"] - prev_p["previsao_floresta_5"]
    prev_p["erro_absoluto_floresta_g"] = prev_p["residuo_floresta_g"].abs()
    guardar_tabela("pinguins_teste_resumo.csv", pd.DataFrame(linhas))
    guardar_tabela("pinguins_teste_previsoes.csv", prev_p)
    guardar_tabela("pinguins_teste_erros_ordenados.csv", prev_p.sort_values("erro_absoluto_floresta_g", ascending=False))
    grafico_regressao(pt, prev_p["previsao_floresta_5"].to_numpy(), "u14_observado_previsto", "observado")
    grafico_regressao(pt, prev_p["previsao_floresta_5"].to_numpy(), "u14_residuos", "residuos")
    guardar_json("execucao_teste.json", {
        "finalidade": "reproducao_da_avaliacao_final_publicada",
        "decisao_sha256": sha256(caminho), "versoes": versoes(),
        "ficheiros_teste_sha256": {nome: sha256(DADOS / nome) for nome in [
            "wisconsin_teste_reservado.csv", "penguins_teste_reservado.csv"]},
        "n_teste_wisconsin": len(wt), "n_teste_pinguins": len(pt),
        "mediana_massa_desenvolvimento_g": float(pdv["body_mass_g"].median()),
        "limiar_wisconsin": decisao["wisconsin"]["limiar"],
    })
    print("Testes concluídos; os modelos e o limiar não foram escolhidos com estes resultados.", flush=True)
    print(tabela.round(4).to_string(index=False))
    print(pd.DataFrame(linhas).round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("etapa", choices=["desenvolvimento", "teste", "tudo"],
                        nargs="?", default="desenvolvimento")
    args = parser.parse_args()
    if args.etapa in ["desenvolvimento", "tudo"]:
        desenvolvimento()
    if args.etapa in ["teste", "tudo"]:
        teste()


if __name__ == "__main__":
    main()
