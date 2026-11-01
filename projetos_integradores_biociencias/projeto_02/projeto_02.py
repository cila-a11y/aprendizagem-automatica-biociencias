#!/usr/bin/env python3
"""Projeto Pima: seleção no desenvolvimento; avaliação final num teste reservado.

Desenvolvimento: python projeto_02.py
Reprodução completa: python projeto_02.py --confirmar-teste
Dependências: numpy, pandas, scikit-learn e matplotlib.
Os dados locais são cópias dos ficheiros públicos de Brian Ripley:
https://www.stats.ox.ac.uk/pub/PRNN/pima.tr
https://www.stats.ox.ac.uk/pub/PRNN/pima.te
"""

from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, precision_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"
FIGURAS = BASE / "figuras"
SEED = 2026
PREDITORES = ["npreg", "glu", "bp", "skin", "bmi", "ped", "age"]
LIMIARES = np.round(np.arange(0.20, 0.8001, 0.05), 2)

# Ordem e regras definidas antes de consultar qualquer resultado do teste.
PROTOCOLO = {
    "seed": SEED,
    "desenvolvimento": "dados/pima.tr; 200 casos completos",
    "teste_reservado": "dados/pima.te; 332 casos completos",
    "preditores": PREDITORES,
    "classe_positiva": "Yes = 1",
    "validacao": "5 folds estratificados, baralhados, comuns a todos os candidatos",
    "selecao_modelo": "Maior media das cinco AUROC de validacao",
    "desempate_modelo": (
        "Igualdade a 12 casas decimais: primeira configuracao na ordem abaixo; "
        "logistica antes da arvore, C crescente, profundidade crescente, "
        "min_samples_leaf decrescente"
    ),
    "grelha_logistica": {"C": [0.1, 1.0, 10.0]},
    "grelha_arvore": {"max_depth": [2, 3, 4], "min_samples_leaf": [10, 5]},
    "limiares": LIMIARES.tolist(),
    "regra_positiva": "probabilidade >= limiar",
    "selecao_limiar": "Maior balanced accuracy nas previsoes OOF do candidato selecionado",
    "desempate_limiar": "Mais proximo de 0.50; persistindo empate, menor limiar",
    "referencia": "DummyClassifier(strategy='prior'): frequencia do treino e classe maioritaria",
    "aviso_oof": (
        "As previsoes OOF servem para selecionar modelo e limiar. As medidas "
        "usadas nessa selecao nao sao estimativas independentes do desempenho."
    ),
}


def ler_dados(nome):
    """Não descarrega dados nem altera as cópias fornecidas."""
    caminho = DADOS / nome
    df = pd.read_csv(caminho, sep=r"\s+")
    assert list(df.columns) == PREDITORES + ["type"]
    assert not df.isna().any().any(), "Este projeto utiliza apenas casos completos."
    assert set(df["type"]) == {"No", "Yes"}
    assert not df[PREDITORES].duplicated().any()
    X = df[PREDITORES].copy()
    y = df["type"].map({"No": 0, "Yes": 1}).to_numpy()
    auditoria = {
        "ficheiro": nome,
        "sha256": hashlib.sha256(caminho.read_bytes()).hexdigest(),
        "n": len(df),
        "positivos": int(y.sum()),
        "negativos": int((y == 0).sum()),
        "valores_em_falta": int(df.isna().sum().sum()),
        "perfis_duplicados": int(X.duplicated().sum()),
    }
    return X, y, auditoria


def candidatos():
    """A ordem desta lista também fixa os desempates."""
    lista = []
    for c in PROTOCOLO["grelha_logistica"]["C"]:
        modelo = Pipeline([
            ("escala", StandardScaler()),
            ("modelo", LogisticRegression(C=c, solver="lbfgs", max_iter=2000,
                                         random_state=SEED)),
        ])
        lista.append((f"logistica_C_{c:g}", "Regressao logistica", {"C": c}, modelo))
    for profundidade in PROTOCOLO["grelha_arvore"]["max_depth"]:
        for folha in PROTOCOLO["grelha_arvore"]["min_samples_leaf"]:
            params = {"max_depth": profundidade, "min_samples_leaf": folha}
            modelo = DecisionTreeClassifier(**params, random_state=SEED)
            lista.append((f"arvore_d_{profundidade}_folha_{folha}", "Arvore", params, modelo))
    return lista


def previsoes_oof(modelo, X, y, particoes):
    """Cada estimador, incluindo a sua escala, só vê o treino do respetivo fold."""
    probabilidades = np.full(len(y), np.nan)
    areas = []
    cobertura = np.zeros(len(y), dtype=int)
    for treino, validacao in particoes:
        assert len(np.intersect1d(treino, validacao)) == 0
        ajustado = clone(modelo).fit(X.iloc[treino], y[treino])
        p = ajustado.predict_proba(X.iloc[validacao])[:, 1]
        probabilidades[validacao] = p
        cobertura[validacao] += 1
        areas.append(float(roc_auc_score(y[validacao], p)))
    assert np.all(cobertura == 1) and np.isfinite(probabilidades).all()
    return probabilidades, areas


def medidas(y, probabilidades, limiar):
    previsto = (probabilidades >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, previsto, labels=[0, 1]).ravel()
    return {
        "limiar": float(limiar),
        "matriz_confusao_linhas_reais_colunas_previstas_No_Yes": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "sensibilidade": float(tp / (tp + fn)),
        "especificidade": float(tn / (tn + fp)),
        "precisao": float(precision_score(y, previsto, zero_division=0)),
        "accuracy": float(accuracy_score(y, previsto)),
        "balanced_accuracy": float(balanced_accuracy_score(y, previsto)),
        "AUROC": float(roc_auc_score(y, probabilidades)),
        "Brier": float(brier_score_loss(y, probabilidades)),
    }


def guardar_json(resultado, nome="resultados.json"):
    (BASE / nome).write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def guardar_figura(fig, nome):
    for extensao in ["pdf", "png"]:
        fig.savefig(FIGURAS / f"{nome}.{extensao}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figura_desenvolvimento(tabela_limiares, limiar):
    FIGURAS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    azul, verde, laranja = "#245B78", "#438B78", "#B86B37"
    fig, ax = plt.subplots(figsize=(8.4, 4.5), constrained_layout=True)
    x = [r["limiar"] for r in tabela_limiares]
    for chave, texto, cor in [
        ("sensibilidade", "Sensibilidade", azul),
        ("especificidade", "Especificidade", laranja),
        ("balanced_accuracy", "Exatidão equilibrada", verde),
    ]:
        ax.plot(x, [r[chave] for r in tabela_limiares], marker="o", markersize=4,
                linewidth=1.8, color=cor, label=texto)
    ax.axvline(limiar, color="#555555", linestyle="--", linewidth=1.2,
               label=f"Limiar escolhido: {limiar:.2f}".replace(".", ","))
    ax.set(xlabel="Limiar de decisão", ylabel="Proporção", ylim=(0, 1.03),
           title="Desenvolvimento: previsões OOF usadas na seleção")
    ax.set_xticks(LIMIARES[::2])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:.1f}".replace(".", ",")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:.1f}".replace(".", ",")))
    ax.grid(alpha=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    guardar_figura(fig, "pima_limiar_desenvolvimento")


def figura_teste(limiar, y_teste, p_teste, metricas):
    FIGURAS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    azul, laranja = "#245B78", "#B86B37"
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    fpr, tpr, _ = roc_curve(y_teste, p_teste)
    axes[0].plot(fpr, tpr, color=azul, linewidth=2.1,
                 label=f"Modelo escolhido · AUROC = {metricas['AUROC']:.3f}".replace(".", ","))
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[0].plot(1 - metricas["especificidade"], metricas["sensibilidade"], "o",
                 color=laranja, markersize=7, label=f"Limiar = {limiar:.2f}".replace(".", ","))
    axes[0].set(xlabel="1 − especificidade", ylabel="Sensibilidade",
                title="Teste final: curva ROC", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:.1f}".replace(".", ",")))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:.1f}".replace(".", ",")))
    axes[0].grid(alpha=0.18)
    matriz = np.array(metricas["matriz_confusao_linhas_reais_colunas_previstas_No_Yes"])
    axes[1].imshow(matriz, cmap="Blues", vmin=0, vmax=matriz.max() * 1.1)
    for (i, j), valor in np.ndenumerate(matriz):
        axes[1].text(j, i, str(valor), ha="center", va="center", fontsize=20,
                     color="white" if valor > matriz.max() * 0.55 else "#183044")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No", "Yes"],
                yticklabels=["No", "Yes"], xlabel="Classe prevista", ylabel="Classe observada",
                title="Teste final: matriz de confusão")
    guardar_figura(fig, "pima_teste_final")


def main(confirmar_teste=False):
    # FASE 1: só é aberto o ficheiro de desenvolvimento.
    X, y, auditoria = ler_dados("pima.tr")
    assert len(y) == 200
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    particoes = list(cv.split(X, y))
    configuracoes = candidatos()
    resumos, probabilidades_oof = [], {}
    for ordem, (nome, familia, params, modelo) in enumerate(configuracoes):
        p, areas = previsoes_oof(modelo, X, y, particoes)
        probabilidades_oof[nome] = p
        resumos.append({"ordem": ordem, "id": nome, "familia": familia, "parametros": params,
                        "AUROC_por_fold": areas, "AUROC_media_folds": float(np.mean(areas)),
                        "AUROC_OOF_agregada": float(roc_auc_score(y, p))})

    # Escolhe a configuração pela AUROC média, antes de escolher o limiar.
    escolhido = max(resumos, key=lambda r: (round(r["AUROC_media_folds"], 12), -r["ordem"]))
    p_oof = probabilidades_oof[escolhido["id"]]
    tabela_limiares = [medidas(y, p_oof, t) for t in LIMIARES]
    escolha_limiar = max(tabela_limiares, key=lambda r: (
        round(r["balanced_accuracy"], 12), -round(abs(r["limiar"] - 0.5), 12), -r["limiar"]
    ))
    limiar = escolha_limiar["limiar"]

    baseline = DummyClassifier(strategy="prior")
    p_baseline_oof, areas_baseline = previsoes_oof(baseline, X, y, particoes)
    # A AUROC por fold é 0,5. Não usar a AUROC agregada da baseline para seleção:
    # a pequena variação da prevalência entre folds altera a ordenação entre folds.
    referencia_cv = {
        "AUROC_por_fold": areas_baseline,
        "AUROC_media_folds": float(np.mean(areas_baseline)),
        "accuracy_classe_maioritaria": float(accuracy_score(y, p_baseline_oof >= 0.5)),
        "balanced_accuracy_classe_maioritaria": float(balanced_accuracy_score(y, p_baseline_oof >= 0.5)),
    }
    resultado = {
        "versoes": {"Python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                    "matplotlib": matplotlib.__version__},
        "protocolo_predefinido": PROTOCOLO,
        "auditoria_desenvolvimento": auditoria,
        "desenvolvimento": {
            "configuracoes": resumos,
            "referencia_cv": referencia_cv,
            "limiares_modelo_escolhido": tabela_limiares,
            "metricas_oof_limiar_050": medidas(y, p_oof, 0.5),
            "metricas_oof_limiar_escolhido": escolha_limiar,
            "indices_validacao_por_fold_base_zero": [v.tolist() for _, v in particoes],
            "probabilidades_oof_modelo_escolhido": p_oof.tolist(),
        },
        "decisoes_fechadas_antes_teste": {
            "modelo": escolhido["id"], "familia": escolhido["familia"],
            "parametros": escolhido["parametros"], "limiar": limiar,
            "ordem_candidato": escolhido["ordem"],
        },
        "registo_decisoes": [
            {"fase": 1, "dados": "nenhum resultado de teste", "decisao": "Fixar protocolo, grelhas e desempates"},
            {"fase": 2, "dados": "pima.tr", "decisao": "Selecionar configuracao pela AUROC media dos folds"},
            {"fase": 3, "dados": "previsoes OOF em pima.tr", "decisao": "Selecionar e fechar limiar"},
        ],
    }
    # Regista decisões e desenvolvimento antes de qualquer acesso ao teste.
    # Estes ficheiros não substituem um resultado final já fornecido ao leitor.
    guardar_json(resultado, "resultados_desenvolvimento.json")
    guardar_json({
        "seed": SEED,
        "sha256_desenvolvimento": auditoria["sha256"],
        "protocolo_predefinido": PROTOCOLO,
        "decisoes_fechadas_antes_teste": resultado["decisoes_fechadas_antes_teste"],
    }, "decisoes.json")
    figura_desenvolvimento(tabela_limiares, limiar)
    if not confirmar_teste:
        print(json.dumps({"decisoes": resultado["decisoes_fechadas_antes_teste"],
                          "estado": "Desenvolvimento concluido; pima.te nao foi lido.",
                          "reproducao_completa": "python projeto_02.py --confirmar-teste"},
                         ensure_ascii=False, indent=2))
        return

    resultado["registo_decisoes"].append({
        "fase": 4, "dados": "pima.tr", "decisao": "Reajustar modelo e referencia em todos os 200 casos"
    })
    modelo_final = clone(configuracoes[escolhido["ordem"]][3]).fit(X, y)
    baseline_final = clone(baseline).fit(X, y)

    # FASE 2: o teste só é lido depois de todas as decisões ficarem fechadas.
    X_teste, y_teste, auditoria_teste = ler_dados("pima.te")
    assert len(y_teste) == 332
    assert not pd.concat([X, X_teste]).duplicated().any(), "Há perfis comuns ao treino e teste."
    p_teste = modelo_final.predict_proba(X_teste)[:, 1]
    p_referencia = baseline_final.predict_proba(X_teste)[:, 1]
    metricas_teste = medidas(y_teste, p_teste, limiar)
    resultado["teste_final"] = {
        "auditoria": auditoria_teste,
        "modelo_escolhido": metricas_teste,
        "referencia_maioritaria": medidas(y_teste, p_referencia, 0.5),
        "probabilidade_constante_referencia": float(p_referencia[0]),
        "probabilidades_modelo_escolhido": p_teste.tolist(),
    }
    resultado["registo_decisoes"].append({
        "fase": 5, "dados": "pima.te", "decisao": "Avaliar uma vez, sem alterar modelo ou limiar"
    })
    guardar_json(resultado)
    figura_teste(limiar, y_teste, p_teste, metricas_teste)
    print(json.dumps({"decisoes": resultado["decisoes_fechadas_antes_teste"],
                      "teste_final": metricas_teste}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmar-teste", action="store_true",
                        help="Repetir o desenvolvimento e executar a avaliacao final reservada.")
    main(confirmar_teste=parser.parse_args().confirmar_teste)
