"""Configuração comum dos dois estudos de caso dos capítulos 14 e 15.

As funções deste módulo não leem dados ao ser importadas. A seleção e a
avaliação final são executadas por programas separados. Todos os caminhos
são relativos ao pacote, e todos os identificadores remetem para linhas.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, log_loss,
    mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
RESULTADOS = RAIZ / "resultados"
SEMENTE = 2026
ENTRADAS_CLASSIFICACAO = ["mean radius", "mean texture"]
ENTRADAS_REGRESSAO = ["flipper_length_mm", "bill_length_mm"]
HASHES_DESENVOLVIMENTO = {
    "wisconsin_desenvolvimento.csv": "997edc11984a788044ef3ceb17595f0b77b1aac91a7186e8afd3c2dde3b6153d",
    "penguins_desenvolvimento.csv": "8f3a9c1a2acef4ce13f8ba13cd30927194b0c30b73d83eaa82d65d269abb03d0",
}
LIMIARES = [0.20, 0.30, 0.50, 0.70]


def sha256_ficheiro(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def guardar_json(nome, objeto):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / nome).write_text(
        json.dumps(objeto, ensure_ascii=False, indent=2,
                   default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def guardar_csv(nome, tabela):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(RESULTADOS / nome, index=False, float_format="%.15g")


def versoes():
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def ler_desenvolvimento(nome):
    caminho = DADOS / nome
    if sha256_ficheiro(caminho) != HASHES_DESENVOLVIMENTO[nome]:
        raise ValueError(f"A cópia de desenvolvimento foi alterada: {nome}")
    tabela = pd.read_csv(caminho)
    assert tabela["id_registo"].is_unique
    assert tabela["id_registo"].is_monotonic_increasing
    return tabela


def configuracao():
    return {
        "semente": SEMENTE,
        "classificacao": {
            "entradas": ENTRADAS_CLASSIFICACAO, "resposta": "maligno",
            "classe_positiva": 1, "partes": 5, "estratificada": True,
            "criterio_escolha": "maior média das AUC por parte",
            "candidatos_em_ordem_de_desempate": ["logistica", "floresta"],
            "logistica": {"C": 1, "solver": "lbfgs", "max_iter": 1000,
                           "imputacao": "mediana", "padronizacao": True},
            "floresta": {"n_estimators": 200, "min_samples_leaf": 5,
                          "max_features": 1, "bootstrap": True,
                          "max_depth": None, "criterion": "gini",
                          "imputacao": "mediana", "n_jobs": 1},
            "limiares": LIMIARES, "sensibilidade_minima_observada": 0.90,
            "escolha_limiar": "maior especificidade entre os elegíveis; em empate, maior limiar",
            "regra_decisao": "p >= limiar",
            "referencia_probabilidade": "frequência de malignidade no treino",
            "referencia_decisao": "sempre benigno; não aplicar o limiar escolhido para o modelo",
        },
        "regressao": {
            "entradas": ENTRADAS_REGRESSAO, "resposta": "body_mass_g",
            "unidade_resposta": "g", "partes": 5, "estratificada": False,
            "criterio_escolha": "menor média dos MAE por parte",
            "candidatos_em_ordem_de_desempate": ["linear", "floresta"],
            "linear": {"imputacao": "mediana", "fit_intercept": True},
            "floresta": {"n_estimators": 200, "min_samples_leaf": 5,
                          "max_features": 1, "bootstrap": True,
                          "max_depth": None, "criterion": "squared_error",
                          "imputacao": "mediana", "n_jobs": 1},
            "referencia": "mediana da massa no treino",
            "exclusao": "apenas respostas ausentes; não imputar a resposta",
        },
    }


def classificadores():
    floresta = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, max_features=1,
        bootstrap=True, max_depth=None, criterion="gini",
        random_state=SEMENTE, n_jobs=1,
    )
    return {
        "referencia": make_pipeline(
            SimpleImputer(strategy="median"), DummyClassifier(strategy="prior")
        ),
        "logistica": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=1, solver="lbfgs", max_iter=1000,
                               random_state=SEMENTE),
        ),
        "floresta": make_pipeline(SimpleImputer(strategy="median"), floresta),
    }


def regressores():
    floresta = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, max_features=1,
        bootstrap=True, max_depth=None, criterion="squared_error",
        random_state=SEMENTE, n_jobs=1,
    )
    return {
        "referencia": make_pipeline(
            SimpleImputer(strategy="median"), DummyRegressor(strategy="median")
        ),
        "linear": make_pipeline(SimpleImputer(strategy="median"), LinearRegression()),
        "floresta": make_pipeline(SimpleImputer(strategy="median"), floresta),
    }


def ajustar(modelo, X, y, contexto, registo_avisos):
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo.fit(X, y)
    registo_avisos.append({
        "contexto": contexto,
        "avisos": [{"tipo": a.category.__name__, "mensagem": str(a.message)} for a in avisos],
    })
    return modelo


def medidas_classificacao(y, p, limiar=.5, decisao=None):
    y, p = np.asarray(y), np.asarray(p)
    d = (p >= limiar).astype(int) if decisao is None else np.asarray(decisao)
    vn, fp, fn, vp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return {
        "n": len(y), "AUC": roc_auc_score(y, p),
        "acerto": accuracy_score(y, d),
        "sensibilidade": vp / (vp + fn), "especificidade": vn / (vn + fp),
        "VPP": vp / (vp + fp) if vp + fp else None,
        "VN": vn, "FP": fp, "FN": fn, "VP": vp,
        "Brier": brier_score_loss(y, p, pos_label=1, scale_by_half=True),
        "perda_logaritmica": log_loss(y, p),
    }


def medidas_regressao(y, p):
    return {
        "n": len(y), "MAE_g": mean_absolute_error(y, p),
        "RMSE_g": root_mean_squared_error(y, p), "R2": r2_score(y, p),
    }
