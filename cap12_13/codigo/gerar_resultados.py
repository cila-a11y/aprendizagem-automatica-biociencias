"""Reproduzir os resultados dos capítulos 12 e 13, sem ler o teste reservado.

Executar com Python 3.12 a partir de qualquer diretório. As entradas são as
duas cópias fixas de desenvolvimento incluídas na pasta dados deste pacote.
Os resultados gerados substituem apenas os ficheiros desta pasta resultados.
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
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


PASTA = Path(__file__).resolve().parents[1]
SAIDA = PASTA / "resultados"
SEMENTE = 2026
HASHES = {
    "wisconsin_desenvolvimento.csv": "997edc11984a788044ef3ceb17595f0b77b1aac91a7186e8afd3c2dde3b6153d",
    "penguins_desenvolvimento.csv": "8f3a9c1a2acef4ce13f8ba13cd30927194b0c30b73d83eaa82d65d269abb03d0",
}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def guardar_json(nome, value):
    with (SAIDA / nome).open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
        f.write("\n")


def guardar_csv(nome, value):
    value.to_csv(SAIDA / nome, index=False, float_format="%.15g")


def ler_dados(nome):
    caminho = PASTA / "dados" / nome
    digest = hashlib.sha256(caminho.read_bytes()).hexdigest()
    if digest != HASHES[nome]:
        raise ValueError(f"A cópia fixa foi alterada: {nome}")
    dados = pd.read_csv(caminho)
    assert dados["id_registo"].is_unique
    assert dados["id_registo"].is_monotonic_increasing
    return dados


def ajustar(modelo, X, y, contexto, registo):
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo.fit(X, y)
    registo.append({
        "contexto": contexto,
        "avisos": [{"classe": w.category.__name__, "mensagem": str(w.message)} for w in avisos],
    })
    return modelo


def metricas_classificacao(y, p):
    return {
        "AUC": roc_auc_score(y, p),
        "acerto_limiar_050": accuracy_score(y, np.asarray(p) >= 0.5),
        "perda_logaritmica": log_loss(y, p),
        "Brier": brier_score_loss(y, p, pos_label=1, scale_by_half=True),
    }


def metricas_regressao(y, p):
    return {
        "MAE_g": mean_absolute_error(y, p),
        "RMSE_g": root_mean_squared_error(y, p),
        "R2": r2_score(y, p),
    }


def registo_particao(dados, treino, validacao, parte):
    return {
        "parte": parte,
        "ids_treino": dados.iloc[treino]["id_registo"].tolist(),
        "ids_validacao": dados.iloc[validacao]["id_registo"].tolist(),
    }


def calibracao_por_faixas(y, p, identificador):
    """Faixas [0,.2), [.2,.4), ..., [.8,1]; omitir pontos se n=0."""
    limites = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    y, p = np.asarray(y), np.asarray(p)
    indice = np.searchsorted(limites[1:-1], p, side="right")
    linhas = []
    for b in range(5):
        m = indice == b
        n = int(m.sum())
        linhas.append({
            "versao": identificador,
            "faixa": b + 1,
            "limite_inferior_incluido": limites[b],
            "limite_superior": limites[b + 1],
            "limite_superior_incluido": b == 4,
            "n": n,
            "n_malignos": int(y[m].sum()),
            "probabilidade_media": float(p[m].mean()) if n else np.nan,
            "fracao_maligna": float(y[m].mean()) if n else np.nan,
        })
    assert sum(row["n"] for row in linhas) == len(y)
    return pd.DataFrame(linhas)


def calcular_classificacao(dados, avisos):
    X = dados[["mean radius", "mean texture"]]
    y = dados["maligno"]
    assert len(dados) == 455
    assert y.value_counts().to_dict() == {0: 285, 1: 170}
    assert X.notna().all().all()
    base = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=1, solver="lbfgs", max_iter=1000, random_state=SEMENTE),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    p_oof = np.full(len(y), np.nan)
    p_referencia = np.full(len(y), np.nan)
    numero_parte = np.zeros(len(y), dtype=int)
    linhas, particoes, ajustes = [], [], []
    for j, (treino, val) in enumerate(cv.split(X, y), 1):
        modelo = ajustar(clone(base), X.iloc[treino], y.iloc[treino], f"logistica_parte_{j}", avisos)
        p_train = modelo.predict_proba(X.iloc[treino])[:, 1]
        p_val = modelo.predict_proba(X.iloc[val])[:, 1]
        p_oof[val] = p_val
        p_referencia[val] = y.iloc[treino].mean()
        numero_parte[val] = j
        treino_m, val_m = metricas_classificacao(y.iloc[treino], p_train), metricas_classificacao(y.iloc[val], p_val)
        linhas.append({
            "parte": j, "n_treino": len(treino), "n_validacao": len(val),
            "benignos_validacao": int((y.iloc[val] == 0).sum()),
            "malignos_validacao": int((y.iloc[val] == 1).sum()),
            **{f"{key}_treino": value for key, value in treino_m.items()},
            **{f"{key}_validacao": value for key, value in val_m.items()},
        })
        particoes.append(registo_particao(dados, treino, val, j))
        scaler = modelo.named_steps["standardscaler"]
        logistica = modelo.named_steps["logisticregression"]
        ajustes.append({
            "parte": j,
            "medianas": modelo.named_steps["simpleimputer"].statistics_,
            "medias_padronizacao": scaler.mean_, "escalas_padronizacao": scaler.scale_,
            "classes": logistica.classes_, "coeficientes": logistica.coef_,
            "termo_constante": logistica.intercept_, "n_iter": logistica.n_iter_,
        })
    assert np.isfinite(p_oof).all() and set(numero_parte) == {1, 2, 3, 4, 5}
    tabela = pd.DataFrame(linhas)
    guardar_csv("classificacao_por_parte.csv", tabela)
    guardar_json("particoes_classificacao.json", particoes)
    guardar_json("ajustes_logistica.json", ajustes)
    previsoes = dados[["id_registo", "mean radius", "mean texture", "maligno"]].copy()
    previsoes["parte"] = numero_parte
    previsoes["p_logistica"] = p_oof
    previsoes["p_frequencia_treino"] = p_referencia
    previsoes["classe_limiar_050"] = (p_oof >= 0.5).astype(int)
    guardar_csv("classificacao_previsoes_reunidas.csv", previsoes)
    resumo_partes = []
    for key in metricas_classificacao(y, p_oof):
        valores = tabela[f"{key}_validacao"]
        resumo_partes.append({"medida": key, "media_partes": valores.mean(), "desvio_partes_ddof1": valores.std(ddof=1)})
    guardar_csv("classificacao_resumo_partes.csv", pd.DataFrame(resumo_partes))
    reunidas = []
    tabelas_calibracao = []
    for nome, p in [("Logistica", p_oof), ("Frequencia no treino", p_referencia)]:
        reunidas.append({"versao": nome, **metricas_classificacao(y, p)})
        tabelas_calibracao.append(calibracao_por_faixas(y, p, nome))
    guardar_csv("classificacao_metricas_reunidas.csv", pd.DataFrame(reunidas))
    guardar_csv("calibracao_faixas.csv", pd.concat(tabelas_calibracao, ignore_index=True))
    return {"resumo_partes": resumo_partes, "metricas_reunidas": reunidas}


def calcular_regressao(dados, avisos):
    assert len(dados) == 275
    reg = dados.dropna(subset=["body_mass_g"]).reset_index(drop=True)
    assert len(reg) == 273
    X = reg[["flipper_length_mm", "bill_length_mm"]]
    y = reg["body_mass_g"]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    divisao = list(cv.split(X, y))
    candidatos = {
        "Mediana": make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="median")),
        "Linear": make_pipeline(SimpleImputer(strategy="median"), LinearRegression()),
    }
    previsoes = reg.copy()
    linhas, resumo = [], []
    for nome, base in candidatos.items():
        p = np.full(len(reg), np.nan)
        partes = np.zeros(len(reg), dtype=int)
        for j, (treino, val) in enumerate(divisao, 1):
            modelo = ajustar(clone(base), X.iloc[treino], y.iloc[treino], f"{nome}_parte_{j}", avisos)
            p[val] = modelo.predict(X.iloc[val])
            partes[val] = j
            linhas.append({"procedimento": nome, "parte": j, "n_treino": len(treino), "n_validacao": len(val), **metricas_regressao(y.iloc[val], p[val])})
        assert np.isfinite(p).all()
        previsoes[f"previsto_{nome.lower()}_g"] = p
        previsoes[f"erro_{nome.lower()}_g"] = y.to_numpy() - p
        bloco = pd.DataFrame([row for row in linhas if row["procedimento"] == nome])
        n = bloco["n_validacao"].to_numpy()
        reunidas = metricas_regressao(y, p)
        assert np.isclose(np.average(bloco["MAE_g"], weights=n), reunidas["MAE_g"])
        assert np.isclose(np.sqrt(np.average(bloco["RMSE_g"] ** 2, weights=n)), reunidas["RMSE_g"])
        for key, pooled in reunidas.items():
            resumo.append({
                "procedimento": nome, "medida": key,
                "media_partes": bloco[key].mean(),
                "desvio_partes_ddof1": bloco[key].std(ddof=1),
                "media_ponderada_n": np.average(bloco[key], weights=n),
                "previsoes_reunidas": pooled,
            })
    previsoes["parte"] = partes
    guardar_csv("regressao_previsoes_reunidas.csv", previsoes)
    guardar_csv("regressao_por_parte.csv", pd.DataFrame(linhas))
    guardar_csv("regressao_resumos.csv", pd.DataFrame(resumo))
    guardar_json("particoes_regressao.json", [registo_particao(reg, tr, va, j) for j, (tr, va) in enumerate(divisao, 1)])
    subgrupos = []
    for nome, grupo in list(previsoes.groupby("species", sort=True)) + [("Total", previsoes)]:
        erros = grupo["erro_linear_g"]
        subgrupos.append({"especie": nome, "n": len(grupo), "erro_medio_assinado_g": erros.mean(), "MAE_g": erros.abs().mean()})
    guardar_csv("regressao_erros_especies.csv", pd.DataFrame(subgrupos))
    return resumo, subgrupos


def calcular_aninhada(dados, avisos):
    """Avaliar a regra que escolhe C dentro de cada treino exterior.

    A melhor média interna orienta a seleção. Não é uma medida exterior.
    Este programa não escolhe um C final usando as 455 amostras juntas.
    """
    X, y = dados[["mean radius", "mean texture"]], dados["maligno"]
    exterior = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEMENTE))
    p_oof = np.full(len(y), np.nan)
    numero_parte = np.zeros(len(y), dtype=int)
    linhas, internas_linhas, particoes, ajustes = [], [], [], []
    for j, (tr, va) in enumerate(exterior.split(X, y), 1):
        interno = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEMENTE)
        pesquisa = GridSearchCV(base, {"logisticregression__C": [0.1, 1, 10]},
                               scoring="roc_auc", cv=interno, refit=True, n_jobs=1,
                               error_score="raise", return_train_score=False)
        ajustar(pesquisa, X.iloc[tr], y.iloc[tr], f"validacao_aninhada_exterior_{j}", avisos)
        p = pesquisa.predict_proba(X.iloc[va])[:, 1]
        p_oof[va] = p
        numero_parte[va] = j
        linhas.append({
            "parte_exterior": j, "n_treino_exterior": len(tr), "n_validacao_exterior": len(va),
            "C_escolhido": pesquisa.best_params_["logisticregression__C"],
            "AUC_media_interna_escolhida": pesquisa.best_score_,
            **{f"{key}_exterior": value for key, value in metricas_classificacao(y.iloc[va], p).items()},
        })
        for k, parametros in enumerate(pesquisa.cv_results_["params"]):
            internas_linhas.append({
                "parte_exterior": j, "C_candidato": parametros["logisticregression__C"],
                **{f"AUC_parte_interna_{i + 1}": pesquisa.cv_results_[f"split{i}_test_score"][k] for i in range(4)},
                "AUC_media_interna": pesquisa.cv_results_["mean_test_score"][k],
                "desvio_interno_ddof0": pesquisa.cv_results_["std_test_score"][k],
                "posicao_na_pesquisa": pesquisa.cv_results_["rank_test_score"][k],
            })
        grupo = registo_particao(dados, tr, va, j)
        grupo["particoes_internas"] = [registo_particao(dados, tr[itr], tr[iva], i)
            for i, (itr, iva) in enumerate(interno.split(X.iloc[tr], y.iloc[tr]), 1)]
        for p_interna in grupo["particoes_internas"]:
            assert set(p_interna["ids_treino"]).isdisjoint(grupo["ids_validacao"])
            assert set(p_interna["ids_validacao"]).isdisjoint(grupo["ids_validacao"])
        particoes.append(grupo)
        escolhido = pesquisa.best_estimator_
        ajustes.append({
            "parte_exterior": j, "C": pesquisa.best_params_["logisticregression__C"],
            "medianas": escolhido.named_steps["simpleimputer"].statistics_,
            "medias": escolhido.named_steps["standardscaler"].mean_,
            "escalas": escolhido.named_steps["standardscaler"].scale_,
            "coeficientes": escolhido.named_steps["logisticregression"].coef_,
            "termo_constante": escolhido.named_steps["logisticregression"].intercept_,
        })
    assert np.isfinite(p_oof).all()
    tabela = pd.DataFrame(linhas)
    guardar_csv("aninhada_por_parte.csv", tabela)
    guardar_csv("aninhada_pesquisas_internas.csv", pd.DataFrame(internas_linhas))
    guardar_json("aninhada_particoes.json", particoes)
    guardar_json("aninhada_ajustes.json", ajustes)
    p = dados[["id_registo", "maligno"]].copy()
    p["parte_exterior"] = numero_parte
    p["p_aninhada"] = p_oof
    guardar_csv("aninhada_previsoes_reunidas.csv", p)
    resumo = {"AUC_media_exterior": tabela["AUC_exterior"].mean(),
              "desvio_AUC_exterior_ddof1": tabela["AUC_exterior"].std(ddof=1),
              "Cs_escolhidos": tabela["C_escolhido"].tolist(),
              "metricas_reunidas": metricas_classificacao(y, p_oof),
              "nota": "As médias internas orientaram escolhas em cada treino exterior; não são confundidas com as avaliações exteriores. Não houve pesquisa final nas 455 amostras."}
    guardar_json("aninhada_resumo.json", resumo)
    return resumo


def calcular_permutacao(dados, avisos):
    X, y = dados[["mean radius", "mean texture"]], dados["maligno"]
    treino, val = train_test_split(np.arange(len(y)), test_size=0.30, stratify=y, random_state=SEMENTE)
    modelo = RandomForestClassifier(n_estimators=200, max_features=1, bootstrap=True,
                                    min_samples_leaf=5, random_state=SEMENTE, n_jobs=1)
    ajustar(modelo, X.iloc[treino], y.iloc[treino], "floresta_permutacao", avisos)
    p = modelo.predict_proba(X.iloc[val])[:, 1]
    auc = roc_auc_score(y.iloc[val], p)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importancia = permutation_importance(modelo, X.iloc[val], y.iloc[val], scoring="roc_auc",
                                             n_repeats=20, random_state=SEMENTE, n_jobs=1)
    avisos.append({"contexto": "permutation_importance", "avisos": [{"classe": a.category.__name__, "mensagem": str(a.message)} for a in w]})
    linhas, resumo = [], []
    for j, coluna in enumerate(X.columns):
        v = importancia.importances[j]
        assert np.isclose(importancia.importances_std[j], v.std(ddof=0))
        for k, queda in enumerate(v, 1):
            linhas.append({"variavel": coluna, "repeticao": k, "AUC_original": auc, "AUC_permutada": auc - queda, "diminuicao_AUC": queda})
        resumo.append({"variavel": coluna, "diminuicao_media_AUC": v.mean(), "desvio_repeticoes_ddof0": v.std(ddof=0), "desvio_amostral_ddof1": v.std(ddof=1)})
    guardar_csv("permutacao_repeticoes.csv", pd.DataFrame(linhas))
    guardar_csv("permutacao_resumo.csv", pd.DataFrame(resumo))
    pv = dados.iloc[val][["id_registo", "mean radius", "mean texture", "maligno"]].copy()
    pv["p_floresta"] = p
    guardar_csv("permutacao_previsoes_validacao.csv", pv)
    guardar_json("particao_permutacao.json", {
        **registo_particao(dados, treino, val, 1),
        "n_treino": len(treino), "n_validacao": len(val),
        "classes_treino": y.iloc[treino].value_counts().sort_index().to_dict(),
        "classes_validacao": y.iloc[val].value_counts().sort_index().to_dict(),
    })
    correlacoes = dados[["mean radius", "mean perimeter", "mean area", "mean texture"]].corr()
    correlacoes.index.name = "variavel"
    correlacoes.to_csv(SAIDA / "correlacoes.csv", float_format="%.15g")
    # Geometria de duas medidas relacionadas. A resposta mantém a posição
    # original em todos os painéis: o par permutado não acompanha o rótulo.
    originais = dados.iloc[val].reset_index(drop=True)
    ordem = np.random.default_rng(SEMENTE).permutation(len(originais))
    geometrias = []
    for painel in ["Original", "Permutacao individual", "Permutacao conjunta"]:
        j_raio = np.arange(len(originais)) if painel == "Original" else ordem
        j_perimetro = ordem if painel == "Permutacao conjunta" else np.arange(len(originais))
        geometrias.append(pd.DataFrame({
            "painel": painel,
            "id_linha_original": originais["id_registo"].to_numpy(),
            "id_dador_raio": originais.iloc[j_raio]["id_registo"].to_numpy(),
            "id_dador_perimetro": originais.iloc[j_perimetro]["id_registo"].to_numpy(),
            "raio_medio": originais.iloc[j_raio]["mean radius"].to_numpy(),
            "perimetro_medio": originais.iloc[j_perimetro]["mean perimeter"].to_numpy(),
            "maligno_linha_original": originais["maligno"].to_numpy(),
        }))
    geometria = pd.concat(geometrias, ignore_index=True)
    guardar_csv("dependencia_geometria.csv", geometria)
    guardar_json("dependencia_permutacao.json", {
        "semente": SEMENTE, "gerador": "numpy.random.default_rng",
        "indices_dadores": ordem, "n": len(val),
        "descricao": "A mesma permutação é usada para o raio sozinho e para o par raio/perímetro. As cores e formas usam sempre a resposta da linha original. Não se reajusta nem se avalia um modelo com estas duas medidas.",
    })
    return {"AUC_original": auc, "n_treino": len(treino), "n_validacao": len(val), "importancias": resumo}


def exemplos_didaticos():
    """Pequenos conjuntos inventados, nunca confundidos com Wisconsin/pinguins."""
    auc = pd.DataFrame({"parte": [1, 1, 2, 2], "classe": [0, 1, 0, 1], "pontuacao": [0.1, 0.2, 0.8, 0.9]})
    guardar_csv("exemplo_ficticio_auc.csv", auc)
    auc_partes = [roc_auc_score(g["classe"], g["pontuacao"]) for _, g in auc.groupby("parte")]
    erros = pd.DataFrame({"parte": [1, 1, 2], "erro_g": [100, 100, 300]})
    guardar_csv("exemplo_ficticio_erros.csv", erros)
    erros_partes = []
    for j, g in erros.groupby("parte"):
        e = g["erro_g"].to_numpy()
        erros_partes.append({"parte": j, "n": len(e), "MAE_g": np.mean(np.abs(e)), "RMSE_g": np.sqrt(np.mean(e ** 2))})
    resumo = {
        "origem": "Todos os valores desta secção são fictícios.",
        "auc": {"partes": auc_partes, "media_partes": np.mean(auc_partes), "reunida": roc_auc_score(auc["classe"], auc["pontuacao"])},
        "erros": {
            "partes": erros_partes,
            "MAE_medio_simples_g": np.mean([r["MAE_g"] for r in erros_partes]),
            "RMSE_medio_simples_g": np.mean([r["RMSE_g"] for r in erros_partes]),
            "MAE_reunido_g": np.mean(np.abs(erros["erro_g"])),
            "RMSE_reunido_g": np.sqrt(np.mean(erros["erro_g"] ** 2)),
        },
    }
    assert resumo["auc"]["media_partes"] == 1 and resumo["auc"]["reunida"] == 0.75
    guardar_json("exemplos_ficticios.json", resumo)
    grupos = pd.DataFrame({"doente": np.repeat(list("ABCDEF"), 2),
                           "medida": [2.1, 2.3, 5.2, 5.0, 2.8, 2.9, 4.8, 4.6, 3.0, 2.7, 5.5, 5.4],
                           "classe": np.repeat([0, 1, 0, 1, 0, 1], 2)})
    partes = []
    for j, (tr, va) in enumerate(GroupKFold(n_splits=3).split(grupos[["medida"]], grupos["classe"], grupos["doente"]), 1):
        gt, gv = sorted(set(grupos.iloc[tr]["doente"])), sorted(set(grupos.iloc[va]["doente"]))
        assert set(gt).isdisjoint(gv)
        partes.append({"parte": j, "doentes_treino": gt, "doentes_validacao": gv, "classes_validacao": grupos.iloc[va]["classe"].value_counts().sort_index().to_dict()})
    guardar_json("exemplo_ficticio_particoes_grupos.json", partes)
    return resumo


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    wisconsin = ler_dados("wisconsin_desenvolvimento.csv")
    pinguins = ler_dados("penguins_desenvolvimento.csv")
    avisos = []
    with threadpool_limits(limits=1):
        classificacao = calcular_classificacao(wisconsin, avisos)
        regressao, subgrupos = calcular_regressao(pinguins, avisos)
        aninhada = calcular_aninhada(wisconsin, avisos)
        permutacao = calcular_permutacao(wisconsin, avisos)
        ficticios = exemplos_didaticos()
    guardar_json("avisos.json", avisos)
    relatorio = {
        "versoes": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "sha256_dados": HASHES,
        "semente": SEMENTE,
        "n_wisconsin": len(wisconsin), "n_pinguins_desenvolvimento": len(pinguins), "n_pinguins_com_massa": int(pinguins["body_mass_g"].notna().sum()),
        "teste_reservado_utilizado": False,
        "convencoes": {
            "classe_positiva": "maligno=1", "decisao": "p>=0.50", "erro_de_previsao": "observado menos previsto (g)",
            "desvios_entre_partes": "ddof=1", "desvios_permutation_importance": "ddof=0 (convenção da função)",
            "Brier": "média de (p-y)^2, escala binária [0,1]",
            "calibracao": "cinco faixas fixas; todos os pontos usam previsões fora do respetivo treino; a curva é uma descrição amostral, não uma estimativa com intervalos inferenciais",
        },
        "classificacao": classificacao, "regressao": regressao,
        "regressao_subgrupos": subgrupos, "validacao_aninhada": aninhada,
        "permutacao": permutacao,
        "exemplos_ficticios": ficticios,
        "n_avisos": sum(len(a["avisos"]) for a in avisos),
    }
    guardar_json("resultados_verificados.json", relatorio)
    print("Resultados reproduzidos sem ler o teste reservado.")
    print(pd.DataFrame(classificacao["resumo_partes"]).to_string(index=False))
    print(pd.DataFrame(classificacao["metricas_reunidas"]).to_string(index=False))
    print("Permutação: AUC original", permutacao["AUC_original"])
    print(pd.DataFrame(permutacao["importancias"]).to_string(index=False))
    print("Avisos:", relatorio["n_avisos"])


if __name__ == "__main__":
    main()
