"""Reproduzir os cálculos do capítulo 11, sem consultar o teste reservado.

Executar de qualquer pasta: python codigo/gerar_resultados.py

As tabelas de validação usam cinco partes fixas e padronização ajustada
dentro de cada treino. As curvas por época usam uma demonstração separada
de 318/137 amostras. Os mapas de probabilidades, quando utilizados, provêm
do ajustamento ao desenvolvimento completo, não de previsões de validação.
"""

import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"
SEMENTE = 2026
COLUNAS = ["mean radius", "mean texture"]
HASH_DADOS = "997edc11984a788044ef3ceb17595f0b77b1aac91a7186e8afd3c2dde3b6153d"
EPOCAS_TABELA = [1, 10, 25, 50, 100, 200]


def guardar_json(nome, objeto):
    (RESULTADOS / nome).write_text(
        json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validar_dados():
    caminho = RAIZ / "dados/wisconsin_desenvolvimento.csv"
    assert hashlib.sha256(caminho.read_bytes()).hexdigest() == HASH_DADOS
    dados = pd.read_csv(caminho)
    assert len(dados) == 455
    assert dados["id_registo"].is_unique
    assert dados["id_registo"].is_monotonic_increasing
    assert dados["maligno"].value_counts().to_dict() == {0: 285, 1: 170}
    assert not dados[COLUNAS + ["maligno"]].isna().any().any()
    return dados


def criar_rede(unidades=8):
    return MLPClassifier(
        hidden_layer_sizes=(unidades,), activation="relu",
        solver="adam", alpha=0.001,
        learning_rate_init=0.01, batch_size=32,
        max_iter=800, n_iter_no_change=30, tol=1e-4,
        random_state=SEMENTE, early_stopping=False,
    )


def procedimentos():
    return {
        "logistica": ("Logística", make_pipeline(
            StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=1000)
        )),
        "floresta_folha_5": ("Floresta folha mínima 5", RandomForestClassifier(
            n_estimators=200, max_features=1, bootstrap=True,
            max_depth=None, min_samples_leaf=5, random_state=SEMENTE, n_jobs=1,
        )),
        "rede_8": ("Rede com 8 unidades", make_pipeline(StandardScaler(), criar_rede(8))),
        "rede_32": ("Rede com 32 unidades", make_pipeline(StandardScaler(), criar_rede(32))),
    }


def medidas(y, p):
    previstas = (p >= 0.5).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, previstas, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "acerto": float(np.mean(previstas == np.asarray(y))),
        "VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp),
        "empates_050": int(np.count_nonzero(p == 0.5)),
    }


def avisos(registos):
    return [{"categoria": w.category.__name__, "mensagem": str(w.message)}
            for w in registos]


def contar_parametros(rede):
    pesos = sum(a.size for a in rede.coefs_)
    constantes = sum(a.size for a in rede.intercepts_)
    return {"pesos": int(pesos), "termos_constantes": int(constantes),
            "total": int(pesos + constantes),
            "formas_pesos": [list(a.shape) for a in rede.coefs_],
            "formas_constantes": [list(a.shape) for a in rede.intercepts_]}


def avaliar(dados):
    X, y = dados[COLUNAS], dados["maligno"]
    particoes = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEMENTE
    ).split(X, y))
    particao_por_linha = np.zeros(len(dados), dtype=int)
    partes = []
    for parte, (treino, validacao) in enumerate(particoes, start=1):
        assert not np.intersect1d(treino, validacao).size
        assert len(treino) == 364 and len(validacao) == 91
        particao_por_linha[validacao] = parte
        partes.append({
            "parte": parte,
            "ids_treino": dados.iloc[treino]["id_registo"].astype(int).tolist(),
            "ids_validacao": dados.iloc[validacao]["id_registo"].astype(int).tolist(),
            "n_treino": len(treino), "n_validacao": len(validacao),
        })
    guardar_json("particoes_classificacao.json", partes)
    previsoes = dados[["id_registo", *COLUNAS, "maligno"]].copy()
    previsoes["parte_validacao"] = particao_por_linha
    linhas, agregadas, resumos, ajustes = [], [], [], []
    for chave, (nome, prototipo) in procedimentos().items():
        p_reunidas = np.empty(len(y))
        for parte, (treino, validacao) in enumerate(particoes, start=1):
            with warnings.catch_warnings(record=True) as capturados:
                warnings.simplefilter("always")
                modelo = clone(prototipo).fit(X.iloc[treino], y.iloc[treino])
            p_treino = modelo.predict_proba(X.iloc[treino])[:, 1]
            p_validacao = modelo.predict_proba(X.iloc[validacao])[:, 1]
            p_reunidas[validacao] = p_validacao
            linha = {"chave": chave, "procedimento": nome, "parte": parte,
                     "n_treino": len(treino), "n_validacao": len(validacao)}
            linha.update({f"treino_{k}": v for k, v in medidas(y.iloc[treino], p_treino).items()})
            linha.update({f"validacao_{k}": v for k, v in medidas(y.iloc[validacao], p_validacao).items()})
            linhas.append(linha)
            detalhe = {"chave": chave, "parte": parte, "avisos": avisos(capturados)}
            if chave.startswith("rede_"):
                rede = modelo[-1]
                detalhe.update({
                    "epocas_fit": int(rede.n_iter_),
                    "loss_interna_final": float(rede.loss_),
                    "loss_interna_minima": float(rede.best_loss_),
                    "log_loss_treino_fim_ultima_epoca": float(log_loss(y.iloc[treino], p_treino)),
                    "parametros": contar_parametros(rede),
                    "out_activation": rede.out_activation_,
                    "t_amostras_processadas": int(rede.t_),
                    "medias_padronizacao": modelo[0].mean_.tolist(),
                    "escalas_padronizacao": modelo[0].scale_.tolist(),
                    "loss_curve": [float(x) for x in rede.loss_curve_],
                })
            ajustes.append(detalhe)
        previsoes[chave] = p_reunidas
        agregadas.append({"chave": chave, "procedimento": nome, **medidas(y, p_reunidas)})
        tabela = pd.DataFrame([x for x in linhas if x["chave"] == chave])
        resumo = {"chave": chave, "procedimento": nome}
        for origem in ["treino", "validacao"]:
            for metrica in ["AUC", "acerto", "log_loss"]:
                valores = tabela[f"{origem}_{metrica}"]
                resumo[f"{origem}_{metrica}_media"] = float(valores.mean())
                resumo[f"{origem}_{metrica}_dp"] = float(valores.std(ddof=1))
        # Todas as partes têm91 amostras. A perda média é aditiva; AUC não.
        assert np.isclose(resumo["validacao_log_loss_media"], agregadas[-1]["log_loss"])
        resumos.append(resumo)
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(RESULTADOS / "metricas_classificacao_por_parte.csv", index=False)
    pd.DataFrame(resumos).to_csv(RESULTADOS / "resumo_classificacao_medias_partes.csv", index=False)
    pd.DataFrame(agregadas).to_csv(RESULTADOS / "metricas_classificacao_previsoes_reunidas.csv", index=False)
    previsoes.to_csv(RESULTADOS / "previsoes_classificacao_validacao.csv", index=False)
    guardar_json("ajustes_validacao.json", ajustes)
    return resumos, agregadas, ajustes


def curvas(dados):
    X, y = dados[COLUNAS], dados["maligno"]
    treino, validacao = train_test_split(
        np.arange(len(dados)), test_size=0.30, stratify=y, random_state=SEMENTE
    )
    assert len(treino) == 318 and len(validacao) == 137
    assert not np.intersect1d(treino, validacao).size
    escala = StandardScaler()
    Xt = escala.fit_transform(X.iloc[treino])
    Xv = escala.transform(X.iloc[validacao])
    yt, yv = y.iloc[treino], y.iloc[validacao]
    guardar_json("particao_curvas.json", {
        "n_treino": len(treino), "n_validacao": len(validacao),
        "classes_treino": {str(k): int(v) for k, v in yt.value_counts().sort_index().items()},
        "classes_validacao": {str(k): int(v) for k, v in yv.value_counts().sort_index().items()},
        "ids_treino": dados.iloc[treino]["id_registo"].astype(int).tolist(),
        "ids_validacao": dados.iloc[validacao]["id_registo"].astype(int).tolist(),
        "medias_padronizacao": escala.mean_.tolist(),
        "escalas_padronizacao": escala.scale_.tolist(),
        "ajustada_apenas_no_treino": True,
    })
    todos, resumos = [], []
    for unidades in [8, 32]:
        rede = criar_rede(unidades)
        linhas, historico_avisos = [], []
        otimizador_inicial = None
        for epoca in range(1, 201):
            with warnings.catch_warnings(record=True) as capturados:
                warnings.simplefilter("always")
                rede.partial_fit(Xt, yt, classes=[0, 1])
            if otimizador_inicial is None:
                otimizador_inicial = id(rede._optimizer)
            assert id(rede._optimizer) == otimizador_inicial
            pt, pv = rede.predict_proba(Xt)[:, 1], rede.predict_proba(Xv)[:, 1]
            mt, mv = medidas(yt, pt), medidas(yv, pv)
            linha = {
                "unidades": unidades, "epoca": epoca,
                "perda_treino": mt["log_loss"], "perda_validacao": mv["log_loss"],
                "AUC_treino": mt["AUC"], "AUC_validacao": mv["AUC"],
                "acerto_treino": mt["acerto"], "acerto_validacao": mv["acerto"],
                "loss_interna": float(rede.loss_), "n_iter_ultima_chamada": int(rede.n_iter_),
                "t_amostras_processadas": int(rede.t_),
                "atualizacoes_adam": int(rede._optimizer.t),
            }
            assert rede.t_ == epoca * len(treino)
            assert rede._optimizer.t == epoca * int(np.ceil(len(treino) / 32))
            linhas.append(linha)
            if capturados:
                historico_avisos.append({"epoca": epoca, "avisos": avisos(capturados)})
        tabela = pd.DataFrame(linhas)
        tabela.to_csv(RESULTADOS / f"curvas_rede_{unidades}.csv", index=False)
        tabela[tabela["epoca"].isin(EPOCAS_TABELA)].to_csv(
            RESULTADOS / f"curvas_rede_{unidades}_epocas_selecionadas.csv", index=False
        )
        melhor = tabela.loc[tabela["perda_validacao"].idxmin()]
        selecionada = tabela[tabela["epoca"].isin(EPOCAS_TABELA)]
        melhor_tabela = selecionada.loc[selecionada["perda_validacao"].idxmin()]
        resumos.append({
            "unidades": unidades, "parametros": contar_parametros(rede),
            "menor_perda_validacao_200_epocas": float(melhor["perda_validacao"]),
            "epoca_menor_perda_validacao": int(melhor["epoca"]),
            "epoca_menor_perda_entre_selecionadas": int(melhor_tabela["epoca"]),
            "menor_perda_treino_200_epocas": float(tabela["perda_treino"].min()),
            "epoca_200": linhas[-1], "avisos": historico_avisos,
            "estado_adam_conservado": True,
            "nota": "Épocas contadas pelo ciclo. As chamadas partial_fit não são a execução fit da validação cruzada.",
        })
        todos.extend(linhas)
    pd.DataFrame(todos).to_csv(RESULTADOS / "curvas_redes.csv", index=False)
    guardar_json("resumo_curvas.json", resumos)
    return resumos


def ajustamentos_completos(dados):
    X, y = dados[COLUNAS], dados["maligno"]
    resumos = []
    intervalos = [(float(X[c].min()), float(X[c].max())) for c in COLUNAS]
    eixos = [np.linspace(inicio - .04 * (fim - inicio), fim + .04 * (fim - inicio), 201)
             for inicio, fim in intervalos]
    grelha_raio, grelha_textura = np.meshgrid(*eixos)
    grelha = pd.DataFrame({COLUNAS[0]: grelha_raio.ravel(),
                          COLUNAS[1]: grelha_textura.ravel()})
    superficies = {"raio": eixos[0], "textura": eixos[1]}
    for unidades in [8, 32]:
        modelo = make_pipeline(StandardScaler(), criar_rede(unidades))
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always")
            modelo.fit(X, y)
        rede, escala = modelo[-1], modelo[0]
        z = escala.transform(X)
        ocultas = np.maximum(0, z @ rede.coefs_[0] + rede.intercepts_[0])
        score = (ocultas @ rede.coefs_[1] + rede.intercepts_[1]).ravel()
        p_manual = expit(score)
        p_estimador = modelo.predict_proba(X)[:, 1]
        np.testing.assert_allclose(p_manual, p_estimador, rtol=0, atol=1e-14)
        superficies[f"p_rede_{unidades}"] = modelo.predict_proba(grelha)[:, 1].reshape(grelha_raio.shape)
        detalhe = {
            "unidades": unidades, "n_treino": len(X),
            "uso": "Ajustamento ao desenvolvimento completo para ilustração; não é avaliação de validação.",
            "epocas_fit": int(rede.n_iter_), "parametros": contar_parametros(rede),
            "pesos": [a.tolist() for a in rede.coefs_],
            "termos_constantes": [a.tolist() for a in rede.intercepts_],
            "medias_padronizacao": escala.mean_.tolist(), "escalas_padronizacao": escala.scale_.tolist(),
            "out_activation": rede.out_activation_, "avisos": avisos(capturados),
            "medidas_ajustamento": medidas(y, p_estimador),
            "erro_maximo_reconstituicao_manual": float(np.max(np.abs(p_manual - p_estimador))),
        }
        guardar_json(f"rede_{unidades}_ajustamento_completo.json", detalhe)
        tabela = dados[["id_registo", *COLUNAS, "maligno"]].copy()
        tabela["probabilidade_ajustamento_completo"] = p_estimador
        tabela.to_csv(RESULTADOS / f"rede_{unidades}_probabilidades_ajustamento.csv", index=False)
        resumos.append({k: v for k, v in detalhe.items() if k not in ["pesos", "termos_constantes"]})
    np.savez_compressed(RESULTADOS / "superficies_probabilidade.npz", **superficies)
    return resumos


def main():
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    dados = validar_dados()
    with threadpool_limits(limits=1):
        medias, reunidas, ajustes = avaliar(dados)
        resumo_curvas = curvas(dados)
        completos = ajustamentos_completos(dados)
    guardar_json("resultados_verificados.json", {
        "ambiente": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "hash_dados": HASH_DADOS, "n_desenvolvimento": len(dados),
        "n_benignas": 285, "n_malignas": 170, "teste_reservado_utilizado": False,
        "medias_partes": medias, "previsoes_reunidas": reunidas,
        "ajustes_validacao": [{k: v for k, v in a.items() if k != "loss_curve"} for a in ajustes],
        "curvas": resumo_curvas, "ajustamentos_completos": completos,
    })
    print(pd.DataFrame(medias)[["procedimento", "validacao_AUC_media", "validacao_AUC_dp",
                               "validacao_acerto_media", "validacao_log_loss_media"]].to_string(index=False))
    print("\nÉpocas fit e avisos das redes em validação:")
    for ajuste in ajustes:
        if ajuste["chave"].startswith("rede_"):
            print(ajuste["chave"], ajuste["parte"], ajuste["epocas_fit"], ajuste["avisos"])
    print("\nCurvas:")
    for resumo in resumo_curvas:
        print(resumo)
    print("\nResultados guardados. O teste reservado não foi utilizado.")


if __name__ == "__main__":
    main()
