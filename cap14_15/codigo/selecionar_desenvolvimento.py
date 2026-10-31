"""Comparar candidatos e registar as escolhas usando só o desenvolvimento.

Não abre os ficheiros de teste, nem a fonte integral dos pinguins. Os resumos
por parte são usados para seleção; não são apresentados como uma avaliação
independente da escolha. Execute avaliar_teste_final.py apenas depois.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from modelos import (
    RAIZ, RESULTADOS, HASHES_DESENVOLVIMENTO, SEMENTE, LIMIARES,
    ENTRADAS_CLASSIFICACAO, ENTRADAS_REGRESSAO, ajustar,
    classificadores, regressores, configuracao, guardar_csv, guardar_json,
    ler_desenvolvimento, medidas_classificacao, medidas_regressao,
    sha256_ficheiro, versoes,
)


def comparar(dados, entradas, resposta, modelos, particoes, tarefa, avisos):
    X, y = dados[entradas], dados[resposta]
    linhas, divisao = [], []
    saida = dados[["id_registo", *entradas, resposta]].copy()
    partes = np.zeros(len(dados), dtype=int)
    for k, (treino, val) in enumerate(particoes, 1):
        partes[val] = k
        divisao.append({"parte": k, "n_treino": len(treino), "n_validacao": len(val),
                        "ids_treino": dados.iloc[treino].id_registo.tolist(),
                        "ids_validacao": dados.iloc[val].id_registo.tolist()})
    assert np.all(partes > 0)
    saida["parte"] = partes
    for nome, prototipo in modelos.items():
        previsoes = np.full(len(dados), np.nan)
        for k, (treino, val) in enumerate(particoes, 1):
            modelo = ajustar(clone(prototipo), X.iloc[treino], y.iloc[treino],
                             f"{tarefa}_{nome}_parte_{k}", avisos)
            if tarefa == "classificacao":
                assert modelo.classes_.tolist() == [0, 1]
                prev = modelo.predict_proba(X.iloc[val])[:, 1]
                m = medidas_classificacao(y.iloc[val], prev)
            else:
                prev = modelo.predict(X.iloc[val])
                m = medidas_regressao(y.iloc[val], prev)
            previsoes[val] = prev
            linhas.append({"procedimento": nome, "parte": k,
                           "n_treino": len(treino), **m})
        assert np.isfinite(previsoes).all()
        saida[f"previsao_{nome}"] = previsoes
    tabela = pd.DataFrame(linhas)
    metricas = ["AUC", "acerto", "Brier"] if tarefa == "classificacao" else ["MAE_g", "RMSE_g", "R2"]
    resumo = []
    for nome in modelos:
        bloco = tabela[tabela.procedimento == nome]
        row = {"procedimento": nome}
        for metrica in metricas:
            row[f"{metrica}_media"] = bloco[metrica].mean()
            row[f"{metrica}_desvio_ddof1"] = bloco[metrica].std(ddof=1)
        resumo.append(row)
    resumo = pd.DataFrame(resumo)
    guardar_csv(f"{tarefa}_validacao_por_parte.csv", tabela)
    guardar_csv(f"{tarefa}_validacao_resumo.csv", resumo)
    guardar_csv(f"{tarefa}_validacao_previsoes.csv", saida)
    guardar_json(f"{tarefa}_particoes.json", divisao)
    return resumo.set_index("procedimento"), saida


def main():
    avisos = []
    w = ler_desenvolvimento("wisconsin_desenvolvimento.csv")
    p_total = ler_desenvolvimento("penguins_desenvolvimento.csv")
    p = p_total.dropna(subset=["body_mass_g"]).reset_index(drop=True)
    assert len(w) == 455 and w.maligno.value_counts().to_dict() == {0: 285, 1: 170}
    assert len(p_total) == 275 and len(p) == 273
    assert p_total.loc[p_total.body_mass_g.isna(), "id_registo"].tolist() == [4, 272]
    assert w[ENTRADAS_CLASSIFICACAO].notna().all().all()
    assert p[ENTRADAS_REGRESSAO].notna().all().all()
    protocolo = configuracao()
    cv_w = list(StratifiedKFold(5, shuffle=True, random_state=SEMENTE).split(w, w.maligno))
    cv_p = list(KFold(5, shuffle=True, random_state=SEMENTE).split(p))
    resumo_w, prev_w = comparar(w, ENTRADAS_CLASSIFICACAO, "maligno", classificadores(), cv_w,
                               "classificacao", avisos)
    resumo_p, _ = comparar(p, ENTRADAS_REGRESSAO, "body_mass_g", regressores(), cv_p,
                           "regressao", avisos)
    candidatos_w = protocolo["classificacao"]["candidatos_em_ordem_de_desempate"]
    candidatos_p = protocolo["regressao"]["candidatos_em_ordem_de_desempate"]
    escolhido_w = max(candidatos_w, key=lambda nome: resumo_w.loc[nome, "AUC_media"])
    escolhido_p = min(candidatos_p, key=lambda nome: resumo_p.loc[nome, "MAE_g_media"])
    limiares = pd.DataFrame([
        {"limiar": limiar, **medidas_classificacao(w.maligno, prev_w[f"previsao_{escolhido_w}"], limiar)}
        for limiar in LIMIARES
    ])
    elegiveis = limiares[limiares.sensibilidade >= .90]
    if elegiveis.empty:
        raise RuntimeError("Nenhum limiar satisfaz o requisito no desenvolvimento; não abrir o teste.")
    melhor = elegiveis.sort_values(["especificidade", "limiar"], ascending=False).iloc[0]
    guardar_csv("limiares_validacao.csv", limiares)
    guardar_json("avisos_desenvolvimento.json", avisos)
    nomes_resultados = [
        f"{tarefa}_{sufixo}"
        for tarefa in ["classificacao", "regressao"]
        for sufixo in ["validacao_por_parte.csv", "validacao_resumo.csv", "validacao_previsoes.csv", "particoes.json"]
    ] + ["limiares_validacao.csv"]
    decisao = {
        "estatuto": "Reprodução didática de um protocolo e de resultados já publicados no livro; não constitui pré-registo nem avaliação prospectiva nova.",
        "leituras_de_dados_nesta_etapa": list(HASHES_DESENVOLVIMENTO),
        "teste_lido_nesta_etapa": False,
        "protocolo": protocolo,
        "escolhas": {"classificacao": escolhido_w, "limiar": float(melhor.limiar),
                     "regressao": escolhido_p},
        "n_desenvolvimento_classificacao": len(w),
        "n_desenvolvimento_regressao_total": len(p_total),
        "n_desenvolvimento_regressao_elegivel": len(p),
        "mediana_desenvolvimento_g": float(p.body_mass_g.median()),
        "frequencia_maligna_desenvolvimento": float(w.maligno.mean()),
        "sha256_desenvolvimento": HASHES_DESENVOLVIMENTO,
        "sha256_codigo": {nome: sha256_ficheiro(RAIZ / "codigo" / nome)
                          for nome in ["modelos.py", "selecionar_desenvolvimento.py"]},
        "sha256_resultados_selecao": {nome: sha256_ficheiro(RESULTADOS / nome) for nome in nomes_resultados},
        "versoes": versoes(),
    }
    guardar_json("decisao_fechada.json", decisao)
    print(resumo_w.to_string())
    print(resumo_p.to_string())
    print(limiares[["limiar", "sensibilidade", "especificidade", "VN", "FP", "FN", "VP"]].to_string(index=False))
    print("Escolhas registadas:", decisao["escolhas"])
    print("O teste não foi lido nesta etapa.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
