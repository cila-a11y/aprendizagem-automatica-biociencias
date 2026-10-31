"""Verificações independentes de dados, divisões e contas dos casos finais.

Não seleciona outro procedimento e não modifica previsões. Depois da
preparação, seleção e avaliação final, executar este programa para conferir
as saídas e escrever resultados/verificacao_resultados.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from avaliar_teste_final import ler_decisao
from modelos import (
    RAIZ, DADOS, RESULTADOS, HASHES_DESENVOLVIMENTO, SEMENTE,
    ENTRADAS_CLASSIFICACAO, ENTRADAS_REGRESSAO,
    classificadores, regressores, guardar_json, sha256_ficheiro, versoes,
)


CONTROLOS = []
PARTICOES_ANTERIORES = {
    "classificacao": "d2cc6e314eb1e825dcd05909c17840e0a1dba3cad746c718c542d99baa781009",
    "regressao": "cbc3bcdcd495f8b659d4fb7e55a082733267170375b3b258fcc490004f67f839",
}
HASHES_TESTE = {
    "wisconsin_teste_reservado.csv": "31bda54e64c273d21ec42ec4a080f6ac7fab36cf2709891ea7a7df3239d09ade",
    "penguins_teste_reservado.csv": "81e9a7144788f1a2ab734c5b6fe445c1a93dd4174af4f9a5441b8862a83d65b4",
}


def verificar(condicao, descricao):
    if not bool(condicao):
        raise AssertionError(descricao)
    CONTROLOS.append(descricao)


def proximos(a, b, descricao):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-9, err_msg=descricao)
    CONTROLOS.append(descricao)


def iguais(a, b, descricao):
    np.testing.assert_array_equal(a, b, err_msg=descricao)
    CONTROLOS.append(descricao)


def ler_json(nome):
    return json.loads((RESULTADOS / nome).read_text(encoding="utf-8"))


def ler_csv(nome):
    return pd.read_csv(RESULTADOS / nome)


def assinatura(objeto):
    return hashlib.sha256(json.dumps(objeto, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def auc_por_pares(y, p):
    positivos, negativos = np.asarray(p)[np.asarray(y) == 1], np.asarray(p)[np.asarray(y) == 0]
    return float(np.mean((positivos[:, None] > negativos) + .5 * (positivos[:, None] == negativos)))


def conferir_particoes(dados, divisoes, tarefa):
    gravadas = ler_json(f"{tarefa}_particoes.json")
    canonicas = []
    for k, (treino, val) in enumerate(divisoes, 1):
        g = gravadas[k - 1]
        iguais(dados.iloc[treino].id_registo, g["ids_treino"], f"{tarefa}: treino da parte {k}")
        iguais(dados.iloc[val].id_registo, g["ids_validacao"], f"{tarefa}: validação da parte {k}")
        verificar(set(g["ids_treino"]).isdisjoint(g["ids_validacao"]), f"{tarefa}: separação na parte {k}")
        canonicas.append({"parte": k, "ids_treino": g["ids_treino"], "ids_validacao": g["ids_validacao"]})
    verificar(assinatura(canonicas) == PARTICOES_ANTERIORES[tarefa],
              f"{tarefa}: partições idênticas às dos capítulos anteriores")


def main():
    decisao = ler_decisao()
    verificar(decisao["escolhas"] == {"classificacao": "logistica", "limiar": .2, "regressao": "floresta"},
              "Escolhas seguem a regra e os resultados de desenvolvimento")
    verificar(decisao["teste_lido_nesta_etapa"] is False,
              "O registo da seleção declara corretamente a etapa sem leitura de teste")
    for nome, esperado in {**HASHES_DESENVOLVIMENTO, **HASHES_TESTE}.items():
        verificar(sha256_ficheiro(DADOS / nome) == esperado, f"Hash da cópia fixa {nome}")
    w = pd.read_csv(DADOS / "wisconsin_desenvolvimento.csv")
    p_total = pd.read_csv(DADOS / "penguins_desenvolvimento.csv")
    p = p_total.dropna(subset=["body_mass_g"]).reset_index(drop=True)
    wt = pd.read_csv(DADOS / "wisconsin_teste_reservado.csv")
    pt = pd.read_csv(DADOS / "penguins_teste_reservado.csv")
    verificar(len(w) == 455 and len(wt) == 114, "Dimensões Wisconsin")
    verificar(len(p_total) == 275 and len(p) == 273 and len(pt) == 69, "Dimensões dos pinguins")
    iguais(p_total.loc[p_total.body_mass_g.isna(), "id_registo"], [4, 272], "Respostas ausentes não imputadas")
    verificar(set(w.id_registo).isdisjoint(wt.id_registo), "Wisconsin: desenvolvimento e teste separados")
    verificar(set(p_total.id_registo).isdisjoint(pt.id_registo), "Pinguins: desenvolvimento e teste separados")
    conferir_particoes(w, list(StratifiedKFold(5, shuffle=True, random_state=SEMENTE).split(w, w.maligno)), "classificacao")
    conferir_particoes(p, list(KFold(5, shuffle=True, random_state=SEMENTE).split(p)), "regressao")

    prev_val = ler_csv("classificacao_validacao_previsoes.csv")
    iguais(prev_val.id_registo, w.id_registo, "Previsões de validação na ordem original")
    limiares = ler_csv("limiares_validacao.csv")
    for _, row in limiares.iterrows():
        d, y = prev_val.previsao_logistica >= row.limiar, w.maligno.to_numpy()
        contagens = [((~d) & (y == 0)).sum(), (d & (y == 0)).sum(), ((~d) & (y == 1)).sum(), (d & (y == 1)).sum()]
        proximos(contagens, row[["VN", "FP", "FN", "VP"]].to_numpy(dtype=float), f"Matriz de validação no limiar {row.limiar}")
        proximos(row.sensibilidade, contagens[3] / 170, f"Sensibilidade de validação no limiar {row.limiar}")
        proximos(row.especificidade, contagens[0] / 285, f"Especificidade de validação no limiar {row.limiar}")
    iguais(limiares.loc[limiares.sensibilidade >= .9, "limiar"], [.2], "Único limiar elegível")
    for tarefa, metricas in [("classificacao", ["AUC", "acerto", "Brier"]), ("regressao", ["MAE_g", "RMSE_g", "R2"])]:
        partes, resumo = ler_csv(f"{tarefa}_validacao_por_parte.csv"), ler_csv(f"{tarefa}_validacao_resumo.csv")
        for _, row in resumo.iterrows():
            bloco = partes[partes.procedimento == row.procedimento]
            for m in metricas:
                proximos(row[f"{m}_media"], bloco[m].mean(), f"{tarefa}, {row.procedimento}: média {m}")
                proximos(row[f"{m}_desvio_ddof1"], bloco[m].std(ddof=1), f"{tarefa}, {row.procedimento}: desvio {m}")

    previsoes_w = ler_csv("classificacao_teste_previsoes.csv")
    iguais(previsoes_w.id_registo, wt.id_registo, "Previsões de classificação de teste na ordem original")
    modelo_w = classificadores()["logistica"].fit(w[ENTRADAS_CLASSIFICACAO], w.maligno)
    proximos(previsoes_w.p_logistica, modelo_w.predict_proba(wt[ENTRADAS_CLASSIFICACAO])[:, 1],
             "Probabilidades finais reproduzidas pelo ajustamento apenas no desenvolvimento")
    iguais(previsoes_w.decisao_logistica, (previsoes_w.p_logistica >= .2).astype(int), "Decisão final usa p >= 0,20")
    iguais(previsoes_w.decisao_referencia, np.zeros(114), "Referência decide sempre benigno")
    m_w = ler_csv("classificacao_teste_medidas.csv").set_index("procedimento")
    for nome in ["referencia", "logistica"]:
        prob, y = previsoes_w[f"p_{nome}"].to_numpy(), wt.maligno.to_numpy()
        row = m_w.loc[nome]
        proximos(row.AUC, auc_por_pares(y, prob), f"AUC final de {nome} por comparação de pares")
        proximos(row.Brier, np.mean((prob - y) ** 2), f"Brier final de {nome} pela fórmula")
    proximos(m_w.loc["logistica", ["VN", "FP", "FN", "VP"]], [54, 18, 2, 40], "Matriz final confirmada")
    proximos(m_w.loc["logistica", "sensibilidade"], 40 / 42, "Sensibilidade final 40/42")
    proximos(m_w.loc["logistica", "especificidade"], 54 / 72, "Especificidade final 54/72")
    proximos(m_w.loc["logistica", "VPP"], 40 / 58, "VPP final 40/58")
    verificar(pd.isna(m_w.loc["referencia", "VPP"]), "VPP da referência não definido")
    roc = ler_csv("classificacao_teste_roc.csv")
    proximos(np.trapezoid(roc.sensibilidade, roc.taxa_falsos_positivos), m_w.loc["logistica", "AUC"], "Área da ROC reproduz a AUC")
    verificar(np.isinf(roc.limiar.iloc[0]), "Primeiro ponto da ROC não prevê positivos")

    previsoes_p = ler_csv("regressao_teste_previsoes.csv")
    iguais(previsoes_p.id_registo, pt.id_registo, "Previsões da regressão de teste na ordem original")
    modelo_p = regressores()["floresta"].fit(p[ENTRADAS_REGRESSAO], p.body_mass_g)
    proximos(previsoes_p.previsto_floresta_g, modelo_p.predict(pt[ENTRADAS_REGRESSAO]),
             "Previsões finais da floresta reproduzidas sem seleção no teste")
    proximos(previsoes_p.previsto_referencia_g, np.repeat(p.body_mass_g.median(), 69), "Referência final é a mediana 4050 g do desenvolvimento")
    m_p = ler_csv("regressao_teste_medidas.csv").set_index("procedimento")
    for nome in ["referencia", "floresta"]:
        erro = pt.body_mass_g.to_numpy() - previsoes_p[f"previsto_{nome}_g"].to_numpy()
        row = m_p.loc[nome]
        proximos(previsoes_p[f"erro_{nome}_g"], erro, f"Erros assinados de {nome}")
        proximos(previsoes_p[f"erro_absoluto_{nome}_g"], np.abs(erro), f"Erros absolutos de {nome}")
        proximos(row.MAE_g, np.mean(np.abs(erro)), f"MAE de {nome} pela fórmula")
        proximos(row.RMSE_g, np.sqrt(np.mean(erro ** 2)), f"RMSE de {nome} pela fórmula")
        proximos(row.R2, 1 - np.sum(erro ** 2) / np.sum((pt.body_mass_g - pt.body_mass_g.mean()) ** 2), f"R² de {nome} pela fórmula")
    resumo = ler_json("teste_resumo.json")
    delta = previsoes_p.erro_absoluto_referencia_g - previsoes_p.erro_absoluto_floresta_g
    proximos(previsoes_p.reducao_erro_absoluto_g, delta, "Diferenças de erros absolutos são pareadas")
    proximos(delta.mean(), m_p.loc["referencia", "MAE_g"] - m_p.loc["floresta", "MAE_g"], "Média das diferenças igual à diferença dos MAE")
    iguais([(delta > 1e-9).sum(), (delta < -1e-9).sum()], [58, 11], "Modelo melhora em 58 linhas e piora em 11")
    maiores = previsoes_p.loc[previsoes_p.erro_absoluto_floresta_g.idxmax()]
    verificar(int(maiores.id_registo) == 262, "Maior erro corresponde ao id 262")
    proximos(maiores.erro_absoluto_floresta_g, 1027.2110121227652, "Maior erro absoluto confirmado")
    contagens = ler_csv("regressao_teste_contagens_por_erro.csv")
    for _, row in contagens.iterrows():
        erros = previsoes_p[f"erro_absoluto_{row.procedimento}_g"]
        verificar(int((erros <= row.limite_g).sum()) == row.n_dentro, f"Contagem {row.procedimento} até {row.limite_g} g")
        proximos(row.fracao_dentro, row.n_dentro / 69, f"Fração {row.procedimento} até {row.limite_g} g")
    verificar(resumo["nova_escolha_no_teste"] is False, "Avaliação final sem nova escolha")
    for nome in ["avisos_desenvolvimento.json", "avisos_teste.json"]:
        verificar(all(not r["avisos"] for r in ler_json(nome)), f"Execução sem avisos: {nome}")
    guardar_json("verificacao_resultados.json", {
        "estado": "aprovado", "n_controlos": len(CONTROLOS), "controlos": CONTROLOS,
        "sha256_dados": {**HASHES_DESENVOLVIMENTO, **HASHES_TESTE},
        "sha256_particoes_anteriores": PARTICOES_ANTERIORES,
        "sha256_decisao": sha256_ficheiro(RESULTADOS / "decisao_fechada.json"),
        "versoes": versoes(),
        "observacao": "As verificações de coerência e reprodução não demonstram pré-registo ou independência histórica das escolhas.",
    })
    print(f"Verificação aprovada: {len(CONTROLOS)} controlos; dados, divisões e contas confirmados.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
