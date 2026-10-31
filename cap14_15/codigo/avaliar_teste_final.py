"""Avaliar as escolhas registadas, sem procurar outros modelos ou limiares.

Executar explicitamente:
    python codigo/avaliar_teste_final.py --confirmar-teste

Os resultados já foram publicados no livro. Este programa reproduz o
protocolo didático; a ordem de execução e os hashes não provam que uma
escolha real foi pré-registada antes de alguém conhecer os resultados.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

from modelos import (
    RAIZ, DADOS, RESULTADOS, ENTRADAS_CLASSIFICACAO, ENTRADAS_REGRESSAO,
    ajustar, classificadores, regressores, configuracao, guardar_csv, guardar_json,
    ler_desenvolvimento, medidas_classificacao, medidas_regressao,
    sha256_ficheiro, versoes,
)


def ler_decisao():
    caminho = RESULTADOS / "decisao_fechada.json"
    decisao = json.loads(caminho.read_text(encoding="utf-8"))
    if decisao["protocolo"] != configuracao() or decisao["teste_lido_nesta_etapa"]:
        raise ValueError("O registo da decisão não corresponde ao protocolo da seleção.")
    for nome, esperado in decisao["sha256_codigo"].items():
        if sha256_ficheiro(RAIZ / "codigo" / nome) != esperado:
            raise ValueError(f"O código mudou depois do registo da decisão: {nome}")
    for nome, esperado in decisao["sha256_resultados_selecao"].items():
        if sha256_ficheiro(RESULTADOS / nome) != esperado:
            raise ValueError(f"O resultado de seleção mudou depois do registo: {nome}")
    resumo_w = pd.read_csv(RESULTADOS / "classificacao_validacao_resumo.csv").set_index("procedimento")
    resumo_p = pd.read_csv(RESULTADOS / "regressao_validacao_resumo.csv").set_index("procedimento")
    limiares = pd.read_csv(RESULTADOS / "limiares_validacao.csv")
    esperadas = {
        "classificacao": max(["logistica", "floresta"], key=lambda n: resumo_w.loc[n, "AUC_media"]),
        "regressao": min(["linear", "floresta"], key=lambda n: resumo_p.loc[n, "MAE_g_media"]),
        "limiar": float(limiares[limiares.sensibilidade >= .90]
                        .sort_values(["especificidade", "limiar"], ascending=False).iloc[0].limiar),
    }
    if decisao["escolhas"] != esperadas:
        raise ValueError("As escolhas registadas não seguem os resultados e a regra declarada.")
    return decisao


def ler_teste(nome):
    proveniencia = json.loads((RESULTADOS / "proveniencia_dados.json").read_text(encoding="utf-8"))
    entrada = next(item for item in proveniencia["ficheiros"] if item["ficheiro"] == nome)
    caminho = DADOS / nome
    if sha256_ficheiro(caminho) != entrada["sha256"]:
        raise ValueError(f"A cópia do teste foi alterada: {nome}")
    tabela = pd.read_csv(caminho)
    assert tabela.id_registo.is_unique and tabela.id_registo.is_monotonic_increasing
    assert len(tabela) == entrada["n"] and tabela.id_registo.tolist() == entrada["ids"]
    return tabela


def main():
    decisao = ler_decisao()
    avisos = []
    w = ler_desenvolvimento("wisconsin_desenvolvimento.csv")
    p = ler_desenvolvimento("penguins_desenvolvimento.csv").dropna(subset=["body_mass_g"])
    escolha_w, escolha_p = decisao["escolhas"]["classificacao"], decisao["escolhas"]["regressao"]
    limiar = decisao["escolhas"]["limiar"]
    modelos_w, modelos_p = classificadores(), regressores()
    ajustados_w = {nome: ajustar(modelos_w[nome], w[ENTRADAS_CLASSIFICACAO], w.maligno,
                                f"classificacao_final_{nome}", avisos)
                   for nome in ["referencia", escolha_w]}
    ajustados_p = {nome: ajustar(modelos_p[nome], p[ENTRADAS_REGRESSAO], p.body_mass_g,
                                f"regressao_final_{nome}", avisos)
                   for nome in ["referencia", escolha_p]}
    # Só agora se abrem as cópias de teste. Não há nova seleção nesta etapa.
    wt = ler_teste("wisconsin_teste_reservado.csv")
    pt_total = ler_teste("penguins_teste_reservado.csv")
    pt = pt_total.dropna(subset=["body_mass_g"]).copy()
    assert set(w.id_registo).isdisjoint(wt.id_registo)
    assert set(p.id_registo).isdisjoint(pt.id_registo)
    assert len(wt) == 114 and wt.maligno.value_counts().to_dict() == {0: 72, 1: 42}
    assert len(pt_total) == 69 and len(pt) == 69
    assert wt[ENTRADAS_CLASSIFICACAO].notna().all().all()
    assert pt[ENTRADAS_REGRESSAO].notna().all().all()

    previsoes_w = wt[["id_registo", *ENTRADAS_CLASSIFICACAO, "maligno"]].copy()
    medidas_w = []
    for nome, modelo in ajustados_w.items():
        prob = modelo.predict_proba(wt[ENTRADAS_CLASSIFICACAO])[:, 1]
        d = np.zeros(len(wt), dtype=int) if nome == "referencia" else (prob >= limiar).astype(int)
        previsoes_w[f"p_{nome}"] = prob
        previsoes_w[f"decisao_{nome}"] = d
        medidas_w.append({"procedimento": nome, "limiar_modelo": None if nome == "referencia" else limiar,
                          **medidas_classificacao(wt.maligno, prob, decisao=d)})
    guardar_csv("classificacao_teste_previsoes.csv", previsoes_w)
    guardar_csv("classificacao_teste_medidas.csv", pd.DataFrame(medidas_w))
    fpr, tpr, limites = roc_curve(wt.maligno, previsoes_w[f"p_{escolha_w}"], drop_intermediate=False)
    guardar_csv("classificacao_teste_roc.csv", pd.DataFrame({"limiar": limites, "taxa_falsos_positivos": fpr,
                                                          "sensibilidade": tpr}))

    previsoes_p = pt[["id_registo", "species", "island", *ENTRADAS_REGRESSAO, "body_mass_g"]].copy()
    medidas_p = []
    for nome, modelo in ajustados_p.items():
        prev = modelo.predict(pt[ENTRADAS_REGRESSAO])
        previsoes_p[f"previsto_{nome}_g"] = prev
        previsoes_p[f"erro_{nome}_g"] = pt.body_mass_g.to_numpy() - prev
        previsoes_p[f"erro_absoluto_{nome}_g"] = np.abs(pt.body_mass_g.to_numpy() - prev)
        medidas_p.append({"procedimento": nome, **medidas_regressao(pt.body_mass_g, prev)})
    melhoria = previsoes_p["erro_absoluto_referencia_g"] - previsoes_p[f"erro_absoluto_{escolha_p}_g"]
    previsoes_p["reducao_erro_absoluto_g"] = melhoria
    guardar_csv("regressao_teste_previsoes.csv", previsoes_p)
    guardar_csv("regressao_teste_medidas.csv", pd.DataFrame(medidas_p))
    limites_g = [250, 500, 1000]
    tolerancias = pd.DataFrame([
        {"procedimento": nome, "limite_g": limite, "n_dentro": int((previsoes_p[f"erro_absoluto_{nome}_g"] <= limite).sum()),
         "n_total": len(pt), "fracao_dentro": float((previsoes_p[f"erro_absoluto_{nome}_g"] <= limite).mean())}
        for nome in ajustados_p for limite in limites_g
    ])
    guardar_csv("regressao_teste_contagens_por_erro.csv", tolerancias)
    maior = previsoes_p.loc[previsoes_p[f"erro_absoluto_{escolha_p}_g"].idxmax()].to_dict()
    referencia, escolhido = medidas_p
    resumo = {
        "estatuto": decisao["estatuto"],
        "sha256_decisao": sha256_ficheiro(RESULTADOS / "decisao_fechada.json"),
        "escolhas": decisao["escolhas"], "classificacao": medidas_w, "regressao": medidas_p,
        "referencia_final_massa_g": float(p.body_mass_g.median()),
        "referencia_final_probabilidade": float(w.maligno.mean()),
        "comparacao_pareada_regressao": {
            "definicao": "erro absoluto da referência menos erro absoluto do modelo, na mesma linha",
            "melhoria_media_g": float(melhoria.mean()),
            "reducao_relativa_MAE": float(1 - escolhido["MAE_g"] / referencia["MAE_g"]),
            "n_modelo_melhor": int((melhoria > 1e-9).sum()),
            "n_modelo_pior": int((melhoria < -1e-9).sum()),
            "n_empates_tolerancia_1e_9_g": int((np.abs(melhoria) <= 1e-9).sum()),
            "maior_melhoria_g": float(melhoria.max()), "maior_agravamento_g": float(-melhoria.min()),
        },
        "maior_erro_modelo": maior,
        "contagens_por_erro": tolerancias.to_dict(orient="records"),
        "intervalos_de_incerteza_estimados": False,
        "nova_escolha_no_teste": False,
        "versoes": versoes(),
    }
    guardar_json("teste_resumo.json", resumo)
    guardar_json("avisos_teste.json", avisos)
    guardar_json("ajustamentos_finais.json", {
        "ids_classificacao": w.id_registo.tolist(), "ids_regressao": p.id_registo.tolist(),
        "medianas_imputacao_classificacao": ajustados_w[escolha_w][0].statistics_,
        "medianas_imputacao_regressao": ajustados_p[escolha_p][0].statistics_,
        "classes": ajustados_w[escolha_w].classes_,
        "coeficientes_logistica": ajustados_w[escolha_w][-1].coef_ if escolha_w == "logistica" else None,
        "termo_constante_logistica": ajustados_w[escolha_w][-1].intercept_ if escolha_w == "logistica" else None,
        "medias_padronizacao": ajustados_w[escolha_w][1].mean_ if escolha_w == "logistica" else None,
        "escalas_padronizacao": ajustados_w[escolha_w][1].scale_ if escolha_w == "logistica" else None,
    })
    print(pd.DataFrame(medidas_w).to_string(index=False))
    print(pd.DataFrame(medidas_p).to_string(index=False))
    print(json.dumps(resumo["comparacao_pareada_regressao"], ensure_ascii=False, indent=2))
    print("Maior erro do modelo:", maior)
    print(tolerancias.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmar-teste", action="store_true",
                        help="abrir os dois testes e reproduzir a avaliação final já publicada")
    args = parser.parse_args()
    if not args.confirmar_teste:
        parser.error("esta etapa abre o teste; acrescente --confirmar-teste após registar a decisão")
    with threadpool_limits(limits=1):
        main()
