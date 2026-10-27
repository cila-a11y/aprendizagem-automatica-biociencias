#!/usr/bin/env python3
"""Reproduzir as operações, tabelas e figuras dos capítulos 1 a 6.

Executar a partir de qualquer pasta:
    python reproducao/fundamentos/reproduzir_fundamentos.py
    python reproducao/fundamentos/reproduzir_fundamentos.py --capitulos 5 6

Usa apenas o ficheiro de desenvolvimento dos pinguins. Os exemplos pequenos
identificados como fictícios são criados no próprio código. Não abre o teste.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="biociencias-matplotlib-")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]
FICHEIRO_DADOS = RAIZ / "dados" / "penguins_desenvolvimento.csv"
MEDIDAS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
NOMES_MEDIDAS = ["Comprimento\ndo bico", "Profundidade\ndo bico", "Barbatana", "Massa"]
CORES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
SEMENTE = 2026


def gravar_json(caminho: Path, objeto: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(objeto, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")


def gravar_tabela(saida: Path, nome: str, tabela: pd.DataFrame,
                  indice: bool = False) -> None:
    caminho = saida / "tabelas" / (nome + ".csv")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    # Uma célula vazia representa uma quantidade em falta ou não definida.
    tabela.to_csv(caminho, index=indice, encoding="utf-8", float_format="%.17g")


def gravar_figura(saida: Path, nome: str, figura: plt.Figure) -> None:
    pasta = saida / "figuras"
    pasta.mkdir(parents=True, exist_ok=True)
    figura.tight_layout()
    figura.savefig(pasta / (nome + ".png"), dpi=180, bbox_inches="tight")
    figura.savefig(pasta / (nome + ".pdf"), bbox_inches="tight",
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figura)


def ler_desenvolvimento() -> pd.DataFrame:
    if not FICHEIRO_DADOS.is_file():
        raise FileNotFoundError(f"Não foi encontrado {FICHEIRO_DADOS}. "
                                "Descomprima o conjunto de materiais completo.")
    dados = pd.read_csv(FICHEIRO_DADOS, float_precision="round_trip")
    esperadas = ["id_registo", "species", "island", *MEDIDAS, "sex", "year"]
    if set(dados.columns) != set(esperadas):
        raise ValueError("As colunas dos pinguins não correspondem ao ficheiro do livro.")
    if dados.empty or dados["id_registo"].isna().any() or dados["id_registo"].duplicated().any():
        raise ValueError("A tabela tem de ser não vazia e ter identificadores únicos.")
    for coluna in MEDIDAS:
        disponiveis = dados[coluna].dropna().to_numpy(dtype=float)
        if not np.isfinite(disponiveis).all() or (disponiveis <= 0).any():
            raise ValueError(f"Verifique os valores de {coluna}: há valores não finitos ou não positivos.")
    return dados


def capitulo_1(dados: pd.DataFrame, saida: Path) -> dict:
    """Leitura, unidades, seleção de colunas e filtro de linhas."""
    gravar_tabela(saida, "cap01_primeiras_linhas", dados.head())
    gravar_tabela(saida, "cap01_bico_massa",
                  dados[["id_registo", "bill_length_mm", "body_mass_g"]])
    mais_pesados = dados.loc[dados["body_mass_g"] >= 4500].copy()
    gravar_tabela(saida, "cap01_massa_pelo_menos_4500g", mais_pesados)
    conversoes = pd.DataFrame({"massa_g": [4200, 3600], "massa_kg": [4.2, 3.6]})
    conversoes["massa_kg"] = conversoes["massa_g"] / 1000
    gravar_tabela(saida, "cap01_conversoes_ficticias", conversoes)
    animais = pd.DataFrame({"animal": ["A", "B", "C"],
                            "comprimento_barbatana_mm": [180, 195, 210],
                            "massa_g": [3400, 4100, 4800]})
    gravar_tabela(saida, "cap01_animais_ficticios", animais)
    return {"linhas": len(dados), "colunas": len(dados.columns),
            "massas_observadas": int(dados["body_mass_g"].notna().sum()),
            "massas_pelo_menos_4500g": len(mais_pesados)}


def capitulo_2(dados: pd.DataFrame, saida: Path) -> dict:
    """Inspeção, resumos descritivos e quatro representações gráficas."""
    inspecao = pd.DataFrame({"variavel": dados.columns,
                             "tipo_pandas": dados.dtypes.astype(str).to_numpy(),
                             "disponiveis": dados.notna().sum().to_numpy(),
                             "em_falta": dados.isna().sum().to_numpy()})
    gravar_tabela(saida, "cap02_inspecao", inspecao)
    resumo = dados[MEDIDAS].describe().T
    resumo["variancia_amostral"] = dados[MEDIDAS].var(ddof=1)
    resumo["amplitude"] = dados[MEDIDAS].max() - dados[MEDIDAS].min()
    resumo.index.name = "variavel"
    gravar_tabela(saida, "cap02_resumos_medidas", resumo, indice=True)
    contagens = dados["species"].value_counts(dropna=False)
    especies = contagens.rename("contagem").to_frame()
    especies["proporcao"] = especies["contagem"] / len(dados)
    especies.index.name = "especie"
    gravar_tabela(saida, "cap02_especies", especies, indice=True)
    repetidas = dados.drop(columns="id_registo").duplicated(keep=False)
    gravar_tabela(saida, "cap02_registos_com_campos_repetidos", dados.loc[repetidas])

    exemplos = pd.DataFrame({"A": [3000, 3500, 4000, 4500, 5000],
                             "B": [3000, 3500, 4000, 4500, 10000]})
    gravar_tabela(saida, "cap02_massas_ficticias", exemplos)
    gravar_tabela(saida, "cap02_resumos_ficticios", exemplos.agg(["mean", "median"]),
                  indice=True)
    omissao = pd.Series([3000.0, 4000.0, np.nan, 5000.0])
    gravar_tabela(saida, "cap02_omissao_ficticia", pd.DataFrame({
        "procedimento": ["usar apenas massas observadas", "substituir a omissao por zero"],
        "media_g": [omissao.mean(), omissao.fillna(0).mean()],
        "numero_de_valores": [omissao.count(), len(omissao)]}))

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(inspecao["variavel"], inspecao["em_falta"], color=CORES[0])
    ax.set(xlabel="Coluna", ylabel="Número de valores em falta",
           title="Omissões no desenvolvimento")
    ax.tick_params(axis="x", rotation=45)
    for rotulo in ax.get_xticklabels():
        rotulo.set_ha("right")
    gravar_figura(saida, "u2_omissos", fig)

    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    contagens.plot.bar(ax=ax, color=CORES[:len(contagens)], rot=0)
    ax.set(xlabel="Espécie", ylabel="Número de observações",
           title="Espécies no desenvolvimento")
    gravar_figura(saida, "u2_barras_especies", fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    dados["body_mass_g"].plot.hist(bins=12, ax=ax, color=CORES[0], edgecolor="white")
    ax.set(xlabel="Massa corporal (g)", ylabel="Número de observações",
           title="Distribuição da massa no desenvolvimento")
    gravar_figura(saida, "u2_hist_massa", fig)

    for coluna, descricao in [("bill_length_mm", "Comprimento do bico"),
                               ("bill_depth_mm", "Profundidade do bico")]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        dados[coluna].plot.hist(bins=12, ax=ax, color=CORES[0], edgecolor="white")
        ax.set(xlabel=f"{descricao} (mm)", ylabel="Número de observações",
               title=f"{descricao} no desenvolvimento")
        gravar_figura(saida, f"exercicio_2_8_{coluna}", fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    para_grafico = dados.dropna(subset=["flipper_length_mm", "body_mass_g"])
    ax.scatter(para_grafico["flipper_length_mm"], para_grafico["body_mass_g"],
               s=27, alpha=0.7, color=CORES[0])
    ax.set(xlabel="Comprimento da barbatana (mm)", ylabel="Massa corporal (g)",
           title="Duas medidas dos pinguins de desenvolvimento")
    gravar_figura(saida, "u2_dispersao_medidas", fig)
    return {"observacoes_dispersao": len(para_grafico),
            "repeticoes_apos_primeira": int(dados.drop(columns="id_registo").duplicated().sum()),
            "media_massa_g": float(dados["body_mass_g"].mean()),
            "mediana_massa_g": float(dados["body_mass_g"].median()),
            "desvio_padrao_amostral_massa_g": float(dados["body_mass_g"].std(ddof=1))}


def capitulo_3(saida: Path) -> dict:
    """Exemplos fictícios de referência, preparação e divisão em partes."""
    treino_massa = pd.DataFrame({"massa_g": [3000, 4000, 5000]})
    validacao_massa = pd.DataFrame({"massa_observada_g": [3500, 4500]})
    validacao_massa["massa_prevista_g"] = treino_massa["massa_g"].mean()
    validacao_massa["erro_absoluto_g"] = (
        validacao_massa["massa_observada_g"] - validacao_massa["massa_prevista_g"]).abs()
    gravar_tabela(saida, "cap03_referencia_treino_ficticio", treino_massa)
    gravar_tabela(saida, "cap03_referencia_validacao_ficticia", validacao_massa)
    comparacao = pd.DataFrame({"observado_g": [3000, 4000, 5000],
                              "referencia_g": [4000, 4000, 4000],
                              "modelo_g": [3200, 3900, 5100]})
    for nome in ["referencia", "modelo"]:
        comparacao[f"erro_absoluto_{nome}_g"] = (
            comparacao["observado_g"] - comparacao[f"{nome}_g"]).abs()
    gravar_tabela(saida, "cap03_comparacao_ficticia", comparacao)

    treino = pd.DataFrame({"comprimento_mm": [180.0, np.nan, 220.0],
                            "ilha": ["Norte", "Sul", "Norte"]}, index=["A", "B", "C"])
    nova = pd.DataFrame({"comprimento_mm": [np.nan], "ilha": ["Sul"]}, index=["D"])
    imputador = SimpleImputer(strategy="median").fit(treino[["comprimento_mm"]])
    codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    codificador.fit(treino[["ilha"]])
    todos = pd.concat([treino, nova])
    preparado = pd.DataFrame({"observacao": todos.index,
                               "parte": ["treino"] * 3 + ["nova"],
                               "comprimento_original_mm": todos["comprimento_mm"].to_numpy(),
                               "comprimento_preparado_mm": imputador.transform(
                                   todos[["comprimento_mm"]]).ravel()})
    indicadores = codificador.transform(todos[["ilha"]])
    for j, categoria in enumerate(codificador.categories_[0]):
        preparado[categoria] = indicadores[:, j].astype(int)
    gravar_tabela(saida, "cap03_preparacao_ficticia", preparado)
    partes = np.repeat(list("ABCDE"), 2)
    divisoes = pd.DataFrame({"observacao": np.arange(1, 11), "parte": partes})
    for parte in "ABCDE":
        divisoes[f"ronda_{parte}"] = np.where(partes == parte, "validacao", "treino")
    gravar_tabela(saida, "cap03_cinco_partes_ficticias", divisoes)
    return {"media_treino_g": float(treino_massa["massa_g"].mean()),
            "mediana_comprimento_treino_mm": float(imputador.statistics_[0]),
            "ordem_categorias": list(codificador.categories_[0])}


def razao(numerador: float, denominador: float) -> float:
    """Manter indefinida uma razão com denominador nulo."""
    return numerador / denominador if denominador > 0 else np.nan


def medidas_classificacao(vp: int, fn: int, fp: int, vn: int) -> dict:
    sensibilidade = razao(vp, vp + fn)
    especificidade = razao(vn, vn + fp)
    return {"VP": vp, "FN": fn, "FP": fp, "VN": vn,
            "n": vp + fn + fp + vn,
            "proporcao_correta": razao(vp + vn, vp + fn + fp + vn),
            "sensibilidade": sensibilidade, "especificidade": especificidade,
            "VPP": razao(vp, vp + fp), "VPN": razao(vn, vn + fn),
            "FPR": razao(fp, fp + vn), "F1": razao(2 * vp, 2 * vp + fp + fn),
            "exatidao_equilibrada": (sensibilidade + especificidade) / 2}


def capitulo_4(saida: Path) -> dict:
    """Avaliar previsões fictícias e explicitar denominadores nulos."""
    y = np.array([1] * 20 + [0] * 80)
    previsto = np.array([1] * 16 + [0] * 4 + [1] * 8 + [0] * 72)
    gravar_tabela(saida, "cap04_cem_amostras_ficticias",
                  pd.DataFrame({"id_amostra": np.arange(1, 101),
                                "observado": y, "previsto": previsto}))
    cenarios = {"exemplo_100_amostras": (16, 4, 8, 72),
                "exercicio_4_1": (8, 2, 5, 35),
                "exercicio_4_3_referencia_A": (0, 5, 0, 95),
                "exercicio_4_3_modelo_B": (4, 1, 7, 88),
                "exercicio_4_6_conjunto_A": (80, 20, 90, 810),
                "exercicio_4_6_conjunto_B": (400, 100, 50, 450),
                "sem_positivos_nem_previsoes_positivas": (0, 0, 0, 10),
                "sem_previsoes_positivas_com_positivos": (0, 5, 0, 5),
                "sem_positivos_observados_com_falsos_positivos": (0, 0, 2, 8)}
    resultados = pd.DataFrame([{"exemplo_ficticio": nome, **medidas_classificacao(*contagens)}
                              for nome, contagens in cenarios.items()])
    gravar_tabela(saida, "cap04_medidas_classificacao_ficticia", resultados)
    tabela = pd.DataFrame([[16, 4], [8, 72]],
                          index=["observado_positivo", "observado_negativo"],
                          columns=["previsto_positivo", "previsto_negativo"])
    tabela.index.name = "classe_observada"
    gravar_tabela(saida, "cap04_matriz_contagens_ficticia", tabela, indice=True)
    gravar_tabela(saida, "cap04_matriz_proporcoes_por_linha_ficticia",
                  tabela.div(tabela.sum(axis=1), axis=0), indice=True)

    regressao = []
    exemplos = [("exemplo_tres_pinguins", [3000, 4000, 5000], [3200, 3800, 5000]),
                ("exercicio_4_5_modelo_A", [3000, 3500, 4000, 4500], [3200, 3400, 4200, 4400]),
                ("exercicio_4_5_referencia_B", [3000, 3500, 4000, 4500], [3800] * 4)]
    resumo = []
    for nome, observado, pred in exemplos:
        residuos = np.asarray(observado) - np.asarray(pred)
        for i, (obs, pre, res) in enumerate(zip(observado, pred, residuos), start=1):
            regressao.append({"exemplo_ficticio": nome, "observacao": i,
                               "observado_g": obs, "previsto_g": pre,
                               "residuo_g": res, "erro_absoluto_g": abs(res)})
        resumo.append({"exemplo_ficticio": nome, "n": len(observado),
                       "media_residuos_g": residuos.mean(), "MAE_g": np.abs(residuos).mean()})
    gravar_tabela(saida, "cap04_regressao_ficticia_por_observacao", pd.DataFrame(regressao))
    gravar_tabela(saida, "cap04_regressao_ficticia_resumo", pd.DataFrame(resumo))
    return {"classe_positiva_ficticia": "presenca_da_bacteria",
            "ordem_matriz": "observado nas linhas; positivo, negativo em ambos os eixos",
            "quantidades_indefinidas": "celulas vazias nos CSV; nao substituidas por zero",
            "exemplo_100_amostras": medidas_classificacao(16, 4, 8, 72)}


def medidas_completas(dados: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, np.ndarray]:
    completos = dados.dropna(subset=MEDIDAS).copy()
    if len(completos) < 5:
        raise ValueError("Não há observações completas suficientes para os exemplos.")
    x = completos[MEDIDAS]
    if (x.var(ddof=0) == 0).any():
        raise ValueError("Há uma medida constante; verifique o ficheiro de entrada.")
    escala = StandardScaler()
    z = escala.fit_transform(x)
    return completos, x, escala, z


def desenhar_grupos(ax: plt.Axes, x: pd.DataFrame, grupos: np.ndarray, titulo: str) -> None:
    for grupo in sorted(np.unique(grupos)):
        selecionados = grupos == grupo
        ax.scatter(x.loc[selecionados, "flipper_length_mm"], x.loc[selecionados, "body_mass_g"],
                   color=CORES[int(grupo) - 1], s=27, alpha=0.75, label=f"Grupo {grupo}")
    ax.set(xlabel="Comprimento da barbatana (mm)", ylabel="Massa corporal (g)", title=titulo)
    ax.legend(frameon=False)


def capitulo_5(dados: pd.DataFrame, saida: Path) -> dict:
    """K-means nas mesmas quatro medidas, com e sem padronização."""
    completos, x, escala, z = medidas_completas(dados)
    resumo = []
    rotulos = pd.DataFrame({"id_registo": completos["id_registo"].to_numpy()})
    for k in [2, 3, 4]:
        modelo = KMeans(n_clusters=k, n_init=20, random_state=SEMENTE)
        grupos = modelo.fit_predict(z) + 1
        rotulos[f"grupo_K{k}"] = grupos
        rotulos[f"silhueta_K{k}"] = silhouette_samples(z, grupos)
        resumo.append({"K": k, "n": len(completos), "inercia": modelo.inertia_,
                       "silhueta_media": silhouette_score(z, grupos), "iteracoes": modelo.n_iter_})
        perfis = x.copy()
        perfis["grupo"] = grupos
        medias = perfis.groupby("grupo")[MEDIDAS].mean()
        medias.insert(0, "n", perfis.groupby("grupo").size())
        gravar_tabela(saida, f"cap05_perfis_K{k}", medias, indice=True)
    gravar_tabela(saida, "cap05_rotulos_e_silhuetas", rotulos)
    gravar_tabela(saida, "cap05_comparacao_K", pd.DataFrame(resumo))
    grupos_padronizados = rotulos["grupo_K3"].to_numpy()
    modelo_bruto = KMeans(n_clusters=3, n_init=20, random_state=SEMENTE)
    grupos_brutos = modelo_bruto.fit_predict(x) + 1
    gravar_tabela(saida, "cap05_comparacao_escalas", pd.DataFrame({
        "id_registo": completos["id_registo"].to_numpy(),
        "grupo_escala_original": grupos_brutos,
        "grupo_escala_padronizada": grupos_padronizados}))
    # A espécie só é consultada aqui, após a obtenção dos grupos.
    grupos_especies = pd.crosstab(pd.Series(grupos_padronizados, name="grupo"),
                                  pd.Series(completos["species"].to_numpy(), name="especie"))
    gravar_tabela(saida, "cap05_grupos_especies_consulta_posterior", grupos_especies, indice=True)
    gravar_tabela(saida, "cap05_padronizacao", pd.DataFrame({"variavel": MEDIDAS,
        "media": escala.mean_, "desvio_padrao_divisor_n": escala.scale_}))

    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    desenhar_grupos(eixos[0], x, grupos_brutos, "Ajuste nas unidades originais")
    desenhar_grupos(eixos[1], x, grupos_padronizados, "Ajuste após padronização")
    gravar_figura(saida, "u5_kmeans_escala", fig)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot([r["K"] for r in resumo], [r["silhueta_media"] for r in resumo],
            marker="o", color=CORES[0])
    ax.set(xlabel="Número de grupos (K)", ylabel="Silhueta média",
           title="Comparação no desenvolvimento", xticks=[2, 3, 4])
    gravar_figura(saida, "u5_silhueta", fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    desenhar_grupos(ax, x, grupos_padronizados, "Três grupos obtidos com quatro medidas padronizadas")
    gravar_figura(saida, "u5_kmeans_grupos", fig)

    # Operação do exemplo fictício: um passo de atribuição e atualização.
    pontos = np.array([[1, 1], [1, 3], [5, 5], [5, 7]])
    centros_iniciais = pontos[[0, 2]]
    distancias = np.linalg.norm(pontos[:, None, :] - centros_iniciais[None, :, :], axis=2)
    grupos = distancias.argmin(axis=1)
    centros = np.array([pontos[grupos == g].mean(axis=0) for g in [0, 1]])
    gravar_tabela(saida, "cap05_iteracao_ficticia", pd.DataFrame({
        "observacao": list("ABCD"), "indice_1": pontos[:, 0], "indice_2": pontos[:, 1],
        "distancia_centro_inicial_1": distancias[:, 0],
        "distancia_centro_inicial_2": distancias[:, 1], "grupo": grupos + 1}))
    gravar_tabela(saida, "cap05_centros_ficticios_atualizados", pd.DataFrame({
        "grupo": [1, 2], "indice_1": centros[:, 0], "indice_2": centros[:, 1]}))
    grupos_exercicio = np.array([[[2, 2], [2, 4]], [[8, 6], [8, 8]]])
    centros_exercicio = grupos_exercicio.mean(axis=1)
    gravar_tabela(saida, "cap05_centros_exercicio_5_2", pd.DataFrame({
        "grupo": [1, 2], "indice_1": centros_exercicio[:, 0],
        "indice_2": centros_exercicio[:, 1]}))
    a = np.array([2.0, 6.0])
    b = np.array([5.0, 3.0])
    gravar_tabela(saida, "cap05_silhuetas_ficticias", pd.DataFrame({
        "observacao": [1, 2], "a": a, "b": b, "silhueta": (b-a)/np.maximum(a, b)}))
    return {"n_desenvolvimento": len(dados), "n_completos": len(completos),
            "n_excluidos_so_desta_exploracao": len(dados) - len(completos),
            "variaveis": MEDIDAS, "semente": SEMENTE, "n_init": 20,
            "contagens_K3": {str(g): int((grupos_padronizados == g).sum()) for g in [1, 2, 3]},
            "silhuetas": {str(r["K"]): float(r["silhueta_media"]) for r in resumo}}


def capitulo_6(dados: pd.DataFrame, saida: Path) -> dict:
    """ACP das quatro medidas padronizadas, sem usar a espécie no ajuste."""
    completos, x, escala, z = medidas_completas(dados)
    acp = PCA(n_components=4)
    valores = acp.fit_transform(z)
    nomes = ["CP1", "CP2", "CP3", "CP4"]
    coordenadas = pd.DataFrame(valores, columns=nomes)
    coordenadas.insert(0, "id_registo", completos["id_registo"].to_numpy())
    coeficientes = pd.DataFrame(acp.components_.T, columns=nomes, index=MEDIDAS)
    coeficientes.index.name = "variavel"
    percentagens = 100 * acp.explained_variance_ratio_
    gravar_tabela(saida, "cap06_coordenadas", coordenadas)
    gravar_tabela(saida, "cap06_coeficientes", coeficientes, indice=True)
    gravar_tabela(saida, "cap06_variabilidade", pd.DataFrame({"componente": nomes,
        "percentagem": percentagens, "percentagem_acumulada": percentagens.cumsum()}))
    gravar_tabela(saida, "cap06_padronizacao", pd.DataFrame({"variavel": MEDIDAS,
        "media": escala.mean_, "desvio_padrao_divisor_n": escala.scale_}))
    gravar_tabela(saida, "cap06_medidas_padronizadas", pd.DataFrame(
        z, columns=MEDIDAS).assign(id_registo=completos["id_registo"].to_numpy())[
            ["id_registo", *MEDIDAS]])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    posicoes = np.arange(4)
    ax.bar(posicoes - 0.18, acp.components_[0], width=0.36, label="CP1", color=CORES[0])
    ax.bar(posicoes + 0.18, acp.components_[1], width=0.36, label="CP2", color=CORES[1])
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set(xticks=posicoes, xticklabels=NOMES_MEDIDAS,
           ylabel="Coeficiente", title="Coeficientes das duas primeiras componentes")
    ax.legend(frameon=False)
    gravar_figura(saida, "u6_acp_coeficientes", fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(np.arange(1, 5), percentagens, color=CORES[0], label="Por componente")
    ax.plot(np.arange(1, 5), percentagens.cumsum(), "o-", color=CORES[1], label="Acumulada")
    ax.set(xlabel="Componente", ylabel="Variabilidade representada (%)",
           title="Quatro medidas padronizadas", xticks=[1, 2, 3, 4], ylim=(0, 105))
    ax.legend(frameon=False)
    gravar_figura(saida, "u6_variancia", fig)
    for usar_especie, nome in [(False, "u6_acp_scores"), (True, "u6_acp_especies")]:
        fig, ax = plt.subplots(figsize=(7, 4.7))
        if usar_especie:
            # As coordenadas foram calculadas acima. A cor não altera a ACP.
            for j, especie in enumerate(sorted(completos["species"].unique())):
                selecao = completos["species"].to_numpy() == especie
                ax.scatter(valores[selecao, 0], valores[selecao, 1], s=27, alpha=0.75,
                           color=CORES[j], label=especie)
            ax.legend(frameon=False)
        else:
            ax.scatter(valores[:, 0], valores[:, 1], s=27, alpha=0.7, color=CORES[0])
        ax.axhline(0, color="0.7", linewidth=0.7)
        ax.axvline(0, color="0.7", linewidth=0.7)
        ax.set(xlabel=f"CP1 ({percentagens[0]:.2f}%)", ylabel=f"CP2 ({percentagens[1]:.2f}%)",
               title="Coordenadas dos pinguins de desenvolvimento")
        gravar_figura(saida, nome, fig)
    # Os exemplos manuais usam os coeficientes arredondados impressos no livro.
    perfis = np.array([[1, -1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    arredondados = np.array([0.463, -0.401, 0.573, 0.545])
    tabela_perfis = pd.DataFrame(perfis, columns=["z1", "z2", "z3", "z4"])
    tabela_perfis.insert(0, "perfil_ficticio", ["quatro_medidas", "A", "B"])
    tabela_perfis["CP1_com_coeficientes_arredondados"] = perfis @ arredondados
    gravar_tabela(saida, "cap06_perfis_ficticios", tabela_perfis)
    gravar_tabela(saida, "cap06_reflexao_ficticia", pd.DataFrame({
        "observacao": [1, 2], "coordenada_original": [2, -1],
        "coordenada_refletida": [-2, 1]}))
    return {"n": len(x), "variaveis": MEDIDAS, "n_components": 4,
            "variabilidade_percentagem": percentagens.tolist(),
            "variabilidade_acumulada_percentagem": percentagens.cumsum().tolist(),
            "convencao_coeficientes": "components_.T; sinais devolvidos pela biblioteca",
            "erro_maximo_reconstrucao_quatro_componentes": float(np.max(
                np.abs(acp.inverse_transform(valores) - z)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capitulos", type=int, nargs="+", choices=range(1, 7),
                        default=list(range(1, 7)), help="Capítulos a executar (por defeito, todos).")
    parser.add_argument("--saida", type=Path, default=PASTA / "resultados",
                        help="Pasta de saída (por defeito, resultados junto do script).")
    argumentos = parser.parse_args()
    saida = argumentos.saida.resolve()
    capitulos = sorted(set(argumentos.capitulos))
    dados = ler_desenvolvimento() if any(c in capitulos for c in [1, 2, 5, 6]) else None
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    resultados = {}
    funcoes_dados = {1: capitulo_1, 2: capitulo_2, 5: capitulo_5, 6: capitulo_6}
    # Limitar bibliotecas numéricas a uma thread evita paralelismo desnecessário
    # neste conjunto pequeno e permite uma execução consistente entre máquinas.
    with threadpool_limits(limits=1):
        for capitulo in capitulos:
            if capitulo in funcoes_dados:
                resultado = funcoes_dados[capitulo](dados, saida)
            else:
                resultado = {3: capitulo_3, 4: capitulo_4}[capitulo](saida)
            resultados[f"capitulo_{capitulo:02d}"] = resultado
            gravar_json(saida / f"capitulo_{capitulo:02d}.json", resultado)
            print(f"Capítulo {capitulo}: concluído.")
    ambiente = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                "matplotlib": matplotlib.__version__}
    manifesto = {"capitulos_executados": capitulos, "ambiente": ambiente,
                 "dados": "dados/penguins_desenvolvimento.csv" if dados is not None else None,
                 "sha256_dados": hashlib.sha256(FICHEIRO_DADOS.read_bytes()).hexdigest()
                 if dados is not None else None,
                 "teste_reservado_utilizado": False}
    gravar_json(saida / "execucao.json", manifesto)
    print(f"Tabelas e figuras guardadas em {saida}")


if __name__ == "__main__":
    main()
