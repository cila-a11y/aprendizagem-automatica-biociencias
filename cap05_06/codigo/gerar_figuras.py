"""Reproduzir as oito figuras e os resultados dos capítulos 5 e 6.

Executar a partir de qualquer pasta:
    python codigo/gerar_figuras.py

As observações reais são exclusivamente as 273 linhas completas do
desenvolvimento. As duas ilustrações geométricas usam dados fictícios
explicitamente identificados. Não se ajusta qualquer análise ao teste.
"""

import json
import os
import platform
import tempfile
from hashlib import sha256
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="acp_mpl_"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

RAIZ = Path(__file__).resolve().parents[1]
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
COLUNAS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
NOMES = ["Comprimento do bico", "Profundidade do bico", "Barbatana", "Massa corporal"]
CORES = ["#1E817F", "#D98B00", "#244A6B", "#9A62A1"]
CINZENTO = "#78838C"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#66717A", "axes.labelcolor": "#28343E",
    "text.color": "#28343E", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "axes.axisbelow": True,
})


def pt(valor, casas=3):
    return f"{valor:.{casas}f}".replace(".", ",")


def estilo(ax, grelha=True):
    if grelha:
        ax.grid(color="#E5E9EC", linewidth=0.6)
    ax.tick_params(length=3, color="#66717A")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}".replace(".", ",")))


def guardar(fig, nome):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    (FIGURAS / "png").mkdir(exist_ok=True)
    temporario = FIGURAS / f"{nome}.tmp.pdf"
    fig.savefig(temporario, bbox_inches="tight",
                metadata={"Title": nome, "Subject": "Ilustração científica reproduzível"})
    if not temporario.read_bytes().rstrip().endswith(b"%%EOF"):
        raise OSError(f"O PDF não foi escrito integralmente: {temporario}")
    temporario.replace(FIGURAS / f"{nome}.pdf")
    fig.savefig(FIGURAS / "png" / f"{nome}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    if temporario.exists() and temporario.stat().st_size == 0:
        temporario.unlink()


def iteracao_ficticia():
    pontos = np.array([[1., 1.], [1., 3.], [5., 5.], [5., 7.]])
    grupos = np.array([0, 0, 1, 1])
    antes = np.array([[1., 1.], [5., 5.]])
    depois = np.array([[1., 2.], [5., 6.]])
    fig, eixos = plt.subplots(1, 2, figsize=(9.2, 4.3), sharex=True, sharey=True)
    for ax, centros, titulo, inercia in zip(
        eixos, [antes, depois],
        ["a) Atribuir aos centros iniciais", "b) Atualizar os centros"], [8., 4.]
    ):
        for i, (ponto, g) in enumerate(zip(pontos, grupos)):
            centro = centros[g]
            ax.plot([ponto[0], centro[0]], [ponto[1], centro[1]],
                    color=CORES[g], linewidth=1.6, alpha=.65, zorder=1)
            ax.scatter(*ponto, color=CORES[g], s=60, zorder=3)
            ax.annotate("ABCD"[i], ponto, xytext=(9, 3), textcoords="offset points")
        ax.scatter(centros[:, 0], centros[:, 1], marker="X", s=150,
                   facecolor="white", edgecolor="#28343E", linewidth=1.6, zorder=4)
        ax.set_title(titulo, loc="left", pad=10)
        ax.text(.04, .96, "Inércia = " + pt(inercia, 0), transform=ax.transAxes,
                va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
        ax.set(xlabel="Índice A", xlim=(0, 6.6), ylim=(0, 8))
        ax.set_aspect("equal", adjustable="box")
        estilo(ax)
        assert np.isclose(((pontos - centros[grupos]) ** 2).sum(), inercia)
    eixos[0].set_ylabel("Índice B")
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=CORES[0], label="Grupo 1"),
        Line2D([], [], marker="o", linestyle="", color=CORES[1], label="Grupo 2"),
        Line2D([], [], marker="X", linestyle="", color="#28343E", markerfacecolor="white", label="Centro"),
    ], loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=.19, wspace=.23)
    guardar(fig, "u5_kmeans_iteracao")


def graficos_grupos(X, rotulos, escala):
    centros = X.assign(grupo=rotulos + 1).groupby("grupo")[COLUNAS].mean()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for g in range(3):
        sel = rotulos == g
        ax.scatter(X.loc[sel, "bill_length_mm"], X.loc[sel, "bill_depth_mm"],
                   s=24, alpha=.8, color=CORES[g], label=f"Grupo {g + 1} (n = {sel.sum()})")
        ax.scatter(centros.iloc[g, 0], centros.iloc[g, 1], marker="X", s=135,
                   color=CORES[g], edgecolor="#182C3B", linewidth=1, zorder=5)
    ax.set(xlabel="Comprimento do bico (mm)", ylabel="Profundidade do bico (mm)")
    ax.legend(frameon=False, fontsize=9)
    estilo(ax)
    guardar(fig, "u5_kmeans_grupos")
    cru = KMeans(n_clusters=3, n_init=20, random_state=2026).fit_predict(X)
    fig, eixos = plt.subplots(1, 2, figsize=(10.2, 4.4), sharex=True, sharey=True)
    for ax, lab, titulo in zip(eixos, [cru, rotulos],
                              ["a) Ajuste nas unidades originais", "b) Ajuste após padronização"]):
        for g in range(3):
            sel = lab == g
            ax.scatter(X.loc[sel, "flipper_length_mm"], X.loc[sel, "body_mass_g"],
                       s=22, color=CORES[g], alpha=.8, label=f"Grupo {g + 1}")
        ax.set_title(titulo, loc="left", pad=10)
        ax.set_xlabel("Comprimento da barbatana (mm)")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        estilo(ax)
    eixos[0].set_ylabel("Massa corporal (g)")
    fig.subplots_adjust(wspace=.12, bottom=.16)
    guardar(fig, "u5_kmeans_escala")


def diagramas_silhueta(ajustes, silhuetas):
    fig, eixos = plt.subplots(1, 3, figsize=(11.2, 4.8), sharex=True, sharey=True)
    n = len(silhuetas[2])
    for ax, k in zip(eixos, [2, 3, 4]):
        lab = ajustes[k].labels_
        valores = silhuetas[k]
        media = valores.mean()
        inicio = 9
        for g in range(k):
            do_grupo = np.sort(valores[lab == g])
            ys = np.arange(inicio, inicio + len(do_grupo))
            ax.barh(ys, do_grupo, height=1.0, color=CORES[g], linewidth=0)
            ax.text(-.95, inicio + (len(do_grupo) - 1) / 2,
                    f"Grupo {g + 1}\nn = {len(do_grupo)}", va="center", ha="left", fontsize=11)
            inicio += len(do_grupo) + 9
        ax.axvline(0, color="#4F5C66", linewidth=.9)
        ax.axvline(media, color="#182C3B", linestyle="--", linewidth=1.4)
        negativas = int((valores < 0).sum())
        ax.set_title(f"K = {k} | média = {pt(media)}\nSilhuetas negativas: {negativas}", fontsize=12, pad=11)
        ax.set(xlim=(-1, 1), ylim=(0, n + 9 * 5), xlabel="Silhueta individual")
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        estilo(ax, False)
    fig.text(.5, .015, "Uma faixa por pinguim; faixas ordenadas dentro de cada grupo. Linha tracejada: média global.",
             ha="center", fontsize=9)
    fig.subplots_adjust(wspace=.13, bottom=.14, top=.83)
    guardar(fig, "u5_silhueta")


def projecao_ficticia():
    cp1 = np.array([-2., -2., 0., 0., 2., 2.])
    cp2 = np.array([-.6, .6, -.3, .3, -.6, .6])
    direcao1 = np.array([1., 1.]) / np.sqrt(2)
    direcao2 = np.array([-1., 1.]) / np.sqrt(2)
    pontos = cp1[:, None] * direcao1 + cp2[:, None] * direcao2
    projecoes = cp1[:, None] * direcao1
    assert np.allclose(pontos.mean(axis=0), 0)
    assert np.isclose(cp1 @ cp2, 0)
    assert np.var(cp1) > np.var(cp2)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"width_ratios": [1, 1]})
    t = np.array([-3., 3.])
    ax.plot(t * direcao1[0], t * direcao1[1], color=CORES[0], linewidth=1.5)
    ax.plot(t * direcao2[0], t * direcao2[1], color=CINZENTO, linewidth=1, linestyle=":")
    ax.text(1.55, 1.95, "CP1", color=CORES[0], fontweight="bold")
    ax.text(-1.95, 1.8, "CP2", color=CINZENTO, fontweight="bold")
    for i, (ponto, proj) in enumerate(zip(pontos, projecoes)):
        ax.plot([ponto[0], proj[0]], [ponto[1], proj[1]], color=CINZENTO, linewidth=1, linestyle="--")
        ax.scatter(*ponto, color=CORES[2], s=40, zorder=4)
        offset = (8, -9) if i % 2 == 0 else (-14, 6)
        ax.annotate("ABCDEF"[i], ponto, xytext=offset, textcoords="offset points", fontsize=10)
    ax.scatter(projecoes[:, 0], projecoes[:, 1], color=CORES[1], marker="D", s=32, zorder=3)
    ax.set(xlabel="Índice A (centrado)", ylabel="Índice B (centrado)", xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    ax.set_title("a) As duas direções principais", loc="left", pad=11)
    ax.set_aspect("equal", adjustable="box")
    estilo(ax)
    bx.axhline(0, color=CORES[0], linewidth=1.5)
    bx.scatter([-2, 0, 2], [0, 0, 0], color=CORES[1], marker="D", s=45, zorder=5)
    for coord, nome in zip([-2, 0, 2], ["A e B", "C e D", "E e F"]):
        bx.text(coord, .16, nome, ha="center", fontsize=11)
    bx.text(.5, .72, "Cada par fica representado\npelo mesmo ponto.", ha="center", transform=bx.transAxes)
    bx.text(.5, .18, "As diferenças na direção CP2\ndeixam de aparecer.", ha="center", transform=bx.transAxes, color=CINZENTO)
    bx.set(xlabel="Coordenada em CP1", xlim=(-3.1, 3.1), ylim=(-1, 1), yticks=[])
    bx.set_xticks([-2, 0, 2])
    bx.spines["left"].set_visible(False)
    bx.set_box_aspect(1)
    bx.set_title("b) Conservar apenas CP1", loc="left", pad=11)
    estilo(bx, False)
    fig.subplots_adjust(wspace=.36, bottom=.16)
    guardar(fig, "u6_acp_projecao")


def graficos_acp(acp, valores, completos):
    fig, eixos = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True, sharex=True)
    ys = np.arange(4)
    for j, ax in enumerate(eixos):
        cs = acp.components_[j]
        ax.barh(ys, cs, color=CORES[0] if j == 0 else CORES[2], height=.55)
        ax.axvline(0, color=CINZENTO, linewidth=.8)
        for y, c in zip(ys, cs):
            ax.text(c + (.025 if c >= 0 else -.025), y, pt(c),
                    va="center", ha="left" if c >= 0 else "right", fontsize=11)
        ax.set_title(f"Coeficientes de CP{j + 1}", loc="left")
        ax.set(xlim=(-.95, 1.02), xlabel="Coeficiente")
        ax.set_yticks(ys, NOMES)
        ax.grid(axis="x", color="#E5E9EC", linewidth=.6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    eixos[0].invert_yaxis()
    fig.subplots_adjust(left=.21, right=.98, bottom=.18, wspace=.16)
    guardar(fig, "u6_acp_coeficientes")
    percent = 100 * acp.explained_variance_ratio_
    acumulada = percent.cumsum()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    xx = np.arange(1, 5)
    ax.bar(xx, percent, color=CORES[0], width=.58)
    bx.plot(xx, acumulada, color=CORES[2], marker="o", markersize=6, linewidth=1.7)
    for eixo, vals in [(ax, percent), (bx, acumulada)]:
        for x, val in zip(xx, vals):
            eixo.text(x, val + 2.7, pt(val, 2) + "%", ha="center", fontsize=11)
        eixo.set(ylim=(0, 112), xticks=xx, xticklabels=["CP1", "CP2", "CP3", "CP4"])
        eixo.set_yticks([0, 20, 40, 60, 80, 100])
        eixo.grid(axis="y", color="#E5E9EC", linewidth=.6)
    ax.set_title("a) Percentagem de cada componente", loc="left")
    bx.set_title("b) Percentagem acumulada", loc="left")
    ax.set_ylabel("Variabilidade representada (%)")
    fig.subplots_adjust(wspace=.13, bottom=.13)
    guardar(fig, "u6_variancia")
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
    xmin, xmax = valores[:, 0].min() - .5, valores[:, 0].max() + .5
    ymin, ymax = valores[:, 1].min() - .5, valores[:, 1].max() + .5
    eixos[0].scatter(valores[:, 0], valores[:, 1], s=23, color=CORES[2], alpha=.7)
    for especie, cor, marca in zip(["Adelie", "Chinstrap", "Gentoo"], CORES[:3], ["o", "^", "s"]):
        sel = completos["species"].to_numpy() == especie
        eixos[1].scatter(valores[sel, 0], valores[sel, 1], s=25, color=cor,
                        marker=marca, alpha=.8, label=especie)
    for ax, titulo in zip(eixos, ["a) Apenas as coordenadas", "b) A espécie acrescentada depois"]):
        ax.axhline(0, color=CINZENTO, linewidth=.7)
        ax.axvline(0, color=CINZENTO, linewidth=.7)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel=f"CP1 ({pt(percent[0], 2)}%)")
        ax.set_title(titulo, loc="left", pad=10)
        ax.set_aspect("equal", adjustable="box")
        estilo(ax)
    eixos[0].set_ylabel(f"CP2 ({pt(percent[1], 2)}%)")
    handles, labels = eixos[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(wspace=.13, bottom=.23)
    guardar(fig, "u6_acp_mapa")


def main():
    caminho = RAIZ / "dados" / "penguins_desenvolvimento.csv"
    if not caminho.exists():
        raise FileNotFoundError("Execute primeiro: python codigo/preparar_dados.py")
    dados = pd.read_csv(caminho)
    completos = dados.dropna(subset=COLUNAS).copy()
    X = completos[COLUNAS]
    assert len(dados) == 275 and len(X) == 273
    assert completos["id_registo"].head().tolist() == [1, 2, 3, 5, 9]
    escala = StandardScaler()
    Z = escala.fit_transform(X)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    resumo = {"fonte": "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv",
              "sha256_desenvolvimento": sha256(caminho.read_bytes()).hexdigest(),
              "versoes": {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
              "n_desenvolvimento": len(dados), "n_completos": len(X),
              "variaveis": COLUNAS, "media_padronizacao": escala.mean_.tolist(),
              "escala_padronizacao": escala.scale_.tolist(), "kmeans": {}}
    ajustes, silhuetas = {}, {}
    with threadpool_limits(limits=1):
        for k in [2, 3, 4]:
            ajuste = KMeans(n_clusters=k, n_init=20, random_state=2026).fit(Z)
            s = silhouette_samples(Z, ajuste.labels_)
            ajustes[k], silhuetas[k] = ajuste, s
            assert np.isclose(s.mean(), silhouette_score(Z, ajuste.labels_))
            tabela = pd.DataFrame({"id_registo": completos["id_registo"].to_numpy(),
                                  "grupo": ajuste.labels_ + 1, "silhueta": s})
            tabela.to_csv(RESULTADOS / f"silhuetas_k{k}.csv", index=False)
            resumo["kmeans"][str(k)] = {"inercia": float(ajuste.inertia_), "media_silhueta": float(s.mean()),
                                        "negativas": int((s < 0).sum()),
                                        "tamanhos": np.bincount(ajuste.labels_).tolist()}
        assert np.allclose([silhuetas[k].mean() for k in [2, 3, 4]],
                           [.5379752876089756, .43375273832930233, .38752310144976976])
        assert [int((silhuetas[k] < 0).sum()) for k in [2, 3, 4]] == [0, 2, 5]
        iteracao_ficticia()
        graficos_grupos(X, ajustes[3].labels_, escala)
        diagramas_silhueta(ajustes, silhuetas)
        acp = PCA(n_components=4)
        valores = acp.fit_transform(Z)
        assert np.allclose(acp.components_ @ acp.components_.T, np.eye(4))
        assert np.allclose(acp.explained_variance_ratio_,
                           [.694500496414, .187406875271, .090474372327, .027618255989])
        assert np.allclose(acp.inverse_transform(valores), Z)
        cov = np.cov(valores, rowvar=False)
        assert np.allclose(cov, np.diag(np.diag(cov)))
        projecao_ficticia()
        graficos_acp(acp, valores, completos)
    nomes = ["CP1", "CP2", "CP3", "CP4"]
    pd.DataFrame(acp.components_.T, index=COLUNAS, columns=nomes).to_csv(RESULTADOS / "acp_coeficientes.csv", index_label="variavel")
    coordenadas = pd.DataFrame(valores, columns=nomes, index=completos["id_registo"])
    coordenadas.to_csv(RESULTADOS / "acp_coordenadas.csv", index_label="id_registo")
    pd.DataFrame({"componente": nomes, "percentagem": 100 * acp.explained_variance_ratio_,
                  "acumulada": 100 * acp.explained_variance_ratio_.cumsum()}).to_csv(RESULTADOS / "acp_variancia.csv", index=False)
    perfis = X.assign(grupo=ajustes[3].labels_ + 1).groupby("grupo")[COLUNAS].mean()
    perfis.insert(0, "n", np.bincount(ajustes[3].labels_))
    perfis.to_csv(RESULTADOS / "kmeans_perfis_k3.csv")
    indices = [int(np.flatnonzero(completos["id_registo"].to_numpy() == i)[0]) for i in [1, 5]]
    a, b = indices
    d4 = float(np.linalg.norm(Z[a] - Z[b]))
    d2 = float(np.linalg.norm(valores[a, :2] - valores[b, :2]))
    d_omitida = float(np.linalg.norm(valores[a, 2:] - valores[b, 2:]))
    assert np.isclose(d4 ** 2, d2 ** 2 + d_omitida ** 2)
    resumo["acp"] = {"coeficientes": acp.components_.T.tolist(),
                     "percentagens": (100 * acp.explained_variance_ratio_).tolist(),
                     "distancias_registos_1_5": {"quatro_medidas": d4, "plano_cp1_cp2": d2,
                                                "cp3_cp4": d_omitida}}
    (RESULTADOS / "resultados_verificados.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print("Oito figuras geradas e verificações numéricas concluídas.")


if __name__ == "__main__":
    main()
