"""Produzir cinco figuras exatas do capítulo 11, em PDF vetorial e PNG.

Executar depois de codigo/gerar_resultados.py. Não reajusta modelos.
Os mapas descrevem ajustamentos ao desenvolvimento completo; as curvas
usam a divisão interna própria e a comparação usa cinco partes de validação.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="redes_mpl_"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from gerar_resultados import validar_dados


RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"
TEAL, LARANJA, AZUL, ROXO = "#1E817F", "#D98B00", "#244A6B", "#9A62A1"
CINZENTO = "#78838C"
CORES_PARTES = [TEAL, LARANJA, AZUL, ROXO, "#82765D"]
MARCADORES = ["o", "s", "^", "v", "P"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#66717A", "axes.labelcolor": "#28343E",
    "text.color": "#28343E", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "axes.axisbelow": True,
})


def formato_pt(valor, _=None):
    return f"{valor:g}".replace(".", ",")


def estilo(ax):
    ax.grid(color="#E5E9EC", linewidth=.6)
    ax.tick_params(length=3, color="#66717A")
    ax.xaxis.set_major_formatter(FuncFormatter(formato_pt))
    ax.yaxis.set_major_formatter(FuncFormatter(formato_pt))


def guardar(fig, nome, titulo):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    (FIGURAS / "png").mkdir(exist_ok=True)
    for extensao, pasta in [("pdf", FIGURAS), ("png", FIGURAS / "png")]:
        fd, temporario_nome = tempfile.mkstemp(prefix=f".{nome}-", suffix=f".tmp.{extensao}", dir=pasta)
        temporario = Path(temporario_nome)
        try:
            with os.fdopen(fd, "wb") as stream:
                extras = {"metadata": {"Title": titulo,
                          "Subject": "Figura científica reproduzível do capítulo 11",
                          "CreationDate": None, "ModDate": None}} if extensao == "pdf" else {"dpi": 180}
                fig.savefig(stream, format=extensao, bbox_inches="tight", **extras)
                stream.flush()
                os.fsync(stream.fileno())
            dados = temporario.read_bytes()
            if extensao == "pdf" and not dados.rstrip().endswith(b"%%EOF"):
                raise OSError("O PDF não foi escrito integralmente.")
            if extensao == "png" and not dados.endswith(b"IEND\xaeB`\x82"):
                raise OSError("O PNG não foi escrito integralmente.")
            temporario.replace(pasta / f"{nome}.{extensao}")
        finally:
            if temporario.exists():
                temporario.unlink()
    plt.close(fig)


def arquitetura():
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.set(xlim=(0, 1), ylim=(-.02, 1.04))
    ax.axis("off")
    xi, xh, xp = .20, .51, .83
    yi = [.68, .34]
    yh = np.linspace(.88, .14, 8)
    yp = .51
    for y1 in yi:
        for y2 in yh:
            seta = FancyArrowPatch((xi, y1), (xh, y2),
                arrowstyle="-|>", mutation_scale=10, shrinkA=17, shrinkB=15,
                color="#A4AFB7", linewidth=.8, alpha=.85, zorder=1)
            ax.add_patch(seta)
    for y1 in yh:
        seta = FancyArrowPatch((xh, y1), (xp, yp),
            arrowstyle="-|>", mutation_scale=10, shrinkA=15, shrinkB=20,
            color="#A4AFB7", linewidth=.9, alpha=.9, zorder=1)
        ax.add_patch(seta)
    ax.scatter([xi, xi], yi, s=1040, facecolors="#F0F3F6", edgecolors=AZUL,
               linewidths=1.4, zorder=3)
    for i, y in enumerate(yi, 1):
        ax.text(xi, y, rf"$x_{i}^*$", ha="center", va="center", fontsize=15, zorder=4)
    ax.scatter(np.repeat(xh, 8), yh, s=760, facecolors="#EDF6F5", edgecolors=TEAL,
               linewidths=1.3, zorder=3)
    for j, y in enumerate(yh, 1):
        ax.text(xh, y, rf"$h_{j}$", ha="center", va="center", fontsize=13, zorder=4)
    ax.scatter([xp], [yp], s=1550, facecolors="#FFF3DB", edgecolors=LARANJA,
               linewidths=1.5, zorder=3)
    ax.text(xp, yp, "$p$", ha="center", va="center", fontsize=18, zorder=4)
    ax.text(xi, 1.015, "Duas entradas\npadronizadas", ha="center", va="center", fontsize=13)
    ax.text(xh, 1.015, "Uma camada oculta\noito unidades ReLU", ha="center", va="center", fontsize=13)
    ax.text(xp, 1.015, "Uma saída\nprobabilidade", ha="center", va="center", fontsize=13)
    ax.text(xi - .065, yi[0], "Raio\nmédio", ha="right", va="center", fontsize=12)
    ax.text(xi - .065, yi[1], "Textura\nmédia", ha="right", va="center", fontsize=12)
    ax.text(.67, .19, r"$z=c+\sum_{j=1}^{8}v_jh_j$", ha="left", fontsize=14)
    ax.text(.67, .11, r"$p=1/(1+e^{-z})$", ha="left", fontsize=14)
    fig.text(.5, .12, r"Cada unidade oculta calcula $h_j=\max(0,w_{j1}x_1^*+w_{j2}x_2^*+b_j)$.",
             ha="center", fontsize=12.5)
    fig.text(.5, .065, "Cada seta tem um peso. Os termos constantes são os oito " + r"$b_j$" + " e " + r"$c$" + ".",
             ha="center", fontsize=12)
    fig.text(.5, .02, "Os termos constantes não representam medidas adicionais da amostra.",
             ha="center", fontsize=11.5, color=CINZENTO)
    fig.subplots_adjust(left=.01, right=.99, top=.95, bottom=.20)
    guardar(fig, "u11_arquitetura", "Rede com duas entradas, oito unidades ocultas e uma probabilidade de saída")


def relu():
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    a = np.linspace(-1.4, 1.5, 501)
    ax.axvline(0, color="#9CA6AE", linewidth=.8)
    ax.plot(a, np.maximum(0, a), color=TEAL, linewidth=3, zorder=3)
    ax.scatter([-.7, .8], [0, .8], color=LARANJA, s=65, edgecolor="white", linewidth=.7, zorder=5)
    ax.plot([.8, .8], [0, .8], color=CINZENTO, linewidth=1, linestyle="--")
    ax.plot([0, .8], [.8, .8], color=CINZENTO, linewidth=1, linestyle="--")
    ax.annotate(r"$a=-0{,}7\;\Rightarrow\;h=0$", xy=(-.7, 0), xytext=(-1.26, .37),
                arrowprops={"arrowstyle": "->", "color": CINZENTO}, fontsize=14,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    ax.annotate(r"$a=0{,}8\;\Rightarrow\;h=0{,}8$", xy=(.8, .8), xytext=(.27, 1.16),
                arrowprops={"arrowstyle": "->", "color": CINZENTO}, fontsize=14,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    ax.set(xlim=(-1.4, 1.5), ylim=(-.12, 1.38),
           xlabel="Combinação ponderada a", ylabel="Saída da unidade h")
    ax.set_xticks([-1.2, -.7, 0, .4, .8, 1.2])
    ax.set_yticks([0, .4, .8, 1.2])
    ax.set_title(r"Ativação ReLU: $h=\max(0,a)$", loc="left", pad=12)
    estilo(ax)
    fig.subplots_adjust(left=.10, right=.98, bottom=.19, top=.87)
    guardar(fig, "u11_relu", "Função ReLU e dois valores de ativação")


def probabilidades(dados):
    superficie = np.load(RESULTADOS / "superficies_probabilidade.npz")
    cores = LinearSegmentedColormap.from_list("probabilidade", ["#F5F7F9", "#6D95B3"])
    fig = plt.figure(figsize=(10.2, 5.65))
    grelha = fig.add_gridspec(1, 2, left=.09, right=.875, top=.80, bottom=.19, wspace=.20)
    ax0 = fig.add_subplot(grelha[0, 0])
    ax1 = fig.add_subplot(grelha[0, 1], sharex=ax0, sharey=ax0)
    for ax, unidades, parametros in [(ax0, 8, 33), (ax1, 32, 129)]:
        p = superficie[f"p_rede_{unidades}"]
        ax.contourf(superficie["raio"], superficie["textura"], p,
                    levels=np.linspace(0, 1, 31), cmap=cores, norm=Normalize(0, 1), zorder=0)
        ax.contour(superficie["raio"], superficie["textura"], p,
                   levels=[.5], colors=["#172F44"], linewidths=1.8, zorder=2)
        for classe, cor, marcador in [(0, TEAL, "o"), (1, LARANJA, "^")]:
            selecao = dados["maligno"] == classe
            ax.scatter(dados.loc[selecao, "mean radius"], dados.loc[selecao, "mean texture"],
                       color=cor, marker=marcador, s=17, edgecolor="white",
                       linewidth=.35, alpha=.9, zorder=3)
        ax.set_title(f"{unidades} unidades ({parametros} parâmetros)", fontsize=13, pad=10)
        ax.set_xlabel("Raio médio (escala da fonte)", fontsize=12)
        estilo(ax)
    ax0.set_ylabel("Textura média (escala da fonte)", fontsize=12)
    ax1.tick_params(labelleft=False)
    barra_ax = fig.add_axes([.91, .19, .018, .61])
    barra = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cores), cax=barra_ax,
                        ticks=[0, .2, .4, .6, .8, 1])
    barra.ax.yaxis.set_major_formatter(FuncFormatter(formato_pt))
    barra.set_label("Probabilidade de malignidade", fontsize=12)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=TEAL, label="Benigno (0)", markersize=6),
        Line2D([], [], marker="^", linestyle="", color=LARANJA, label="Maligno (1)", markersize=6),
        Line2D([], [], color="#172F44", linewidth=1.8, label="Probabilidade 0,50"),
    ], loc="upper center", bbox_to_anchor=(.49, 1.01), ncol=3,
       frameon=False, title="Símbolos: classe de referência; linha: limiar de decisão", title_fontsize=11)
    fig.text(.5, .04, "Ajustamentos às mesmas 455 amostras de desenvolvimento; não são previsões de validação.",
             ha="center", fontsize=11.5, color=CINZENTO)
    guardar(fig, "u11_probabilidades", "Mapas de probabilidades das redes ajustadas ao desenvolvimento completo")


def curvas():
    fig, eixos = plt.subplots(2, 2, figsize=(9.6, 7.2), sharey="row")
    for coluna, unidades in enumerate([8, 32]):
        tabela = pd.read_csv(RESULTADOS / f"curvas_rede_{unidades}.csv")
        for linha in [0, 1]:
            ax = eixos[linha, coluna]
            mostrada = tabela if linha == 0 else tabela[tabela["epoca"] >= 20]
            for chave, cor, traco in [("perda_treino", AZUL, "--"), ("perda_validacao", LARANJA, "-")]:
                ax.plot(mostrada["epoca"], mostrada[chave], color=cor, linestyle=traco, linewidth=1.8)
            ax.set_title(f"{unidades} unidades" if linha == 0 else "Pormenor a partir da época 20", loc="left", fontsize=13)
            ax.set(xlim=(0, 202) if linha == 0 else (20, 202),
                   ylim=(.18, .56) if linha == 0 else (.19, .28))
            ax.set_xticks([0, 50, 100, 150, 200] if linha == 0 else [20, 50, 100, 150, 200])
            ax.set_yticks([.2, .3, .4, .5] if linha == 0 else [.20, .22, .24, .26, .28])
            ax.set_xlabel("Época")
            if coluna == 0:
                ax.set_ylabel("Perda logarítmica média")
            estilo(ax)
    fig.legend(handles=[
        Line2D([], [], color=AZUL, linestyle="--", linewidth=1.8, label="Treino (318 amostras)"),
        Line2D([], [], color=LARANJA, linestyle="-", linewidth=1.8, label="Validação (137 amostras)"),
    ], loc="lower center", bbox_to_anchor=(.51, .055), ncol=2, frameon=False, fontsize=12)
    fig.text(.51, .018, "Perda calculada no fim de cada época, sem acrescentar a penalização dos pesos.",
             ha="center", fontsize=11.5, color=CINZENTO)
    fig.subplots_adjust(left=.10, right=.98, bottom=.20, top=.94, wspace=.20, hspace=.55)
    guardar(fig, "u11_curvas", "Perdas por época das redes com oito e trinta e duas unidades")


def comparacao():
    tabela = pd.read_csv(RESULTADOS / "metricas_classificacao_por_parte.csv")
    chaves = ["logistica", "floresta_folha_5", "rede_8", "rede_32"]
    nomes = ["Logística", "Floresta (folha mínima 5)", "Rede (8 unidades)", "Rede (32 unidades)"]
    fig, ax = plt.subplots(figsize=(9.7, 4.8))
    for yi, chave in enumerate(chaves):
        valores = tabela[tabela["chave"] == chave].sort_values("parte")["validacao_AUC"].to_numpy()
        assert len(valores) == 5
        for j, (valor, cor, marcador) in enumerate(zip(valores, CORES_PARTES, MARCADORES)):
            ax.scatter(valor, yi + (j - 2) * .075, color=cor, marker=marcador,
                       s=42, edgecolor="white", linewidth=.45, zorder=3)
        ax.scatter(valores.mean(), yi, color="#171D23", marker="D", s=50,
                   edgecolor="white", linewidth=.45, zorder=4)
    ax.set(xlim=(.86, .985), ylim=(3.5, -.5), xlabel="AUC de validação (maior é melhor)")
    ax.set_xticks([.86, .88, .90, .92, .94, .96, .98])
    estilo(ax)
    ax.set_yticks(range(4), nomes, fontsize=12)
    handles = [Line2D([], [], color=cor, marker=marcador, linestyle="", markersize=6,
                      label=f"Parte {i}") for i, (cor, marcador) in enumerate(zip(CORES_PARTES, MARCADORES), 1)]
    handles.append(Line2D([], [], color="#171D23", marker="D", linestyle="", markersize=6,
                          label="Média das 5 partes"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11.5,
               bbox_to_anchor=(.58, .005))
    fig.subplots_adjust(left=.32, right=.98, top=.95, bottom=.31)
    guardar(fig, "u11_comparacao", "AUC dos quatro procedimentos nas mesmas cinco partes de validação")


def main():
    dados = validar_dados()
    arquitetura()
    relu()
    probabilidades(dados)
    curvas()
    comparacao()
    print("Cinco figuras geradas a partir dos resultados; nenhum modelo foi reajustado.")


if __name__ == "__main__":
    main()
