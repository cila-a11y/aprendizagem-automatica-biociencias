"""Produzir as sete figuras dos capítulos 9 e 10 a partir dos resultados.

Executar primeiro codigo/gerar_resultados.py e depois este script.
Não volta a ajustar modelos, não lê o teste e não altera números.
As figuras são guardadas em PDF vetorial e em PNG para consulta rápida.
"""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="arvores_mpl_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from validar_dados import validar


RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"
TEAL, LARANJA, AZUL, ROXO = "#1E817F", "#D98B00", "#244A6B", "#9A62A1"
CINZENTO = "#78838C"
CORES_PARTES = [TEAL, LARANJA, AZUL, ROXO, "#82765D"]
MARCADORES = ["o", "s", "^", "v", "P"]
CHAVES_ARVORES = ["arvore_profundidade_2", "arvore_profundidade_4", "arvore_sem_limite"]
CHAVES_COMUNS = [*CHAVES_ARVORES, "floresta_folha_1", "floresta_folha_5"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#66717A", "axes.labelcolor": "#28343E",
    "text.color": "#28343E", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "axes.axisbelow": True,
})


def pt(valor, casas=1):
    return f"{valor:.{casas}f}".replace(".", ",")


def ler_json(nome):
    return json.loads((RESULTADOS / nome).read_text(encoding="utf-8"))


def estilo(ax):
    ax.grid(color="#E5E9EC", linewidth=.6)
    ax.tick_params(length=3, color="#66717A")
    formato = FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(formato)
    ax.yaxis.set_major_formatter(formato)


def guardar(fig, nome, titulo):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    (FIGURAS / "png").mkdir(exist_ok=True)
    fd, temporario_nome = tempfile.mkstemp(prefix=f".{nome}-", suffix=".tmp.pdf", dir=FIGURAS)
    temporario = Path(temporario_nome)
    try:
        with os.fdopen(fd, "wb") as stream:
            fig.savefig(stream, format="pdf", bbox_inches="tight", metadata={
                "Title": titulo, "Subject": "Figura científica reproduzível dos capítulos 9 e 10",
                "CreationDate": None, "ModDate": None,
            })
            stream.flush()
            os.fsync(stream.fileno())
        if not temporario.read_bytes().rstrip().endswith(b"%%EOF"):
            raise OSError("O PDF não foi escrito integralmente.")
        temporario.replace(FIGURAS / f"{nome}.pdf")
        fig.savefig(FIGURAS / "png" / f"{nome}.png", dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
        if temporario.exists():
            temporario.unlink()


def desenhar_arvore(tarefa):
    nos = ler_json(f"arvore_{tarefa}_completa.json")
    posicoes = {0: (.50, .86), 1: (.25, .55), 4: (.75, .55),
                2: (.115, .17), 3: (.37, .17), 5: (.63, .17), 6: (.885, .17)}
    fig, ax = plt.subplots(figsize=(9.2, 6.1 if tarefa == "classificacao" else 5.3))
    ax.set(xlim=(0, 1), ylim=(-.035, 1.02))
    ax.axis("off")
    for no in nos:
        if no["folha"] is not None:
            continue
        x, y = posicoes[no["no"]]
        for chave, texto in [("filho_esquerdo", "Sim"), ("filho_direito", "Não")]:
            xf, yf = posicoes[no[chave]]
            altura_pai = .245 if tarefa == "classificacao" else .205
            e_folha = no[chave] in [2, 3, 5, 6]
            altura_filho = (.285 if e_folha else .245) if tarefa == "classificacao" else (.245 if e_folha else .205)
            ax.plot([x, xf], [y - altura_pai / 2, yf + altura_filho / 2],
                    color=CINZENTO, linewidth=1.3, zorder=1)
            ax.text((x + xf) / 2, (y - altura_pai / 2 + yf + altura_filho / 2) / 2,
                    texto, ha="center", va="center", fontsize=12.5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2}, zorder=3)
    for no in nos:
        x, y = posicoes[no["no"]]
        folha = no["folha"]
        if tarefa == "classificacao":
            benignas = f"{no['benignas']} " + ("benigna" if no["benignas"] == 1 else "benignas")
            malignas = f"{no['malignas']} " + ("maligna" if no["malignas"] == 1 else "malignas")
            if folha:
                texto = (f"Folha {folha}\n{no['n_amostras']} amostras\n"
                         f"{benignas}\n{malignas}\np = {pt(no['p_maligno'], 3)}")
                largura, altura = .205, .285
                cor_fundo = "#EDF6F5" if no["p_maligno"] < .5 else "#FFF3DB"
                cor_borda = TEAL if no["p_maligno"] < .5 else LARANJA
            else:
                nome = "Raio médio" if no["variavel"] == "mean radius" else "Textura média"
                texto = (f"{nome} ≤ {pt(no['limiar'], 3)}\n{no['n_amostras']} amostras\n"
                         f"{benignas}\n{malignas}")
                largura, altura = .305, .245
                cor_fundo, cor_borda = "#F0F3F6", AZUL
        else:
            if folha:
                texto = f"Folha {folha}\n{no['n_amostras']} pinguins\nPrevisão\n{pt(no['media_massa_g'])} g"
                largura, altura = .205, .245
                cor_fundo, cor_borda = "#EDF6F5", TEAL
            else:
                nome = "Barbatana" if no["variavel"] == "flipper_length_mm" else "Bico"
                casas = 2 if no["variavel"] == "bill_length_mm" else 1
                texto = f"{nome} ≤ {pt(no['limiar'], casas)} mm\n{no['n_amostras']} pinguins\nMédia {pt(no['media_massa_g'])} g"
                largura, altura = .305, .205
                cor_fundo, cor_borda = "#F0F3F6", AZUL
        caixa = FancyBboxPatch((x - largura / 2, y - altura / 2), largura, altura,
                              boxstyle="round,pad=0.008", facecolor=cor_fundo,
                              edgecolor=cor_borda, linewidth=1.2, zorder=2)
        ax.add_patch(caixa)
        ax.text(x, y, texto, ha="center", va="center", fontsize=13,
                linespacing=1.18, zorder=3)
    nota = ("Ajuste às 455 amostras de desenvolvimento. p = proporção maligna na folha."
            if tarefa == "classificacao" else "Ajuste aos 273 pinguins de desenvolvimento com massa observada.")
    fig.text(.5, .015, nota, ha="center", fontsize=11.5, color=CINZENTO)
    fig.subplots_adjust(left=.01, right=.99, bottom=.06, top=.99)
    guardar(fig, f"u9_arvore_{tarefa}", f"Árvore de {tarefa} com profundidade máxima dois")


def regioes_classificacao(wisconsin):
    nos = ler_json("arvore_classificacao_completa.json")
    x = wisconsin["mean radius"].to_numpy()
    y = wisconsin["mean texture"].to_numpy()
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    t0, t1, t4 = nos[0]["limiar"], nos[1]["limiar"], nos[4]["limiar"]
    retangulos = {
        2: (xmin, t0, ymin, t1), 3: (xmin, t0, t1, ymax),
        5: (t0, xmax, ymin, t4), 6: (t0, xmax, t4, ymax),
    }
    cores = matplotlib.colors.LinearSegmentedColormap.from_list(
        "probabilidade", ["#F5F7F9", "#6D95B3"]
    )
    fig, ax = plt.subplots(figsize=(8.3, 5.6))
    for no, (xe, xd, yi, ys) in retangulos.items():
        ax.add_patch(Rectangle((xe, yi), xd - xe, ys - yi,
                               facecolor=cores(nos[no]["p_maligno"]), edgecolor="none", zorder=0))
    ax.plot([t0, t0], [ymin, ymax], color=AZUL, linewidth=1.7)
    ax.plot([xmin, t0], [t1, t1], color=AZUL, linewidth=1.7)
    ax.plot([t0, xmax], [t4, t4], color=AZUL, linewidth=1.7)
    for classe, cor, marcador, nome in [(0, TEAL, "o", "Benigno (0)"), (1, LARANJA, "^", "Maligno (1)")]:
        selecionadas = wisconsin["maligno"].to_numpy() == classe
        ax.scatter(x[selecionadas], y[selecionadas], s=21, color=cor, marker=marcador,
                   edgecolor="white", linewidth=.35, alpha=.88, zorder=4,
                   label=nome, clip_on=False)
    # A folha A tem muitos pontos: as probabilidades ficam fora do painel
    # para que nenhuma caixa de texto oculte observações.
    posicoes = {2: (7.65, 11.7), 3: (7.9, 36.3), 5: (24.1, 12.5), 6: (24.1, 35.7)}
    for no, (xi, yi) in posicoes.items():
        ax.text(xi, yi, nos[no]['folha'],
                fontsize=14, weight="bold", ha="center", va="center", color=AZUL, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93, "pad": 2.6})
    ax.set(xlabel="Raio médio (escala da fonte)", ylabel="Textura média (escala da fonte)",
           xlim=(xmin, xmax), ylim=(ymin, ymax))
    estilo(ax)
    barra = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cores), ax=ax,
                        fraction=.05, pad=.025, ticks=[0, .2, .4, .6, .8, 1])
    barra.set_label("Probabilidade na folha", fontsize=12)
    barra.ax.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: pt(valor, 1)))
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=12,
               title="Classe de referência", title_fontsize=11, bbox_to_anchor=(.48, 1.015))
    probabilidades = "      ".join(
        f"{nos[no]['folha']}: p = {pt(nos[no]['p_maligno'], 3)}" for no in [2, 3, 5, 6]
    )
    fig.text(.5, .025, probabilidades, ha="center", fontsize=11.5, color=AZUL)
    fig.subplots_adjust(left=.11, right=.91, top=.83, bottom=.24)
    guardar(fig, "u9_regioes_classificacao", "Regiões e probabilidades da árvore curta de classificação")


def complexidade():
    cls = pd.read_csv(RESULTADOS / "resumo_classificacao_medias_partes.csv").set_index("chave")
    reg = pd.read_csv(RESULTADOS / "resumo_regressao_medias_partes.csv").set_index("chave")
    fig, eixos = plt.subplots(1, 2, figsize=(10.2, 4.7))
    labels = ["Profundidade 2\nFolha mínima 5", "Profundidade 4\nFolha mínima 5", "Sem limite\nFolha mínima 1"]
    for ax, tabela, medida, titulo, limites in [
        (eixos[0], cls, "AUC", "a) Classificação", (.80, 1.025)),
        (eixos[1], reg, "MAE_g", "b) Regressão", (-15, 470)),
    ]:
        for origem, cor, marcador, estilo_linha in [("treino", AZUL, "o", "--"), ("validacao", LARANJA, "s", "-")]:
            valores = tabela.loc[CHAVES_ARVORES, f"{origem}_{medida}_media"].to_numpy()
            ax.plot(range(3), valores, marker=marcador, color=cor,
                    linestyle=estilo_linha, linewidth=1.8, markersize=7)
            for xi, valor in enumerate(valores):
                dy = 9 if origem == "treino" else -19
                if medida == "MAE_g":
                    dy = -20 if origem == "treino" and xi < 2 else 9
                ax.annotate(pt(valor, 3 if medida == "AUC" else 1), (xi, valor),
                            xytext=(0, dy), textcoords="offset points", ha="center", fontsize=12, color=cor,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95, "pad": 1.2})
        ax.set_title(titulo, loc="left", pad=12)
        ax.set_ylabel("AUC média" if medida == "AUC" else "MAE médio (g)")
        ax.set(xlim=(-.25, 2.25), ylim=limites)
        estilo(ax)
        ax.set_xticks(range(3), labels, fontsize=11)
        if medida == "AUC":
            ax.set_yticks([.8, .85, .9, .95, 1])
    fig.legend(handles=[
        Line2D([], [], color=AZUL, marker="o", linestyle="--", label="Treino"),
        Line2D([], [], color=LARANJA, marker="s", label="Validação"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(.5, .005))
    fig.subplots_adjust(left=.08, right=.99, top=.89, bottom=.25, wspace=.3)
    guardar(fig, "u9_complexidade", "Complexidade das árvores e resultados de treino e validação")


def bootstrap():
    dados = ler_json("bootstrap_ficticio.json")
    original = dados["treino_original"]
    cores = ["#D8EEEB", "#F8E5B9", "#D4E4F0", "#E8D8EC",
             "#E1E8CF", "#F0D5CB", "#D4E9E2", "#DEE0E6"]
    cor_letra = dict(zip(original, cores))
    fig, ax = plt.subplots(figsize=(10.0, 4.9))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ys = [.83, .61, .39, .17]
    linhas = [("Treino", original, "8 indivíduos distintos", "")]
    for a in dados["amostras"]:
        linhas.append((f"Árvore {a['arvore_ficticia']}", a["sorteios"],
                       f"{a['n_distintos']} indivíduos distintos", "Ausentes: " + ", ".join(a["ausentes"])))
    for yi, (nome, letras, resumo, ausentes) in zip(ys, linhas):
        ax.text(.015, yi, nome, va="center", ha="left", fontsize=15, color=AZUL)
        for j, letra in enumerate(letras):
            xi, largura = .165 + j * .064, .052
            ax.add_patch(FancyBboxPatch((xi, yi - .055), largura, .11,
                                       boxstyle="round,pad=0.005", linewidth=.8,
                                       facecolor=cor_letra[letra], edgecolor="#AAB4BD"))
            ax.text(xi + largura / 2, yi, letra, ha="center", va="center", fontsize=16, weight="bold")
        ax.text(.71, yi + .023, resumo, va="center", ha="left", fontsize=13.5)
        if ausentes:
            ax.text(.71, yi - .029, ausentes, va="center", ha="left", fontsize=12.5, color=CINZENTO)
    ax.text(.5, .99, "Um treino fictício; três amostras com reposição", ha="center", va="top", fontsize=15)
    fig.text(.5, .015, "Cada amostra tem oito posições. Repetir um registo não cria um novo indivíduo.",
             ha="center", fontsize=12, color=CINZENTO)
    fig.subplots_adjust(left=.015, right=.985, bottom=.05, top=.98)
    guardar(fig, "u10_bootstrap", "Amostragem com reposição em oito indivíduos fictícios")


def agregacao():
    fig, eixos = plt.subplots(1, 2, figsize=(10.0, 4.5))
    maximo_contagem = 0
    histogramas = []
    for tarefa, bins in [("classificacao", np.linspace(0, 1, 11)), ("regressao", np.arange(3000, 4301, 100))]:
        tabela = pd.read_csv(RESULTADOS / f"agregacao_{tarefa}_arvores.csv")
        contagens, limites = np.histogram(tabela["saida_individual"], bins=bins)
        assert contagens.sum() == 200
        maximo_contagem = max(maximo_contagem, int(contagens.max()))
        histogramas.append((tarefa, tabela["saida_individual"].to_numpy(), limites))
        pd.DataFrame({"limite_inferior": limites[:-1], "limite_superior": limites[1:],
                      "arvores": contagens}).to_csv(RESULTADOS / f"agregacao_{tarefa}_histograma.csv", index=False)
    for ax, (tarefa, valores, bins) in zip(eixos, histogramas):
        resumo = ler_json(f"agregacao_{tarefa}.json")
        media = resumo["previsao_floresta"]
        ax.hist(valores, bins=bins, color=TEAL, alpha=.8, edgecolor="white", linewidth=.7)
        ax.axvline(media, color=LARANJA, linewidth=2.1)
        texto_media = f"Média = {pt(media, 3)}" if tarefa == "classificacao" else f"Média = {pt(media, 1)} g"
        ax.text(.035, .96, texto_media, transform=ax.transAxes, va="top", fontsize=13, color=AZUL,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95, "pad": 2})
        ax.set_title("a) Wisconsin, registo 1" if tarefa == "classificacao" else "b) Pinguins, registo 1", loc="left", pad=13)
        ax.set_xlabel("Probabilidade de malignidade por árvore" if tarefa == "classificacao" else "Massa prevista por árvore (g)")
        ax.set_ylabel("Número de árvores")
        ax.set_ylim(0, (maximo_contagem // 10 + 1) * 10)
        ax.set_xlim(bins[0], bins[-1])
        estilo(ax)
    eixos[1].set_xticks([3000, 3400, 3800, 4200])
    fig.text(.5, .015, "Cada painel reúne 200 previsões para a mesma observação, excluída de todo o treino da floresta.",
             ha="center", fontsize=11, color=CINZENTO)
    fig.subplots_adjust(left=.075, right=.99, top=.88, bottom=.2, wspace=.30)
    guardar(fig, "u10_agregacao", "Saídas de 200 árvores e a média da floresta para duas observações de validação")


def comparacao():
    fig, eixos = plt.subplots(2, 1, figsize=(8.0, 7.8))
    nomes_comuns = ["Árvore profundidade 2", "Árvore profundidade 4", "Árvore sem limite",
                    "Floresta folha mínima 1", "Floresta folha mínima 5"]
    for ax, tarefa, primeira, nome, metrica, limites in [
        (eixos[0], "classificacao", "logistica", "Logística", "AUC", (.75, 1.0)),
        (eixos[1], "regressao", "linear", "Regressão linear", "MAE_g", (270, 480)),
    ]:
        tabela = pd.read_csv(RESULTADOS / f"metricas_{tarefa}_por_parte.csv")
        chaves = [primeira, *CHAVES_COMUNS]
        for yi, chave in enumerate(chaves):
            grupo = tabela[tabela["chave"] == chave].sort_values("parte")
            valores = grupo[f"validacao_{metrica}"].to_numpy()
            for j, (valor, cor, marcador) in enumerate(zip(valores, CORES_PARTES, MARCADORES)):
                ax.scatter(valor, yi + (j - 2) * .075, color=cor, marker=marcador,
                           s=35, edgecolor="white", linewidth=.45, zorder=3)
            ax.scatter(valores.mean(), yi, color="#171D23", marker="D", s=40,
                       edgecolor="white", linewidth=.45, zorder=4)
        ax.set(xlim=limites, ylim=(5.5, -.6))
        estilo(ax)
        ax.set_yticks(range(6), [nome, *nomes_comuns], fontsize=12)
        ax.set_xlabel("AUC de validação (maior é melhor)" if tarefa == "classificacao" else "MAE de validação em g (menor é melhor)")
        ax.set_title("a) Classificação" if tarefa == "classificacao" else "b) Regressão", loc="left", pad=10)
    handles = [Line2D([], [], color=cor, marker=marcador, linestyle="", markersize=6,
                      label=f"Parte {i}") for i, (cor, marcador) in enumerate(zip(CORES_PARTES, MARCADORES), 1)]
    handles.append(Line2D([], [], color="#171D23", marker="D", linestyle="", markersize=6, label="Média das 5 partes"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(.56, .005))
    fig.subplots_adjust(left=.32, right=.98, top=.95, bottom=.19, hspace=.65)
    guardar(fig, "u10_comparacao", "Comparação dos procedimentos nas mesmas cinco partes de validação")


def main():
    wisconsin, _ = validar()
    desenhar_arvore("classificacao")
    regioes_classificacao(wisconsin)
    desenhar_arvore("regressao")
    complexidade()
    bootstrap()
    agregacao()
    comparacao()
    print("Sete figuras geradas a partir dos resultados verificados; nenhum modelo foi reajustado.")


if __name__ == "__main__":
    main()
