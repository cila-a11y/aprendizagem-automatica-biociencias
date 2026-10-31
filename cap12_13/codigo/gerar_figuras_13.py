"""Quatro figuras científicas do capítulo 13, sem voltar a ajustar modelos.

Executar depois de gerar_resultados.py. Os PDFs são vetoriais; as cópias PNG
permitem consulta rápida. Os pontos e as permutações provêm dos resultados
guardados. O deslocamento horizontal nos erros por espécie serve apenas
para tornar visíveis as observações; não altera nenhum erro de previsão.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="interpretacao_mpl_"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"
TEAL, LARANJA, AZUL, ROXO = "#1E817F", "#D98B00", "#244A6B", "#9A62A1"
CINZENTO, ESCURO = "#78838C", "#28343E"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#66717A", "axes.labelcolor": ESCURO,
    "text.color": ESCURO, "pdf.fonttype": 42,
    "savefig.facecolor": "white", "axes.axisbelow": True,
})


def formato_pt(valor, _=None):
    return f"{valor:g}".replace(".", ",")


def estilo(ax, eixo="both"):
    ax.grid(axis=eixo, color="#E5E9EC", linewidth=.6)
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
                          "Subject": "Figura científica reproduzível do capítulo 13",
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


def calibracao():
    dados = pd.read_csv(RESULTADOS / "calibracao_faixas.csv")
    dados = dados.loc[dados["versao"] == "Logistica"].copy()
    assert dados["n"].sum() == 455 and list(dados["n"]) == [235, 46, 36, 26, 112]
    fig, (ax, barras) = plt.subplots(1, 2, figsize=(9.9, 4.9), gridspec_kw={"width_ratios": [1.05, 1]})
    ax.plot([0, 1], [0, 1], color=CINZENTO, linestyle="--", linewidth=1.3, zorder=1)
    ax.scatter(dados["probabilidade_media"], dados["fracao_maligna"],
               s=90, color=TEAL, edgecolor="white", linewidth=.8, zorder=3)
    deslocamentos = [(10, 8), (10, 8), (10, -14), (10, -13), (-25, -17)]
    for (_, row), deslocamento in zip(dados.iterrows(), deslocamentos):
        ax.annotate(f"F{row['faixa']}", (row["probabilidade_media"], row["fracao_maligna"]),
                    xytext=deslocamento, textcoords="offset points", fontsize=11.5,
                    color=TEAL, weight="bold")
    ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025),
           xlabel="Probabilidade média prevista", ylabel="Fração observada de malignos")
    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("A  Probabilidades e frequências", loc="left", pad=13)
    estilo(ax)

    y = np.arange(5)
    barras.barh(y, dados["n"], color=TEAL, height=.60, alpha=.75)
    for pos, n in zip(y, dados["n"]):
        barras.text(n + 5, pos, str(n), va="center", ha="left", fontsize=12)
    barras.set(xlim=(0, 275), xlabel="Número de amostras", ylim=(4.6, -.6))
    barras.set_title("B  Dimensão de cada faixa", loc="left", pad=13)
    estilo(barras, "x")
    barras.set_yticks(y, ["F1  [0; 0,2[", "F2  [0,2; 0,4[", "F3  [0,4; 0,6[",
                           "F4  [0,6; 0,8[", "F5  [0,8; 1]"])
    barras.set_xticks([0, 50, 100, 150, 200, 250])
    barras.spines["left"].set_visible(False)
    barras.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=.08, right=.98, bottom=.21, top=.86, wspace=.58)
    fig.text(.5, .04, "455 previsões fora do respetivo treino; cinco faixas fixas, sem unir os pontos.",
             ha="center", fontsize=11.5, color=CINZENTO)
    guardar(fig, "u13_calibracao", "Calibração descritiva e contagem das cinco faixas de probabilidade")


def erros_especies():
    dados = pd.read_csv(RESULTADOS / "regressao_previsoes_reunidas.csv")
    resumo = pd.read_csv(RESULTADOS / "regressao_erros_especies.csv").set_index("especie")
    assert len(dados) == 273
    especies = ["Adelie", "Chinstrap", "Gentoo"]
    fig, ax = plt.subplots(figsize=(8.9, 5.5))
    rng = np.random.default_rng(2026)
    ax.axhline(0, color=ESCURO, linestyle="--", linewidth=1.2, zorder=2)
    for j, (especie, cor) in enumerate(zip(especies, [TEAL, LARANJA, AZUL])):
        erros = dados.loc[dados["species"] == especie, "erro_linear_g"].to_numpy()
        deslocamentos = rng.uniform(-.19, .19, len(erros))
        ax.scatter(j + deslocamentos, erros, color=cor, alpha=.55, s=26,
                   edgecolor="white", linewidth=.4, zorder=3)
        media = resumo.loc[especie, "erro_medio_assinado_g"]
        assert np.isclose(erros.mean(), media)
        ax.plot([j - .24, j + .24], [media, media], color=ESCURO, linewidth=1.5, zorder=4)
        ax.scatter([j], [media], color=ESCURO, edgecolor="white", linewidth=.9,
                   marker="D", s=85, zorder=5)
    limite_baixo = np.floor(dados["erro_linear_g"].min() / 250) * 250
    limite_alto = np.ceil(dados["erro_linear_g"].max() / 250) * 250
    ax.set(xlim=(-.65, 2.65), ylim=(limite_baixo - 90, limite_alto + 90),
           ylabel="Erro de previsão: observado - previsto (g)")
    estilo(ax, "y")
    ax.set_xticks(range(3), [f"{e}\nn = {int(resumo.loc[e, 'n'])}" for e in especies])
    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.set_title("Erros fora do treino, por espécie", loc="left", pad=15)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=CINZENTO, markersize=6, label="Uma amostra"),
        Line2D([], [], marker="D", linestyle="", color=ESCURO, markersize=7, label="Erro médio da espécie"),
    ], loc="upper center", bbox_to_anchor=(.5, 1.025), frameon=False, ncol=2, fontsize=11)
    fig.subplots_adjust(left=.13, right=.98, bottom=.20, top=.87)
    fig.text(.5, .04, "273 pinguins; a espécie apenas organiza a leitura dos erros, não foi usada como preditor.",
             ha="center", fontsize=11, color=CINZENTO)
    guardar(fig, "u13_erros_especies", "Erros de previsão da regressão linear, agrupados por espécie")


def permutacao():
    dados = pd.read_csv(RESULTADOS / "permutacao_repeticoes.csv")
    resumo = pd.read_csv(RESULTADOS / "permutacao_resumo.csv").set_index("variavel")
    assert len(dados) == 40
    fig, ax = plt.subplots(figsize=(8.9, 4.0))
    rng = np.random.default_rng(2026)
    ax.axvline(0, color=CINZENTO, linewidth=1.2, linestyle="--", zorder=1)
    for pos, var, cor in [(1, "mean radius", TEAL), (0, "mean texture", LARANJA)]:
        quedas = dados.loc[dados["variavel"] == var, "diminuicao_AUC"].to_numpy()
        assert len(quedas) == 20
        ax.scatter(quedas, pos + rng.uniform(-.16, .16, len(quedas)),
                   s=38, color=cor, edgecolor="white", linewidth=.5, alpha=.85, zorder=3)
        media = resumo.loc[var, "diminuicao_media_AUC"]
        ax.plot([media, media], [pos - .24, pos + .24], color=ESCURO, linewidth=1.2, zorder=4)
        ax.scatter([media], [pos], marker="D", s=88, color=ESCURO,
                   edgecolor="white", linewidth=.9, zorder=5)
    ax.set(xlim=(-.015, dados["diminuicao_AUC"].max() + .04), ylim=(-.52, 1.65),
           xlabel="Queda de AUC após a permutação")
    estilo(ax, "x")
    ax.set_yticks([0, 1], ["Textura média", "Raio médio"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Vinte permutações de cada variável", loc="left", pad=15)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=CINZENTO, markersize=6, label="Uma permutação"),
        Line2D([], [], marker="D", linestyle="", color=ESCURO, markersize=7, label="Média das 20 permutações"),
    ], loc="upper center", bbox_to_anchor=(.53, 1.015), frameon=False, ncol=2, fontsize=11)
    auc = float(dados["AUC_original"].iloc[0])
    fig.text(.5, .035, f"Floresta fixa; mesmas 137 amostras; AUC original = {auc:.3f}.".replace(".", ",", 1),
             ha="center", fontsize=11.5, color=CINZENTO)
    fig.subplots_adjust(left=.18, right=.98, bottom=.23, top=.83)
    guardar(fig, "u13_permutacao", "Quarenta diminuições de AUC, vinte permutações por variável")


def dependencia():
    dados = pd.read_csv(RESULTADOS / "dependencia_geometria.csv")
    paineis = ["Original", "Permutacao individual", "Permutacao conjunta"]
    titulos = ["A  Valores originais", "B  Só o raio permutado", "C  O par permutado junto"]
    fig, eixos = plt.subplots(1, 3, figsize=(10.5, 4.4), sharex=True, sharey=True)
    for ax, painel, titulo in zip(eixos, paineis, titulos):
        bloco = dados.loc[dados["painel"] == painel]
        assert len(bloco) == 137
        for classe, cor, simbolo in [(0, TEAL, "o"), (1, LARANJA, "^")]:
            selecao = bloco["maligno_linha_original"] == classe
            ax.scatter(bloco.loc[selecao, "raio_medio"], bloco.loc[selecao, "perimetro_medio"],
                       color=cor, marker=simbolo, s=23, alpha=.85, edgecolor="white", linewidth=.4)
        estilo(ax)
        ax.set_title(titulo, fontsize=12, pad=10)
        ax.set_xlabel("Raio médio")
        ax.set_xticks([10, 15, 20, 25])
    eixos[0].set_ylabel("Perímetro médio")
    original = dados.loc[dados["painel"] == "Original", ["raio_medio", "perimetro_medio"]].sort_values(["raio_medio", "perimetro_medio"])
    junto = dados.loc[dados["painel"] == "Permutacao conjunta", ["raio_medio", "perimetro_medio"]].sort_values(["raio_medio", "perimetro_medio"])
    assert np.allclose(original.to_numpy(), junto.to_numpy())
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=TEAL, markersize=6, label="Referência benigna (0)"),
        Line2D([], [], marker="^", linestyle="", color=LARANJA, markersize=6, label="Referência maligna (1)"),
    ], loc="upper center", bbox_to_anchor=(.52, 1.025), frameon=False, ncol=2, fontsize=11.5)
    fig.text(.5, .07, "As cores e os símbolos mantêm a resposta da linha original nos três painéis.",
             ha="center", fontsize=11, color=CINZENTO)
    fig.text(.5, .015, "Mesmas 137 amostras; medidas na escala da fonte; mesma permutação em B e C.",
             ha="center", fontsize=11, color=CINZENTO)
    fig.subplots_adjust(left=.075, right=.985, bottom=.24, top=.82, wspace=.17)
    guardar(fig, "u13_dependencia", "Permutação individual e conjunta de duas medidas relacionadas")


def main():
    calibracao()
    erros_especies()
    permutacao()
    dependencia()
    print("Quatro figuras do capítulo 13 guardadas em PDF vetorial e PNG; não foram reajustados modelos.")


if __name__ == "__main__":
    main()
