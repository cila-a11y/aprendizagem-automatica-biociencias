"""Gerar três esquemas exatos do capítulo 12, em PDF vetorial e PNG.

As contagens da validação aninhada são determinadas pelo desenho indicado.
O esquema de grupos usa as divisões produzidas por GroupKFold no exemplo
fictício do capítulo. Não ajusta modelos nem consulta dados de teste.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="validacao_mpl_"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np
from sklearn.model_selection import GroupKFold


RAIZ = Path(__file__).resolve().parents[1]
FIGURAS = RAIZ / "figuras"
AZUL = "#244A6B"
LARANJA = "#D98B00"
TEAL = "#1E817F"
CINZENTO = "#78838C"
TEXTO = "#28343E"
CLARO_AZUL = "#EFF3F7"
CLARO_LARANJA = "#FFF3DC"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.labelcolor": TEXTO, "text.color": TEXTO,
    "axes.edgecolor": "#66717A", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False,
})


def guardar(fig, nome, titulo):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    (FIGURAS / "png").mkdir(exist_ok=True)
    for extensao, pasta in [("pdf", FIGURAS), ("png", FIGURAS / "png")]:
        fd, caminho = tempfile.mkstemp(
            prefix=f".{nome}-", suffix=f".tmp.{extensao}", dir=pasta
        )
        temporario = Path(caminho)
        try:
            with os.fdopen(fd, "wb") as stream:
                extras = ({"metadata": {
                    "Title": titulo,
                    "Subject": "Esquema científico reproduzível do capítulo 12",
                    "CreationDate": None, "ModDate": None,
                }} if extensao == "pdf" else {"dpi": 180})
                fig.savefig(stream, format=extensao, bbox_inches="tight", **extras)
                stream.flush()
                os.fsync(stream.fileno())
            conteudo = temporario.read_bytes()
            if extensao == "pdf" and not conteudo.rstrip().endswith(b"%%EOF"):
                raise OSError("O PDF não foi escrito integralmente.")
            if extensao == "png" and not conteudo.endswith(b"IEND\xaeB`\x82"):
                raise OSError("O PNG não foi escrito integralmente.")
            temporario.replace(pasta / f"{nome}.{extensao}")
        finally:
            if temporario.exists():
                temporario.unlink()
    plt.close(fig)


def caixa(ax, x, y, w, h, texto, cor=AZUL, fundo=CLARO_AZUL, tamanho=13):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.009",
        facecolor=fundo, edgecolor=cor, linewidth=1.3,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, texto,
            ha="center", va="center", fontsize=tamanho, linespacing=1.25)


def seta(ax, inicio, fim, cor=CINZENTO, largura=1.3):
    ax.add_patch(FancyArrowPatch(
        inicio, fim, arrowstyle="-|>", mutation_scale=13,
        color=cor, linewidth=largura, shrinkA=2, shrinkB=3,
    ))


def validacao_aninhada():
    assert 455 == 364 + 91
    assert 364 == 4 * 91
    assert 273 == 3 * 91
    assert 3 * 4 == 12
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.02, .975, "Uma ronda da validação aninhada", fontsize=15,
            weight="bold", va="top")
    caixa(ax, .20, .867, .60, .07, "455 amostras de desenvolvimento",
          cor=CINZENTO, fundo="#F6F7F8", tamanho=14)
    seta(ax, (.35, .861), (.325, .812))
    seta(ax, (.65, .861), (.84, .812))
    caixa(ax, .025, .732, .60, .075, "Treino exterior: 364 amostras",
          tamanho=14)
    caixa(ax, .705, .732, .27, .075, "Validação exterior\n91 amostras",
          cor=LARANJA, fundo=CLARO_LARANJA, tamanho=13.5)

    ax.text(.325, .690, "Escolha interior entre C = 0,1; 1; 10",
            ha="center", fontsize=13.5, weight="bold")
    esquerda, passo, largura, altura = .095, .133, .125, .041
    y_base = .605
    for k in range(4):
        ax.text(esquerda + k * passo + largura / 2, .658,
                f"Parte {k + 1}", ha="center", fontsize=11.5)
    for r in range(4):
        y = y_base - .053 * r
        ax.text(.075, y + altura / 2, str(r + 1), ha="right", va="center",
                fontsize=12)
        for k in range(4):
            valida = k == r
            cor = LARANJA if valida else AZUL
            x = esquerda + k * passo
            ax.add_patch(Rectangle((x, y), largura, altura, facecolor=cor,
                                   edgecolor="white", linewidth=.6))
            ax.text(x + largura / 2, y + altura / 2, "91",
                    ha="center", va="center", fontsize=12,
                    color=TEXTO if valida else "white")
    ax.text(.035, .552, "Ronda", rotation=90, va="center", ha="center",
            fontsize=11.5)
    ax.text(.325, .409, "Em cada ronda: 273 no treino, 91 na validação",
            ha="center", fontsize=12)
    ax.text(.325, .373, "3 valores de C × 4 rondas = 12 ajustamentos",
            ha="center", fontsize=12.5)
    ax.text(.84, .620, "Não participa\nna escolha de C\nnem no ajustamento",
            ha="center", va="center", fontsize=12.5, linespacing=1.4)

    seta(ax, (.325, .357), (.325, .318))
    caixa(ax, .025, .245, .60, .066,
          "Escolher C pela AUC média interior", tamanho=13)
    seta(ax, (.325, .239), (.325, .202))
    caixa(ax, .025, .115, .60, .081,
          "Reajustar a configuração escolhida\nnas 364 amostras do treino exterior",
          tamanho=13)
    caixa(ax, .705, .115, .27, .081,
          "Prever e avaliar\nas 91 exteriores", cor=LARANJA,
          fundo=CLARO_LARANJA, tamanho=13)
    seta(ax, (.633, .155), (.699, .155), cor=AZUL)
    seta(ax, (.84, .725), (.84, .204), cor=LARANJA)
    # Interromper a seta vertical no texto evita uma linha através das letras.
    ax.add_patch(Rectangle((.731, .535), .218, .17,
                           facecolor="white", edgecolor="none", zorder=4))
    ax.text(.84, .620, "Não participa\nna escolha de C\nnem no ajustamento",
            ha="center", va="center", fontsize=12.5, linespacing=1.4, zorder=5)
    ax.text(.5, .060, "Repetir nas cinco rondas exteriores.", ha="center", fontsize=12.5)
    ax.text(.5, .019, "As 114 amostras de teste reservado não entram neste esquema.",
            ha="center", fontsize=12.5, color=CINZENTO)
    fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.99)
    guardar(fig, "u12_validacao_aninhada", "Uma ronda exterior com seleção interior de C")


def grupos():
    ids = np.repeat(np.array(list("ABCDEF")), 2)
    medidas = np.array([2.1, 2.3, 5.2, 5.0, 2.8, 2.9,
                        4.8, 4.6, 3.0, 2.7, 5.5, 5.4]).reshape(-1, 1)
    classes = np.repeat([0, 1, 0, 1, 0, 1], 2)
    divisoes = list(GroupKFold(n_splits=3).split(medidas, classes, groups=ids))
    coberturas = np.zeros(12, dtype=int)
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.set(xlim=(-.5, 11.5), ylim=(-.5, 2.5))
    for ronda, (treino, validacao) in enumerate(divisoes):
        assert set(ids[treino]).isdisjoint(ids[validacao])
        coberturas[validacao] += 1
        for i in range(12):
            valida = i in validacao
            ax.add_patch(Rectangle((i - .5, ronda - .42), 1, .84,
                                   facecolor=LARANJA if valida else AZUL,
                                   edgecolor="white", linewidth=1))
            ax.text(i, ronda, "V" if valida else "T", ha="center", va="center",
                    color=TEXTO if valida else "white", fontsize=13)
        print(f"Ronda {ronda + 1}: validação dos doentes "
              + ", ".join(sorted(set(ids[validacao]))))
    assert np.all(coberturas == 1)
    for x in [1.5, 3.5, 5.5, 7.5, 9.5]:
        ax.axvline(x, color="white", linewidth=3)
    ax.invert_yaxis()
    ax.set_yticks(range(3), [f"Ronda {r + 1}" for r in range(3)])
    ax.set_xticks(range(12), [str(i) for i in range(1, 13)])
    ax.set_xlabel("Linhas do exemplo", fontsize=13, labelpad=10)
    ax.tick_params(axis="both", length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    topo = ax.secondary_xaxis("top")
    topo.set_xticks(np.arange(.5, 12, 2), list("ABCDEF"))
    topo.set_xlabel("Doente", fontsize=13, labelpad=8)
    topo.tick_params(length=0, pad=6)
    topo.spines["top"].set_visible(False)
    ax.legend(handles=[Patch(facecolor=AZUL, label="T: treino"),
                       Patch(facecolor=LARANJA, label="V: validação")],
              loc="upper center", bbox_to_anchor=(.5, -.38), ncol=2,
              frameon=False, fontsize=13)
    fig.text(.51, .020,
             "GroupKFold com três partes: separação por doente, sem estratificação.",
             ha="center", fontsize=12, color=CINZENTO)
    fig.subplots_adjust(left=.13, right=.98, top=.77, bottom=.35)
    guardar(fig, "u12_grupos", "Divisões reais de GroupKFold para seis doentes fictícios")


def tempo():
    origens = [60, 80, 100]
    respostas = [90, 110, 130]
    instante_ajuste = 100
    assert respostas[0] <= instante_ajuste < respostas[1]
    assert all(fim - inicio == 30 for inicio, fim in zip(origens, respostas))
    fig = plt.figure(figsize=(9.5, 4.7))
    ax = fig.add_axes([.17, .28, .55, .54])
    ax.set(xlim=(55, 136), ylim=(-.5, 2.55))
    cores = [AZUL, LARANJA, TEAL]
    for inicio, fim, y, cor in zip(origens, respostas, [2, 1, 0], cores):
        ax.plot([inicio, fim], [y, y], color=cor, linewidth=5, solid_capstyle="butt")
        ax.scatter([inicio], [y], facecolor="white", edgecolor=cor,
                   s=90, linewidth=2, zorder=5)
        ax.scatter([fim], [y], facecolor=cor, edgecolor="white",
                   s=95, linewidth=.6, zorder=5)
        ax.text((inicio + fim) / 2, y + .19, "30 dias", ha="center", fontsize=12.5)
    ax.axvline(instante_ajuste, color=CINZENTO, linestyle="--", linewidth=1.4, zorder=1)
    ax.text(100, 2.52, "Ajustamento\ndia 100", ha="center", va="bottom",
            fontsize=12.5, color=TEXTO)
    ax.set_xticks([60, 80, 90, 100, 110, 130])
    ax.set_yticks([2, 1, 0], ["Origem\ndia 60", "Origem\ndia 80", "Previsão\ndia 100"])
    ax.set_xlabel("Dia", fontsize=13, labelpad=10)
    ax.tick_params(axis="x", length=3, labelsize=12)
    ax.tick_params(axis="y", length=0, labelsize=13, pad=12)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(CINZENTO)
    for x in [60, 80, 90, 110, 130]:
        ax.axvline(x, color="#E7ECEF", linewidth=.6, zorder=0)
    estados = [
        "Conhecida no dia 90\nPode entrar no treino",
        "Conhecida no dia 110\nAinda não pode entrar\nno treino do dia 100",
        "Conhecida no dia 130\nPermite avaliar depois",
    ]
    for y, estado, cor in zip([2, 1, 0], estados, cores):
        xy = ax.transData.transform((0, y))
        altura = fig.transFigure.inverted().transform(xy)[1]
        fig.text(.745, altura, estado, ha="left", va="center",
                 fontsize=12.5, linespacing=1.3, color=cor)
    fig.text(.5, .96, "Prever uma ocorrência nos 30 dias seguintes",
             ha="center", fontsize=15, weight="bold")
    fig.text(.5, .105,
             "Neste exemplo, cada resposta fica disponível no fim da sua janela.",
             ha="center", fontsize=12.5)
    fig.text(.5, .035,
             "Círculo aberto: início excluído. Círculo cheio: fim incluído e resposta conhecida.",
             ha="center", fontsize=11.5, color=CINZENTO)
    guardar(fig, "u12_tempo", "Disponibilidade da resposta e previsão a trinta dias")


if __name__ == "__main__":
    validacao_aninhada()
    grupos()
    tempo()
    print("Três figuras do capítulo 12 produzidas em PDF vetorial e PNG.")
