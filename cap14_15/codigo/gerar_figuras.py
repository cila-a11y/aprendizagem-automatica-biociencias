"""Gera as duas figuras do capítulo 14 a partir das previsões já guardadas.

Este programa não ajusta modelos, não seleciona métodos nem procura limiares.
Lê apenas resultados da avaliação final, verifica as contas usadas nos
gráficos e escreve versões PDF vetoriais e PNG. Executar a partir de qualquer
pasta: python codigo/gerar_figuras.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

# Uma cache privada evita interferência com outros processos de composição.
_CACHE = tempfile.TemporaryDirectory(prefix="cap14_matplotlib_")
os.environ["MPLCONFIGDIR"] = _CACHE.name

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"
AZUL = "#244A6B"
VERDE = "#1E817F"
LARANJA = "#D98B00"
CINZENTO = "#78838C"
ESCURO = "#28343E"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,
    "text.color": ESCURO,
    "axes.labelcolor": ESCURO,
    "axes.edgecolor": CINZENTO,
    "xtick.color": ESCURO,
    "ytick.color": ESCURO,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

CONTROLOS: list[str] = []


def conferir(condicao: bool, descricao: str) -> None:
    if not bool(condicao):
        raise AssertionError(descricao)
    CONTROLOS.append(descricao)


def conferir_proximo(a, b, descricao: str) -> None:
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, err_msg=descricao)
    CONTROLOS.append(descricao)


def decimal_pt(valor, _pos=None) -> str:
    return f"{valor:g}".replace(".", ",")


def milhar_pt(valor, _pos=None) -> str:
    return f"{valor:,.0f}".replace(",", "\u2009")


def sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def guardar(fig, nome: str, assunto: str) -> list[dict]:
    """Fecha cada ficheiro antes de o tornar visível pelo nome definitivo."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    saidas = []
    for formato in ("pdf", "png"):
        destino = FIGURAS / f"{nome}.{formato}"
        fd, temporario = tempfile.mkstemp(
            prefix=f".{nome}_", suffix=f".tmp.{formato}", dir=FIGURAS
        )
        tmp = Path(temporario)
        try:
            opcoes = {"dpi": 200} if formato == "png" else {"metadata": {
                "Title": nome,
                "Author": "Material de apoio ao livro",
                "Subject": assunto,
                "CreationDate": None,
                "ModDate": None,
            }}
            with os.fdopen(fd, "wb") as fluxo:
                fig.savefig(fluxo, format=formato, **opcoes)
                fluxo.flush()
                os.fsync(fluxo.fileno())
            conteudo = tmp.read_bytes()
            if formato == "pdf":
                conferir(conteudo.startswith(b"%PDF-") and
                         conteudo.rstrip().endswith(b"%%EOF"),
                         f"{destino.name}: cabeçalho e fecho PDF completos")
            else:
                conferir(conteudo.startswith(b"\x89PNG\r\n\x1a\n") and
                         conteudo.endswith(b"IEND\xaeB`\x82"),
                         f"{destino.name}: cabeçalho e fecho PNG completos")
            tmp.replace(destino)
            saidas.append({"ficheiro": str(destino.relative_to(RAIZ)),
                           "bytes": destino.stat().st_size,
                           "sha256": sha256(destino),
                           "largura_polegadas": float(fig.get_size_inches()[0]),
                           "altura_polegadas": float(fig.get_size_inches()[1])})
        finally:
            if tmp.exists():
                tmp.unlink()
    plt.close(fig)
    return saidas


def classificacao() -> tuple[list[dict], dict]:
    previsoes = pd.read_csv(RESULTADOS / "classificacao_teste_previsoes.csv")
    roc = pd.read_csv(RESULTADOS / "classificacao_teste_roc.csv")
    medidas = pd.read_csv(RESULTADOS / "classificacao_teste_medidas.csv")
    linha = medidas.loc[medidas.procedimento.eq("logistica")].iloc[0]
    y = previsoes.maligno.to_numpy(dtype=int)
    p = previsoes.p_logistica.to_numpy(dtype=float)
    decisao = previsoes.decisao_logistica.to_numpy(dtype=int)
    conferir(len(y) == 114 and sum(y == 1) == 42 and sum(y == 0) == 72,
             "Classificação: 114 registos, 42 malignos e 72 benignos")
    conferir(np.array_equal(decisao, (p >= .20).astype(int)),
             "Todas as decisões representadas aplicam o limiar fixado 0,20")
    matriz = np.array([[np.sum((y == r) & (decisao == c)) for c in (0, 1)]
                       for r in (0, 1)])
    conferir(np.array_equal(matriz, [[54, 18], [2, 40]]),
             "Matriz calculada das previsões igual a [[54,18],[2,40]]")

    # Conferência independente de todos os vértices da ROC, incluindo empates.
    limiares = np.r_[np.inf, np.unique(p)[::-1]]
    fpr = np.array([np.mean(p[y == 0] >= t) for t in limiares])
    tpr = np.array([np.mean(p[y == 1] >= t) for t in limiares])
    conferir_proximo(roc.limiar, limiares, "ROC: todos os limiares observados")
    conferir_proximo(roc.taxa_falsos_positivos, fpr, "ROC: abscissas empíricas")
    conferir_proximo(roc.sensibilidade, tpr, "ROC: ordenadas empíricas")
    positivos, negativos = p[y == 1], p[y == 0]
    auc = float(np.mean((positivos[:, None] > negativos[None, :]) +
                       .5 * (positivos[:, None] == negativos[None, :])))
    conferir_proximo(auc, linha.AUC, "AUC conferida pelas comparações entre pares")
    conferir(f"{auc:.3f}" == "0.946", "AUC arredondada da figura é 0,946")
    fpr_fixo = matriz[0, 1] / matriz[0].sum()
    tpr_fixo = matriz[1, 1] / matriz[1].sum()
    conferir_proximo([fpr_fixo, tpr_fixo], [.25, 40 / 42],
                     "Ponto do limiar 0,20: FPR 0,25 e sensibilidade 40/42")

    fig, axs = plt.subplots(1, 2, figsize=(11.2, 5.7),
                            gridspec_kw={"width_ratios": [.96, 1]})
    fig.subplots_adjust(left=.15, right=.982, bottom=.20, top=.86, wspace=.42)
    ax = axs[0]
    # A cor diferencia decisões certas e erradas; as contagens são explícitas.
    ax.set(xlim=(-.5, 1.5), ylim=(1.5, -.5), aspect="equal")
    nomes = [["VN", "FP"], ["FN", "VP"]]
    for r in (0, 1):
        for c in (0, 1):
            cor = VERDE if r == c else "#965900"
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1,
                                    facecolor="#E1EFEE" if r == c else "#FCEDD5",
                                    edgecolor="white", linewidth=3))
            ax.text(c, r - .10, str(matriz[r, c]), ha="center", va="center",
                    fontsize=30, fontweight="bold", color=cor)
            ax.text(c, r + .20, nomes[r][c], ha="center", va="center",
                    fontsize=14, color=ESCURO)
    ax.set_xticks([0, 1], ["Benigno (0)", "Maligno (1)"])
    ax.set_yticks([0, 1], ["Benigno (0)", "Maligno (1)"])
    ax.set_xticks([-.5, .5, 1.5], minor=True)
    ax.set_yticks([-.5, .5, 1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Decisão do modelo", labelpad=12)
    ax.set_ylabel("Referência", labelpad=10)
    ax.set_title("A  Decisões com limiar 0,20", loc="left", pad=15, color=AZUL)

    ax = axs[1]
    ax.plot([0, 1], [0, 1], color=CINZENTO, linestyle=(0, (4, 3)), linewidth=1.5)
    # Ligam-se os vértices observados, sem alisamento nem supressão de pontos.
    ax.plot(fpr, tpr, color=VERDE, linewidth=2.5, zorder=3)
    ax.scatter([fpr_fixo], [tpr_fixo], color=LARANJA, edgecolor="white",
               linewidth=1.2, s=105, zorder=4)
    ax.annotate("Limiar fixado 0,20\nSensibilidade 40/42",
                xy=(fpr_fixo, tpr_fixo), xytext=(.36, .55),
                ha="left", va="center", fontsize=12.5,
                arrowprops={"arrowstyle": "->", "color": ESCURO,
                            "linewidth": 1.1, "connectionstyle": "arc3,rad=-.18"},
                bbox={"boxstyle": "round,pad=.32", "fc": "white", "ec": "none"})
    ax.text(.98, .08, "AUC = 0,946", ha="right", va="bottom", color=AZUL,
            fontsize=16, fontweight="bold")
    ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), aspect="equal")
    ax.set_xlabel("Taxa de falsos positivos", labelpad=10)
    ax.set_ylabel("Sensibilidade", labelpad=8)
    ax.set_title("B  Curva ROC no teste", loc="left", pad=15, color=AZUL)
    for eixo in (ax.xaxis, ax.yaxis):
        eixo.set_major_locator(MultipleLocator(.2))
        eixo.set_major_formatter(FuncFormatter(decimal_pt))
    ax.grid(alpha=.14, linewidth=.7)
    ax.set_axisbelow(True)
    fig.text(.54, .048,
             "114 amostras de teste; modelo e limiar escolhidos no desenvolvimento.",
             ha="center", va="center", fontsize=13, color=ESCURO)
    saidas = guardar(fig, "u14_classificacao",
                     "Matriz de confusão e ROC empírica do teste; limiar fixado 0,20.")
    return saidas, {"n": len(y), "matriz": matriz.tolist(), "AUC": auc,
                    "vertices_ROC": len(fpr), "limiar_fixo": .2,
                    "FPR_limiar_fixo": fpr_fixo,
                    "sensibilidade_limiar_fixo": tpr_fixo}


def ecdf_exata(valores: np.ndarray, xmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Coordenadas de uma função de distribuição empírica direita-contínua.

    Cada valor distinto produz um salto com a multiplicidade observada.
    A duplicação eventual de x=0 permite representar corretamente erros nulos.
    """
    unicos, multiplicidades = np.unique(valores, return_counts=True)
    x = np.r_[0., unicos, xmax]
    y = np.r_[0., np.cumsum(multiplicidades) / len(valores), 1.]
    return x, y


def regressao() -> tuple[list[dict], dict]:
    d = pd.read_csv(RESULTADOS / "regressao_teste_previsoes.csv")
    conferir(len(d) == 69 and d.id_registo.is_unique,
             "Regressão: os mesmos 69 registos em ambos os painéis")
    observado = d.body_mass_g.to_numpy(dtype=float)
    previsto = d.previsto_floresta_g.to_numpy(dtype=float)
    referencia = d.previsto_referencia_g.to_numpy(dtype=float)
    erro_f = d.erro_absoluto_floresta_g.to_numpy(dtype=float)
    erro_r = d.erro_absoluto_referencia_g.to_numpy(dtype=float)
    conferir_proximo(referencia, np.full(69, 4050.), "Referência constante de 4050 g")
    # Os CSV têm 15 algarismos significativos; admite-se só erro de arredondamento.
    np.testing.assert_allclose(erro_f, np.abs(observado - previsto), rtol=0, atol=1e-8)
    CONTROLOS.append("Erros absolutos da floresta conferidos com as massas e previsões")
    conferir_proximo(erro_r, np.abs(observado - referencia),
                     "Erros absolutos da referência conferidos")
    nf, nr = int(sum(erro_f <= 500)), int(sum(erro_r <= 500))
    conferir((nf, nr) == (60, 29), "Até 500 g: floresta 60/69; referência 29/69")
    i_maior = int(np.argmax(erro_f))
    conferir(int(d.iloc[i_maior].id_registo) == 262,
             "Maior erro da floresta no teste corresponde ao registo 262")
    # Prolongar o eixo torna visível também o último salto até à fração 1.
    xmax = float(math.ceil(max(erro_f.max(), erro_r.max()) / 250) * 250 + 250)
    xf, yf = ecdf_exata(erro_f, xmax)
    xr, yr = ecdf_exata(erro_r, xmax)
    # Verifica a avaliação direita-contínua, incluindo o ponto de leitura 500 g.
    for nome, erros, x, y in [("floresta", erro_f, xf, yf),
                              ("referência", erro_r, xr, yr)]:
        conferir(np.all(np.diff(x) >= 0) and np.all(np.diff(y) >= 0),
                 f"ECDF {nome}: degraus não decrescentes, sem alisamento")
        for valor in np.r_[np.unique(erros), 500.]:
            indice = np.searchsorted(x, valor, side="right") - 1
            conferir_proximo(y[indice], np.mean(erros <= valor),
                             f"ECDF {nome} avaliada corretamente em {valor:.10g} g")

    fig, axs = plt.subplots(1, 2, figsize=(11.2, 5.7))
    fig.subplots_adjust(left=.09, right=.97, bottom=.20, top=.87, wspace=.32)
    ax = axs[0]
    inferior = math.floor(min(observado.min(), previsto.min()) / 500) * 500
    superior = math.ceil(max(observado.max(), previsto.max()) / 500) * 500
    ax.plot([inferior, superior], [inferior, superior], color=CINZENTO,
            linestyle=(0, (4, 3)), linewidth=1.6, label="Previsão = observação")
    ax.scatter(observado, previsto, s=38, facecolor=VERDE, edgecolor="white",
               linewidth=.5, alpha=.88, zorder=3)
    ax.scatter([observado[i_maior]], [previsto[i_maior]], s=72,
               color=LARANJA, edgecolor="white", linewidth=.8, zorder=4)
    ax.annotate("Registo 262", xy=(observado[i_maior], previsto[i_maior]),
                xytext=(observado[i_maior] - 60, previsto[i_maior] - 590),
                ha="center", va="center", fontsize=12.5,
                arrowprops={"arrowstyle": "->", "color": ESCURO,
                            "linewidth": 1.1},
                bbox={"boxstyle": "round,pad=.22", "fc": "white", "ec": "none"})
    ax.set(xlim=(inferior - 100, superior + 100),
           ylim=(inferior - 100, superior + 100), aspect="equal")
    ax.set_xlabel("Massa observada (g)", labelpad=9)
    ax.set_ylabel("Massa prevista pela floresta (g)", labelpad=9)
    ax.set_title("A  Previsões no teste", loc="left", pad=14, color=AZUL)
    for eixo in (ax.xaxis, ax.yaxis):
        eixo.set_major_locator(MultipleLocator(1000))
        eixo.set_major_formatter(FuncFormatter(milhar_pt))
    ax.grid(alpha=.15, linewidth=.7)
    ax.set_axisbelow(True)

    ax = axs[1]
    ax.step(xf, yf, where="post", color=VERDE, linewidth=2.3,
            label="Floresta aleatória")
    ax.step(xr, yr, where="post", color=LARANJA, linewidth=2.1,
            linestyle=(0, (5, 2.5)), label="Referência de 4 050 g")
    ax.axvline(500, color=CINZENTO, linestyle=(0, (2, 3)), linewidth=1.3, zorder=1)
    ax.scatter([500], [nf / 69], color=VERDE, edgecolor="white", linewidth=.8,
               s=70, zorder=5)
    ax.scatter([500], [nr / 69], color=LARANJA, edgecolor="white", linewidth=.8,
               s=70, zorder=5)
    ax.annotate("60/69", xy=(500, nf / 69), xytext=(770, .73),
                color=VERDE, fontsize=14, fontweight="bold", ha="left",
                arrowprops={"arrowstyle": "-", "color": VERDE, "linewidth": 1},
                bbox={"boxstyle": "round,pad=.14", "fc": "white", "ec": "none"})
    ax.annotate("29/69", xy=(500, nr / 69), xytext=(790, .32),
                color="#965900", fontsize=14, fontweight="bold", ha="left",
                arrowprops={"arrowstyle": "-", "color": LARANJA, "linewidth": 1},
                bbox={"boxstyle": "round,pad=.14", "fc": "white", "ec": "none"})
    ax.set(xlim=(0, xmax), ylim=(0, 1.035))
    ax.set_xlabel("Erro absoluto (g)", labelpad=9)
    ax.set_ylabel("Fração com erro até esse valor", labelpad=9)
    ax.set_title("B  Distribuição dos erros absolutos",
                 loc="left", pad=14, fontsize=14.5, color=AZUL)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_major_formatter(FuncFormatter(milhar_pt))
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_formatter(FuncFormatter(decimal_pt))
    ax.grid(alpha=.15, linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, framealpha=.97,
              facecolor="white", edgecolor="none", borderpad=.5)
    fig.text(.54, .058,
             "69 pinguins de teste. 500 g é um ponto de leitura descritiva,",
             ha="center", va="center", fontsize=13, color=ESCURO)
    fig.text(.54, .018,
             "não um limite de aceitação científica estabelecido.",
             ha="center", va="center", fontsize=13, color=ESCURO)
    saidas = guardar(fig, "u14_regressao",
                     "Previsões e distribuições empíricas dos erros absolutos no teste.")
    return saidas, {"n": len(d), "referencia_g": 4050, "leitura_g": 500,
                    "n_floresta_ate_500g": nf, "n_referencia_ate_500g": nr,
                    "registo_maior_erro_floresta": 262,
                    "maior_erro_floresta_g": float(erro_f[i_maior]),
                    "ecdf": "direita-contínua; saltos exatos, multiplicidades conservadas"}


def main() -> None:
    fontes = ["classificacao_teste_previsoes.csv", "classificacao_teste_roc.csv",
              "classificacao_teste_medidas.csv", "regressao_teste_previsoes.csv"]
    hashes_antes = {nome: sha256(RESULTADOS / nome) for nome in fontes}
    sc, dc = classificacao()
    sr, dr = regressao()
    conferir(hashes_antes == {nome: sha256(RESULTADOS / nome) for nome in fontes},
             "Nenhum ficheiro de previsões ou medidas foi modificado")
    relatorio = {
        "descricao": "Figuras de resultados já calculados; nenhum modelo foi reajustado.",
        "fontes_sha256": hashes_antes,
        "classificacao": dc,
        "regressao": dr,
        "saidas": sc + sr,
        "numero_controlos_aprovados": len(CONTROLOS),
        "controlos_aprovados": CONTROLOS,
        "inspecao_visual": "A realizar separadamente nas imagens renderizadas dos PDFs.",
    }
    destino = RESULTADOS / "verificacao_figuras.json"
    destino.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    print(f"Duas figuras PDF e PNG geradas; {len(CONTROLOS)} controlos aprovados.")


if __name__ == "__main__":
    main()
