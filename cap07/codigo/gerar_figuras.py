"""Reproduzir os resultados e as três figuras do capítulo 7.

Executar a partir de qualquer pasta:
    python codigo/gerar_figuras.py

As análises reais usam apenas os 273 registos de desenvolvimento com massa
observada. As previsões de validação são obtidas em cinco partes, sem que a
observação prevista participe no ajustamento correspondente. O conjunto de
teste não é lido. Os três padrões da última figura são dados fictícios.
"""

import json
import os
import platform
import tempfile
from hashlib import sha256
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="regressao_mpl_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits


RAIZ = Path(__file__).resolve().parents[1]
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
DADOS = RAIZ / "dados" / "penguins_desenvolvimento.csv"
HASH_DADOS = "8f3a9c1a2acef4ce13f8ba13cd30927194b0c30b73d83eaa82d65d269abb03d0"
SEMENTE = 2026
CORES = ["#1E817F", "#D98B00", "#244A6B", "#9A62A1"]
CINZENTO = "#78838C"
DESTAQUES = {1: (CORES[1], "o"), 3: (CORES[3], "D")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#66717A", "axes.labelcolor": "#28343E",
    "text.color": "#28343E", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "axes.axisbelow": True,
})


def pt(valor, casas=1):
    """Vírgula decimal apenas na apresentação; os cálculos não são arredondados."""
    return f"{valor:.{casas}f}".replace(".", ",")


def estilo(ax):
    ax.grid(color="#E5E9EC", linewidth=.6)
    ax.tick_params(length=3, color="#66717A")
    fmt = FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def guardar(fig, nome, titulo):
    """Guardar o PDF completo antes de o colocar no destino definitivo."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    (FIGURAS / "png").mkdir(exist_ok=True)
    fd, nome_temporario = tempfile.mkstemp(
        prefix=f".{nome}-", suffix=".tmp.pdf", dir=FIGURAS
    )
    temporario = Path(nome_temporario)
    try:
        with os.fdopen(fd, "wb") as stream:
            fig.savefig(
                stream, format="pdf", bbox_inches="tight",
                metadata={
                    "Title": titulo,
                    "Subject": "Ilustração científica reproduzível do capítulo 7",
                    "CreationDate": None, "ModDate": None,
                },
            )
            stream.flush()
            os.fsync(stream.fileno())
        if not temporario.read_bytes().rstrip().endswith(b"%%EOF"):
            raise OSError(f"O PDF não foi escrito integralmente: {temporario}")
        temporario.replace(FIGURAS / f"{nome}.pdf")
        fig.savefig(FIGURAS / "png" / f"{nome}.png", dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
        if temporario.exists():
            temporario.unlink()


def medidas(observadas, previstas):
    return {
        "MAE_g": float(mean_absolute_error(observadas, previstas)),
        "RMSE_g": float(mean_squared_error(observadas, previstas) ** .5),
        "R2": float(r2_score(observadas, previstas)),
    }


def procedimento_linear():
    return make_pipeline(SimpleImputer(strategy="median"), LinearRegression())


def analisar():
    hash_obtido = sha256(DADOS.read_bytes()).hexdigest()
    if hash_obtido != HASH_DADOS:
        raise ValueError("A cópia dos dados difere da utilizada neste capítulo.")
    desenvolvimento = pd.read_csv(DADOS)
    dados = desenvolvimento.dropna(subset=["body_mass_g"]).copy()
    colunas_simples = ["flipper_length_mm"]
    colunas_multipla = ["flipper_length_mm", "bill_length_mm"]
    y = dados["body_mass_g"]
    X = dados[colunas_multipla]
    assert len(desenvolvimento) == 275 and len(dados) == 273
    assert desenvolvimento.loc[
        desenvolvimento["body_mass_g"].isna(), "id_registo"
    ].tolist() == [4, 272]
    assert dados["id_registo"].is_unique
    assert dados["id_registo"].is_monotonic_increasing
    assert not X.isna().any().any()

    # A lista conserva exatamente as mesmas divisões para os quatro procedimentos.
    particoes = list(KFold(
        n_splits=5, shuffle=True, random_state=SEMENTE
    ).split(X, y))
    n_parte = np.zeros(len(dados), dtype=int)
    particoes_ids = []
    for numero, (treino, validacao) in enumerate(particoes, start=1):
        n_parte[validacao] = numero
        assert not np.intersect1d(treino, validacao).size
        particoes_ids.append({
            "parte": numero,
            "ids_treino": dados.iloc[treino]["id_registo"].astype(int).tolist(),
            "ids_validacao": dados.iloc[validacao]["id_registo"].astype(int).tolist(),
        })
    assert sorted(len(v) for _, v in particoes) == [54, 54, 55, 55, 55]
    assert np.all(n_parte > 0)

    configuracoes = {
        "referencia_media": ("Referência média", DummyRegressor(strategy="mean"), colunas_multipla),
        "referencia_mediana": ("Referência mediana", DummyRegressor(strategy="median"), colunas_multipla),
        "regressao_simples": ("Regressão simples", procedimento_linear(), colunas_simples),
        "regressao_multipla": ("Regressão múltipla", procedimento_linear(), colunas_multipla),
    }
    previstas = {}
    metricas = []
    metricas_partes = []
    for chave, (nome, modelo, colunas) in configuracoes.items():
        x = dados[colunas]
        p = cross_val_predict(modelo, x, y, cv=particoes)
        previstas[chave] = p
        metricas.append({"procedimento": nome, **medidas(y, p)})
        for numero, (_, validacao) in enumerate(particoes, start=1):
            metricas_partes.append({
                "procedimento": nome, "parte_validacao": numero,
                "n_validacao": len(validacao),
                **medidas(y.iloc[validacao], p[validacao]),
            })

    esperados = np.array([
        [681.5954225420733, 811.4392984185596, -.009754114193107899],
        [672.57326007326, 832.065083460154, -.061739935665028334],
        [320.816473775671, 400.20773392177557, .7543739006833899],
        [320.0462953804463, 398.9573461895999, .7559063452032163],
    ])
    tabela_metricas = pd.DataFrame(metricas)
    assert np.allclose(
        tabela_metricas[["MAE_g", "RMSE_g", "R2"]].to_numpy(),
        esperados, rtol=1e-10, atol=1e-9,
    )

    completos = {}
    coeficientes = []
    metricas_treino = {}
    for chave, colunas in [("regressao_simples", colunas_simples),
                           ("regressao_multipla", colunas_multipla)]:
        modelo = procedimento_linear().fit(dados[colunas], y)
        completos[chave] = modelo
        reg = modelo.named_steps["linearregression"]
        coeficientes.append({
            "procedimento": configuracoes[chave][0], "termo": "ordenada_na_origem",
            "unidade": "g", "coeficiente": float(reg.intercept_),
        })
        for coluna, coef in zip(colunas, reg.coef_):
            coeficientes.append({
                "procedimento": configuracoes[chave][0], "termo": coluna,
                "unidade": "g/mm", "coeficiente": float(coef),
            })
        ajustadas = modelo.predict(dados[colunas])
        metricas_treino[chave] = medidas(y, ajustadas)
        assert abs(np.sum(y.to_numpy() - ajustadas)) < 1e-6
    assert np.allclose(
        completos["regressao_simples"].named_steps["linearregression"].coef_,
        [49.005657113894436], rtol=1e-12,
    )
    assert np.allclose(
        completos["regressao_multipla"].named_steps["linearregression"].coef_,
        [46.802306572844344, 8.683748127906327], rtol=1e-12,
    )

    tabela_previsoes = dados[
        ["id_registo", "body_mass_g", "flipper_length_mm", "bill_length_mm"]
    ].reset_index(drop=True)
    tabela_previsoes["parte_validacao"] = n_parte
    for chave, p in previstas.items():
        tabela_previsoes[f"prevista_{chave}_g"] = p
        tabela_previsoes[f"erro_{chave}_g"] = y.to_numpy() - p
    escolhidos = tabela_previsoes[
        tabela_previsoes["id_registo"].isin([1, 2, 3, 5, 9])
    ]
    assert np.allclose(
        escolhidos["prevista_regressao_simples_g"],
        [3216.7392222548, 3481.0628194098, 3926.7067026633,
         3811.4065882291, 3826.2394623146], atol=1e-8, rtol=0,
    )

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabela_previsoes.to_csv(RESULTADOS / "previsoes_validacao.csv", index=False)
    tabela_metricas.to_csv(RESULTADOS / "metricas_validacao.csv", index=False)
    pd.DataFrame(metricas_partes).to_csv(RESULTADOS / "metricas_por_parte.csv", index=False)
    pd.DataFrame(coeficientes).to_csv(RESULTADOS / "coeficientes_ajustamento_completo.csv", index=False)
    tabela_previsoes[["id_registo", "parte_validacao"]].to_csv(
        RESULTADOS / "atribuicao_partes.csv", index=False
    )
    (RESULTADOS / "particoes_validacao.json").write_text(
        json.dumps(particoes_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Guardar também a geometria de treino dos dois registos destacados.
    selecao = dados.loc[dados["id_registo"].isin(DESTAQUES)].copy()
    selecao["prevista_treino_g"] = completos["regressao_simples"].predict(
        selecao[colunas_simples]
    )
    selecao["residuo_treino_g"] = selecao["body_mass_g"] - selecao["prevista_treino_g"]
    selecao[["id_registo", "flipper_length_mm", "body_mass_g",
             "prevista_treino_g", "residuo_treino_g"]].to_csv(
        RESULTADOS / "registos_destacados_ajustamento.csv", index=False
    )
    metadata = {
        "ficheiro_dados": "dados/penguins_desenvolvimento.csv",
        "sha256_dados": hash_obtido,
        "registos_desenvolvimento": len(desenvolvimento),
        "registos_avaliados": len(dados), "ids_excluidos_sem_massa": [4, 272],
        "omissos_nos_preditores_dos_registos_avaliados": X.isna().sum().to_dict(),
        "semente_particoes": SEMENTE,
        "regra_particoes": "KFold(n_splits=5, shuffle=True, random_state=2026), na ordem do CSV",
        "tamanhos_treino_validacao": [[len(t), len(v)] for t, v in particoes],
        "convencao_avaliacao": "Medidas calculadas sobre as 273 previsões de validação reunidas.",
        "convencao_erro": "massa observada menos massa prevista, em gramas",
        "teste_utilizado": False,
        "metricas_validacao": metricas,
        "metricas_treino_ajustamentos_completos": metricas_treino,
        "coeficientes_ajustamentos_completos": coeficientes,
        "registos_exemplo_validacao": escolhidos.to_dict(orient="records"),
        "gama_barbatana_mm": [float(X["flipper_length_mm"].min()), float(X["flipper_length_mm"].max())],
        "previsoes_reta_completa_g": {
            str(x): float(p) for x, p in zip(
                [190, 200, 210, 220], completos["regressao_simples"].predict(
                    pd.DataFrame({"flipper_length_mm": [190, 200, 210, 220]})
                )
            )
        },
        "versoes": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    return dados, previstas["regressao_simples"], completos, metadata


def ajuste_simples(dados, modelo):
    fig, ax = plt.subplots(figsize=(7.7, 4.8))
    x = dados["flipper_length_mm"]
    y = dados["body_mass_g"]
    ax.scatter(x, y, color=CORES[0], s=24, alpha=.67,
               edgecolors="none", label="Observações de desenvolvimento")
    xx = np.linspace(x.min(), x.max(), 150)
    yy = modelo.predict(pd.DataFrame({"flipper_length_mm": xx}))
    ax.plot(xx, yy, color=CORES[2], linewidth=1.9, label="Reta ajustada aos 273 pinguins")
    posicoes_texto = {1: (173.8, 5030), 3: (195.7, 2680)}
    for registo, (cor, marcador) in DESTAQUES.items():
        linha = dados.loc[dados["id_registo"] == registo].iloc[0]
        observado = float(linha["body_mass_g"])
        xi = float(linha["flipper_length_mm"])
        previsto = float(modelo.predict(pd.DataFrame({"flipper_length_mm": [xi]}))[0])
        residuo = observado - previsto
        ax.plot([xi, xi], [observado, previsto], color=cor, linewidth=2.2, zorder=4)
        ax.scatter([xi], [previsto], color="white", edgecolor=cor,
                   marker=marcador, s=65, linewidth=1.7, zorder=5)
        ax.scatter([xi], [observado], color=cor, edgecolor="#28343E",
                   marker=marcador, s=75, linewidth=.7, zorder=6)
        sinal = "+" if residuo > 0 else ""
        ax.annotate(
            f"Registo {registo}\nResíduo de treino: {sinal}{pt(residuo)} g",
            xy=(xi, observado), xytext=posicoes_texto[registo], textcoords="data",
            fontsize=10.5, color="#28343E", ha="left", va="center",
            arrowprops={"arrowstyle": "->", "color": CINZENTO, "lw": .9,
                        "relpos": (.15, 0 if registo == 1 else 1)},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93, "pad": 2.2},
        )
    ax.set(xlabel="Comprimento da barbatana (mm)", ylabel="Massa corporal (g)",
           xlim=(169, 234), ylim=(2420, 6350))
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    estilo(ax)
    fig.subplots_adjust(left=.12, right=.98, bottom=.14, top=.98)
    guardar(fig, "u7_ajuste_simples", "Reta ajustada ao desenvolvimento e resíduos de ajustamento")


def diagnostico_validacao(dados, previstas):
    observadas = dados["body_mass_g"].to_numpy()
    erros = observadas - previstas
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.7))
    ax.scatter(observadas, previstas, s=22, color=CORES[0], alpha=.68, edgecolors="none")
    ax.plot([2400, 6400], [2400, 6400], linestyle="--", color=CORES[2], linewidth=1.3)
    ax.set(xlabel="Massa observada (g)", ylabel="Massa prevista (g)",
           xlim=(2400, 6400), ylim=(2400, 6400))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("a) Observado e previsto", loc="left", pad=12)
    bx.scatter(previstas, erros, s=22, color=CORES[0], alpha=.68, edgecolors="none")
    bx.axhline(0, color=CORES[2], linewidth=1.3, linestyle="--")
    limite = max(1000, int(np.ceil(np.abs(erros).max() / 200) * 200))
    bx.set(xlabel="Massa prevista (g)", ylabel="Observado - previsto (g)",
           xlim=(2400, 6400), ylim=(-limite, limite))
    bx.set_box_aspect(1)
    bx.set_title("b) Erros de previsão", loc="left", pad=12)
    deslocacoes = {
        "observado": {1: (24, -27), 3: (-39, 31)},
        "erro": {1: (-25, 28), 3: (17, -28)},
    }
    ids = dados["id_registo"].to_numpy()
    for registo, (cor, marcador) in DESTAQUES.items():
        i = int(np.flatnonzero(ids == registo)[0])
        for eixo, ponto, tipo in [
            (ax, (observadas[i], previstas[i]), "observado"),
            (bx, (previstas[i], erros[i]), "erro"),
        ]:
            eixo.scatter(*ponto, color=cor, marker=marcador, s=70,
                         edgecolor="#28343E", linewidth=.7, zorder=6)
            eixo.annotate(
                f"Registo {registo}", ponto, textcoords="offset points",
                xytext=deslocacoes[tipo][registo], fontsize=9.5,
                ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": cor, "lw": 1},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .91, "pad": 1.8},
            )
    for eixo in [ax, bx]:
        estilo(eixo)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=CORES[0], label="Outros registos"),
        Line2D([], [], marker="o", linestyle="", color=CORES[1], label="Registo 1"),
        Line2D([], [], marker="D", linestyle="", color=CORES[3], label="Registo 3"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.085, right=.99, bottom=.22, top=.89, wspace=.35)
    guardar(fig, "u7_diagnostico_validacao", "Previsões e erros da regressão simples por validação cruzada")


def padroes_ficticios():
    rng = np.random.default_rng(SEMENTE)
    previstas = np.linspace(3000, 5500, 80)
    t = np.linspace(-1, 1, len(previstas))
    z = rng.normal(size=len(previstas))
    erros = {
        "sem_padrao_evidente": 150 * z,
        "curvatura": 650 * (t ** 2 - np.mean(t ** 2)) + 105 * z,
        "dispersao_crescente": (50 + 360 * (t + 1) / 2) * z,
    }
    titulos = ["a) Sem padrão evidente", "b) Curvatura", "c) Dispersão crescente"]
    fig, eixos = plt.subplots(1, 3, figsize=(11.2, 3.9), sharex=True, sharey=True)
    limite = max(1000, int(np.ceil(max(np.abs(e).max() for e in erros.values()) / 200) * 200))
    linhas = []
    for ax, (cenario, erro), titulo in zip(eixos, erros.items(), titulos):
        ax.scatter(previstas, erro, color=CORES[0], s=22, alpha=.82, edgecolors="none")
        ax.axhline(0, color=CORES[2], linestyle="--", linewidth=1.2)
        ax.set_title(titulo, loc="left", fontsize=14, pad=11)
        ax.set(xlim=(2850, 5650), ylim=(-limite, limite))
        ax.set_xticks([3000, 4000, 5000])
        ax.set_yticks([-1000, -500, 0, 500, 1000])
        estilo(ax)
        ax.tick_params(labelsize=13)
        for i, (prevista, e) in enumerate(zip(previstas, erro), start=1):
            linhas.append({
                "cenario_ficticio": cenario, "id_ficticio": i,
                "massa_prevista_g": float(prevista),
                "massa_observada_g": float(prevista + e),
                "erro_previsao_g": float(e),
            })
    eixos[0].set_ylabel("Observado - previsto (g)", fontsize=14)
    fig.supxlabel("Massa prevista (g)", fontsize=14, y=.035)
    fig.suptitle("Dados fictícios em todos os painéis", fontsize=14,
                 color=CINZENTO, fontstyle="italic", y=.98)
    fig.subplots_adjust(left=.085, right=.99, bottom=.2, top=.8, wspace=.13)
    tabela = pd.DataFrame(linhas)
    assert len(tabela) == 240
    assert np.allclose(
        tabela["massa_observada_g"] - tabela["massa_prevista_g"],
        tabela["erro_previsao_g"],
    )
    tabela.to_csv(RESULTADOS / "padroes_ficticios.csv", index=False)
    guardar(fig, "u7_padroes_residuos", "Três padrões fictícios de erros de previsão")
    return {
        "semente": SEMENTE, "observacoes_por_cenario": 80,
        "fonte": "Dados fictícios gerados exclusivamente para a ilustração.",
        "construcao": {
            "massa_prevista_g": "80 valores igualmente espaçados entre 3000 e 5500",
            "t": "80 valores igualmente espaçados entre -1 e 1",
            "z": "default_rng(2026).normal(size=80), o mesmo vetor nos três painéis",
            "erro_sem_padrao_evidente": "150*z",
            "erro_curvatura": "650*(t**2 - mean(t**2)) + 105*z",
            "erro_dispersao_crescente": "(50 + 360*(t+1)/2)*z",
            "massa_observada_g": "massa_prevista_g + erro_previsao_g",
        },
    }


def main():
    with threadpool_limits(limits=1):
        dados, previstas, completos, metadata = analisar()
        ajuste_simples(dados, completos["regressao_simples"])
        diagnostico_validacao(dados, previstas)
        metadata["ilustracoes_ficticias"] = padroes_ficticios()
    (RESULTADOS / "resultados_verificados.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("273 registos de desenvolvimento; quatro procedimentos verificados.")
    print("Criadas três figuras em PDF e PNG e os resultados reproduzíveis.")
    print("O conjunto de teste não foi utilizado.")


if __name__ == "__main__":
    main()
