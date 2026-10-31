"""Reproduzir os resultados e as cinco figuras do capítulo 8.

Executar a partir de qualquer pasta:
    python codigo/gerar_figuras.py

As análises reais usam apenas os 455 registos de desenvolvimento.
Cada probabilidade de validação é obtida sem usar essa amostra no treino
correspondente. O mapa de fronteiras, devidamente identificado, usa um
ajustamento separado a todo o desenvolvimento. O teste não é lido.
"""

import json
import os
import platform
import tempfile
from hashlib import sha256
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="logistica_mpl_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados" / "wisconsin_desenvolvimento.csv"
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
HASH_DADOS = "997edc11984a788044ef3ceb17595f0b77b1aac91a7186e8afd3c2dde3b6153d"
COLUNAS = ["mean radius", "mean texture"]
SEMENTE = 2026
TEAL, LARANJA, AZUL, ROXO = "#1E817F", "#D98B00", "#244A6B", "#9A62A1"
CINZENTO = "#78838C"
LIMIARES = [0.20, 0.30, 0.50, 0.70]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#66717A", "axes.labelcolor": "#28343E",
    "text.color": "#28343E", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "axes.axisbelow": True,
})


def pt(valor, casas=2):
    return f"{valor:.{casas}f}".replace(".", ",")


def estilo(ax):
    ax.grid(color="#E5E9EC", linewidth=.6)
    ax.tick_params(length=3, color="#66717A")
    formato = FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(formato)
    ax.yaxis.set_major_formatter(formato)


def guardar(fig, nome, titulo):
    """Concluir a escrita do PDF antes de substituir o ficheiro final."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    (FIGURAS / "png").mkdir(exist_ok=True)
    fd, temporario_nome = tempfile.mkstemp(
        prefix=f".{nome}-", suffix=".tmp.pdf", dir=FIGURAS
    )
    temporario = Path(temporario_nome)
    try:
        with os.fdopen(fd, "wb") as stream:
            fig.savefig(stream, format="pdf", bbox_inches="tight", metadata={
                "Title": titulo,
                "Subject": "Ilustração científica reproduzível do capítulo 8",
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


def procedimento():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, solver="lbfgs", max_iter=1000),
    )


def medidas(y, p, limiar):
    tn, fp, fn, tp = confusion_matrix(y, p >= limiar, labels=[0, 1]).ravel()
    return {
        "limiar": float(limiar), "VN": int(tn), "FP": int(fp),
        "FN": int(fn), "VP": int(tp),
        "proporcao_corretas": float((tn + tp) / (tn + fp + fn + tp)),
        "sensibilidade": float(tp / (tp + fn)),
        "especificidade": float(tn / (tn + fp)),
        "taxa_falsos_positivos": float(fp / (tn + fp)),
        "VPP": float(tp / (tp + fp)) if tp + fp else None,
    }


def analisar():
    hash_obtido = sha256(DADOS.read_bytes()).hexdigest()
    if hash_obtido != HASH_DADOS:
        raise ValueError("A cópia dos dados difere da utilizada no capítulo.")
    dados = pd.read_csv(DADOS)
    X, y = dados[COLUNAS], dados["maligno"]
    assert len(dados) == 455
    assert y.value_counts().to_dict() == {0: 285, 1: 170}
    assert dados["id_registo"].is_unique and dados["id_registo"].is_monotonic_increasing
    assert not X.isna().any().any()
    particoes = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEMENTE
    ).split(X, y))
    modelo = procedimento()
    p = cross_val_predict(modelo, X, y, cv=particoes, method="predict_proba")[:, 1]
    assert not hasattr(modelo[-1], "classes_")
    auc_partes = cross_validate(
        modelo, X, y, cv=particoes, scoring="roc_auc"
    )["test_score"]
    auc_agregada = float(roc_auc_score(y, p))
    assert np.isclose(auc_partes.mean(), .9521155830753354, atol=1e-12)
    assert np.isclose(auc_partes.std(ddof=1), .00988101781404845, atol=1e-12)
    assert np.isclose(auc_agregada, .9518059855521156, atol=1e-12)

    numero_parte = np.zeros(len(dados), dtype=int)
    partes, metricas_partes = [], []
    for numero, (treino, validacao) in enumerate(particoes, start=1):
        assert len(treino) == 364 and len(validacao) == 91
        assert np.bincount(y.iloc[validacao]).tolist() == [57, 34]
        assert not np.intersect1d(treino, validacao).size
        numero_parte[validacao] = numero
        partes.append({
            "parte": numero,
            "ids_treino": dados.iloc[treino]["id_registo"].astype(int).tolist(),
            "ids_validacao": dados.iloc[validacao]["id_registo"].astype(int).tolist(),
        })
        metricas_partes.append({
            "parte": numero, "n_treino": len(treino), "n_validacao": len(validacao),
            "benignos_validacao": 57, "malignos_validacao": 34,
            "AUC": float(auc_partes[numero - 1]),
            **medidas(y.iloc[validacao], p[validacao], .50),
        })
        assert np.isclose(roc_auc_score(y.iloc[validacao], p[validacao]), auc_partes[numero - 1])

    tabela_metricas = pd.DataFrame([medidas(y, p, t) for t in LIMIARES])
    assert tabela_metricas[["VN", "FP", "FN", "VP"]].values.tolist() == [
        [225, 60, 10, 160], [247, 38, 19, 151],
        [268, 17, 31, 139], [278, 7, 49, 121],
    ]
    tabela_previsoes = dados[["id_registo", *COLUNAS, "maligno"]].copy()
    tabela_previsoes["parte_validacao"] = numero_parte
    tabela_previsoes["p_maligno_validacao"] = p
    tabela_previsoes["classe_limiar_050"] = (p >= .50).astype(int)
    exemplos = tabela_previsoes[tabela_previsoes["id_registo"].isin([1, 4])]
    assert np.allclose(exemplos["p_maligno_validacao"], [.722231100186363, .050958988966104], atol=1e-10)

    modelo.fit(X, y)
    ajuste, escala = modelo[-1], modelo[0]
    assert ajuste.classes_.tolist() == [0, 1]
    assert np.allclose(ajuste.coef_[0], [3.31483462, .86306003], atol=1e-8)
    assert np.isclose(ajuste.intercept_[0], -.67221019, atol=1e-8)
    coeficientes = [
        {"termo": "constante", "coeficiente": float(ajuste.intercept_[0]),
         "media_treino": None, "escala_treino": None},
    ]
    for j, coluna in enumerate(COLUNAS):
        coeficientes.append({
            "termo": coluna, "coeficiente": float(ajuste.coef_[0, j]),
            "media_treino": float(escala.mean_[j]), "escala_treino": float(escala.scale_[j]),
        })
    fp_roc, tp_roc, t_roc = roc_curve(y, p, pos_label=1, drop_intermediate=False)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabela_previsoes.to_csv(RESULTADOS / "previsoes_validacao.csv", index=False)
    tabela_previsoes[["id_registo", "parte_validacao"]].to_csv(
        RESULTADOS / "atribuicao_partes.csv", index=False
    )
    tabela_metricas.to_csv(RESULTADOS / "metricas_por_limiar.csv", index=False)
    pd.DataFrame(metricas_partes).to_csv(RESULTADOS / "metricas_por_parte.csv", index=False)
    pd.DataFrame(coeficientes).to_csv(RESULTADOS / "coeficientes_ajustamento_completo.csv", index=False)
    pd.DataFrame({"limiar": t_roc, "taxa_fp": fp_roc, "sensibilidade": tp_roc}).to_csv(
        RESULTADOS / "roc_validacao.csv", index=False
    )
    (RESULTADOS / "particoes_validacao.json").write_text(
        json.dumps(partes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    positivos, negativos = p[y == 1], p[y == 0]
    n_pares_superiores = int(np.sum(positivos[:, None] > negativos[None, :]))
    n_empates = int(np.sum(positivos[:, None] == negativos[None, :]))
    assert np.isclose((n_pares_superiores + .5 * n_empates) / (170 * 285), auc_agregada)
    metadata = {
        "ficheiro_dados": "dados/wisconsin_desenvolvimento.csv",
        "sha256_dados": hash_obtido,
        "registos_avaliados": 455, "benignos": 285, "malignos": 170,
        "omissos_nas_duas_entradas": X.isna().sum().to_dict(),
        "classe_positiva": "maligno=1", "coluna_predict_proba": 1,
        "regra_decisao": "classe=1 se p_maligno >= limiar; caso contrário, classe=0",
        "procedimento": "StandardScaler + LogisticRegression(C=1, solver='lbfgs', max_iter=1000)",
        "penalizacao": "L2, predefinida nesta configuração em scikit-learn 1.8.0; constante não penalizada",
        "semente_particoes": SEMENTE,
        "regra_particoes": "StratifiedKFold(n_splits=5, shuffle=True, random_state=2026), na ordem do CSV",
        "auc_por_parte": auc_partes.tolist(),
        "auc_media_partes": float(auc_partes.mean()),
        "auc_desvio_padrao_partes_ddof1": float(auc_partes.std(ddof=1)),
        "auc_probabilidades_reunidas": auc_agregada,
        "auc_agregada_pares": {
            "pares_positivo_negativo": 170 * 285,
            "positivo_com_probabilidade_superior": n_pares_superiores,
            "empates": n_empates,
        },
        "metricas_por_limiar": tabela_metricas.to_dict(orient="records"),
        "coeficientes_ajustamento_completo": coeficientes,
        "registos_exemplo_validacao": exemplos.to_dict(orient="records"),
        "probabilidade_ajuste_completo_nas_medias": float(
            modelo.predict_proba(pd.DataFrame([escala.mean_], columns=COLUNAS))[0, 1]
        ),
        "razoes_odds_uma_unidade_padronizada": np.exp(ajuste.coef_[0]).tolist(),
        "sigmoide_exemplos": {
            str(z): float(1 / (1 + np.exp(-z))) for z in [-2, 0, 1.2, 2, 3.2]
        },
        "teste_utilizado": False,
        "versoes": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    return dados, p, modelo, metadata


def sigmoide():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    z = np.linspace(-5.5, 5.5, 500)
    ax.plot(z, 1 / (1 + np.exp(-z)), color=AZUL, linewidth=2.4)
    ax.axhline(.5, color=CINZENTO, linewidth=1, linestyle="--")
    ax.axvline(0, color=CINZENTO, linewidth=1, linestyle="--")
    textos = {-2: (-4.8, .29), 0: (-2.5, .63), 2: (.45, .94)}
    for valor in [-2, 0, 2]:
        probabilidade = 1 / (1 + np.exp(-valor))
        ax.scatter([valor], [probabilidade], color=TEAL, edgecolor="white", s=75, zorder=5)
        ax.annotate(
            f"z = {valor}; p = {pt(probabilidade, 3)}", (valor, probabilidade),
            xytext=textos[valor], fontsize=12,
            arrowprops={"arrowstyle": "-", "color": CINZENTO, "lw": .9},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95, "pad": 1.8},
        )
    ax.set(xlabel="Combinação linear, z", ylabel="Probabilidade produzida, p",
           xlim=(-5.5, 5.5), ylim=(-.03, 1.06))
    ax.set_yticks([0, .25, .5, .75, 1])
    estilo(ax)
    fig.subplots_adjust(left=.12, right=.98, bottom=.18, top=.98)
    guardar(fig, "u8_sigmoide", "A função logística transforma z numa probabilidade")


def probabilidades(dados, p):
    rng = np.random.default_rng(SEMENTE)
    y = dados["maligno"].to_numpy()
    alturas = y + rng.uniform(-.16, .16, len(dados))
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.axvline(.50, color=AZUL, linewidth=1.5, linestyle="--")
    ax.text(.51, .47, "Limiar 0,50", color=AZUL, fontsize=12, va="center")
    for classe, cor in [(0, TEAL), (1, LARANJA)]:
        selecao = y == classe
        ax.scatter(p[selecao], alturas[selecao], color=cor, s=23,
                   alpha=.72, edgecolor="none")
    for id_registo, cor, marcador, texto in [
        (1, AZUL, "o", (.63, 1.48)), (4, ROXO, "D", (.025, 1.48))
    ]:
        i = int(np.flatnonzero(dados["id_registo"].to_numpy() == id_registo)[0])
        ax.scatter(p[i], alturas[i], s=85, marker=marcador,
                   facecolor=LARANJA, edgecolor=cor, linewidth=2, zorder=6)
        ax.annotate(
            f"Registo {id_registo}: p = {pt(p[i], 3)}", (p[i], alturas[i]),
            xytext=texto, fontsize=12, color=cor,
            arrowprops={"arrowstyle": "->", "color": cor, "lw": 1},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93, "pad": 2},
        )
    ax.set(xlabel="Probabilidade de malignidade obtida por validação cruzada",
           ylabel="Classe de referência", xlim=(-.025, 1.025), ylim=(-.32, 1.68))
    estilo(ax)
    ax.set_yticks([0, 1], ["Benigno (0)\n285 amostras", "Maligno (1)\n170 amostras"])
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.grid(axis="y", visible=False)
    fig.text(.56, .015, "Cada ponto é uma amostra; a separação vertical é apenas visual.",
             ha="center", fontsize=11, color=CINZENTO)
    fig.subplots_adjust(left=.21, right=.985, bottom=.2, top=.96)
    guardar(fig, "u8_probabilidades", "Probabilidades de validação nas duas classes de referência")


def fronteiras(dados, modelo):
    X = dados[COLUNAS]
    xx = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 230)
    yy = np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 230)
    gx, gy = np.meshgrid(xx, yy)
    grade = pd.DataFrame({COLUNAS[0]: gx.ravel(), COLUNAS[1]: gy.ravel()})
    gp = modelo.predict_proba(grade)[:, 1].reshape(gx.shape)
    cores = ["#F5F7F9", "#E6EBF0", "#CDD9E4", "#ADC2D4", "#83A5C1", "#557F9F"]
    limites = [0, .1, .2, .4, .6, .8, 1]
    mapa = ListedColormap(cores)
    norma = BoundaryNorm(limites, mapa.N)
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    fundo = ax.contourf(gx, gy, gp, levels=limites, cmap=mapa, norm=norma, alpha=.88)
    curvas = ax.contour(gx, gy, gp, levels=[.2, .5, .7],
                       colors=[AZUL, AZUL, AZUL], linewidths=[1.6, 2.3, 1.6],
                       linestyles=[":", "-", "--"])
    # Etiquetas colocadas em pontos escolhidos para não cobrir observações.
    escala, ajuste = modelo[0], modelo[-1]
    posicoes = []
    for t, textura in [(.2, 35.2), (.5, 31.5), (.7, 27.8)]:
        z = np.log(t / (1 - t))
        raio = escala.mean_[0] + escala.scale_[0] * (
            z - ajuste.intercept_[0]
            - ajuste.coef_[0, 1] * (textura - escala.mean_[1]) / escala.scale_[1]
        ) / ajuste.coef_[0, 0]
        posicoes.append((raio, textura))
    ax.clabel(curvas, fmt={.2: "t = 0,20", .5: "t = 0,50", .7: "t = 0,70"},
              inline=True, inline_spacing=7, fontsize=12, manual=posicoes)
    for classe, cor, marcador, nome in [
        (0, TEAL, "o", "Benigno (0)"), (1, LARANJA, "^", "Maligno (1)")
    ]:
        selecao = dados["maligno"] == classe
        ax.scatter(X.loc[selecao, COLUNAS[0]], X.loc[selecao, COLUNAS[1]],
                   color=cor, marker=marcador, s=23, edgecolor="white",
                   linewidth=.35, alpha=.87, label=nome, zorder=4,
                   clip_on=False)
    barra = fig.colorbar(fundo, ax=ax, boundaries=limites, ticks=[0, .2, .4, .6, .8, 1],
                        pad=.025, fraction=.05)
    barra.set_label("Probabilidade no ajuste completo", fontsize=12)
    barra.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pt(v, 1)))
    barra.ax.tick_params(labelsize=11)
    ax.set(xlabel="Raio médio (escala da fonte)", ylabel="Textura média (escala da fonte)",
           xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
    ax.set_title("Modelo ajustado às 455 amostras de desenvolvimento", fontsize=13, pad=12)
    legenda = ax.legend(title="Classe de referência", frameon=True, facecolor="white",
                        framealpha=.95, loc="upper right", fontsize=11, title_fontsize=11)
    estilo(ax)
    fig.subplots_adjust(left=.105, right=.88, bottom=.145, top=.89)
    fig.canvas.draw()
    caixa_legenda = legenda.get_window_extent().expanded(1.02, 1.04)
    pontos_ecran = ax.transData.transform(X.to_numpy())
    assert not any(caixa_legenda.contains(x, y) for x, y in pontos_ecran), (
        "A legenda sobrepõe-se a uma observação e deve ser reposicionada."
    )
    guardar(fig, "u8_fronteiras", "Probabilidades e fronteiras do ajuste completo de regressão logística")


def roc_construcao():
    nomes = np.array(list("ABCDEF"))
    classes = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([.90, .70, .55, .45, .30, .10])
    fpr, tpr, limiares = roc_curve(classes, p, drop_intermediate=False)
    assert np.allclose(fpr, [0, 0, 1/3, 1/3, 2/3, 2/3, 1])
    assert np.allclose(tpr, [0, 1/3, 1/3, 2/3, 2/3, 1, 1])
    assert np.isclose(roc_auc_score(classes, p), 2/3)
    pd.DataFrame({"amostra_ficticia": nomes, "classe_referencia": classes,
                  "p_maligno": p}).to_csv(RESULTADOS / "amostras_roc_ficticias.csv", index=False)
    linhas = [{"limiar": "acima de 0.90", "amostras_previstas_positivas": "nenhuma",
               "VP": 0, "FP": 0, "taxa_fp": 0., "sensibilidade": 0.}]
    for i, t in enumerate(p):
        m = medidas(classes, p, t)
        linhas.append({"limiar": float(t), "amostras_previstas_positivas": ", ".join(nomes[p >= t]),
                       "VP": m["VP"], "FP": m["FP"],
                       "taxa_fp": m["taxa_falsos_positivos"], "sensibilidade": m["sensibilidade"]})
    pd.DataFrame(linhas).to_csv(RESULTADOS / "roc_construcao_ficticia.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.0, 5.3))
    ax.plot([0, 1], [0, 1], color=CINZENTO, linestyle="--", linewidth=1)
    ax.plot(fpr, tpr, color=AZUL, linewidth=2.1, marker="o", markersize=6)
    deslocacoes = [
        (18, 15), (16, -20), (14, -23), (-70, 17),
        (14, -27), (-126, 14), (-88, 14),
    ]
    rotulos = ["Nenhuma previsão\npositiva", "A; t = 0,90", "+ B; t = 0,70",
               "+ C; t = 0,55", "+ D; t = 0,45", "+ E; t = 0,30", "+ F; t = 0,10"]
    for x, y, deslocamento, rotulo in zip(fpr, tpr, deslocacoes, rotulos):
        ax.annotate(rotulo, (x, y), xytext=deslocamento, textcoords="offset points",
                    fontsize=11.5,
                    arrowprops={"arrowstyle": "-", "color": CINZENTO, "lw": .8},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92, "pad": 1.7})
    ax.scatter([1/3], [2/3], s=145, facecolor="none", edgecolor=ROXO, linewidth=2.3, zorder=5)
    ax.text(.04, .91, "O limiar 0,50 dá\no ponto assinalado.",
            transform=ax.transAxes, fontsize=11.5, color=ROXO, va="top")
    ax.set(xlabel="Taxa de falsos positivos = FP / 3", ylabel="Sensibilidade = VP / 3",
           xlim=(-.035, 1.04), ylim=(-.035, 1.115))
    ax.set_aspect("equal", adjustable="box")
    estilo(ax)
    ax.set_xticks([0, 1/3, 2/3, 1], ["0", "1/3", "2/3", "1"])
    ax.set_yticks([0, 1/3, 2/3, 1], ["0", "1/3", "2/3", "1"])
    ax.set_title("Seis amostras fictícias: três positivas e três negativas", fontsize=12.5, pad=12)
    fig.subplots_adjust(left=.12, right=.97, bottom=.15, top=.89)
    guardar(fig, "u8_roc_construcao", "Construção da curva ROC com seis amostras fictícias")
    return {"amostras": 6, "positivas": 3, "negativas": 3, "auc": 2/3,
            "fonte": "Exemplo inteiramente fictício A-F, conservado do exercício 8.6."}


def roc_validacao(dados, p):
    y = dados["maligno"]
    fpr, tpr, _ = roc_curve(y, p, pos_label=1, drop_intermediate=False)
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    ax.plot(fpr, tpr, color=AZUL, linewidth=2, label="ROC das probabilidades reunidas")
    ax.plot([0, 1], [0, 1], color=CINZENTO, linewidth=1.1, linestyle="--",
            label="Referência diagonal")
    deslocacoes = {.20: (.37, .89), .30: (.35, .72), .50: (.30, .55), .70: (.23, .38)}
    cores = {.20: TEAL, .30: ROXO, .50: LARANJA, .70: AZUL}
    for t in LIMIARES:
        m = medidas(y, p, t)
        x, yy = m["taxa_falsos_positivos"], m["sensibilidade"]
        ax.scatter([x], [yy], s=62, color=cores[t], edgecolor="white", linewidth=.8, zorder=5)
        ax.annotate(
            f"t = {pt(t)}\n({pt(x, 3)}; {pt(yy, 3)})", (x, yy),
            xytext=deslocacoes[t], fontsize=11.5, color=cores[t],
            arrowprops={"arrowstyle": "-", "color": cores[t], "lw": 1.0},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95, "pad": 2.2},
        )
    ax.text(.98, .12, f"AUC agregada = {pt(roc_auc_score(y, p), 4)}",
            transform=ax.transAxes, ha="right", fontsize=12.5, color=AZUL)
    ax.set(xlabel="Taxa de falsos positivos (1 - especificidade)",
           ylabel="Sensibilidade", xlim=(-.025, 1.025), ylim=(-.025, 1.025))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
    estilo(ax)
    fig.legend(loc="lower center", frameon=False, fontsize=11, ncol=2,
               bbox_to_anchor=(.5, -.002))
    ax.set_title("455 probabilidades obtidas por validação cruzada", fontsize=12.5, pad=12)
    fig.subplots_adjust(left=.13, right=.98, bottom=.22, top=.89)
    guardar(fig, "u8b_roc", "Curva ROC das probabilidades reunidas de validação cruzada")


def main():
    with threadpool_limits(limits=1):
        dados, p, modelo, metadata = analisar()
        sigmoide()
        probabilidades(dados, p)
        fronteiras(dados, modelo)
        metadata["roc_ficticia"] = roc_construcao()
        roc_validacao(dados, p)
    (RESULTADOS / "resultados_verificados.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Cinco figuras e resultados verificados, usando apenas os dados de desenvolvimento.")
    print(f"AUC média entre partes: {metadata['auc_media_partes']:.12f}")
    print(f"AUC das probabilidades reunidas: {metadata['auc_probabilidades_reunidas']:.12f}")


if __name__ == "__main__":
    main()
