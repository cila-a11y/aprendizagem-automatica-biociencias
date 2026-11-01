"""Projeto 1: variação morfológica e agrupamento de caranguejos.

Executar: python projeto_01.py
Entrada: dados/crabs.csv, distribuído com este projeto.
Saídas: resultados.json e três figuras, em PDF e PNG, na pasta figuras.
O programa usa apenas dados locais e não necessita de GPU nem de Internet.
"""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import os

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEMENTE = 2026
N_INIT = 20
MEDIDAS = ["FL", "RW", "CL", "CW", "BD"]
FIGURAS = BASE / "figuras"
FIGURAS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "figure.dpi": 120, "savefig.dpi": 180,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
AZUL, LARANJA = "#276A9B", "#C86822"
CORES_GRUPOS = ["#3A7A65", "#805399"]


def assinatura_particao(rotulos):
    """Comparar partições sem confundir os números arbitrários dos grupos."""
    conjuntos = [tuple(np.flatnonzero(rotulos == g)) for g in np.unique(rotulos)]
    return tuple(sorted(conjuntos))


def guardar_figura(figura, nome):
    for eixo in figura.axes:
        for escala in (eixo.xaxis, eixo.yaxis):
            if isinstance(escala.get_major_formatter(), ScalarFormatter):
                escala.set_major_formatter(FuncFormatter(
                    lambda valor, _: f"{valor:g}".replace(".", ",")
                ))
    figura.savefig(FIGURAS / f"{nome}.pdf", bbox_inches="tight")
    figura.savefig(FIGURAS / f"{nome}.png", bbox_inches="tight")
    plt.close(figura)


def tabela_numerica(tabela):
    """Converter uma tabela pequena para números e listas legíveis em JSON."""
    return {
        "linhas": [str(x) for x in tabela.index],
        "colunas": [str(x) for x in tabela.columns],
        "valores": tabela.to_numpy().tolist(),
    }


def main():
    ficheiro = BASE / "dados" / "crabs.csv"
    dados = pd.read_csv(ficheiro)

    # Apenas estas cinco colunas entram na escala, na PCA e no K-means.
    # Os contadores rownames/index e os rótulos sp/sex não são atributos.
    X = dados.loc[:, MEDIDAS].astype(float)
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Existem medidas em falta ou não finitas; investigar antes de ajustar.")

    pca_mm = PCA(svd_solver="full").fit(X)
    escala = StandardScaler()
    Z = escala.fit_transform(X)
    pca_z = PCA(svd_solver="full").fit(Z)
    coordenadas = pca_z.transform(Z)

    # Cada execução já usa 20 inicializações. Repetimos com dez sementes
    # para observar a sensibilidade ao acaso do algoritmo, na mesma amostra.
    resumos = []
    modelos = {}
    for k in range(2, 7):
        execucoes = [
            KMeans(n_clusters=k, n_init=N_INIT, random_state=s).fit(Z)
            for s in range(SEMENTE, SEMENTE + 10)
        ]
        melhor = min(execucoes, key=lambda modelo: modelo.inertia_)
        modelos[k] = melhor
        inercias = [float(modelo.inertia_) for modelo in execucoes]
        particoes = {assinatura_particao(modelo.labels_) for modelo in execucoes}
        resumos.append({
            "k": k,
            "inercia_minima": min(inercias),
            "inercia_maxima": max(inercias),
            "silhouette": float(silhouette_score(Z, melhor.labels_)),
            "particoes_distintas": len(particoes),
            "inercia_semente_2026": inercias[0],
        })

    # Escolha interpretativa de uma partição grosseira, justificada no texto.
    # Não decorre do número de combinações de sexo e cor, ainda não consultadas.
    k_escolhido = 2
    modelo = modelos[k_escolhido]
    ordem = np.argsort(modelo.cluster_centers_.mean(axis=1))
    mapa = {int(antigo): novo + 1 for novo, antigo in enumerate(ordem)}
    grupos = np.array([mapa[int(g)] for g in modelo.labels_])
    centros_z = modelo.cluster_centers_[ordem]
    centros_mm = escala.inverse_transform(centros_z)
    perfis = X.assign(grupo=grupos).groupby("grupo").mean()
    tamanhos = pd.Series(grupos).value_counts().sort_index()

    # A análise já está construída. Só agora usamos os rótulos conhecidos.
    rotulos = dados["sp"].astype(str) + "/" + dados["sex"].astype(str)
    ordem_rotulos = ["B/F", "B/M", "O/F", "O/M"]
    cruzamento = pd.crosstab(pd.Series(grupos, name="grupo"), rotulos)
    cruzamento = cruzamento.reindex(index=[1, 2], columns=ordem_rotulos, fill_value=0)
    correlacao = X.corr()
    valores_fora_diagonal = correlacao.to_numpy()[~np.eye(5, dtype=bool)]

    resultados = {
        "dados": {
            "ficheiro": "dados/crabs.csv",
            "fonte_csv": "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/crabs.csv",
            "fonte_autor": "https://www.stats.ox.ac.uk/pub/PRNN/crabs.dat",
            "sha256": hashlib.sha256(ficheiro.read_bytes()).hexdigest(),
            "n_observacoes": len(dados), "n_colunas_csv": len(dados.columns),
            "atributos": MEDIDAS, "unidade": "mm",
            "valores_em_falta_csv": int(dados.isna().sum().sum()),
            "vetores_de_medidas_duplicados": int(X.duplicated().sum()),
            "contagem_rotulos": rotulos.value_counts().sort_index().to_dict(),
            "correlacao_minima": float(valores_fora_diagonal.min()),
            "correlacao_maxima": float(valores_fora_diagonal.max()),
            "correlacoes": tabela_numerica(correlacao),
        },
        "pca": {
            "variancia_explicada_mm": pca_mm.explained_variance_ratio_.tolist(),
            "variancia_explicada_padronizada": pca_z.explained_variance_ratio_.tolist(),
            "coeficientes_pc1_mm": pca_mm.components_[0].tolist(),
            "coeficientes_pc1_padronizada": pca_z.components_[0].tolist(),
            "componentes_padronizadas": pca_z.components_.tolist(),
        },
        "kmeans": {
            "semente_referencia": SEMENTE, "n_init": N_INIT,
            "sementes_estabilidade": list(range(SEMENTE, SEMENTE + 10)),
            "inercia_k1": float(np.square(Z - Z.mean(axis=0)).sum()),
            "comparacao": resumos, "k_escolhido": k_escolhido,
            "contagens": {str(k): int(v) for k, v in tamanhos.items()},
            "centros_mm": centros_mm.tolist(), "centros_padronizados": centros_z.tolist(),
            "cruzamento_com_rotulos": tabela_numerica(cruzamento),
        },
        "versoes": {nome: importlib.metadata.version(nome) for nome in
                    ["numpy", "pandas", "matplotlib", "scikit-learn"]},
    }
    (BASE / "resultados.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Figura 1: correlação, variância e projeções. A cor e o símbolo servem
    # apenas para interpretar projeções calculadas sem esses rótulos.
    fig, eixos = plt.subplots(2, 2, figsize=(7.4, 6.9), layout="constrained")
    ax = eixos[0, 0]
    ax.imshow(correlacao, vmin=0.85, vmax=1, cmap="Blues")
    ax.set_xticks(range(5), MEDIDAS)
    ax.set_yticks(range(5), MEDIDAS)
    for i in range(5):
        for j in range(5):
            valor = correlacao.iloc[i, j]
            ax.text(j, i, f"{valor:.3f}".replace(".", ","), ha="center", va="center",
                    color="white" if valor > 0.95 else "#152C40", fontsize=8.5)
    ax.set_title("A. Medidas fortemente correlacionadas")

    ax = eixos[0, 1]
    posicoes = np.arange(1, 6)
    ax.bar(posicoes - 0.19, 100 * pca_mm.explained_variance_ratio_, 0.38,
           color=AZUL, label="Medidas em mm")
    ax.bar(posicoes + 0.19, 100 * pca_z.explained_variance_ratio_, 0.38,
           color=LARANJA, label="Medidas padronizadas")
    ax.set_xticks(posicoes, [f"PC{i}" for i in posicoes])
    ax.set_ylim(0, 113)
    ax.set_ylabel("Variância explicada (%)")
    ax.set_title("B. Predomínio da primeira componente")
    ax.text(0.72, 101, f"{100 * pca_mm.explained_variance_ratio_[0]:.2f}".replace(".", ","),
            color=AZUL, ha="center", fontsize=9)
    ax.text(1.32, 96, f"{100 * pca_z.explained_variance_ratio_[0]:.2f}".replace(".", ","),
            color=LARANJA, ha="center", fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    for ax, (i, j), titulo in zip(
        eixos[1], [(0, 1), (1, 2)],
        ["C. Tamanho e um contraste de forma", "D. Contrastes com menor variância"]
    ):
        for cor, sexo, nome in [
            ("B", "F", "Azul, fêmea"), ("B", "M", "Azul, macho"),
            ("O", "F", "Laranja, fêmea"), ("O", "M", "Laranja, macho")
        ]:
            selecao = (dados["sp"] == cor) & (dados["sex"] == sexo)
            ax.scatter(coordenadas[selecao, i], coordenadas[selecao, j], s=17,
                       marker="o" if sexo == "F" else "^", alpha=0.8,
                       color=AZUL if cor == "B" else LARANJA, label=nome,
                       edgecolors="none")
        ax.set_xlabel(f"PC{i + 1} (dados padronizados)")
        ax.set_ylabel(f"PC{j + 1} (dados padronizados)")
        ax.set_title(titulo)
        ax.grid(alpha=0.15)
    eixos[1, 0].legend(loc="upper left", frameon=False, fontsize=7.5)
    guardar_figura(fig, "pca_caranguejos")

    # Figura 2: a inércia melhora com k; os perfis esclarecem o que foi agrupado.
    fig, eixos = plt.subplots(1, 2, figsize=(7.4, 3.5), layout="constrained")
    eixos[0].plot([r["k"] for r in resumos], [r["inercia_minima"] for r in resumos],
                  "o-", color=AZUL)
    eixos[0].set_xticks(range(2, 7))
    eixos[0].set_xlabel("Número de grupos (k)")
    eixos[0].set_ylabel("Inércia mínima (dados padronizados)")
    eixos[0].set_title("A. Redução gradual da inércia")
    eixos[0].grid(alpha=0.2)
    for i, cor in enumerate(CORES_GRUPOS):
        eixos[1].plot(MEDIDAS, centros_z[i], "o-", color=cor,
                     label=f"Grupo {i + 1} (n={tamanhos.loc[i + 1]})")
    eixos[1].axhline(0, color="#707070", linewidth=0.8)
    eixos[1].set_ylabel("Centro nas unidades padronizadas")
    eixos[1].set_title("B. Perfis dos dois grupos escolhidos")
    eixos[1].legend(frameon=False, fontsize=9)
    eixos[1].grid(axis="y", alpha=0.2)
    guardar_figura(fig, "kmeans_caranguejos")

    # Figura suplementar: comparar a partição escolhida com os rótulos.
    fig, eixos = plt.subplots(1, 2, figsize=(7.4, 3.4), layout="constrained")
    for g, cor in enumerate(CORES_GRUPOS, start=1):
        selecao = grupos == g
        eixos[0].scatter(coordenadas[selecao, 0], coordenadas[selecao, 1],
                        s=20, color=cor, alpha=0.75, label=f"Grupo {g}")
    eixos[0].set_xlabel("PC1 (dados padronizados)")
    eixos[0].set_ylabel("PC2 (dados padronizados)")
    eixos[0].set_title("A. Partição em duas regiões")
    eixos[0].legend(frameon=False)
    eixos[1].imshow(cruzamento, cmap="Blues", vmin=0, vmax=50, aspect="auto")
    eixos[1].set_xticks(range(4), ordem_rotulos)
    eixos[1].set_yticks([0, 1], ["Grupo 1", "Grupo 2"])
    for i in range(2):
        for j in range(4):
            valor = int(cruzamento.iloc[i, j])
            eixos[1].text(j, i, str(valor), ha="center", va="center",
                         color="white" if valor > 28 else "#152C40", fontsize=12)
    eixos[1].set_xlabel("B: azul; O: laranja; F: fêmea; M: macho")
    eixos[1].set_title("B. Rótulos usados após o ajustamento")
    guardar_figura(fig, "grupos_e_rotulos")

    print(f"Observações: {len(dados)}; atributos usados: {', '.join(MEDIDAS)}")
    print("Variância da PC1, em mm / padronizada:",
          f"{100 * pca_mm.explained_variance_ratio_[0]:.4f}% /",
          f"{100 * pca_z.explained_variance_ratio_[0]:.4f}%")
    print(pd.DataFrame(resumos).round(5).to_string(index=False))
    print("\nPerfis médios (mm):\n", perfis.round(3).to_string())
    print("\nCruzamento com rótulos:\n", cruzamento.to_string())
    print("\nResultados e figuras guardados na pasta do projeto.")


if __name__ == "__main__":
    main()
