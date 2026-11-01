#!/usr/bin/env python3
"""Projeto 3: massa corporal de pintos que não participaram no treino.

Desenvolvimento: python projeto_03.py
Reprodução completa: python projeto_03.py --confirmar-teste
Validar apenas a divisão: python projeto_03.py --validar-apenas

Não utiliza a rede. A semente 2026 e a divisão por animal são fixas.
O ficheiro divisao.json é escrito ANTES de ajustar qualquer modelo e
reutilizado em execuções seguintes. A seleção usa MAE médio por animal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

# Evita escrever a configuração do Matplotlib numa pasta pessoal protegida.
_MPL_CACHE = tempfile.TemporaryDirectory(prefix="projeto03_mpl_")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE.name)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


SEED = 2026
HERE = Path(__file__).resolve().parent
SCHEDULE = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21]
DIETS = [1, 2, 3, 4]
FEATURES = ["Time", "Diet"]
DATA_SHA256 = "db0256706f76174732c06a29547ba3247debbbaacef488ccc255ba54fac2d703"
COLORS = {1: "#0072B2", 2: "#009E73", 3: "#D55E00", 4: "#CC79A7"}
FAMILY_NAMES = {
    "mediana": "Mediana do treino",
    "idade": "Regressão: idade",
    "idade_dieta": "Regressão: idade + dieta",
    "arvore": "Árvore de regressão",
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_data(path: Path) -> tuple[pd.DataFrame, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError("O CSV não corresponde à versão auditada deste projeto.")
    data = pd.read_csv(path)
    required = ["rownames", "weight", "Time", "Chick", "Diet"]
    assert list(data.columns) == required, "Colunas inesperadas."
    assert not data.isna().any().any(), "Há células em falta."
    assert len(data) == 578 and data["Chick"].nunique() == 50
    assert not data.duplicated(["Chick", "Time"]).any()
    assert data.groupby("Chick")["Diet"].nunique().eq(1).all()
    assert sorted(data["Time"].unique().tolist()) == SCHEDULE
    assert sorted(data["Diet"].unique().tolist()) == DIETS
    assert data["weight"].gt(0).all()
    # Conserva o rótulo original de Chick: nunca usa códigos de níveis de R.
    return data, digest


def get_split(data: pd.DataFrame, digest: str, path: Path) -> dict:
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        if split["semente"] != SEED or split["sha256_dados"] != digest:
            raise ValueError("A divisão guardada não corresponde à semente/dados.")
        return split

    # Esta tabela contém identidades e dietas, sem consultar weight.
    animals = data[["Chick", "Diet"]].drop_duplicates()
    rng = np.random.default_rng(SEED)
    test, development, fold_for_chick = [], [], {}
    for diet in DIETS:
        ids = sorted(animals.loc[animals["Diet"].eq(diet), "Chick"].tolist())
        ids = [int(x) for x in rng.permutation(ids)]
        n_test = 4 if diet == 1 else 2
        test.extend(ids[:n_test])
        remaining = ids[n_test:]
        development.extend(remaining)
        for position, chick in enumerate(remaining):
            fold_for_chick[str(chick)] = position % 4 + 1
    split = {
        "semente": SEED,
        "sha256_dados": digest,
        "unidade_divisao": "Chick",
        "procedimento": "permutação dos identificadores dentro de cada dieta",
        "criada_antes_de_qualquer_ajuste": True,
        "animais_desenvolvimento": sorted(development),
        "animais_teste": sorted(test),
        "parte_validacao_por_animal": dict(
            sorted(fold_for_chick.items(), key=lambda item: int(item[0]))
        ),
    }
    write_json(path, split)  # Tem de ocorrer antes de qualquer model.fit().
    return split


def diet_counts(data: pd.DataFrame, ids: set[int]) -> dict[str, int]:
    animals = data.loc[data["Chick"].isin(ids), ["Chick", "Diet"]].drop_duplicates()
    return {str(d): int(animals["Diet"].eq(d).sum()) for d in DIETS}


def validate_split(data: pd.DataFrame, split: dict) -> dict:
    all_ids = set(data["Chick"].tolist())
    development = set(split["animais_desenvolvimento"])
    test = set(split["animais_teste"])
    fold_map = {int(k): int(v) for k, v in split["parte_validacao_por_animal"].items()}
    assert len(development) == 40 and len(test) == 10
    assert len(split["animais_desenvolvimento"]) == len(development)
    assert len(split["animais_teste"]) == len(test)
    assert not development & test, "Há animais no desenvolvimento e no teste."
    assert development | test == all_ids
    assert set(fold_map) == development
    assert set(fold_map.values()) == {1, 2, 3, 4}
    assert diet_counts(data, test) == {"1": 4, "2": 2, "3": 2, "4": 2}
    assert diet_counts(data, development) == {"1": 16, "2": 8, "3": 8, "4": 8}
    seen, checks = set(), []
    for fold in range(1, 5):
        validation = {c for c, f in fold_map.items() if f == fold}
        training = development - validation
        assert len(validation) == 10 and len(training) == 30
        assert not training & validation and not training & test and not validation & test
        assert not seen & validation
        assert diet_counts(data, validation) == {"1": 4, "2": 2, "3": 2, "4": 2}
        seen |= validation
        checks.append({
            "parte": fold,
            "animais_treino": sorted(training),
            "animais_validacao": sorted(validation),
            "n_registos_treino": int(data["Chick"].isin(training).sum()),
            "n_registos_validacao": int(data["Chick"].isin(validation).sum()),
            "sobreposicao_animais": 0,
            "animais_por_dieta_validacao": diet_counts(data, validation),
        })
    assert seen == development
    assert not set(FEATURES) & {"Chick", "rownames", "weight"}
    return {
        "valido": True,
        "n_animais_desenvolvimento": 40,
        "n_animais_teste": 10,
        "n_registos_desenvolvimento": int(data["Chick"].isin(development).sum()),
        "n_registos_teste": int(data["Chick"].isin(test).sum()),
        "sobreposicao_desenvolvimento_teste": 0,
        "cada_animal_validado_uma_vez": True,
        "preditores": FEATURES,
        "partes": checks,
    }


def candidate_specs() -> list[dict]:
    candidates = [
        {"id": "mediana", "familia": "mediana", "parametros": {}},
        {"id": "idade", "familia": "idade", "parametros": {}},
        {"id": "idade_dieta", "familia": "idade_dieta", "parametros": {}},
    ]
    # Ordem fixa, independente dos resultados; em empate prefere a árvore mais simples.
    for depth in [2, 3, 4, 5]:
        for leaf in [12, 8, 4]:
            candidates.append({
                "id": f"arvore_p{depth}_f{leaf}",
                "familia": "arvore",
                "parametros": {"max_depth": depth, "min_samples_leaf": leaf},
            })
    return candidates


def make_model(spec: dict) -> Pipeline:
    if spec["familia"] == "mediana":
        return Pipeline([("modelo", DummyRegressor(strategy="median"))])
    if spec["familia"] == "idade":
        transformer = ColumnTransformer([("idade", "passthrough", ["Time"])])
    else:
        transformer = ColumnTransformer([
            ("idade", "passthrough", ["Time"]),
            ("dieta", OneHotEncoder(
                categories=[DIETS], drop="first", handle_unknown="error", sparse_output=False
            ), ["Diet"]),
        ], remainder="drop")
    estimator = (
        DecisionTreeRegressor(random_state=SEED, **spec["parametros"])
        if spec["familia"] == "arvore" else LinearRegression()
    )
    return Pipeline([("preparacao", transformer), ("modelo", estimator)])


def evaluate(data: pd.DataFrame, prediction: np.ndarray) -> tuple[dict, pd.DataFrame]:
    assert len(data) == len(prediction) and np.isfinite(prediction).all()
    scores = data[["rownames", "Chick", "Diet", "Time", "weight"]].copy()
    scores["previsao"] = prediction
    # Convenção do livro: erro de previsão = observado menos previsto.
    scores["erro"] = scores["weight"] - scores["previsao"]
    scores["erro_absoluto"] = scores["erro"].abs()
    per_chick = scores.groupby(["Chick", "Diet"], as_index=False).agg(
        n_registos=("erro_absoluto", "size"),
        mae=("erro_absoluto", "mean"),
        erro_medio=("erro", "mean"),
    )
    return {
        "mae_medio_por_animal": float(per_chick["mae"].mean()),
        "mae_registos_reunidos": float(scores["erro_absoluto"].mean()),
        "n_animais": int(len(per_chick)),
        "n_registos": int(len(scores)),
        "por_animal": per_chick.to_dict(orient="records"),
    }, scores


def run_cross_validation(development: pd.DataFrame, split: dict, specs: list[dict]):
    fold_map = {int(k): int(v) for k, v in split["parte_validacao_por_animal"].items()}
    fold_labels = development["Chick"].map(fold_map)
    results, all_predictions = [], {}
    for spec in specs:
        oof = pd.Series(index=development.index, dtype=float)
        fold_results = []
        for fold in range(1, 5):
            training = development.loc[fold_labels.ne(fold)]
            validation = development.loc[fold_labels.eq(fold)]
            assert not set(training["Chick"]) & set(validation["Chick"])
            model = make_model(spec)
            model.fit(training[FEATURES], training["weight"])
            prediction = model.predict(validation[FEATURES])
            oof.loc[validation.index] = prediction
            fold_scores, _ = evaluate(validation, prediction)
            fold_results.append({"parte": fold, **fold_scores})
        assert oof.notna().all()
        pooled, predictions = evaluate(development, oof.loc[development.index].to_numpy())
        # A seleção usa os 40 erros individuais reunidos, não a média dos erros por linha.
        fold_mean = float(np.mean([f["mae_medio_por_animal"] for f in fold_results]))
        assert np.isclose(fold_mean, pooled["mae_medio_por_animal"], rtol=0, atol=1e-10)
        predictions["parte_validacao"] = fold_labels.to_numpy()
        predictions["candidato"] = spec["id"]
        all_predictions[spec["id"]] = predictions
        results.append({**spec, "validacao": pooled, "por_parte": fold_results})
    # A ordem predefinida resolve empates exatos; o teste ainda não foi utilizado.
    ranking = sorted(results, key=lambda r: r["validacao"]["mae_medio_por_animal"])
    selected_by_family = {}
    for result in ranking:
        selected_by_family.setdefault(result["familia"], result)
    representatives = [selected_by_family[f] for f in FAMILY_NAMES]
    return results, representatives, ranking[0], all_predictions


def configure_plots() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18,
        "savefig.dpi": 180, "pdf.fonttype": 42,
    })


def save_figure(fig: plt.Figure, stem: str, folder: Path) -> None:
    for axis in fig.axes:
        for scale in (axis.xaxis, axis.yaxis):
            if isinstance(scale.get_major_formatter(), ScalarFormatter):
                scale.set_major_formatter(FuncFormatter(
                    lambda value, _: f"{value:g}".replace(".", ",")
                ))
    fig.savefig(folder / f"{stem}.png", facecolor="white")
    fig.savefig(folder / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def make_figures(development, representatives, test_results, winner_id, predictions, folder):
    configure_plots()
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), sharex=True, sharey=True,
                             layout="constrained")
    for diet, ax in zip(DIETS, axes.flat):
        subset = development.loc[development["Diet"].eq(diet)]
        for _, bird in subset.groupby("Chick"):
            ax.plot(bird["Time"], bird["weight"], color=COLORS[diet], alpha=0.28, lw=1.0)
        means = subset.groupby("Time")["weight"].mean()
        ax.plot(means.index, means.values, color=COLORS[diet], lw=2.5, label="Média por idade")
        ax.set_title(f"Dieta {diet} · {subset['Chick'].nunique()} pintos")
        ax.set_xticks([0, 4, 8, 12, 16, 21])
        ax.set_xlim(-0.5, 21.5)
        ax.set_ylim(20, 400)
    for ax in axes[-1]:
        ax.set_xlabel("Idade (dias)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Massa corporal (g)")
    fig.suptitle("Trajetórias no conjunto de desenvolvimento", fontsize=13)
    save_figure(fig, "trajetorias_desenvolvimento", folder)

    fig, ax = plt.subplots(figsize=(8.6, 4.5), layout="constrained")
    for position, result in enumerate(representatives):
        fold_values = [f["mae_medio_por_animal"] for f in result["por_parte"]]
        ax.scatter(np.array([-0.16, -0.05, 0.05, 0.16]) + position, fold_values,
                   color="#72838C", s=32, alpha=0.85, zorder=3,
                   label="Cada parte (10 pintos)" if position == 0 else None)
        value = result["validacao"]["mae_medio_por_animal"]
        ax.scatter(position, value, marker="D", s=70, color="#0072B2", zorder=4,
                   label="40 erros individuais reunidos" if position == 0 else None)
        ax.annotate(f"{value:.2f} g".replace(".", ","), (position, value), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(4), ["Mediana", "Regressão\nidade", "Regressão\nidade + dieta", "Árvore"])
    ax.set_ylabel("MAE médio por pinto (g)")
    ax.set_title("Validação cruzada por animal · quatro partes")
    ax.set_ylim(0, max(f["mae_medio_por_animal"] for r in representatives for f in r["por_parte"]) * 1.18)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save_figure(fig, "comparacao_validacao", folder)

    if test_results is None:
        return
    winner = next(r for r in test_results if r["id"] == winner_id)
    scores = predictions[winner_id]
    per_bird = pd.DataFrame(winner["teste"]["por_animal"]).sort_values("Chick")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    for diet in DIETS:
        part = scores.loc[scores["Diet"].eq(diet)]
        axes[0].scatter(part["weight"], part["previsao"], color=COLORS[diet],
                        s=22, alpha=0.68, label=f"Dieta {diet}")
    upper = float(max(scores["weight"].max(), scores["previsao"].max()) * 1.05)
    axes[0].plot([0, upper], [0, upper], color="#26343B", lw=1, ls="--")
    axes[0].set(xlim=(0, upper), ylim=(0, upper), xlabel="Massa observada (g)",
                ylabel="Massa prevista (g)", title="Previsões em animais novos")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].barh(range(len(per_bird)), per_bird["mae"],
                 color=[COLORS[int(d)] for d in per_bird["Diet"]])
    axes[1].set_yticks(range(len(per_bird)), [f"Pinto {int(c)}" for c in per_bird["Chick"]])
    axes[1].axvline(winner["teste"]["mae_medio_por_animal"], color="#26343B", ls="--", lw=1)
    axes[1].set(xlabel="MAE por pinto (g)", title="Cada animal tem o mesmo peso")
    axes[1].invert_yaxis()
    fig.suptitle(f"Teste final · {FAMILY_NAMES[winner['familia']]} escolhida em validação", fontsize=12)
    save_figure(fig, "teste_modelo_escolhido", folder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=HERE / "dados" / "ChickWeight.csv")
    parser.add_argument("--saida", type=Path, default=HERE)
    parser.add_argument("--validar-apenas", action="store_true")
    parser.add_argument("--confirmar-teste", action="store_true",
                        help="Executar a fase final depois de fixar as decisões de desenvolvimento.")
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    data, digest = load_data(args.dados)
    split = get_split(data, digest, args.saida / "divisao.json")
    checks = validate_split(data, split)
    write_json(args.saida / "validacao_divisao.json", checks)
    if args.validar_apenas:
        print("Divisão válida: 40 animais de desenvolvimento; 10 de teste; sobreposição zero.")
        return

    development = data.loc[data["Chick"].isin(split["animais_desenvolvimento"])].copy()
    test = data.loc[data["Chick"].isin(split["animais_teste"])].copy()
    cv, representatives, winner, oof = run_cross_validation(development, split, candidate_specs())
    selection = {
        "criterio": "menor MAE médio dos 40 animais com previsão fora de amostra",
        "vencedor": winner["id"],
        "parametros": winner["parametros"],
        "fixada_antes_da_avaliacao_final": True,
        "representantes_pre_fixados": [r["id"] for r in representatives],
    }
    write_json(args.saida / "selecao_antes_teste.json", selection)

    development_report = {
        "fase": "desenvolvimento; teste reservado",
        "semente": SEED,
        "sha256_dados": digest,
        "ambiente": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
        "divisao": checks,
        "selecao": selection,
        "candidatos_validacao": cv,
        "definicao_erro": "observado menos previsto",
    }
    write_json(args.saida / "desenvolvimento.json", development_report)
    pd.concat([oof[r["id"]] for r in representatives], ignore_index=True).to_csv(
        args.saida / "previsoes_validacao.csv", index=False, float_format="%.10f")
    if not args.confirmar_teste:
        make_figures(development, representatives, None, None, None, args.saida / "figuras")
        print("Desenvolvimento concluído; o teste continua reservado.")
        print("Escolhido pela validação:", winner["id"])
        print("Modelo | MAE CV por animal (g)")
        for result in representatives:
            print(f"{result['id']} | {result['validacao']['mae_medio_por_animal']:.6f}")
        print("Para reproduzir a fase final: python projeto_03.py --confirmar-teste")
        return

    # Fase final única: comparar procedimentos já fixados; nenhuma seleção posterior.
    test_results, test_predictions = [], {}
    linear_parameters = {}
    for result in representatives:
        model = make_model(result)
        model.fit(development[FEATURES], development["weight"])
        prediction = model.predict(test[FEATURES])
        metrics, prediction_frame = evaluate(test, prediction)
        test_predictions[result["id"]] = prediction_frame
        test_results.append({
            "id": result["id"], "familia": result["familia"], "parametros": result["parametros"],
            "escolhido_em_validacao": result["id"] == winner["id"],
            "mae_cv_por_animal": result["validacao"]["mae_medio_por_animal"],
            "mae_cv_registos_reunidos": result["validacao"]["mae_registos_reunidos"],
            "teste": metrics,
        })
        if result["familia"] in {"idade", "idade_dieta"}:
            estimator = model.named_steps["modelo"]
            names = model.named_steps["preparacao"].get_feature_names_out().tolist()
            linear_parameters[result["id"]] = {
                "intercecao": float(estimator.intercept_),
                "coeficientes": {n: float(v) for n, v in zip(names, estimator.coef_)},
            }

    counts = data.groupby("Chick").agg(Diet=("Diet", "first"), n=("weight", "size"),
                                        ultima_idade=("Time", "max"))
    results = {
        "projeto": "Previsão da massa corporal com medições repetidas",
        "semente": SEED,
        "ambiente": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
        "dados": {
            "sha256": digest, "n_registos": len(data), "n_animais": int(data["Chick"].nunique()),
            "n_celulas_em_falta": int(data.isna().sum().sum()),
            "idades_programadas": SCHEDULE,
            "n_visitas_ausentes": 50 * len(SCHEDULE) - len(data),
            "n_animais_com_seguimento_completo": int(counts["n"].eq(12).sum()),
            "animais_com_seguimento_incompleto": counts.loc[counts["n"].lt(12)].reset_index().to_dict("records"),
            "url_csv": "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ChickWeight.csv",
            "url_documentacao": "https://search.r-project.org/R/refmans/datasets/html/ChickWeight.html",
        },
        "protocolo": {
            "pergunta": "Novos animais nas idades observadas, sob dietas representadas",
            "preditores": FEATURES,
            "colunas_excluidas_dos_preditores": ["Chick", "rownames"],
            "metrica_principal": "média dos MAE individuais, com igual peso por animal",
            "metrica_secundaria": "MAE de todos os registos reunidos",
            "definicao_erro": "observado menos previsto",
            "nao_ha_imputacao_de_pesos_ausentes": True,
            "uma_fase_final_de_teste_sem_nova_selecao": True,
            "confirmar_teste": True,
            "ajustes_cv": len(cv) * 4,
            "ajustes_finais": len(test_results),
        },
        "divisao": checks,
        "selecao": selection,
        "candidatos_validacao": cv,
        "resultados_finais": test_results,
        "parametros_regressoes_ajustadas_desenvolvimento": linear_parameters,
    }
    write_json(args.saida / "resultados.json", results)
    pd.concat([p.assign(candidato=k) for k, p in test_predictions.items()], ignore_index=True).to_csv(
        args.saida / "previsoes_teste.csv", index=False, float_format="%.10f")
    make_figures(development, representatives, test_results, winner["id"], test_predictions,
                 args.saida / "figuras")
    print("Escolhido antes do teste:", winner["id"])
    print("Modelo | MAE CV/animal | MAE teste/animal | MAE teste/registo (g)")
    for result in test_results:
        print(f"{result['id']} | {result['mae_cv_por_animal']:.6f} | "
              f"{result['teste']['mae_medio_por_animal']:.6f} | "
              f"{result['teste']['mae_registos_reunidos']:.6f}")


if __name__ == "__main__":
    main()
