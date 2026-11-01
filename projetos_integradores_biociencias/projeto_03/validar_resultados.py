#!/usr/bin/env python3
"""Audita os grupos e recalcula as métricas a partir dos CSV de previsões.

Executar depois de projeto_03.py --confirmar-teste.
Não ajusta modelos nem altera ficheiros.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    data = pd.read_csv(HERE / "dados" / "ChickWeight.csv")
    split = json.loads((HERE / "divisao.json").read_text(encoding="utf-8"))
    results = json.loads((HERE / "resultados.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((HERE / "dados" / "ChickWeight.csv").read_bytes()).hexdigest()
    require(digest == split["sha256_dados"] == results["dados"]["sha256"], "Hash dos dados incoerente.")
    development = set(split["animais_desenvolvimento"])
    test = set(split["animais_teste"])
    require(len(development) == 40 and len(test) == 10, "Contagens de animais incorretas.")
    require(development.isdisjoint(test), "Sobreposição entre desenvolvimento e teste.")
    require(development | test == set(data.Chick), "A divisão não cobre todos os animais.")
    fold_map = {int(k): int(v) for k, v in split["parte_validacao_por_animal"].items()}
    require(set(fold_map) == development, "Identidades das partes de validação incorretas.")
    require(set(fold_map.values()) == {1, 2, 3, 4}, "Índices das partes incorretos.")
    animal_diets = data[["Chick", "Diet"]].drop_duplicates().set_index("Chick").Diet
    for fold in range(1, 5):
        validation = {chick for chick, value in fold_map.items() if value == fold}
        training = development - validation
        require(len(validation) == 10 and len(training) == 30, "Dimensão de uma parte incorreta.")
        require(validation.isdisjoint(training | test), "Sobreposição de animais numa parte.")
        counts = animal_diets.loc[list(validation)].value_counts().to_dict()
        require(counts == {1: 4, 2: 2, 3: 2, 4: 2}, "Composição de dietas incorreta.")

    candidates = {c["id"]: c for c in results["candidatos_validacao"]}
    final = {c["id"]: c for c in results["resultados_finais"]}
    expected_ids = set(results["selecao"]["representantes_pre_fixados"])
    require(expected_ids == set(final), "Mudaram os procedimentos na fase final.")
    best = min(candidates.values(), key=lambda c: c["validacao"]["mae_medio_por_animal"])
    require(best["id"] == results["selecao"]["vencedor"], "A seleção não segue a métrica principal.")
    for filename, expected_animals, stage in [
        ("previsoes_validacao.csv", development, "validacao"),
        ("previsoes_teste.csv", test, "teste"),
    ]:
        predictions = pd.read_csv(HERE / filename)
        require(set(predictions.candidato) == expected_ids, "Candidatos nas previsões incoerentes.")
        expected_rows = set(data.loc[data.Chick.isin(expected_animals), "rownames"])
        for model_id, part in predictions.groupby("candidato"):
            require(set(part.Chick) == expected_animals, "Previsões de animais inesperados.")
            require(set(part.rownames) == expected_rows, "Faltam ou sobram observações.")
            require(not part.rownames.duplicated().any(), "Observação prevista mais de uma vez.")
            original = data.set_index("rownames").loc[part.rownames]
            for column in ["Chick", "Diet", "Time", "weight"]:
                require(np.array_equal(part[column].to_numpy(), original[column].to_numpy()),
                        f"Dados de referência alterados em {column}.")
            residual = part.weight - part.previsao
            error = residual.abs()
            require(np.allclose(part.erro, residual, atol=1e-8, rtol=0),
                    "Convenção do erro de previsão incorreta: deve ser observado menos previsto.")
            require(np.allclose(part.erro_absoluto, error, atol=1e-8, rtol=0),
                    "Os erros absolutos não coincidem com as previsões.")
            per_animal = error.groupby(part.Chick).mean()
            metrics = candidates[model_id]["validacao"] if stage == "validacao" else final[model_id]["teste"]
            require(np.isclose(per_animal.mean(), metrics["mae_medio_por_animal"], atol=1e-8, rtol=0),
                    "O MAE por animal não coincide com o CSV.")
            require(np.isclose(error.mean(), metrics["mae_registos_reunidos"], atol=1e-8, rtol=0),
                    "O MAE reunido não coincide com o CSV.")
            if stage == "validacao":
                require(np.array_equal(part.Chick.map(fold_map), part.parte_validacao),
                        "A parte atribuída a uma previsão de validação está errada.")
    print("Auditoria concluída: zero sobreposição de animais; divisão completa e estratificada.")
    print("MAE por animal e MAE reunido reproduzidos a partir de todas as previsões guardadas.")
    print("Vencedor da validação:", results["selecao"]["vencedor"])


if __name__ == "__main__":
    main()
