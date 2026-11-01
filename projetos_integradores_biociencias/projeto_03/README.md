# Projeto 3 — Massa corporal com medições repetidas

Prevê-se a massa de **novos pintos**, usando idade e dieta, nas idades observadas entre o nascimento e o dia 21. A unidade de divisão e a unidade principal de avaliação são os animais.

## Executar

```bash
python projeto_03.py
python projeto_03.py --validar-apenas
python projeto_03.py --confirmar-teste
python validar_resultados.py
```

O primeiro comando executa apenas o desenvolvimento e guarda as decisões; não calcula nem escreve previsões ou métricas do teste. O argumento `--confirmar-teste` permite reproduzir a análise final depois de fixar as decisões. A execução de desenvolvimento não substitui os ficheiros de resultados finais já existentes.

O programa utiliza apenas `dados/ChickWeight.csv`. Não faz transferências pela Internet. Foi executado com Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 e Matplotlib 3.10.8. A computação usa CPU e uma procura pequena: 60 ajustes de validação e, quando confirmados, quatro ajustes finais.

## Protocolo fixado

- Semente 2026; 40 pintos de desenvolvimento e dez de teste, com divisão aleatória dentro de cada dieta.
- Quatro partes de validação, cada uma com dez animais (4/2/2/2 por dieta); todos os registos de um pinto acompanham o seu identificador.
- A divisão é guardada em `divisao.json` antes de ajustar modelos. Execuções posteriores reutilizam-na e verificam o hash do CSV.
- A seleção minimiza a média dos 40 MAE individuais calculados com previsões fora de amostra.
- A árvore usa profundidades máximas 2/3/4/5 e mínimos de 4/8/12 **registos** por folha. Compara-se com mediana do treino e duas regressões, uma apenas com idade e outra com idade e dieta categórica.
- `Chick` e `rownames` nunca entram nos preditores. Os níveis das dietas são metadados previamente conhecidos, não valores aprendidos com os resultados do teste.
- O vencedor e os quatro procedimentos finais são guardados em `selecao_antes_teste.json` antes de uma única fase de avaliação final. Os resultados do teste não alteram a seleção.
- Os erros de previsão seguem a convenção do livro: **observado menos previsto**. Um erro positivo indica uma previsão abaixo do valor observado.

## Ficheiros gerados

`desenvolvimento.json` contém a seleção e os resultados de validação. Depois da confirmação, `resultados.json` contém também a avaliação final, versões, parâmetros, auditoria dos dados e erros por animal. `previsoes_validacao.csv` e `previsoes_teste.csv` permitem recalcular os erros. `validacao_divisao.json` regista as verificações dos grupos. A pasta `figuras` contém dois gráficos de desenvolvimento e um de teste em PDF e PNG.

Resultado da execução de referência: a árvore de profundidade máxima 5 e mínimo de quatro registos por folha apresenta MAE por animal de **23,607785 g na validação** e **18,048911 g no teste**. O MAE reunido no teste é **17,862305 g**. Estes números descrevem a divisão fixada, não uma garantia para outra população.

## Dados e proveniência

- Documentação oficial: https://search.r-project.org/R/refmans/datasets/html/ChickWeight.html
- CSV público: https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ChickWeight.csv
- Fonte citada pelo R: Crowder, M. J., e Hand, D. J. (1990). *Analysis of Repeated Measures*. Chapman & Hall, Londres, exemplo 5.3, pp. 74–78.
- SHA-256 do CSV incluído: `db0256706f76174732c06a29547ba3247debbbaacef488ccc255ba54fac2d703`.

Há 578 registos, 50 animais e zero células em falta, mas cinco trajetórias são incompletas. Não se imputam as 22 visitas ausentes nem se atribui uma causa à sua ausência. A análise não estima efeitos causais de dietas nem valida previsões para idades ou contextos não representados.
