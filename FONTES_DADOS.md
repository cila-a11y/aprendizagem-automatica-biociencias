# Fontes e licenças dos dados

Os materiais utilizam três conjuntos públicos. Os nomes das colunas, as transformações efetuadas e a separação entre desenvolvimento e teste estão descritos em [dados/README.md](dados/README.md). As licenças seguintes aplicam-se aos respetivos dados de origem e às cópias aqui distribuídas; esta página não atribui uma licença ao texto do livro nem ao código.

## Palmer Penguins

**Fonte:** [projeto Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/), com dados de Kristen Gorman e do programa Palmer Station LTER. A tabela simplificada está disponível em [penguins.csv](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv).

**Licença dos dados:** [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), conforme indicado pelo projeto.

**Referência do recurso:** Horst, A. M., Hill, A. P., e Gorman, K. B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. [DOI: 10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218).

A cópia `dados/penguins_original.csv` conserva a tabela simplificada de 344 registos. A preparação acrescenta `id_registo` a partir da posição da linha, reserva 20% dos registos com a semente 2026 e guarda as duas partes pela ordem do identificador. Conservam-se as colunas e os valores omissos da fonte. A seleção das observações que participam em cada análise é feita explicitamente nos respetivos scripts.

Ficheiros distribuídos: `penguins_original.csv`, `penguins_desenvolvimento.csv` e `penguins_teste_reservado.csv`, na pasta `dados`.

## Breast Cancer Wisconsin Diagnostic

**Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). A preparação usa a cópia incluída no scikit-learn, através de [`load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

**Licença dos dados:** [Creative Commons Attribution 4.0 International — CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), conforme indicado pelo UCI.

**Atribuição:** Wolberg, W., Mangasarian, O., Street, N., e Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Conjunto de dados]. UCI Machine Learning Repository. [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).

A preparação conserva as 30 características da cópia do scikit-learn e acrescenta `id_registo`. Recodifica a resposta como `maligno = 1 - target`, para que 1 signifique maligno e 0 benigno. Reserva 20% dos 569 registos com a semente 2026 e estratificação pela resposta, guardando cada parte pela ordem do identificador. Estas são as transformações aplicadas às cópias distribuídas; os valores das características não são padronizados nos CSV.

Ficheiros distribuídos: `dados/wisconsin_desenvolvimento.csv` e `dados/wisconsin_teste_reservado.csv`.

## Abalone

**Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1/abalone), incluindo os ficheiros originais [abalone.data](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data) e [abalone.names](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names).

**Licença dos dados:** [Creative Commons Attribution 4.0 International — CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), conforme indicado pelo UCI.

**Atribuição:** Nash, W., Sellers, T., Talbot, S., Cawthorn, A., e Ford, W. (1994). *Abalone* [Conjunto de dados]. UCI Machine Learning Repository. [DOI: 10.24432/C55C7W](https://doi.org/10.24432/C55C7W).

A preparação conserva as 4177 linhas publicadas, acrescenta os nomes das colunas e `id_registo`, reserva 20% dos registos com a semente 2026 e guarda as duas partes pela ordem do identificador. Não há estratificação. Os valores contínuos mantêm a escala publicada; não são reconvertidos em unidades físicas. Não se aplica a transformação de anéis em idade à resposta utilizada no estudo.

Os originais acompanham os materiais em `reproducao/abalone/originais`. As cópias preparadas são `dados/abalone_desenvolvimento.csv` e `dados/abalone_teste_reservado.csv`.

## Exemplos fictícios

Os seis registos A01–A06 da pasta `exemplo_previsoes` foram construídos para praticar a avaliação de previsões recebidas em ficheiros. Não são registos de Wisconsin, de Abalone ou de participantes reais. As pequenas tabelas criadas pelos scripts para ilustrar preparação, métricas ou divisão por doente também são identificadas como fictícias nos respetivos exemplos.

Os ficheiros de referência e previsão em `reproducao/abalone/resultados` têm outra origem: são exportações da avaliação do estudo Abalone. Conservam os identificadores para permitir a ligação entre as linhas, mesmo quando a ordem dos dois ficheiros difere.

## Verificar a origem das cópias

O ficheiro [reproducao/dados/origens.json](reproducao/dados/origens.json) regista a fonte, a preparação e os hashes das cópias iniciais. O estudo Abalone conserva o [protocolo](reproducao/abalone/protocolo.json) e os registos de preparação e resultados na sua pasta. Os hashes permitem conferir a identidade dos ficheiros; não substituem a leitura do significado das variáveis ou das condições de avaliação.
