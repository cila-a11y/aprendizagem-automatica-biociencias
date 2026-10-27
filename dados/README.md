# Dados usados nos exemplos

Os CSV desta pasta são as cópias usadas no livro. Utilize-as para reproduzir as análises: uma nova divisão dos dados pode produzir resultados diferentes.

## Ficheiros e divisão dos dados

| Conjunto | Desenvolvimento | Teste reservado | Resposta nos estudos preditivos |
| --- | --- | --- | --- |
| Palmer Penguins | `penguins_desenvolvimento.csv`: 275 linhas | `penguins_teste_reservado.csv`: 69 linhas | `body_mass_g`, massa corporal em gramas |
| Wisconsin Diagnostic Breast Cancer | `wisconsin_desenvolvimento.csv`: 455 linhas | `wisconsin_teste_reservado.csv`: 114 linhas | `maligno`: 1 para maligno e 0 para benigno |
| Abalone | `abalone_desenvolvimento.csv`: 3341 linhas | `abalone_teste_reservado.csv`: 836 linhas | `Rings`, contagem de anéis |

O desenvolvimento serve para explorar os dados, ajustar procedimentos e comparar alternativas por validação. O teste reservado serve para avaliar o procedimento depois de fixadas essas escolhas. Na reprodução do percurso do livro, execute primeiro as etapas de desenvolvimento e só depois a avaliação final. Conhecer os resultados publicados e repetir o cálculo não cria uma nova avaliação independente para escolhas posteriores.

As divisões reservam 20% dos registos, usam `random_state=2026` e são guardadas pela ordem de `id_registo`. Wisconsin usa estratificação pela resposta; os outros dois conjuntos usam divisão aleatória sem estratificação. Dentro de cada conjunto, desenvolvimento e teste não partilham identificadores.

`id_registo` identifica a linha na tabela de origem, começando em 1. Os identificadores pertencem a cada conjunto: o registo 1 de Wisconsin não tem relação com o registo 1 de Abalone. Esta coluna permite ligar respostas e previsões e conferir as divisões; não é uma variável preditora nem demonstra independência biológica entre linhas.

Os arquivos [dados_desenvolvimento.zip](../dados_desenvolvimento.zip) e [dados_teste_reservado.zip](../dados_teste_reservado.zip) contêm, respetivamente, os três CSV de desenvolvimento e os três CSV de teste. Ao descompactá-los, os ficheiros ficam na pasta `dados`.

## Palmer Penguins

Cada linha descreve um pinguim observado. O ficheiro `penguins_original.csv` conserva a tabela simplificada de origem, com 344 linhas e sem o identificador acrescentado às cópias de desenvolvimento e teste.

| Coluna | Significado | Tipo ou unidade |
| --- | --- | --- |
| `id_registo` | Identificação da linha de origem | Identificador |
| `species` | Espécie: Adelie, Chinstrap ou Gentoo | Categoria |
| `island` | Ilha: Biscoe, Dream ou Torgersen | Categoria |
| `bill_length_mm` | Comprimento do bico | Milímetros |
| `bill_depth_mm` | Profundidade do bico | Milímetros |
| `flipper_length_mm` | Comprimento da barbatana | Milímetros |
| `body_mass_g` | Massa corporal | Gramas |
| `sex` | Sexo registado: `male` ou `female` | Categoria; pode estar em falta |
| `year` | Ano de observação | Ano |

No desenvolvimento, duas linhas não têm as quatro medidas corporais e onze linhas não têm sexo registado. K-means e ACP usam as 273 linhas completas nas quatro medidas. A regressão usa as 273 linhas com massa corporal observada; nesta cópia, essas linhas também têm os dois comprimentos utilizados como entradas. Os campos em falta são lidos como valores omissos, nunca como zero.

Os agrupamentos e a ACP usam comprimento e profundidade do bico, comprimento da barbatana e massa corporal. A espécie pode ser acrescentada à representação depois do ajuste para interpretar os resultados. A regressão principal prevê a massa a partir dos comprimentos da barbatana e do bico. Uma omissão numa coluna que não participa numa análise, como `sex` nestes exemplos, não justifica eliminar essa linha.

A divisão aleatória representa a mistura de ilhas e anos disponível. Avaliar a previsão para outro local ou período exige uma divisão que corresponda a essa utilização.

## Wisconsin Diagnostic Breast Cancer

Cada linha reúne características de núcleos celulares em imagens de uma amostra mamária obtida por punção aspirativa. A resposta é o diagnóstico registado para a amostra. O exemplo estuda classificação de amostras nesse contexto; não estima risco de doença futura.

Os CSV contêm as 30 características fornecidas pelo scikit-learn, `id_registo` e `maligno`. Os modelos principais usam apenas `mean radius` e `mean texture`.

| Coluna ou família | Significado |
| --- | --- |
| `id_registo` | Identificador acrescentado; não é preditor |
| `mean radius` | Média das medidas de raio dos núcleos |
| `mean texture` | Média da característica de textura, baseada na variabilidade dos níveis de cinzento |
| `mean ...` | Médias das dez características de origem |
| `... error` | Erros-padrão das dez características de origem |
| `worst ...` | Resumos de valores extremos das dez características, segundo a definição da fonte |
| `maligno` | Resposta: 1 para maligno e 0 para benigno |

As dez características são `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry` e `fractal dimension`. Conserve os nomes, espaços e escala dos CSV; não atribua unidades físicas que a documentação não estabelece. A [descrição do conjunto no scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset) e a [fonte UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) apresentam as definições de origem.

Na distribuição do scikit-learn, `target=0` corresponde a maligno e `target=1` a benigno. A preparação aplica `maligno = 1 - target`. Os CSV desta pasta já incluem essa transformação: não volte a recodificar a resposta.

O desenvolvimento contém 170 amostras malignas e 285 benignas; o teste contém 42 malignas e 72 benignas. Não existem omissões nestes CSV. As proporções descrevem esta coleção de amostras e não estimam a frequência de doença numa população geral.

## Abalone

Cada linha representa um exemplar na tabela publicada. O estudo prevê a contagem de anéis da concha a partir de três medidas escolhidas antes do ajustamento: `Length`, `Diameter` e `Whole_weight`.

| Coluna | Significado | Uso no estudo |
| --- | --- | --- |
| `id_registo` | Número da linha de origem | Identificador |
| `Sex` | Categorias M, F e I; I identifica um exemplar juvenil | Não utilizada |
| `Length` | Comprimento da concha | Entrada |
| `Diameter` | Diâmetro perpendicular ao comprimento | Entrada |
| `Height` | Altura do exemplar com o corpo na concha | Não utilizada |
| `Whole_weight` | Massa do exemplar inteiro | Entrada |
| `Shucked_weight` | Massa da carne | Não utilizada |
| `Viscera_weight` | Massa das vísceras | Não utilizada |
| `Shell_weight` | Massa da concha depois de seca | Não utilizada |
| `Rings` | Contagem de anéis registada | Resposta |

A documentação indica que os valores contínuos foram divididos por 200 antes da disponibilização. Os CSV conservam esses valores. Assim, um valor numérico de `Length` não deve receber diretamente a unidade milímetros, nem um valor de `Whole_weight` a unidade gramas. `Rings` mantém a unidade contagem de anéis; os erros de previsão são apresentados nessa mesma unidade.

A fonte associa `Rings + 1.5` a uma convenção de idade. O estudo do livro avalia a previsão da contagem registada, sem tratar essa transformação como uma idade cronológica individual conhecida com exatidão. A categoria I de `Sex` significa juvenil; não é um terceiro sexo nem um nível de uma escala numérica ordenada.

Na utilização proposta, as três entradas podem ser medidas antes de processar o exemplar. As massas de carne, vísceras e concha seca não entram porque exigem outro tipo de preparação. A análise não inclui automaticamente todas as colunas numéricas.

Não há omissões na tabela pública utilizada. A documentação informa que foram removidos registos com omissões antes da sua publicação. A preparação destes materiais conserva todas as 4177 linhas publicadas e verifica se as três entradas são finitas e positivas e se `Rings` é uma contagem inteira positiva. Uma violação interrompe o procedimento; não é eliminada silenciosamente.

A tabela não fornece os identificadores de local, data ou lote necessários para organizar uma avaliação por esses grupos. Os resultados referem-se à divisão aleatória aqui descrita; a transferência para outra região, época ou processo de medição permanece por avaliar.

## Preparação e proveniência

Os scripts e registos correspondentes são:

- [Preparação dos pinguins e de Wisconsin](../reproducao/dados/preparar_dados_iniciais.py) e [registo de origem](../reproducao/dados/origens.json).
- [Preparação e análise Abalone](../reproducao/abalone/codigo/reproduzir_abalone.py), [protocolo](../reproducao/abalone/protocolo.json) e [originais](../reproducao/abalone/originais/).
- [Fontes, atribuições e licenças dos dados](../FONTES_DADOS.md).

Para repetir as análises, comece pelas cópias fornecidas. A preparação dos dados permite conferir a origem e as divisões; o ajustamento e a avaliação são etapas próprias, descritas no [guia principal](../README.md).
