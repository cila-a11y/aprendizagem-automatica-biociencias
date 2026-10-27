# Fundamentos, exploração de grupos e componentes principais

Este percurso acompanha os capítulos 1 a 6. Usa o ficheiro `dados/penguins_desenvolvimento.csv`, fornecido na pasta principal dos materiais. Os exemplos pequenos de preparação e de avaliação são fictícios e estão identificados no código e nos nomes das tabelas.

## Executar

Instale primeiro as dependências indicadas no `README.md` principal. A partir da pasta principal dos materiais, execute:

```bash
python reproducao/fundamentos/reproduzir_fundamentos.py
```

Para executar apenas alguns capítulos:

```bash
python reproducao/fundamentos/reproduzir_fundamentos.py --capitulos 5 6
```

O programa localiza os dados a partir da sua própria localização. Também pode ser chamado por um caminho absoluto, a partir de outra pasta. Não necessita de acesso à Internet nem abre os dados de teste.

## Encontrar os resultados

As saídas ficam em `reproducao/fundamentos/resultados/`.

| Capítulo | Operações | Tabelas principais |
|---|---|---|
| 1 | Ler, selecionar colunas, filtrar massas e converter unidades | `tabelas/cap01_primeiras_linhas.csv`, `tabelas/cap01_massa_pelo_menos_4500g.csv` |
| 2 | Contar omissões e categorias, procurar repetições e resumir medidas | `tabelas/cap02_inspecao.csv`, `tabelas/cap02_resumos_medidas.csv`, `tabelas/cap02_especies.csv` |
| 3 | Construir uma referência, preparar entradas no treino e acompanhar cinco partes | `tabelas/cap03_preparacao_ficticia.csv`, `tabelas/cap03_cinco_partes_ficticias.csv` |
| 4 | Avaliar classificações e previsões quantitativas fornecidas | `tabelas/cap04_medidas_classificacao_ficticia.csv`, `tabelas/cap04_regressao_ficticia_resumo.csv` |
| 5 | Padronizar quatro medidas, ajustar K-means e comparar K = 2, 3 e 4 | `tabelas/cap05_perfis_K3.csv`, `tabelas/cap05_comparacao_K.csv`, `tabelas/cap05_rotulos_e_silhuetas.csv` |
| 6 | Ajustar a ACP, apresentar coeficientes, coordenadas e variabilidade | `tabelas/cap06_coeficientes.csv`, `tabelas/cap06_coordenadas.csv`, `tabelas/cap06_variabilidade.csv` |

Em `figuras/`, cada gráfico está disponível em PNG e PDF. Os nomes `u2_`, `u5_` e `u6_` identificam os gráficos dos capítulos respetivos. Os dois histogramas `exercicio_2_8_` permitem explorar o comprimento e a profundidade do bico. A execução regenera estas representações a partir dos dados; o aspeto gráfico pode diferir da paginação do livro.

Os ficheiros `capitulo_01.json` a `capitulo_06.json` contêm resumos da execução. `execucao.json` regista os capítulos executados, as versões das bibliotecas e a identificação digital do ficheiro de dados. Pode abrir os CSV numa folha de cálculo ou com `pandas.read_csv`.

## Convenções para interpretar as saídas

- Os pinguins têm 275 registos de desenvolvimento. Nos capítulos 5 e 6 são usadas as 273 observações completas nas quatro medidas. O sexo não participa nessa seleção.
- A variância amostral dos resumos usa divisor `n - 1`. A escala aplicada por `StandardScaler` usa divisor `n`.
- Os grupos são numerados a partir de 1, acrescentando uma unidade aos rótulos produzidos por K-means. A espécie é consultada apenas depois de obter os grupos.
- Os coeficientes da ACP correspondem a `components_.T`. Uma inversão consistente dos sinais dos coeficientes e das coordenadas de uma componente representa a mesma direção.
- A matriz de confusão tem observados nas linhas e previstos nas colunas, com a classe positiva em primeiro lugar. Nos exemplos fictícios, positivo significa presença da bactéria.
- Uma célula vazia nas tabelas de avaliação representa uma medida não definida por ter denominador nulo. Por exemplo, o VPP não é substituído por zero quando não há previsões positivas.
- Nos exemplos de regressão, o resíduo é o valor observado menos o previsto e os erros estão em gramas.

Para conservar uma execução numa pasta diferente, acrescente `--saida caminho/da/pasta`. A execução habitual substitui apenas os resultados dos capítulos pedidos.
