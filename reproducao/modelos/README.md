# Modelos supervisionados

Scripts e resultados dos capítulos 7–13 e da avaliação final dos exemplos de Wisconsin e Palmer Penguins que abre o capítulo 14.

## Executar

Instale as dependências indicadas no `README.md` principal. Na pasta principal do repositório, execute:

```bash
python reproducao/modelos/reproduzir_modelos.py desenvolvimento
python reproducao/modelos/reproduzir_modelos.py teste
```

O primeiro comando lê apenas os ficheiros de desenvolvimento, calcula as comparações e guarda `resultados/decisao.json`. O segundo verifica essa decisão, reajusta os procedimentos escolhidos no desenvolvimento e abre os testes reservados. Para executar o percurso completo de uma vez:

```bash
python reproducao/modelos/reproduzir_modelos.py tudo
```

Os caminhos são resolvidos a partir da localização do script. Também pode executar o ficheiro indicando o seu caminho absoluto, a partir de outra pasta. Não é necessária ligação à Internet. Uma nova execução substitui os resultados gerados com os mesmos nomes.

## Encontrar os resultados

Os CSV e JSON ficam em `resultados/`. Os gráficos ficam em `figuras/`, nos formatos PNG e PDF. Os nomes dos gráficos conservam o número do capítulo a que dizem respeito.

| Capítulo | Conteúdo | Ficheiros principais em `resultados/` |
|---|---|---|
| 7 | Regressões simples e múltipla; referências média e mediana; previsões e resíduos | `pinguins_validacao_resumo.csv`, `pinguins_previsoes_validacao.csv`, `coeficientes_desenvolvimento_completo.csv` |
| 8 | Logística, limiares, matriz de confusão, ROC, curva VPP–sensibilidade e AP | `wisconsin_logistica_limiares.csv`, `wisconsin_matriz_050.csv`, `wisconsin_logistica_roc.csv`, `wisconsin_logistica_vpp_sensibilidade.csv` |
| 9 | Árvores de profundidade 2, 4 e sem limite | `wisconsin_validacao_resumo.csv`, `pinguins_validacao_resumo.csv`, `arvore_classificacao_nos.csv`, `arvore_regressao_nos.csv` |
| 10 | Florestas com folha mínima 1 e 5 | As duas tabelas `*_validacao_resumo.csv` e as tabelas `*_validacao_por_parte.csv` |
| 11 | Redes com 8 e 32 unidades; curvas durante 200 épocas | `wisconsin_validacao_resumo.csv`, `rede_8_curva_200_epocas.csv`, `rede_32_curva_200_epocas.csv` |
| 12 | Médias por parte, medidas agregadas, preparação dentro do treino e divisão por doente | `*_validacao_por_parte.csv`, `demonstracao_preparacao_correta_e_indevida.csv`, `exemplo_ficticio_divisao_grupos.csv` |
| 13 | Calibração, resíduos, importância por permutação e correlação entre entradas | `wisconsin_calibracao.csv`, `wisconsin_permutacao_resumo.csv`, `wisconsin_permutacao_repeticoes.csv`, `wisconsin_correlacoes.csv` |
| 14 | Logística no teste Wisconsin e floresta no teste dos pinguins | `wisconsin_teste_resumo.csv`, `pinguins_teste_resumo.csv`, `*_teste_previsoes.csv` |

`wisconsin_partes.csv` e `pinguins_partes.csv` identificam a parte em que cada registo foi avaliado. As tabelas de previsões conservam `id_registo`, a resposta observada e as previsões de cada procedimento. Assim, pode relacionar um resultado com a linha que lhe deu origem sem depender da posição da linha.

## Convenções para comparar com o livro

- A classe positiva é `maligno = 1`. Uma probabilidade igual ao limiar recebe classe positiva.
- As massas e os erros de regressão são expressos em gramas. As duas linhas sem massa no desenvolvimento são identificadas em `pinguins_sem_resposta.csv`; não se imputa a resposta para avaliar previsões.
- No capítulo 7, MAE, RMSE e R² são calculados sobre todas as previsões fora do treino reunidas. Essas colunas começam por `agregado_` na tabela de resumo. Nas comparações posteriores, o critério é a média das medidas nas cinco partes. Não são, em geral, o mesmo resumo.
- `*_DP_partes` usa desvio-padrão amostral (`ddof=1`). Na importância por permutação usa-se o desvio devolvido pelo scikit-learn (`ddof=0`); a coluna identifica essa convenção. Nenhum destes desvios é um intervalo de confiança.
- A validação cruzada das redes usa `fit`. As curvas de aprendizagem são outra demonstração: divisão interna 70/30 e 200 chamadas a `partial_fit`. Os ficheiros mantêm as duas avaliações separadas.
- Na curva VPP–sensibilidade, a última linha é o ponto convencional sem limiar. Uma célula vazia para VPP corresponde a uma razão sem denominador positivo; não foi substituída por zero. O primeiro limiar da curva ROC pode ser infinito, representando nenhuma previsão positiva.
- `demonstracao_preparacao_correta_e_indevida.csv` contém um contraste deliberadamente incorreto do capítulo 12. Essa preparação global não participa na seleção dos modelos nem na avaliação final.
- O protocolo completo está em `protocolo.json`. Os resultados usam as sementes, a ordem das linhas e as configurações aí indicadas. Consulte também as versões das bibliotecas registadas em `resultados/decisao.json`.

## Ler a avaliação final

No desenvolvimento, a logística é escolhida pela maior AUC média e o limiar 0,20 pela regra de sensibilidade de pelo menos 90%, seguida da maior especificidade. Para os pinguins, a floresta com folha mínima 5 é escolhida pelo menor MAE médio. A etapa final compara apenas os procedimentos escolhidos com as respetivas referências.

O teste Wisconsin contém 114 amostras; o teste dos pinguins contém 69 animais. Os resultados de referência incluem AUC aproximadamente 0,946 para a logística e MAE aproximadamente 293,7 g para a floresta. A reprodução destes resultados conhecidos permite estudar o procedimento. Não constitui uma nova avaliação independente de decisões tomadas depois de os conhecer.

Não altere sementes ou configurações para tentar obter números mais favoráveis no teste. Para explorar alternativas, use o desenvolvimento e trate os resultados como uma nova análise, com as suas próprias escolhas e limitações.
