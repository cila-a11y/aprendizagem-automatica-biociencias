# Avaliar previsões guardadas em ficheiros

Execute a partir da pasta principal:

```bash
python reproducao/previsoes/avaliar_previsoes.py classificacao
python reproducao/previsoes/avaliar_previsoes.py abalone
```

O primeiro comando reproduz a atividade fictícia do apêndice A, com seis
amostras. Guarda a tabela ligada por identificador, os erros, a matriz de
confusão, as medidas e as figuras em `exemplo_previsoes/resultados/`.
Uma medida com denominador zero fica assinalada como `nao_definido`.
As curvas exigem pontuações finitas e ambas as classes de referência.

O segundo comando liga as exportações reais do estudo Abalone e guarda os
resultados em `reproducao/previsoes/resultados_abalone/`. Reconstitui a mesma
avaliação do capítulo 14; não é uma nova validação independente.

Ambos os comandos verificam tabelas não vazias, identificadores não omissos
e únicos, cobertura dos identificadores esperados e correspondência
completa entre referência e previsões. Não ajustam modelos nem alteram os
limiares ou as previsões recebidas.

Para avaliar outros ficheiros com o mesmo formato:

```bash
python reproducao/previsoes/avaliar_previsoes.py classificacao --referencia meus_dados/referencia.csv --previsoes meus_dados/previsoes.csv --ids-esperados meus_dados/amostras_previstas.csv --saida meus_resultados
```

O ficheiro dos identificadores esperados precisa da coluna `id_registo`.
Se não for fornecido, o script apenas confirma a correspondência entre os
dois ficheiros: não pode detetar amostras ausentes de ambos.
Na classificação, o significado de `maligno`, de 0/1 e do sentido crescente
de `score_positivo` deve coincidir com o descrito na nota do procedimento.
Uma pontuação não é automaticamente uma probabilidade calibrada.
