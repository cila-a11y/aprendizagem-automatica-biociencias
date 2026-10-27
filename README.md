# Aprendizagem automática em Biociências

Materiais de apoio a *Aprendizagem automática em Biociências — Uma introdução
com exemplos em Python*. Aqui encontra os dados, os programas e os resultados
necessários para executar os exemplos e reproduzir os estudos.

## Começar

1. Descarregue o repositório em **Code → Download ZIP** e descompacte-o.
2. Siga o [guia de execução](GUIA_EXECUCAO.md) para trabalhar no seu computador
   ou no Google Colab.
3. Escolha o percurso correspondente ao capítulo que está a ler.

| Percurso | Conteúdo | Instruções |
|---|---|---|
| Capítulos 1–6 | Python, tabelas, gráficos, avaliação, K-means e ACP | [Fundamentos](reproducao/fundamentos/README.md) |
| Capítulos 7–13 | Regressão, classificação, árvores, florestas, redes, validação e interpretação | [Modelos](reproducao/modelos/README.md) |
| Capítulo 14 | Prever a contagem de anéis em abalones | [Estudo Abalone](reproducao/abalone/LEIA_ME.txt) |
| Capítulo 14 e apêndice A | Ligar e avaliar previsões recebidas em ficheiros | [Previsões em ficheiros](reproducao/previsoes/README.md) |
| Apêndice B | Fontes, variáveis, omissões e divisões dos dados | [Dicionário dos dados](dados/README.md) |

## Reproduzir os estudos

Depois de instalar as dependências, abra o terminal na pasta principal:

```bash
python reproduzir.py desenvolvimento
python reproduzir.py teste
python reproduzir.py previsoes
python reproduzir.py verificar
```

`desenvolvimento` executa os exemplos e as comparações sem abrir as respostas
do teste. `teste` lê as decisões guardadas e reproduz as avaliações finais.
`previsoes` executa as atividades de ligação por identificador.
`verificar` confere os dados, os ficheiros e os principais resultados.
Para reconstruir também as divisões dos dados e executar a sequência completa:

```bash
python reproduzir.py tudo
```

Os dados e os resultados de referência já estão incluídos. Reproduzir um
estudo depois de conhecer os seus resultados permite verificar e compreender
o procedimento; não constitui uma nova avaliação independente das escolhas.

## Ficheiros para trabalhar num caderno

- [Fundamentos e exploração](fundamentos.ipynb).
- [Modelos e validação](modelos.ipynb).
- [Estudo Abalone](estudo_abalone.ipynb).
- [Previsões recebidas em ficheiros](previsoes_ficheiros.ipynb).

Os arquivos [dados_desenvolvimento.zip](dados_desenvolvimento.zip) e
[dados_teste_reservado.zip](dados_teste_reservado.zip) conservam os nomes e
as pastas usados nos blocos do livro. Para acompanhar esses blocos num
caderno pessoal, comece pelo arquivo de desenvolvimento e abra o de teste
apenas na etapa correspondente.

## Ambiente e fontes

A execução de referência usa Python 3.12 e as versões indicadas em
[requisitos.txt](requisitos.txt). Os programas guardam tabelas CSV e figuras
PNG/PDF; o guia indica onde encontrar cada saída.

As fontes, atribuições, licenças dos dados e alterações feitas na preparação
estão em [FONTES_DADOS.md](FONTES_DADOS.md).
