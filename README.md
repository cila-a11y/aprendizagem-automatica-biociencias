# Aprendizagem automática em Biociências

Material de apoio a *Aprendizagem automática em Biociências — Uma introdução
com exemplos em Python*. Aqui encontra os dados, os programas Python, os
resultados de referência e as figuras dos capítulos 5 a 15, organizados
pelos mesmos capítulos e nomes de ficheiros usados no livro.

## Descarregar e começar

**[Descarregar os materiais](https://github.com/cila-a11y/aprendizagem-automatica-biociencias/archive/refs/heads/main.zip)**

A ligação descarrega um ZIP com os materiais do repositório.
Descompacte-o e conserve a organização das pastas. Comece por
[LEIA-ME.txt](LEIA-ME.txt) e siga a sequência de [COMANDOS.txt](COMANDOS.txt)
para o capítulo pretendido. Esses ficheiros incluem instruções para o
computador local e para o Google Colab.

Para acompanhar apenas os blocos de código impressos no livro, estão também
disponíveis os dois arquivos de dados:

- [Descarregar dados_desenvolvimento.zip](https://github.com/cila-a11y/aprendizagem-automatica-biociencias/raw/refs/heads/main/dados_desenvolvimento.zip): dois CSV, um de pinguins e outro de Wisconsin, dentro de `dados/`.
- [Descarregar dados_teste_reservado.zip](https://github.com/cila-a11y/aprendizagem-automatica-biociencias/raw/refs/heads/main/dados_teste_reservado.zip): os dois CSV destinados à avaliação final do capítulo 14, também dentro de `dados/`.

O pacote completo já inclui estes dois arquivos. O apêndice A do livro
explica quando usar cada um. Comece pelo desenvolvimento e abra o teste na
etapa correspondente.

## Localizar os materiais de cada capítulo

| Capítulos | Conteúdo | Pasta e instruções |
|---|---|---|
| 5 e 6 | K-means e análise de componentes principais | [cap05_06](cap05_06/LEIA-ME.txt) |
| 7 | Regressão linear | [cap07](cap07/LEIA-ME.txt) |
| 8 | Regressão logística | [cap08](cap08/LEIA-ME.txt) |
| 9 e 10 | Árvores de decisão e florestas aleatórias | [cap09_10](cap09_10/LEIA-ME.txt) |
| 11 | Redes neuronais | [cap11](cap11/LEIA-ME.txt) |
| 12 e 13 | Validação, interpretação e diagnóstico | [cap12_13](cap12_13/LEIA-ME.txt) |
| 14 e 15 | Estudos finais com Wisconsin e pinguins e comunicação dos resultados | [cap14_15](cap14_15/LEIA-ME.txt) |

Cada pasta é autónoma: `codigo/` contém os programas; `dados/`, as entradas;
`resultados/`, as tabelas, previsões e registos de execução; e `figuras/`,
os PDF e PNG. Siga o `LEIA-ME.txt` dessa pasta sem juntar programas de
capítulos diferentes, pois alguns têm o mesmo nome.

## Preparar o ambiente e executar

A execução de referência usa Python 3.12.14 e as versões fixadas em
[requisitos.txt](requisitos.txt). Num terminal, a partir da pasta principal
extraída do pacote:

```bash
python -m pip install -r requisitos.txt
python verificar_pacote.py
```

O segundo comando confere a cópia recebida antes de alterar ou regenerar
ficheiros. Depois, execute os comandos do capítulo em
[COMANDOS.txt](COMANDOS.txt). Por exemplo, para o capítulo 7:

```bash
python cap07/codigo/gerar_figuras.py
```

Para os estudos finais dos capítulos 14 e 15, a ordem é:

```bash
python cap14_15/codigo/selecionar_desenvolvimento.py
python cap14_15/codigo/avaliar_teste_final.py --confirmar-teste
python cap14_15/codigo/gerar_figuras.py
python cap14_15/codigo/verificar_resultados.py
```

A seleção usa apenas o desenvolvimento e guarda as escolhas. A avaliação
confere esse registo e exige `--confirmar-teste` antes de abrir os testes.
Os programas substituem as suas saídas; guarde separadamente experiências
que queira conservar.

Os dados e resultados de teste apresentados no livro acompanham o pacote.
A sua reprodução permite conferir e compreender o procedimento; não
constitui uma nova avaliação independente. Os exemplos são didáticos.

## Fontes e verificação

- [FONTES_DADOS.txt](FONTES_DADOS.txt): fontes, atribuições, licenças dos dados e preparação das cópias de Palmer Penguins e Wisconsin.
- [VERIFICACAO_EXECUCAO.txt](VERIFICACAO_EXECUCAO.txt): ambiente, comandos e comparações da execução de referência.
- [manifesto_sha256.json](manifesto_sha256.json) e [verificar_pacote.py](verificar_pacote.py): integridade dos ficheiros da distribuição e dos dois arquivos de dados.

O repositório contém o material de apoio ao leitor. O texto do livro e os
seus ficheiros LaTeX não fazem parte desta distribuição.
