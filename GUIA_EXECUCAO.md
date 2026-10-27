# Guia de execução

## 1. Preparar a pasta

Descompacte o ZIP completo. Trabalhe na pasta que contém `reproduzir.py`,
`requisitos.txt`, `dados/` e `reproducao/`. Não é necessário descarregar
os dados novamente: as cópias utilizadas nos estudos acompanham os programas.

O cálculo dos estudos funciona sem acesso à Internet depois de instalar
as dependências. A preparação só consulta a fonte pública se faltar um
ficheiro original; a cópia recebida é conferida pela sua identificação SHA-256.

## 2. Instalar num computador

Utilize Python 3.12. Abra um terminal na pasta principal e crie um ambiente
de trabalho separado:

```bash
python -m venv .venv
```

No Windows, instale e execute com:

```powershell
.venv\Scripts\python -m pip install -r requisitos.txt
.venv\Scripts\python reproduzir.py desenvolvimento
```

No macOS ou Linux:

```bash
.venv/bin/python -m pip install -r requisitos.txt
.venv/bin/python reproduzir.py desenvolvimento
```

Se o seu computador usa o comando `python3`, use-o para criar o ambiente.
Nos exemplos seguintes, `python` representa o interpretador desse ambiente;
pode substituí-lo pelo caminho usado acima.

As versões fixadas permitem comparar os resultados com a execução de
referência. Noutros ambientes podem ocorrer pequenas diferenças numéricas,
de sinais da ACP, de rótulos equivalentes dos grupos ou de apresentação
gráfica. Os programas não ocultam avisos de ajustamento.

Para abrir os cadernos no seu computador, pode instalar JupyterLab no mesmo
ambiente com `python -m pip install jupyterlab` e iniciá-lo com
`python -m jupyterlab`. Abra um dos ficheiros `.ipynb` da pasta principal.
Esta aplicação não é necessária para executar os scripts no terminal.

## 3. Escolher um percurso

| Comando | O que executa |
|---|---|
| `python reproduzir.py fundamentos` | Capítulos 1–6, usando apenas desenvolvimento e exemplos fictícios identificados |
| `python reproduzir.py modelos` | Ajustamento, comparação e diagnóstico dos modelos dos capítulos 7–13 |
| `python reproduzir.py abalone` | Comparação dos três procedimentos do capítulo 14, sem abrir o teste |
| `python reproduzir.py desenvolvimento` | Os três percursos anteriores |
| `python reproduzir.py teste` | Avaliações finais de Wisconsin, pinguins e Abalone, a partir das decisões guardadas |
| `python reproduzir.py previsoes` | Atividade fictícia do apêndice A e ligação das exportações reais Abalone |
| `python reproduzir.py dados` | Reconstrução das divisões e dos dois arquivos ZIP de dados |
| `python reproduzir.py verificar` | Verificação dos ficheiros e resultados incluídos, sem reajustar modelos |
| `python reproduzir.py tudo` | Preparação, desenvolvimento, teste, atividades e verificação, por esta ordem |

As decisões dos estudos acompanham os resultados de referência. Para
reconstruir a sequência, execute desenvolvimento antes de teste. Conserve
os parâmetros descritos para reproduzir o livro. Pode alterá-los para
explorar outra hipótese, identificando essa análise como sua e revendo o
plano de avaliação; consultar o teste não o transforma em novos dados.

## 4. Encontrar as saídas

| Material | Pasta |
|---|---|
| Tabelas e resumos dos capítulos 1–6 | `reproducao/fundamentos/resultados/` |
| Figuras dos capítulos 2, 5 e 6 | `reproducao/fundamentos/resultados/figuras/` |
| Comparações, previsões por registo e avaliações dos modelos | `reproducao/modelos/resultados/` |
| Figuras dos modelos | `reproducao/modelos/figuras/` |
| Plano, decisões, previsões e erros Abalone | `reproducao/abalone/resultados/` |
| Figuras Abalone | `figuras/` |
| Atividade fictícia com seis amostras | `exemplo_previsoes/` e `exemplo_previsoes/resultados/` |
| Avaliação das exportações Abalone por identificador | `reproducao/previsoes/resultados_abalone/` |

Os CSV podem ser abertos com `pandas.read_csv` ou numa folha de cálculo.
Os JSON documentam configurações, decisões e o ambiente da execução.
As tabelas guardam mais casas decimais do que as impressas no livro.
A execução habitual regenera os resultados nas respetivas pastas.

## 5. Trabalhar no Google Colab

1. Abra o Colab e escolha **Ficheiro → Carregar caderno**.
2. Carregue um dos quatro ficheiros `.ipynb` da pasta principal.
3. Na primeira célula de preparação, carregue o ZIP completo dos materiais.
   O caderno localiza e descompacta a pasta necessária.
4. Execute a célula de instalação antes das células de análise. Se o Colab
   pedir para reiniciar o ambiente após a instalação, reinicie e volte a
   executar as células pela ordem indicada.
5. Leia as instruções entre as células e inspecione as saídas.
6. Guarde uma cópia pessoal do caderno. Descarregue também os resultados
   antes de terminar a sessão: os ficheiros temporários do Colab não ficam
   guardados apenas por guardar o caderno.

Para copiar diretamente os blocos do livro para um caderno seu, basta
carregar `dados_desenvolvimento.zip` e seguir as instruções do apêndice A.
As figuras e as exportações do capítulo 14 exigem o pacote completo.

## 6. Resolver dificuldades comuns

- **Ficheiro não encontrado:** confirme que conservou a estrutura das
  pastas. Os scripts localizam os dados a partir da sua própria posição.
- **Biblioteca não encontrada:** instale `requisitos.txt` no mesmo Python
  com que executa o programa ou o caderno.
- **Resultado diferente:** confirme as versões, os dados, a ordem dos
  identificadores, a semente e a configuração. Não altere a semente para
  procurar uma execução mais favorável.
- **Erro de identificadores:** investigue duplicações, ausências ou
  amostras inesperadas; não junte tabelas pela posição das linhas.
- **Medida não definida:** consulte o denominador e as classes presentes.
  Uma célula vazia ou `nao_definido` não representa um desempenho de zero.

As tabelas de dados, o dicionário e as notas de cada percurso permitem
relacionar cada saída com a pergunta e com as escolhas do estudo.
