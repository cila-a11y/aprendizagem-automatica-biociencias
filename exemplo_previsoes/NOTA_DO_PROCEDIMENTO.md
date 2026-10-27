# Previsões recebidas em ficheiros

Estas seis amostras e as suas previsões são fictícias. Servem para a atividade
do apêndice A e não pertencem ao conjunto Wisconsin. Nenhum modelo foi
ajustado para produzir estas pontuações.

A resposta de referência e a classe prevista usam 0 para benigno e 1 para
maligno. As pontuações maiores favorecem a classe positiva. As classes
fornecidas foram construídas com o limiar 0,50; a avaliação conserva-as e não
procura outro limiar. O limiar não é o do estudo Wisconsin.

Os ficheiros têm ordens de linhas diferentes. A correspondência é feita
por `id_registo`, preservando os zeros iniciais. `ids_esperados.csv` regista
as seis amostras previstas nesta atividade. A03 deve surgir como falso
positivo e A04 como falso negativo.

Os resultados descrevem apenas este exemplo didático. Para usar o script
com outros ficheiros, registe a origem das respostas, o modelo e a versão
que produziram as previsões, os dados usados no ajustamento e nas escolhas,
a definição da classe positiva, o sentido das pontuações, o limiar fixado
e os identificadores previstos no plano de avaliação.
