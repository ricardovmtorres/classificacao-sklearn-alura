import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)
# a função head imprime apenas os 5 primeiros dados
cabecalho = dados.head()
print(cabecalho)

# separando os dados dos resultados
# x = dados[["home","how_it_works","contact"]]
# y = dados["bought"]
# print(x)
# print(y)

# renomeando para portugues
mapa = {
    "home": "principal",
    "how_it_works": "como_funciona",
    "contact": "contato",
    "bought": "comprou"
}
dados = dados.rename(columns = mapa)
# print(dados)
x = dados[["principal","como_funciona","contato"]]
y = dados["comprou"]
print(x.head())
print(y.head())

# seprando os dados de treino e de teste
formato_dados = dados.shape
print(formato_dados)

# TREINAMENTO 1
# considerando a quantidade de 99 linhas de dados
# os dados de treino são os 75 primeiros
treino_x = x[:75]
treino_y = y[:75]
print("X de treino", treino_x.shape)
print("Y de treino", treino_y.shape)
# e os dados de teste serão seguintes
teste_x = x[75:]
teste_y = y[75:]
print("X de teste", teste_x.shape)
print("Y de teste", teste_y.shape)

print("Treinaremos com %d elementos e testaremmos com %d elementos" % (len(treino_x), len(teste_x)))

# etapa de treinamento similar ao do primeiro_treino.py
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC(dual=True)
model.fit(treino_x, treino_y) # labels / etiquetas

previsoes = model.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

# TREINAMENTO 2
# usando os recursos do sklearn pra separar as bases em treino e teste
from sklearn.model_selection import train_test_split

# são passados por parâmetro o x e y da base inteira e o tamanho da base de teste desejada
# são retornados os 4 arrays de de treino e teste em x e y 
# treino2_x, teste2_x, treino2_y, teste2_y = train_test_split(x, y, test_size=0.25)
SEED = 20
# para evitar resultados diferentes, foi acrescido o parametro random_state
# random_state define a ordem de aleatoriedade durante a separação do conjuntos de treino e teste
treino2_x, teste2_x, treino2_y, teste2_y = train_test_split(x, y, random_state=SEED, test_size=0.25)
print("Treinaremos com %d elementos e testaremmos com %d elementos" % (len(treino2_x), len(teste2_x)))

model2 = LinearSVC(dual=True)
model2.fit(treino2_x, treino2_y) # labels / etiquetas

previsoes2 = model2.predict(teste2_x)
acuracia2 = accuracy_score(teste2_y, previsoes2) * 100
print("A acurácia 2 foi %.2f%%" % acuracia2)

# analizando as proporções das compras no treino e teste
proporcao_treino2 = treino2_y.value_counts()
proporcao_teste2 = teste2_y.value_counts()
print(proporcao_treino2)
print(proporcao_teste2)


# TREINAMENTO 3
# para evitar a DESPROPORÇÃO de resultados entre treino e teste, foi acrescido o parametro stratify
# stratify separa os conjuntos de treino e teste proporcionalmente de acordo com y
treino3_x, teste3_x, treino3_y, teste3_y = train_test_split(x, y, random_state=SEED, test_size=0.25, stratify=y)
print("Treinaremos com %d elementos e testaremmos com %d elementos" % (len(treino3_x), len(teste3_x)))

model3 = LinearSVC(dual=True)
model3.fit(treino3_x, treino3_y) # labels / etiquetas

previsoes3 = model3.predict(teste3_x)
acuracia3 = accuracy_score(teste3_y, previsoes3) * 100
print("A acurácia 3 foi %.2f%%" % acuracia3)

# analizando as proporções das compras no treino e teste
# nesse treinamento 3 as proporções das compras são de 2 para 1 de não-compra
proporcao_treino3 = treino3_y.value_counts()
proporcao_teste3 = teste3_y.value_counts()
print(proporcao_treino3)
print(proporcao_teste3)