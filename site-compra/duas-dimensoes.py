# Nesse caso será analizada a mesma situção do site de compras
# Porem, para múltiplos sites 
# Vamos prever se o site será finalizado ou não dentro do contezto inserido
import pandas as pd

uri  = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)
print(dados.head())
a_renomear = {
    'expected_hours' : 'horas_esperadas',
    'price' : 'preco',
    'unfinished' : 'nao_finalizado'
}
dados = dados.rename(columns = a_renomear)
print(dados.head())

troca = {
    0 : 1,
    1 : 0
}
dados['finalizado'] = dados.nao_finalizado.map(troca)
print(dados.tail())

import matplotlib.pyplot as plt
import seaborn as sns
# TODO: REVISAR CODIGO
sns.scatterplot(x = "horas_esperadas", y="preco", data=dados)
# sns.scatterplot(x="horas_esperadas", y="preco", hue="finalizado", data=dados)
sns.relplot(x="horas_esperadas", y="preco", col="finalizado", data=dados)

sns.relplot(x="horas_esperadas", y="preco", hue="finalizado", col="finalizado", data=dados)
plt.show()

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, test_size = 0.25, stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC(dual=True, C=0.1)
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

import numpy as np
previsoes_do_guilherme = np.ones(540)
acuracia = accuracy_score(teste_y, previsoes_do_guilherme) * 100
print("A acurácia do Guilherme foi %.2f%%" % acuracia)