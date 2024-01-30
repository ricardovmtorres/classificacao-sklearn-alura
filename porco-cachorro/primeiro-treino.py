# features (1 sim, 0 não)
# pelo longo? 
# perna curta?
# late?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 => porco, 0 => cachorro
# dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
# classes = [1,1,1,0,0,0]
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1,1,1,0,0,0]

from sklearn.svm import LinearSVC

model = LinearSVC(dual=True)
# modelos e classes pros dados serem interpretados
# model.fit(dados, classes)
model.fit(treino_x, treino_y) # labels / etiquetas

animal_misterioso = [1,1,1]
# o predict espera uma lista (varios itens) nesse caso temos apenas 1
# predict1=model.predict([animal_misterioso])
# sprint("Previsão 1")
# print(predict1)

# teste com vaios animais
animal1 = [1,1,1]
animal2 = [1,1,0]
animal3 = [0,1,1]

# testes = [animal1,animal2,animal3]
# teste_y = [0, 1, 1]
teste_x = [animal1,animal2,animal3]
teste_y = [0, 1, 1] # labels / etiquetas

predict2 = model.predict(teste_x)
print(predict2)


print("Previsão 2")
print(predict2 == teste_y)

corretos = (predict2 == teste_y).sum()
print("Total de acertos")
print(corretos)

total = len(teste_x)
taxa_de_acerto = corretos/total
print("Taxa de acerto ", taxa_de_acerto)
print("Taxa de acerto %", taxa_de_acerto *100)

from sklearn.metrics import accuracy_score
taxa_de_acerto = accuracy_score(teste_y, predict2)*100
print("Taxa de acerto 2: %.2f" % taxa_de_acerto)