import numpy as np
import matplotlib.pyplot as plt

#Gráfico de linhas
#plt.plot([1, 2, 3, 4, 5], [0.97, 0.97, 0.97, 0.97, 0.97])

#plt.title('Gráfico da média das execuções (Dígitos)')
#plt.xlabel('Execução nº:')
#plt.ylabel('Accuracy/Precisão')

#plt.show()

#Gráfico de barras (cor azul)
numExecucoes = ['Support vector machines', 'Decision Trees', 'Nearest Neighbors']
taxaPrecisao = [0.97, 0.748, 0.74]

plt.bar(numExecucoes, taxaPrecisao, color ="red")

plt.xticks(numExecucoes)
plt.xlabel('Algoritmos usados:')
plt.ylabel('Accuracy/acurácia média')
plt.title('Gráfico dos valores de accuracy média dos algoritmos usados (Dígitos)')
plt.show()