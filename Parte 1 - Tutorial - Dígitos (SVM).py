import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
# O conjunto de dados da iris
iris = datasets.load_iris()
# O conjunto de dados dos dígitos
digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Treino: %i' % label)
    
# Para aplicar um classificador nestes dados precisamos planificar a imagem, transformando os dados em uma matriz
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#Criar um classificador
classifier = svm.SVC(gamma=0.001)

#Aprendemos os dígitos da primeira metade dos dígitos
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Agora prever o valor do dígito da segunda metade
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Relatório de classificação para classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))

print("Matriz de confusão (Confusion matrix):\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))

for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Predição: %i' % prediction)
    
plt.show()

#print(digits.data)
#digits.target
#digits.images[0]
