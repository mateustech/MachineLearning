import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR

# Lendo todas as imagens
atletico_g = cv2.imread('imagens/teste.jpg')
corinthians_g = cv2.imread('imagens/teste1.jpg')
flamengo_g = cv2.imread('imagens/teste2.png')
palmeiras_g = cv2.imread('imagens/teste3.jpg')
ceara_g = cv2.imread('imagens/teste5.jpg')
test_ceara_g = cv2.imread('imagens/treino/botafogo_2.jpg')

# Redimensionando imagens para 10 x 10 px
atletico = cv2.resize(atletico_g, (10,10))
corinthians = cv2.resize(corinthians_g, (10,10))
flamengo = cv2.resize(flamengo_g, (10,10))
palmeiras = cv2.resize(palmeiras_g, (10,10))
ceara = cv2.resize(ceara_g, (10,10))
test_ceara = cv2.resize(test_ceara_g, (10,10))

# Concat all arrays to one
X = np.concatenate((atletico, corinthians, flamengo, palmeiras, ceara), axis=0)

# Concatenando as matrizes
y = [1,2,3,4,5]

# Set y as a array
y = np.array(y)

# remodelando y
Y = y.reshape(-1)

# remodelando X com comprimento de y
X = X.reshape(len(y), -1)

# Criando o classificador
classifier_linear = SVC(kernel='linear')

print(40 * '-')
print('Iniciado treino do modelo SVC')

# Treinando o classificador com imagens e índices
classifier_linear.fit(X,Y)

print('Treinamento Finalizado')
print(40 * '-')

# Prevendo a categoria da imagem 
prediction = classifier_linear.predict(test_ceara.reshape(1,-1))

# Pontuação de acerto 
score = classifier_linear.score(X,Y)

# mostrando previsão
print('Resultado: {}'.format(prediction))

# Mostrar pontuação de previsão
print('Pontuação de acerto: {:.1f}%'.format(score * 100))

# Definindo resultado como imagem da previsão
if prediction == 1:
	result = atletico_g
elif prediction == 2:
	result = corinthians_g
elif prediction == 3:
	result = flamengo_g
elif prediction == 4:
	result = palmeiras_g
elif prediction == 5:
	result = ceara_g

# Mostrar imagem com base na previsão
cv2.imshow("Resultado", result)
# Mostrar a imagem testada
cv2.imshow("Teste", test_ceara_g)
#Espere pela chave
cv2.waitKey(0)

print('---------------------------------------')


# Crie o classificador 
classifier_linear_regression = SVR(kernel='linear')

print('Iniciando treino SVR')

# Treine o classificador com imagens e índices
classifier_linear_regression.fit(X,Y)

print('Final do treinamento')
print(40 * '-')

# Prever a categoria de imagem
prediction = classifier_linear_regression.predict(test_ceara.reshape(1,-1))

# Pontuação de acerto 
score = classifier_linear_regression.score(X,Y)

# mostrar resultado
print('Resultado: {}'.format(prediction))

# mostrar pontuação de precisão
print('Pontuação de precisão: {:.1f}%'.format(score * 100))

# Definindo imagem baseado nos resultados
if prediction == 1:
	result = atletico_g
elif prediction == 2:
	result = corinthians_g
elif prediction == 3:
	result = flamengo_g
elif prediction == 4:
	result = palmeiras_g
elif prediction == 5:
	result = ceara_g

# Mostrar imagem com base na previsão
cv2.imshow("Resultado", result)
# Show the image tested
cv2.imshow("Teste", test_ceara_g)
# Wait for key
cv2.waitKey(0)
