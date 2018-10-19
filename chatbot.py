import imp
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

json_data = open('corpus_eleicao.json').read()
corpus = json.loads(json_data)

intencoes_treino, perguntas_treino = list(), list()
for intencao in corpus:
    for pergunta in corpus[intencao]:
        intencoes_treino.append(intencao)
        perguntas_treino.append(pergunta)

vocabulario = TfidfVectorizer().fit(perguntas_treino)
perguntas_tokenizadas = vocabulario.transform(perguntas_treino)

modelo = SVC(probability=True)
modelo.fit(perguntas_tokenizadas, intencoes_treino)

while True:
    entrada = input()
    entrada_tokenizada = vocabulario.transform([entrada])
    print(modelo.predict(entrada_tokenizada))