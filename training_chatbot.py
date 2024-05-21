import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay
import random

data_file = open('intents.json', 'r', encoding='uft-8').read()
intents = json.loads(data_file)

lematizer = WordNetLemmatizer()

words=[]
classes = []
documents = []
ignore_words = ['?', '!']

# Recorre cada intención y sus patrones en el archivo JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokeniza las palabras en cada patrón y las agrega a la lista de palabras
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Agrega el par (patrón, etiqueta) a la lista de documentos
        documents.append((w, intent['tag']))
        # Si la etiqueta no está en la lista de clases, la agrega
        if intent['tag'] not in classes:
            classes.append(intent['tag'])