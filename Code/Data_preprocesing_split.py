"""
Script para obtener conjuntos de datos para entrenamiento
y prueba cuando se obtiene una partición 80%-20%.
"""

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from gensim.models import KeyedVectors

def normalize_text(text):
    if isinstance(text,float):
        text = '-'
    else:
      text = text.strip()
    return text

tokenizer = RegexpTokenizer(r'\w+')
STOPWORDS = set(stopwords.words('spanish'))
def clean_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    clean_text = ''
    for token in tokens:
        if token not in STOPWORDS:
            clean_text += ' {}'.format(token)
    clean_text = clean_text.strip()
    return clean_text.strip()

# Lectura de archivo de entrenamiento #
df_train = pd.read_csv('./Data/data_train.csv', encoding='utf8')
df_train['texto'] = df_train['texto'].apply(normalize_text)
df_train['texto'] = df_train['texto'].apply(clean_text)

# Lectura de archivo para prueba #
df_test = pd.read_csv('./Data/data_test.csv', encoding='utf8')
df_test['texto'] = df_test['texto'].apply(normalize_text)
df_test['texto'] = df_test['texto'].apply(clean_text)

# Carga del modelo w2vec #
EMBEDDINGS_PATH = './Models/sbw_vectors.bin'
embedding_model = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=True)

# Obtención de datos para entrenamiento y prueba #
texts_train = [text for text in df_train['texto']]
texts_test = [text for text in df_test['texto']]

max_len = 300
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_train)

x_train = tokenizer.texts_to_sequences(texts_train)
x_test = tokenizer.texts_to_sequences(texts_test)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

labels_train = [label for label in df_train['clase']]
labels_test= [label for label in df_test['clase']]

y_train = pd.get_dummies(labels_train).values
y_test = pd.get_dummies(labels_test).values

#Guardado de pesos word2vec#
input_dim = len(tokenizer.word_index)
embedding_matrix = np.zeros((input_dim+1,300))
for word, i in tokenizer.word_index.items():
    if word in embedding_model:
        embedding_matrix[i] = embedding_model[word]
np.savez_compressed('./Data/Split/Train/embedding_matrix_train_base.npz',embedding_matrix,)

# Guardado de tokenizer#
with open('./Data/Split/Train/tokenizer_train_base.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Guardado de los conjuntos de entrenamiento y prueba #
np.savez_compressed('./Data/Split/Train/x_train_base.npz',x_train,)
np.savez_compressed('./Data/Split/Train/y_train_base.npz',y_train,)
np.savez_compressed('./Data/Split/Test/x_test_base.npz',x_test,)
np.savez_compressed('./Data/Split/Test/y_test_base.npz',y_test,)
