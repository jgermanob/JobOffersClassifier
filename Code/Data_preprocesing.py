import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold
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

df = pd.read_csv('./Data/Data_For_Backtranslation.csv', encoding='utf8')
df['texto'] = df['texto'].apply(normalize_text)
df['texto'] = df['texto'].apply(clean_text)

# Obtención de textos y clases #
texts = [text for text in df['texto']]
labels = [label for label in df['clase']]

# Carga del modelo w2vec #
EMBEDDINGS_PATH = './Models/SBW-vectors-300-min5.bin'
embedding_model = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=True)

# Obtención de datos para entrenamiento y prueba #
kf = StratifiedKFold(n_splits=5, random_state=42)
fold = 1
max_len = 300
for train_index, test_index in kf.split(texts,labels):
    print("{}-fold".format(fold))
    texts_train = [texts[index] for index in train_index]
    texts_test = [texts[index] for index in test_index]
    y = pd.get_dummies(labels).values
    y_train, y_test = y[train_index], y[test_index]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts_train)
    x_train = tokenizer.texts_to_sequences(texts_train)
    x_test = tokenizer.texts_to_sequences(texts_test)
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)
    
    #Guardado de pesos word2vec#
    input_dim = len(tokenizer.word_index)
    embedding_matrix = np.zeros((input_dim+1,300))
    for word, i in tokenizer.word_index.items():
        if word in embedding_model:
            embedding_matrix[i] = embedding_model[word]
    np.savez_compressed('./Data/K-Folds/{}-fold/embedding_matrix_baseline.npz'.format(fold),embedding_matrix,)

    # Guardado de los conjuntos de entrenamiento y prueba #
    np.savez_compressed('./Data/K-Folds/{}-fold/x_train_baseline.npz'.format(fold),x_train,)
    np.savez_compressed('./Data/K-Folds/{}-fold/y_train_baseline.npz'.format(fold),y_train,)
    np.savez_compressed('./Data/K-Folds/{}-fold/x_test_baseline.npz'.format(fold),x_test,)
    np.savez_compressed('./Data/K-Folds/{}-fold/y_test_baseline.npz'.format(fold),y_test,)

    # Guardado de tokenizer#
    with open('./Data/K-Folds/{}-fold/tokenizer_baseline.pickle'.format(fold), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fold += 1

