import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

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


# Obtención de datos para entrenamiento y prueba #
texts_train = [text for text in df_train['texto']]
texts_test = [text for text in df_test['texto']]

y_train = [label for label in df_train['clase']]
y_test= [label for label in df_test['clase']]

# Extracción de características #
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(texts_train)
x_test = vectorizer.transform(texts_test)

# Clasificadores #
svm = LinearSVC()
naive_bayes = MultinomialNB()
log_reg = LogisticRegression()
random_forest = RandomForestClassifier()

def train_evaluate(x_train, y_train, x_test, y_test, classifier, title):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    report = classification_report(y_test, y_pred)
    print(title)
    print(report)
    output_file = open('./Results/Split/{}.txt'.format(title), 'w')
    output_file.write(report)
    output_file.close()

train_evaluate(x_train, y_train, x_test, y_test, classifier=svm, title='SVM_base')
train_evaluate(x_train, y_train, x_test, y_test, classifier=naive_bayes, title='Naive-Bayes_base')
train_evaluate(x_train, y_train, x_test, y_test, classifier=log_reg, title='Logistic_Regression_base')
train_evaluate(x_train, y_train, x_test, y_test, classifier=random_forest, title='Random_Forest_base')





