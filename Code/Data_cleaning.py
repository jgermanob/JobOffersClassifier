"""
Script para limpieza inicial del conjunto de datos, se realiza:
1) Normalización de textos
2) Elimina aviso de privacidad
3) Guarda el texto limpio en un archivo csv
"""


import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

#Funciones para preprocesamiento de los datos#
"""
Agrega un guión si no hay texto de la oferta y 
devuelve el texto sin espacios en los extremos.
"""
def normalize_text(text):
    if isinstance(text,float):
        text = '-'
    else:
      text = text.strip()
    return text

tokenizer = RegexpTokenizer(r'\w+')
STOPWORDS = set(stopwords.words('spanish'))
privacy_policy = 'el contenido de este aviso es de propiedad del anunciante los requisitos de la posición son definidos y administrados por el anunciante sin que bumeran sea responsable por ello'
PRIVACY_POLICY_RE = re.compile(privacy_policy)

def clean_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    clean_text = ''
    for token in tokens:
        clean_text += ' {}'.format(token)
    clean_text = clean_text.strip()
    clean_text = PRIVACY_POLICY_RE.sub('',clean_text)
    return clean_text.strip()

PRIVACY_POLICY_RE = re.compile(r'el contenido de este aviso es de propiedad del anunciante. los requisitos de la posición son definidos y administrados por el anunciante sin que bumeran sea responsable por ello.')
def clean_text2(text):
    text = text.lower()
    text = text.strip()
    clean_text = PRIVACY_POLICY_RE.sub('',text)
    return clean_text.strip()

DATA_PATH = './Data/xample_bumeran2Utf8.csv'
df = pd.read_csv(DATA_PATH, encoding='utf8')

# Limpieza del texto #
df = df.reset_index(drop=True)
df['texto'] = df['texto'].apply(normalize_text)
df['texto'] = df['texto'].apply(clean_text2)
df.drop_duplicates(subset='texto', keep='first', inplace=True)
df = df.reset_index(drop=True)


# Creación de nuevo dataframe para utilizar en la etapa de backtranslation#
new_df = pd.DataFrame()
new_df['texto'] = df['texto']
new_df['clase'] = df['area_trab']

print(new_df['clase'].value_counts())

new_df.to_csv('./Data/Data_For_Backtranslation.csv', encoding='utf8', index=False)
