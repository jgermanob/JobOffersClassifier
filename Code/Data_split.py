import pandas as pd
from sklearn.model_selection import train_test_split

# Lectura de archivo y obtención de etiquetas #
df = pd.read_csv('./Data/Data_For_Backtranslation.csv', encoding='utf8')
labels = set(df['clase'].values.tolist())

# Divisón de conjunto de datos por clase #
test_texts = []
test_labels = []
for label in labels:
    df_class = df[(df['clase']==label)]
    x_train, x_test, y_train, y_test = train_test_split(df_class['texto'], df_class['clase'], train_size=0.8, random_state=42)
    train_path = './Data/Split/Train/train_{}.xlsx'.format(label)
    df_train = pd.DataFrame()
    df_train['texto'] = x_train
    df_train['clase'] = y_train
    df_train.reset_index(drop=True)
    df_train.to_excel(train_path, encoding='utf8', index=False)
    for text in x_test:
        test_texts.append(text)
    for label in y_test:
        test_labels.append(label)

df_test = pd.DataFrame()
df_test['texto'] = test_texts
df_test['clase'] = test_labels
df_test.reset_index(drop=True)
print(df_test['clase'].value_counts())
df_test.to_csv('./Data/data_test.csv', encoding='utf8', index=False)

