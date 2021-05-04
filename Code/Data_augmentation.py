"""
Script para obtener un dataframe de entrenamiento
que contenga las intancias aumentadas de las clases 
minoritarias (aquellas con menos de 1000 instancias)
"""

import pandas as pd

majority_classes = ['Producción', 'Secretaria', 'Gastronomía', 'Finanzas', 'Salud',
                    'Mercadotecnia', 'Logística', 'Recursos Humanos', 'Oficios', 'Tecnología',
                    'Call Center', 'Administración', 'Ventas']

minority_classes = ['Minería', 'Comercio Exterior', 'Gerencia', 'Comunicación', 'Seguros',
                    'Construcción', 'Legales', 'Diseño', 'Educación', 'Ingeniería']

augmented_columns = ['texto','back_en', 'back_te', 'back_zh', 'back_ro', 'back_ar', 'back_ja',
                     'back_jv', 'back_ko', 'back_vi', 'back_tr', 'back_yo']


# Dataframe de entrenamiento #
df_train = pd.DataFrame()

# Obtención de datos para clases mayoritarias #
for maj_class in majority_classes:
    path = './Data/Split/Train/train_{}.xlsx'.format(maj_class)
    maj_class_df = pd.read_excel(path)
    df_train = df_train.append(maj_class_df, ignore_index=True)

# Clases minoritarias #
for min_class in minority_classes:
    augmented_path = './Data/Augmented data/train_{}.xlsx'.format(min_class)
    min_class_df = pd.read_excel(augmented_path)
    classes = []
    texts = []
    for col in augmented_columns:
        for text in min_class_df[col]:
            texts.append(text)
        for element in min_class_df['clase']:
            classes.append(element)
    augmented_df = pd.DataFrame()
    augmented_df['texto'] = texts
    augmented_df['clase'] = classes
    df_train = df_train.append(augmented_df, ignore_index=True)



print(df_train['clase'].value_counts())
# Guarda el dataframe en un archivo csv #
df_train.to_csv('./Data/augmented_data_train.csv', index=False)  
        
