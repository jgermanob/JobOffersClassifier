"""
Script para obtener un dataframe de entrenamiento
que contenga los datos de la partición de entrenamiento
sin las instancias aumentadas
"""

import pandas as pd

majority_classes = ['Producción', 'Secretaria', 'Gastronomía', 'Finanzas', 'Salud',
                    'Mercadotecnia', 'Logística', 'Recursos Humanos', 'Oficios', 'Tecnología',
                    'Call Center', 'Administración', 'Ventas']

minority_classes = ['Minería', 'Comercio Exterior', 'Gerencia', 'Comunicación', 'Seguros',
                    'Construcción', 'Legales', 'Diseño', 'Educación', 'Ingeniería']

# Dataframe de entrenamiento #
AUGMENTED_DATA_PATH_TRAIN = './Data/augmented_data_train.csv'
df_train = pd.DataFrame()

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
    for text in min_class_df['texto']:
        texts.append(text)
    for element in min_class_df['clase']:
        classes.append(element)
    
    augmented_df = pd.DataFrame()
    augmented_df['texto'] = texts
    augmented_df['clase'] = classes
    df_train = df_train.append(augmented_df, ignore_index=True)

print(df_train['clase'].value_counts())
df_train.to_csv('./Data/data_train.csv', index=False)  
        
