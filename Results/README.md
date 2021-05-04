# Resultados

* Se utilizaron 2 arquitecturas de redes neuronales para realizar la clasificación: LSTM y CNN.

## Validación cruzada k=5

#### Resultados con backtranslation
| Algoritmo | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- |--- | 			
| LSTM | 0.788 |  0.754 | 0.732 | 0.736  |
| CNN | 0.798 |  0.782  | 0.74 | 0.754 |

#### Resultados sin backtranslation
| Algoritmo | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- |--- | 			
| LSTM | 0.652 | 0.578 | 0.538 | 0.55 |
| CNN | 0.696 | 0.674 | 0.56 | 0.594 |

## División de conjunto de datos. 80%-entrenamiento, 20%-prueba.

#### Resultados con backtranslation
| Algoritmo | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- |--- | 			
| LSTM | 0.68 |  0.59 | 0.56 | 0.57  |
| CNN | 0.70 |  0.66 | 0.57 | 0.60 |
| SVM | 0.25 | 0.21 | 0.51 | 0.30 |
| Naive-Bayes | 0.21 | 0.27 | 0.42 | 0.29 |
| Logistic Regression | 0.26 | 0.23 | 0.51 | 0.31 |
| Random Forest | 0.24 | 0.28 | 0.52 | 0.33 |

#### Resultados sin backtranslation
| Algoritmo | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- |--- | 			
| LSTM | 0.69 | 0.59 | 0.55 | 0.57 |
| CNN | 0.71 | 0.68 | 0.59 | 0.62 |
| SVM | 0.68 | 0.59 | 0.56 | 0.57 |
| Naive-Bayes | 0.61 | 0.60 | 0.32 | 0.37 |
| Logistic Regression |0.69|0.62|0.55|0.58|
| Random Forest |0.70|0.74|0.51|0.58|