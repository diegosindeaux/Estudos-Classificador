import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore
from imblearn.pipeline import Pipeline # type: ignore
from imblearn.under_sampling import RandomUnderSampler # type: ignore
import joblib # type: ignore

# Aqui minha ideia foi ver quais seriam os melhores parâmetros para um undersampling, por ter um tempo de execução mais rápido que todos os outros, eu poderia testar mais valores.

data = pd.read_csv('data_training.csv')

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([ #acabei fazendo com pipeline
    ('under', RandomUnderSampler(random_state=42)), # testei outros tamanhos de distribuição, porém, o padrão se mostrou o melhor
    ('rf', RandomForestClassifier(random_state=42))
])

parameter_grid_search = {
    'rf__n_estimators': [50, 100, 150, 200],
    'rf__max_depth': [5, 10, 20, 30],
    'rf__min_samples_split': [1, 3, 5, 10],
    'rf__min_samples_leaf': [None, 1, 2, 4],
    'rf__bootstrap': [True, False]
}

grid = GridSearchCV(estimator=pipeline, param_grid=parameter_grid_search, cv=2, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print(grid.best_params_) # peguei os parâmetros que foram dados aqui e utilizei no algoritmo que vai gerar a AI oficial. (arquivo -> oficial_document_reader.py)