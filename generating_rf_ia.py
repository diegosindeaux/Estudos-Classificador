import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore
import joblib # type: ignore

data = pd.read_csv('data_training.csv')

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_parameters = { # utilizei os melhores parâmetros apontados pelo teste de parâmetros que fiz.
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'bootstrap': False
}

best_rf = RandomForestClassifier(random_state=42, **best_parameters)
best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) # aqui, me preocupei novamente, pois estavam dando em média de 55 1's que não foram preditos (gastos estavam em $ 31.205).

joblib.dump(best_rf, 'random_forest.pkl')

loaded_model = joblib.load('random_forest.pkl')

new_data = pd.read_csv('data_testing.csv')
Z = new_data.drop('class', axis=1)

predictions = loaded_model.predict(Z)
real_results = new_data['class']

print(accuracy_score(real_results, predictions))
print(classification_report(real_results, predictions))
print(confusion_matrix(real_results, predictions)) # e o padrão se seguiu aqui, com 99 1's que não foram preditos (gastos em $ 56.585)