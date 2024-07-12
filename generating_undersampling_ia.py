import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore
from imblearn.under_sampling import RandomUnderSampler # type: ignore
import joblib # type: ignore

data = pd.read_csv('data_training.csv')

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print(X_train.shape, X_train_under.shape) #coloquei para ver a diferença de tamanho que seria.

best_parameter = { # primeiramente utilizei os dados que foram gerados pelo parameter_for_rf.py
    'n_estimators': 150,
    'max_depth': 20,
    'min_samples_split': 3,
    'min_samples_leaf': 2,
    'bootstrap': False
} # Esses são os parâmetros que o arquivo paramether_for_undersampling.py me deu.

best_rf = RandomForestClassifier(random_state=42, **best_parameter)
best_rf.fit(X_train_under, y_train_under)

importances = best_rf.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
feature_importances.to_csv('feature_importances.csv') # criei um csv com as features mais importantes e assim, pude informar ao contratante.

y_pred = best_rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) # aqui, a diferença já se mostrou colossal, tanto em questão de tempo, quanto de valor de gasto ($ 14.220)

joblib.dump(best_rf, 'undersampling.pkl')
loaded_model = joblib.load('undersampling.pkl')

new_data = pd.read_csv('data_testing.csv') # como teste final, eu sempre coloco para testar em relação ao ano atual.
Z = new_data.drop('class', axis=1)

predictions = loaded_model.predict(Z)

print(accuracy_score(new_data['class'], predictions))
print(classification_report(new_data['class'], predictions))
print(confusion_matrix(new_data['class'], predictions)) # aqui a diferença foi mais significante ainda, a redução em gastos chegou em $ 20.875 (63% menor que o random forest menor)