import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore

data = pd.read_csv('data_training.csv')

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier() # Aqui foi quando eu decidi por utilizar o random forest pelo tamanho do dataset, a quantidade de features e os outliers.
model.fit(X_train, y_train)

feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances)

parameter_grid_search = { # Tentei alguns números para ter meus parâmetros, o que foi apontado como melhor eu utilizei, mesmo estando com um tempo de execuçā alto.
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [3, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

classifier = RandomForestClassifier(random_state=42)
grid = GridSearchCV(estimator=classifier, param_grid=parameter_grid_search, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print(grid.best_params_)

best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)

print(accuracy_score(y_test, y_pred_best)) # Nesse momento me assustei um pouco, como a acurácia estava muito alta (0,99), achei que poderia, mesmo usando random forest, feito overfitting.