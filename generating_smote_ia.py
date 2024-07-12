import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
import joblib # type: ignore

data = pd.read_csv('data_training.csv')

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# eu entendi que o que estava acontecendo era que o meu modelo de teste tinham porquíssimas ocorrências de teste para o tanto de não ocorrências, entāo,
# fiz um SMOTE para tentar aumentar minha precisão de dados.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

best_parameter = { #utilizei os mesmos parâmetros que tinha conseguido do RandomForest de antes
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'bootstrap': False
}

best_rf = RandomForestClassifier(random_state=42, **best_parameter)
best_rf.fit(X_train_smote, y_train_smote)

y_pred = best_rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred)) # já percebi uma melhora significativa, o número de 1's que não foram acertados já caiu para 32 (gastos estavam em $ 20.950)

joblib.dump(best_rf, 'SMOTE.pkl')
loaded_model = joblib.load('SMOTE.pkl')

new_data = pd.read_csv('data_testing.csv')
Z = new_data.drop('class', axis=1)

predictions = loaded_model.predict(Z)

print(accuracy_score(new_data['class'], predictions))
print(classification_report(new_data['class'], predictions))
print(confusion_matrix(new_data['class'], predictions)) # aqui também a melhora foi muito significante de 99 foi para 55 (gastos em $ 36.715)

# porém, ainda achava que estava dando um valor muito alto, resolvi testar o undersampling para ver como seria.