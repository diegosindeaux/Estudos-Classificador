import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore
import joblib # type: ignore
from functions import convert_to_float

best_ia = joblib.load('undersampling.pkl')

new_data = pd.read_csv('air_system_present_year.csv')
new_data = new_data.drop('class', axis=1)

new_data = new_data.map(convert_to_float) # percebi que normalmente esse relatório é preenchido como string, então resolvi deixar aqui para ter certeza.

rf_predictions = best_ia.predict(new_data)

new_data['class'] = np.where(rf_predictions == 1, 'repair', 'good')

new_data.to_csv('data_testing_with_predictions.csv', index=False)
