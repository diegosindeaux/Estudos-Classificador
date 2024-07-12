import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from functions import convert_to_float

data = pd.read_csv('air_system_previous_years.csv')
data_now = pd.read_csv('air_system_present_year.csv')

print(data)
print(data_now)

# primeiro eu resolvi trocar o 'neg' (quando o caminhão teve defeito em qualquer outro sistema que não seja ar) para 0
# e o 'pos' (quando ele teve defeito no sistema de ar) para 1
data['class'] = data['class'].map({'neg': 0, 'pos': 1})

print(data)

# troquei o 'na' que estava uma string para o NaN (nulo) para poder trabalhar melhor com os dados.
data.replace('na', np.nan, inplace=True)


# na tentativa de ver se existia alguma correlação (esperadamente frustrada)
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlações')
plt.show()

# aqui eu estava querendo ver quantas colunas e linhas tinham 'na' nos dados para saber como poderia posseguir com eles.
missing_data_columns = data.isnull().any(axis=0).sum()
missing_data_lines = data.isnull().any(axis=1).sum()
print(missing_data_columns, missing_data_lines)

# analisei as porcentagem que cada coluna e cada linha tinham de números nulos para ver o que poderia fazer com eles.
missing_data_percentage_column = data.isnull().mean() * 100
missing_data_percentage_line = data.isnull().mean(axis=1) * 100
print(missing_data_percentage_column, missing_data_percentage_line)

# defini um limite para as linhas e colunas com dados faltantes, assim, seria mais preciso o trabalho do nulos seguintes.
threshold = 70
columns_drop = missing_data_percentage_column[missing_data_percentage_column > threshold].index
lines_drop = missing_data_percentage_line[missing_data_percentage_line > threshold].index
data_cleaned = data.drop(columns=columns_drop, index=lines_drop)

data_cleaned

# mapa de calor das correlações (mais uma tentativa frustrada, porém, já deu uma melhorada na visualização).
plt.figure(figsize=(12, 8))
sns.heatmap(data_cleaned.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Nova correlação')
plt.show()

# minha última tratativa de dados será para completar os nulos faltantes com a média da coluna dele. (está comentada para o código não quebrar)
# data_cleaned.fillna(data_cleaned.mean(), inplace=True)

# como foi acusado um erro que existia string no dataset, resolvi ver aonde estavam essas strings.
is_string = data_cleaned.map(lambda x: isinstance(x, str))
print(is_string)

data_cleaned = data_cleaned.map(convert_to_float) # percebi que na verdade todo o dataset estava no formato errado e resolvi converter para float.

# agora eu pude confiar e fazer o tratamento dos nulos pela médias de sua coluna.
data_cleaned.fillna(data_cleaned.mean(), inplace=True)
data_cleaned = data_cleaned.to_csv('data_training.csv')


# repeti o processo com os dados do ano atual
data_now['class'] = data_now['class'].map({'neg': 0, 'pos': 1})

data_now.replace('na', np.nan, inplace=True)

now_missing_data_percentage_column = data_now.isnull().mean() * 100
now_missing_data_percentage_line = data_now.isnull().mean(axis=1) * 100

threshold = 70
now_columns_drop = now_missing_data_percentage_column[now_missing_data_percentage_column > threshold].index
now_lines_drop = now_missing_data_percentage_line[now_missing_data_percentage_line > threshold].index
data_now_cleaned = data_now.drop(columns=now_columns_drop, index=now_lines_drop)

data_now_cleaned = data_now_cleaned.map(convert_to_float)
data_now_cleaned.fillna(data_now_cleaned.mean(), inplace=True)
data_now_cleaned = data_now_cleaned.to_csv('data_testing.csv')