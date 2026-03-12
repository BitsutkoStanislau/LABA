import pandas as pd

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('C:/Users/Asus/my_ml_project/netflix_titles.csv')
print(df.head())
print(df.info())
cols = df.columns.tolist()
print(cols)
print(df['type'])
nan_matrix = df.isnull()
print(nan_matrix)

director_mode = df['director'].mode()[0]
cast_mode = df['cast'].mode()[0]
df = df.fillna({'director': director_mode, 'cast': cast_mode})
nan_matrix = df.isnull()
print(nan_matrix)

scaler = MinMaxScaler()
scaler.fit(df[['release_year']])
df[['release_year']] = scaler.transform(df[['release_year']])

df = pd.get_dummies(df, columns = ['country'], drop_first = True)
print(df.head())