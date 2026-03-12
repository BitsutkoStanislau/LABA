import pandas as pd

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('C:/Users/Asus/my_ml_project/netflix_titles.csv')
print(df.head())

print(df.info())
print(df.isnull().sum())

for column in df.columns:
    if df[column].isnull().any():
        value = df[column].mode()
        df[column] = df[column].fillna(value[0])

print(df.info())
print(df.isnull().sum())

num_col = df.select_dtypes(include = ['int64']).columns
print(df[num_col].head(10))
scaler = MinMaxScaler()
df[num_col] = scaler.fit_transform(df[num_col])
print(df[num_col].head(10))

categ_col = df.select_dtypes(include = ['str']).columns
df = pd.get_dummies(df, columns = categ_col, drop_first = True)

print(df.head(10))