import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report

df = pd.read_csv('processed_dataset.csv')
print(df.info())
for column in df.columns:
    if df[column].dtype == 'bool':
        df[column] = df[column].astype(int)

X_lr = df.drop(['release_year'], axis=1)
y_lr = df['release_year']

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_lr, y_lr, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train_lr, y_train_lr)

y_pred_test_lr = linear_model.predict(X_test_lr)

MSE = mean_squared_error(y_test_lr, y_pred_test_lr)
RMSE =  np.sqrt(MSE)
MAE = mean_absolute_error(y_test_lr, y_pred_test_lr)

print('Среднеквадратичная ошибка (MSE): ' + str(MSE))
print('Корень среднеквадратичной ошибки (RMSE): ' + str(RMSE))
print('Средняя абсолютная ошибка (MAE): ' + str(MAE))

type_cols = [col for col in df.columns if col.startswith('type_')]
if type_cols:
    X_log = df.drop(type_cols, axis=1)
    y_log = df[type_cols[0]]

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_log, y_train_log)

y_pred_test_log = logreg_model.predict(X_test_log)

accuracy = accuracy_score(y_test_log, y_pred_test_log)
report = classification_report(y_test_log, y_pred_test_log)

print('Точность модели : ' + str(accuracy))
print('Отчет по классификации:')
print(report)