import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation

%matplotlib inline
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

csv_path = "https://raw.githubusercontent.com/PytAZU/BTCUSDPRICE/main/BTC-USD.csv"
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values('Date')
df.head()
df.shape

ax = df.plot(x='Date', y='Close');
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (USD)")

close_price = df.Close.values.reshape(-1, 1)
scaler = StandardScaler()
standardized_close = scaler.fit_transform(close_price)
print(standardized_close.shape)

seq_len = 200


def to_sequences(data, seq_len):
    d = [data[index: index + seq_len] for index in range(len(data) - seq_len)]
    return np.array(d)

def preprocess(data_raw, seq_len, train_split):
    data = to_sequences(data_raw, seq_len)
    X = data[:, :-1, :]  # Все столбцы кроме последнего в каждой последовательности
    y = data[:, -1, :]   # Последний столбец в каждой последовательности

    # Используем train_test_split для разделения данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, shuffle=False)

    return X_train, y_train, X_test, y_test

# Применяем функцию preprocess
X_train, y_train, X_test, y_test = preprocess(standardized_close, seq_len, train_split=0.95)

X_train.shape

DROPOUT = 0.5
WINDOW_SIZE = seq_len - 1
model = Sequential()
model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE, X_train.shape[-1])))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

model.add(Dense(units=1))
model.add(Activation('linear'))

model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

BATCH_SIZE = 512

history = model.fit(
    X_train,
    y_train,
    epochs=250,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_split=0.1
)

model.evaluate(X_test, y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_hat = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)

plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')

plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')

plt.show();

