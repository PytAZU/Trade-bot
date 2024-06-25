import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler

# Функция для генерации случайного массива
def generate_random_array(size):
    return [random.randint(-1000000, 1000000) for _ in range(size)]

# Исходные данные
data = generate_random_array(130)
print("Training data: ", data)

# Масштабирование данных
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()

window_size = 3

# Подготовка обучающих данных
X = []
y = []
for i in range(len(data_scaled) - window_size):
    X.append(data_scaled[i:i + window_size])
    y.append(data_scaled[i + window_size])

X = np.array(X)
y = np.array(y)

# Преобразование формата данных для RNN
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Создание модели
model = Sequential()
model.add(Input(shape=(window_size, 1)))
model.add(LSTM(100, return_sequences=True))  # Первый LSTM слой возвращает последовательности
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))  # Второй LSTM слой возвращает последовательности
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))  # Третий LSTM слой возвращает последнее состояние
model.add(Dropout(0.3))
model.add(Dense(1))

# Компиляция модели
model.compile(optimizer=SGD(learning_rate=0.001), loss=Huber())

# Обучение модели
history = model.fit(X, y, epochs=200, batch_size=1, verbose=2)

# Тестовые данные
test_data = generate_random_array(180)
print("Testing data: ", test_data)

# Масштабирование тестовых данных
test_data_scaled = scaler.transform(np.array(test_data).reshape(-1, 1)).flatten()

# Подготовка тестовых данных
X_test = []
for i in range(170 - window_size + 1):  # 170 элементов для предсказания следующих 10
    X_test.append(test_data_scaled[i:i + window_size])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Проверка формы данных
print("Shape of X_test:", X_test.shape)

# Предсказание следующих 10 элементов
predictions = []
current_input = X_test[-1]  # Последнее окно из 170 элементов

for _ in range(10):
    # Предсказание следующего элемента
    next_pred = model.predict(current_input.reshape(1, window_size, 1))
    print("Predicted value (scaled):", next_pred[0, 0])  # Печать предсказанного значения (масштабированного)
    predictions.append(next_pred[0, 0])
    # Обновление current_input для следующего предсказания
    current_input = np.append(current_input[1:], next_pred[0, 0]).reshape(window_size, 1)

# Печать предсказанных значений (масштабированных)
print("Predictions (scaled):", predictions)

# Обратное масштабирование предсказанных данных
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Печать предсказанных значений (обратное масштабирование)
print("Predictions (inverse scaled):", predictions)

# Построение графиков
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

# График изменения ошибки
ax1.plot(history.history['loss'])
ax1.set_title('Model Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

# График предсказаний
ax2.plot(test_data, label='Original Test Data')
ax2.scatter(range(len(test_data)), test_data, color='blue', s=10)  # Добавление точек для исходных данных
predicted_indices = list(range(170, 180))
ax2.plot(predicted_indices, predictions, label='Predicted Data', color='red')
ax2.scatter(predicted_indices, predictions, color='red', s=10)  # Добавление точек для предсказанных данных
ax2.legend()
ax2.set_title('Original Test Data and Predicted Data')
ax2.set_xlabel('Index')
ax2.set_ylabel('Value')

# Отображение графиков
plt.tight_layout()
plt.show()
