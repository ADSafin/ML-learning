import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(x_train.shape[1],)))
model.add(Dense(1))

model.compile(loss="mse",
              optimizer="adam",
              metrics=["mae"]) # mae — mean absolute error (средняя абсолютная ошибка).
print(model.summary())

history = model.fit(x_train, y_train,
                    batch_size=1,
                    epochs=100,
                    validation_split=0.2,
                    verbose=2)

model.save('boston_house.h5')

mse, mae = model.evaluate(x_test, y_test, verbose=0)
print("Средняя абсолютная ошибка (тысяч долларов):", mae)

pred = model.predict(x_test)
print("Предсказанная стоимость: ", pred[11][0], ", правильная стоимость: ", y_test[11])

parameters = []
for index in range(0, 102):
  parameters.append(x_test[index][11])
results = []
for index in range(0, 102):
  results.append(pred[index][0])
plt.plot(parameters, results, 'bo', label='Prices')
plt.xlabel('Доля афроамериканцев в разбивке по городам')
plt.ylabel('Средняя цена')
plt.legend()
plt.show()
