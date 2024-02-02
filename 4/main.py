from keras.models import load_model
from IPython.display import Image
from keras import utils
import numpy as np

classes = ['0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9']
model = load_model('mnist_dense.h5')

model.summary()

img_path = '6 (2).jpg'
Image(img_path, width=150, height=150)

img = utils.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")
# Преобразуем картинку в массив
x = utils.img_to_array(img)
# Меняем форму массива в плоский вектор
x = x.reshape(1, 784)
# Инвертируем изображение
x = 255 - x
# Нормализуем изображение
x /= 255

prediction = model.predict(x)
prediction = np.argmax(prediction)
print("Номер класса:", prediction)
print("Название класса:", classes[prediction])