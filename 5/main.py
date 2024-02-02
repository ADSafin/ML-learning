from keras.models import load_model
from IPython.display import Image
from keras import utils
import numpy as np

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
model = load_model('fashion_mnist_dense.h5')

model.summary()

img_path1 = 'coat2.jpg'
#img_path2 = 'trousers.jpg'
#img_path3 = 'velvet traction.jpg'

Image(img_path1, width=150, height=150)

img = utils.load_img(img_path1, target_size=(28, 28), color_mode = "grayscale")
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
