import numpy as np
from IPython.display import Image
from keras.models import load_model
from PIL import Image

model = load_model('CIFAR10.h5')

def classify_image(filename):
    img = Image.open(filename)
    img = img.resize((32, 32))
    img = np.array(img)
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0) # Добавляем размерность батча
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    return class_idx

file_path = '1.jpg'
class_idx = classify_image(file_path)
class_names = ['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
print(f"Классификация изображения: Класс {class_idx} - {class_names[class_idx]}")