import numpy as np
import sys
class PartyNN(object):
    def __init__(self, learning_rate=0.1):
        # случайные исходные веса от 0 к 1 уровню и от 1 к выходному
        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5,(2, 3))
        self.weights_1_2 = np.random.normal(0.0, 1, (1, 2))
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def predict(self, inputs):
        # метод обучения и проход через сигмоиды
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)
        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        return outputs_2
    def train(self, inputs, expected_predict):
        # реализация метода обучения обратного распространения
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)
        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        actual_predict = outputs_2[0]
        error_layer_2 = np.array([actual_predict - expected_predict])
        gradient_layer_2 = actual_predict * (1 - actual_predict)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        self.weights_1_2 -= (np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * self.learning_rate
def MSE(y, Y):
    return np.mean((y-Y)**2)
train = [
 ([0, 0, 0], 1),
 ([0, 0, 1], 0),
 ([0, 1, 0], 1),
 ([0, 1, 1], 1),
 ([1, 0, 0], 0),
 ([1, 0, 1], 0),
 ([1, 1, 0], 1),
 ([1, 1, 1], 0)]
print('Обучение моделей с разницей эпох')
epochs_list = [15, 30, 45, 60, 75]
predict_list = []
epochs = 15
learning_rate = 0.1
network = PartyNN(learning_rate=learning_rate)
for i in range(0, len(epochs_list)):
    epochs = epochs_list[i]
    for e in range(epochs):
        inputs_ = []
        counter = 0
        correct_predictions = []
        for input_stat, correct_predict in train:
            network.train(np.array(input_stat), correct_predict)
            inputs_.append(np.array(input_stat))
            correct_predictions.append(np.array(correct_predict))

            # # выполняем предсказание сети
            if network.predict(np.array(input_stat)) > 0.5 and correct_predict == 1:
                counter = counter + 1
            if network.predict(np.array(input_stat)) <= 0.5 and correct_predict == 0:
                counter = counter + 1
            # # посмотрим кол-во  предсказывания сети
    predict_list.append(counter)
    counter = 0
    train_loss = MSE(network.predict(np.array(inputs_).T), np.array(correct_predictions))
    sys.stdout.write("\r Количество эпох: {},Progress: {}, Training loss:{}".format(epochs_list[i],str(100 * e/float(epochs))[:4], str(train_loss)[:5]))
    print("")
    for input_stat, correct_predict in train:
        print("For input: {} the prediction is: {}, expected:{}".format(str(input_stat),str(network.predict(np.array(input_stat)) > 0.5),str(correct_predict == 1)))
    for input_stat, correct_predict in train:
        print("For input: {} the prediction is: {}, expected:{}".format(str(input_stat),str(network.predict(np.array(input_stat))),str(correct_predict == 1)))
    print(network.weights_0_1)
    print(network.weights_1_2)
print('Количество верных ответов')
print(predict_list)