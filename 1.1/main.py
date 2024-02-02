import numpy as np

storm = [0, 0, 0, 0, 1, 1, 1, 1]
pilot = [0, 0, 1, 1, 0, 0, 1, 1]
wind = [0, 1, 0, 1, 0, 1, 0, 1]

expected = [1, 0, 1, 1, 0, 0, 1, 0]


def activation_function(x):
    if x < 0.5:
        return 1
    else:
        return 0


def predict(storm, pilot, wind):
    inputs = np.array([storm, pilot, wind])

    weights_input_to_hiden_1 = [0.6, 0, 0.6]
    weights_input_to_hiden_2 = [-0.4, 0.91, -0.4]

    weights_input_to_hiden = np.array([weights_input_to_hiden_1, weights_input_to_hiden_2])
    weights_hiden_to_output = np.array([-1, 1])
    hiden_input = np.dot(weights_input_to_hiden, inputs)
    hiden_output = np.array([activation_function(x) for x in hiden_input])

    output = np.dot(weights_hiden_to_output, hiden_output)

    return activation_function(output) == 1


for i in range(0, 8):
    print(f"{i + 1} набор данных")
    print("ожидание:" + str(expected[i] == 1))
    print("результат:" + str(predict(storm[i], pilot[i], wind[i])))
