import numpy as np

from ch01.some_functions import sigmoid, softmax
from dataset import mnist


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_train, y_train, x_test, y_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = mnist.pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data()
    network = init_network()

    accuracy = 0
    for i in range(len(x_test)):
        y = predict(network, x_test[i])
        p = np.argmax(y)
        if p == y_test[i]:
            accuracy += 1

    print("Accuracy:" + str(float(accuracy) / len(x_test)))
