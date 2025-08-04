import numpy as np
import matplotlib.pylab as plt
import sys, os
from dataset.mnist import load_mnist
from PIL import Image

def step_function(x):
    y = x > 0
    return y.astype(int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3

    return a3

def import_mnist():
    sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
    # 第一次调用会花费几分钟……
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=False)
    print(t_train.shape)  # (60000,)
    print(x_test.shape)  # (10000, 784)
    print(t_test.shape)  # (10000,)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

if __name__ == '__main__':
    sys.path.append(os.pardir)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=False)
    img = x_train[0]
    label = t_train[0]
    print(label)  # 5
    print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # 把图像的形状变成原来的尺寸
    print(img.shape)  # (28, 28)
    img_show(img)