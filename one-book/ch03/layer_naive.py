import numpy as np

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.x.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)


def main1():
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print(price)

    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple, dapple_num, dtax)

def main2():
    W = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3, 4, 5])
    affine = Affine(W, b)
    affine.forward(np.array([1, 1, 1, 1, 1]))
    affine.backward(np.array([1, 2, 3, 4, 5]))
    print(affine.dW)
    print(affine.db)

if __name__ == '__main__':
    main2()