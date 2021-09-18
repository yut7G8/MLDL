import sys, os
sys.path.append(os.pardir)
import numpy as np
from activationfunc import Activation
from lossfunc import Loss
from differential import Differential

class Neuralnet:

    def __init__(self) -> None:
        self.X = np.array([1.0, 0.5])
        self.W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        self.B1 = np.array([0.1, 0.2, 0.3])
        self.W2 = np.array([0.1, 0.4], [0.2, 0.5], [0.3, 0.6])
        self.B2 = np.array([0.1, 0.2])
        self.W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.B3 = np.array([0.1, 0.2])
        self.Activation = Activation()

    def layor3(self):
        A1 = np.dot(self.X, self.W1) + self.B1
        Z1 = self.Activation.sigmoid(A1)
        A2 = np.dot(Z1, self.W2) + self.B2
        Z2 = self.Activation.sigmoid(A2)
        A3 = np.dot(Z2, self.W3) + self.B3
        Y = self.Activation.identity_func(A3)

        return Y

class SimpleNet:

    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = Activation().softmax(z)
        loss = Loss.cross_entropy_error(y, t)

        return loss

class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size,
                weight_init_std=0.01):
        # 重みパラメータの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = Activation().sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = Activation().softmax(a2)

        return y

    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)

        return Loss().cross_entropy_error_minibatch(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        diff = Differential()

        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = diff.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = diff.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = diff.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = diff.numerical_gradient(loss_W, self.params['b2'])

        return grads
    