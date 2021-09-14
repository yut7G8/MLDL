import numpy as np
from activationfunc import Activation
from perceptron import Gate

class Neuralnet():

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


