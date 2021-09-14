import numpy as np
import matplotlib.pylab as plt

class Activation():

    def __init__(self) -> None:
        pass

    def step_func(self, x):
        # if x > 0:
        #     return 1
        # else:
        #     return 0
        y = x > 0
        return y.astype(np.int)
    
    def plot_step(self, x):
        y = self.step_func(x)
        plt.plt(x, y)
        plt.ylim(-0.1, 1.1)
        plt.show()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def plot_sigmoid(self, x):
        y = self.sigmoid(x)
        plt.plt(x, y)
        plt.ylim(-0.1, 1.1)
        plt.show()
    
    def relu(self, x):
        return np.matrix(0, x)
        
    def identity_func(self, x):
        return x

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y
    
