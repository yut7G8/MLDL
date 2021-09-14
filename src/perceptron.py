import numpy as np

class Gate():

    def __init__(self) -> None:
        self.w1, self.w2, self.theta = 0.5, 0.5, 0.7
        self.w = np.array([0.5, 0.5])
        self.beta = - self.theta

    def AND(self, x1, x2):
        tmp = x1 * self.w1 + x2 *self.w2
        if tmp <= self.theta:
            return 0
        elif tmp > self.theta:
            return 1

    def AND(self, x1, x2):
        x = np.array([x1, x2])
        tmp = np.sum(self.w*x)+self.beta
        if tmp <= 0:
            return 0
        else:
            return 1
    
    def NAND(self, x1, x2):
        x = np.array([x1, x2])
        w = - self.w
        b = self.theta
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else:
            return 1
    
    def OR(self, x1, x2):
        x = np.array([x1, x2])
        b = -0.2
        tmp = np.sum(self.w*x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def XOR(self, x1, x2):
        s1 = self.NAND(x1, x2)
        s2 = self.OR(x1, x2)
        y = self.AND(s1, s2)
        
        return y

    



