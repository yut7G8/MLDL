import numpy as np
from numpy.core.fromnumeric import shape
from module.activationfunc import Activation

class DNN():

    def __init__(self, w1=np.ones((3, 2)), w2=np.ones((3, 2)), w3=np.ones((3, 2))) -> None:
        """
        各層での重みパラメータは全て初期値1
        """
        self.x = np.array([1, 1, 1])
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self):
        """
        順伝播

        Returns
        -------
        S1, S2, S3, y1, y2, y3, E
            各層での重みづけ総和S, y, 誤差評価関数E 

        """
        acfunc = Activation()
        S1 = np.dot(self.x, self.w1)
        y1 = acfunc.sigmoid(S1)
        y1 = np.append(1, y1) # バイアスの追加
        S2 = np.dot(y1, self.w2)
        y2 = acfunc.sigmoid(S2)
        y2 = np.append(1, y2) # バイアスの追加
        S3 = np.dot(y2, self.w2)
        y3 = acfunc.sigmoid(S3)
        E = np.sum((y3-1)**2)

        return (S1, S2, S3, y1, y2, y3, E)
    
    def easy_backward(self, res, alpha=1):
        """
        誤差逆伝搬法

        Prameters
        -------
        res : list
            順伝播で返されるS1, S2, S3, y1, y2, y3, E
        alpha : int or float
            ゲイン. デフォルト値1

        Returns
        -------
        W1, W2, W3 : np.array
            修正後の重みパラメータ

        """
        y1 = res[3]
        y2 = res[4]
        y3 = res[5]

        # 3層目
        ES3 = alpha*y3*(1-y3)*2*(y3-1) # dE/dS 右端
        # print(ES3)
        # print(ES3.shape)
        EW3 = y2.reshape(1, -1).T*ES3
        W3 = self.easy_grad(EW3, self.w3)
        print(W3)

        # 2層目
        ES2 = ((alpha*y2[1:]*(1-y2[1:])).reshape(1, -1).T)*(np.dot(W3[1:], ES3.reshape(1, -1).T))
        ES2 = ES2.T
        EW2 = y1.reshape(1, -1).T*ES2
        W2 = self.easy_grad(EW2, self.w2)
        print(W2)

        # 1層目
        ES1 = ((alpha*y1[1:]*(1-y1[1:])).reshape(1, -1).T)*(np.dot(W2[1:], ES2.reshape(1, -1).T))
        ES1 = ES1.T
        EW1 = self.x.reshape(1, -1).T*ES1
        W1 = self.easy_grad(EW1, self.w1)
        print(W1)

        return (W1, W2, W3)

    
    def easy_grad(self, EW, W):
        """
        勾配法(簡略版) wの修正

        Prameters
        -------
        EW : np.array
            dE/dW
        W : np.array
            修正前の重みパラメータ
        
        Returns
        -------
        W : np.array
            修正後の重みパラメータ

        """
        
        for n in range(len(EW)):
            for m in range(len(EW[0])):
                if EW[n][m] > 0:
                    W[n][m] -= 0.01
                else:
                    W[n][m] += 0.01
        return W
