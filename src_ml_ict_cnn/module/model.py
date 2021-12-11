import numpy as np
from module.activationfunc import Activation

class CNN():

    def __init__(self, X, b, W=np.ones((2, 1)), wb1=np.ones((3, 1)), w3=np.ones((3, 2))) -> None:
        """
        Parameters
        -------
        X : np.array
            入力 [x1, x2, x3, x4]
        b : np.array
            目標出力 [b1, b2]
        W : np.array
            畳み込み/コンボリューションの重み.初期値は全て1
        wb1 : np.array
            x0(ダミー)の結線にかかる重み.初期値は全て1
        w3 : np,array
            全結合層の重みパラメータ.初期値は全て1

        """
        self.X = X
        self.b = b
        self.W = W
        self.wb1 = wb1
        self.w3 = w3
    
    def forward(self, E_list=[], pooling_size=2):
        """
        順伝播

        Parameters
        -------
        E_list : list
            誤差を格納するlist
        pooling_size : int
            プーリング層のサイズ.本問では2
        
        Returns
        -------
        S1, S2, S3, y1, y2, y3, E, x, S2_i, E_list
            順伝播の各結果を返す

        """
        acfunc = Activation()
        # Xを計算しやすいよう変形する
        for s in range(len(self.W)+1):
            try:
                x = np.concatenate([x, self.X[0+s:len(self.W)+s].reshape(1, -1)])
            except:
                x = self.X[0+s:len(self.W)+s].reshape(1, -1)
        # 畳み込み層(l=1)
        S1 = np.dot(x, self.W) + self.wb1
        y1 = acfunc.sigmoid(S1) # 3*1配列
        
        # プーリング層(Max Pooling, l=2)
        # y1を1*3配列に転置後、プーリングしやすうように変形
        y1 = y1.T.reshape(-1)
        for ps in range(pooling_size):
            try:
                y1tmp = np.concatenate([y1tmp, y1[0+ps:pooling_size+ps].reshape(1, -1)])
            except:
                y1tmp = y1[0+ps:pooling_size+ps].reshape(1, -1)
        # print(y1tmp)
        S2 = np.max(y1tmp, axis=1)

        # 最大値の位置を取得しておく
        S2_i = list(np.argmax(y1tmp, axis=1))
        for i in range(len(S2_i)):
            S2_i[i] += i
        y2 = S2 # 恒等写像
        # y2 = y2.reshape(1, -1).T # 2*1配列に

        # 全結合層
        y2 = np.append(1, y2) # ダミーの追加
        S3 = np.dot(y2, self.w3)
        y3 = acfunc.sigmoid(S3)
        E = np.sum((y3-self.b)**2)
        print(E)
        E_list.append(E)

        return (S1, S2, S3, y1, y2, y3, E, x, S2_i, E_list)

    def backward(self, res, alpha=1):
        """
        誤差逆伝播法(誤差逆伝搬法)

        Parameters
        -------
        res : tuple
            順伝播の計算結果を格納している
        alpha : int or float
            ゲイン.デフォルト値は1
        
        Returns
        -------
        W, wb1, W3 : np.array
            修正後の重み

        """
        y1 = res[3]
        y2 = res[4]
        y3 = res[5]
        x = res[7]
        S2_i = res[8]

        # 全結合層の誤差逆伝搬法
        ES3 = alpha*y3*(1-y3)*2*(y3-self.b)
        # print(ES3)
        EW3 = y2.reshape(1, -1).T*ES3
        W3 = self.grad(EW3, self.w3)
        # print(EW3)
        # print(W3)

        # 畳み込み層の誤差逆伝搬法
        ES1 = ((alpha*y2[1:]*(1-y2[1:])).reshape(1, -1).T)*(np.dot(W3[1:], ES3.reshape(1, -1).T))
        ES1 = ES1.T
        EW1 = np.dot(ES1, np.array([x[S2_i[0]], x[S2_i[1]]])).T
        W = self.grad(EW1, self.W)
        # print(EW1)
        # print(W)
        # ダミー結線の重み更新
        tmp = self.wb1
        wb1 = self.grad(EW1, np.array([self.wb1[S2_i[0]], self.wb1[S2_i[1]]]))
        tmp = np.delete(tmp, S2_i[0], axis=0)
        tmp = np.delete(tmp, S2_i[1]-1, axis=0)
        tmp = np.insert(tmp, S2_i[0], wb1[0], axis=0)
        wb1 = np.insert(tmp, S2_i[1], wb1[1], axis=0)
        # print(wb1)

        return (W, wb1, W3) 


        
    def grad(self, EW, W, ep=0.01):
        """
        勾配法 wの修正

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
        
        # for n in range(len(EW)):
        #     for m in range(len(EW[0])):
        #         W[n][m] -= ep*EW[n][m]
        W -= ep*EW
                
        return W