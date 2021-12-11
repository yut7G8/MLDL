import numpy as np
from module.model import CNN
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def main():
     
     # ここで入力、目標出力の決める
     X = np.array([1, 1, 0, 0])
     b = np.array([0, 1])
     num = 1 # 入力パターン2種類 1or2 
     epoch = 1000
     E_list = []

     cnn = CNN(X, b)
     # 順伝播
     res = cnn.forward()
     # 誤差逆伝播法(誤差逆伝搬法)
     W, wb1, W3 = cnn.backward(res)

     for i in tqdm(range(epoch-1)):
          cnn = CNN(X, b, W, wb1, W3)
          res = cnn.forward(res[-1])
          W, wb1, W3 = cnn.backward(res)

     plt.figure()
     E_df = pd.DataFrame(res[-1])
     E_df.plot(legend=False)
     plt.title("E{}_transition".format(num))
     plt.xlabel("epoch")
     plt.ylabel("E{} (Loss Function)".format(num))
     plt.savefig("./result/E{}_transition.jpeg".format(num))

if __name__ == "__main__":
    main()