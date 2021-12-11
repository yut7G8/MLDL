import numpy as np
from module.model import DNN

def main():

    dnn = DNN()
    res = dnn.forward()
    # print(type(res))
    for r in res:
        print(r)
    # 誤差逆伝搬法
    W1, W2, W3 = dnn.easy_backward(list(res))
    dnn2 = DNN(W1, W2, W3)
    res2 = dnn.forward()
    for r in res2:
        print(r)

if __name__ == "__main__":
    main()