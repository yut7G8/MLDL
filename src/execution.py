import sys, os

from numpy.lib import twodim_base
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np
import pickle
from activationfunc import Activation

class Execution():

    def __init__(self) -> None:
        pass

    def get_data(self):
        (x_train, t_train), (x_test, t_test) = \
            load_mnist(flatten=True, normalize=False)

        return x_train, t_train, x_test, t_test
    
    def init_network():
        with open("data/sample_weight.pkl", 'rb') as f:
            network = pickle.load(f)

        return network

    def predict(network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x, W1) + b1
        z1 = Activation.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = Activation.sigmoid(a2)
        a3 = np.dot(z2, W1) + b3
        y = Activation.sigmoid(a3)

        return y
    
    def img_shoe(self, img):
        pil_img = Image.fromarray(np.uint8(img))
        pil_img.show()

    def numrecog(self):
        x, t = self.get_data()
        network = self.init_network()

        accuracy_cnt = 0

        for i in range(len(x)):
            y = self.predict(network, x[i])
            p = np.argmax(y)
            if p == t[i]:
                accuracy_cnt += 1
        
        print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    
    def numrecog_batch(self):
        x_train, t_train, x_test, t_tset = self.get_data()
        network = self.init_network()

        batch_size = 100
        accuracy_cnt = 0

        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size]
            y_batch = self.predict(network, x_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy_cnt += np.sum(p == t_tset[i:i+batch_size])
        
        print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))
    
    def numrecog_minibatch(self):
        x_train, t_train, x_test, t_tset = self.get_data()
        network = self.init_network()

        train_size = x_train.shape[0]
        batch_size = 10
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        