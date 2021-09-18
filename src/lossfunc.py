import numpy as np

class Loss:
    def __init__(self) -> None:
        pass
    
    def mean_squared_error(self, y, t):
        return 0.5 * np.sum((y - t)**2)

    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))
    
    def cross_entropy_error_minibatch(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        
        return -np.sum(t + np.log(y + 1e-7)) / batch_size