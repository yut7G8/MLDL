import numpy as np

class Loss():
    def __init__(self) -> None:
        pass
    
    def mean_squared_error(self, y, t):
        return 0.5 * np.sum((y - t)**2)

    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))
    
    

