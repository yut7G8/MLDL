import numpy as np

class Activation:
    
    def __init__(self) -> None:
        pass

    def sigmoid(self, s, alpha=1):
        """
        シグモイド関数
        """
        return 1 / (1 + np.exp(-alpha*s))