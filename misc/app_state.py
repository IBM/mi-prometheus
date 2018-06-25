from misc.singleton import SingletonMetaClass
import torch

class AppState(metaclass=SingletonMetaClass):
    def __init__(self):
        self.visualize = False
        self.dtype=torch.FloatTensor
        self.dtype_long=torch.LongTensor
