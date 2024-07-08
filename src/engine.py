import numpy as np
from typing import *

Array = Union[np.ndarray, list, int, float]
class Tensor:
    '''
    A class that stores tensor objects and their gradients
    '''
    def __init__(self, data: Array, _children = (), _opt = ''):
       self.data = data if isinstance(data, np.ndarray) else np.array(data)
       self._opt = _opt
       self._prev = set(_children)

    def __repr__(self):
        return f"Tensor(data={self.data}, dtype={type(self.data)})"


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        return out
    
    def __radd__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(other.data + self.data, (other, self), '+')
