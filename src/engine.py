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
    #    self.requires_grad = False
    #    self.grad = np.zeros_like(self.data) if self.requires_grad else None
       self.grad = np.zeros_like(self.data)
       self._backward = lambda: None

    def __repr__(self):
        # return f"Tensor(data={self.data}, dtype={type(self.data)})"
        return f"Tensor(data={self.data}, grad={self.grad}, children={self._prev})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # if self.requires_grad:
                self.grad += out.grad
            # if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.data + self.data, (other, self), '+')

        def _backward():
            # if self.requires_grad:
                self.grad += out.grad
            # if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward

        return out
    

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        print(reversed(topo))
        for v in reversed(topo):
            v._backward()
        
