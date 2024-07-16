import numpy as np
from typing import *

Array = Union[np.ndarray, list, int, float]
class Tensor:
    '''
    A class that stores tensor objects and their gradients
    '''
    def __init__(self, data, _children = (), _opt = '', requires_grad = False):
       self.data = data.astype(float) if isinstance(data, np.ndarray) else np.array(data, dtype=float)
       if self.data.ndim == 0:  # If it's a scalar, reshape it to (1,)
            self.data = self.data.reshape(1)
       self._opt = _opt
       self._prev = set(_children)
       self.requires_grad = requires_grad
       self.grad = np.zeros_like(self.data) if self.requires_grad else None
       self.grad_fn = lambda: None
       self.shape = self.data.shape

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def set_requires_grad(self):
        '''
        Manages the required_grad parameter for a given node.
        If a node is a leaf node ie no children, then it does not require grad.
        '''
        if self._prev != set():
            self.requires_grad = True

    def initialize_grad(self):
        '''
        After setting requires_grad, initialize the tensor gradient
        '''
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # out.set_requires_grad()
        out = Tensor(self.data + other.data, (self, other), '+')
        out.set_requires_grad()
        out.initialize_grad()
        
        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                if grad_self.shape != self.shape:
                    for axis, (dim_self, dim_out) in enumerate(zip(self.shape, grad_self.shape)):
                        if dim_self != dim_out:
                            grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad = self.grad + grad_self

            if other.requires_grad:
                grad_other = out.grad
                if grad_other.shape != other.shape:
                    for axis, (dim_other, dim_out) in enumerate(zip(other.shape, grad_other.shape)):
                        if dim_other != dim_out:
                            grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad = other.grad + grad_other
        out.grad_fn = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data*other.data, (other, self), '*')
        out.set_requires_grad()
        out.initialize_grad()

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * other.data
                if grad_self.shape != self.shape:
                    for axis, (dim_self, dim_out) in enumerate(zip(self.shape, grad_self.shape)):
                        if dim_self != dim_out:
                            grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad = self.grad + grad_self

            if other.requires_grad:
                grad_other = out.grad * self.data
                if grad_other.shape != other.shape:
                    for axis, (dim_other, dim_out) in enumerate(zip(other.shape, grad_other.shape)):
                        if dim_other != dim_out:
                            grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other
        
        out.grad_fn = _backward
        return out
    
    def __pow__(self, power: float):
        out = Tensor(self.data ** power, (self,), '**')
        out.set_requires_grad()
        out.initialize_grad()

        def _backward():
            if self.requires_grad:
                grad_self = power * (self.data ** (power - 1)) * out.grad
                if grad_self.shape != self.shape:
                    for axis, (dim_self, dim_out) in enumerate(zip(self.shape, grad_self.shape)):
                        if dim_self != dim_out:
                            grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad = self.grad + grad_self
        out.grad_fn = _backward
        return out

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self*other
    
    def __neg__(self): 
        return self * -1
    
    def __sub__(self, other): 
        return self + (-Tensor(other))

    def __rsub__(self, other):
        return other + (-Tensor(self))
    
    def __truediv__(self, other): 
        return self * Tensor(other)**-1

    def __rtruediv__(self, other): 
        return Tensor(other) * self**-1

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
        # if self.grad == np.zeros_like(self.data):
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            print(v)
            v.grad_fn()
