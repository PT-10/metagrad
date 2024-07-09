from engine import Tensor
import numpy as np

tensor_a = Tensor([1,2])
tensor_b = [3,1]
out = tensor_a + tensor_b
out.backward()
# print("Addition:", out)
# out.grad = np.ones_like(out.data)
# out._backward()
# print(out.grad)
# print("Gradients of children")
# print(tensor_a.grad)
# print(tensor_b.grad)
print(out)