from engine import Tensor

tensor_a = Tensor([1,2])
tensor_b = [3,1]
out = tensor_b + tensor_a
print("Addition:", out)
print(out._prev)