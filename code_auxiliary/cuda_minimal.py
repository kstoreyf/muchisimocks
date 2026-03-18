import torch

print('torch.cuda.is_available():', torch.cuda.is_available())
print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
from sympy import S
print("imported sympy")
