import math
import torch.nn as nn
import torch

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gelu_tensor = 0.5*x*(1 + torch.tanh(torch.sqrt(torch.tensor(2.0/math.pi)) * (x+0.044715*x**3)))
        return  gelu_tensor
