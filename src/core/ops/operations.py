import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

class SeparableConv(nn.Module):
    """Separable Convolution"""
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=1, dilation=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.op(x)
class DilatedConv(nn.Module):
    """Dilated Convolution"""
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=2, dilation=2): 
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.op(x)
class Identity(nn.Module): 
    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self,stride):
        super().__init__()
        self.stride=stride
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride].mul(0.)

# Dictionary of operations: 
OPS = {
    'none': lambda C, stride: Zero(stride),
    'avg_pool_3x3': lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride: Identity() if stride == 1 else nn.AvgPool2d(2, stride=stride),
    'sep_conv_3x3': lambda C, stride: SeparableConv(C, C, 3, stride, 1),
    'sep_conv_5x5': lambda C, stride: SeparableConv(C, C, 5, stride, 2),
    'dil_conv_3x3': lambda C, stride: DilatedConv(C, C, 3, stride, 2, 2),
    'dil_conv_5x5': lambda C, stride: DilatedConv(C, C, 5, stride, 4, 2),
    'conv_1x1': lambda C, stride: nn.Sequential(
        nn.Conv2d(C, C, 1, stride=stride, bias=False),
        nn.BatchNorm2d(C),
        nn.ReLU(inplace=True)
    ),
    'conv_3x3': lambda C, stride: nn.Sequential(
        nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(C),
        nn.ReLU(inplace=True)
    ),
}

## src/core/operations.py