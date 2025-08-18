import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from hardware.predictor import HardwarePredictor
from core.ops.operations import OPS

class MixedOp(nn.Module):
    """
    MixedOp

    A weighted combination of candidate operations for differentiable architecture search (DARTS).

    Attributes:
        _ops (nn.ModuleList): Candidate operations.
        operations (List[str]): Names of the operations.

    Methods:
        forward(x, weights):
            Returns weighted sum of operation outputs.

        get_hardware_cost(weights, predictor, input_shape):
            Estimates expected hardware cost (latency, memory, FLOPs, energy) 
            based on operation weights and input shape.
"""

    
    def __init__(self, C: int, stride: int, operations: List[str]):
        super().__init__()
        self._ops = nn.ModuleList()
        self.operations = operations
        
        for op_name in operations:
            op = OPS[op_name](C, stride)
            self._ops.append(op)
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Forward pass with architecture weights"""
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    
    def get_hardware_cost(self, weights: torch.Tensor, predictor: HardwarePredictor, 
                         input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Calculate expected hardware cost based on operation weights"""
        total_cost = {'latency': 0.0, 'memory': 0.0, 'flops': 0.0, 'energy': 0.0}
        
        for w, op_name in zip(weights, self.operations):
            arch = {'layers': [{'op': op_name, 'stride': 1}]}
            op_cost = predictor.predict_costs(arch, input_shape)
            
            for metric in total_cost:
                total_cost[metric] += w.item() * op_cost[metric]
        
        return total_cost

"""Usage example"""


# OPS = {
#     'conv_3x3': lambda C, stride: torch.nn.Conv2d(C, C, 3, stride, 1),
#     'sep_conv_3x3': lambda C, stride: torch.nn.Conv2d(C, C, 3, stride, 1)
# }

# predictor = HardwarePredictor(platform="mobile")
# mixed_op = MixedOp(C=16, stride=1, operations=['conv_3x3', 'sep_conv_3x3'])
# x = torch.randn(1, 16, 32, 32)
# weights = torch.tensor([0.6, 0.4])
# output = mixed_op(x, weights)
# print("Output shape:", output.shape)
# costs = mixed_op.get_hardware_cost(weights, predictor, input_shape=(1, 16, 32, 32))
# #print results
# print("Expected hardware costs:", costs)
