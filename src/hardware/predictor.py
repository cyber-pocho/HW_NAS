import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class HardwarePredictor: 
    """
HardwarePredictor

Estimates hardware costs (latency, memory, FLOPs, energy) for a neural network 
architecture on a given platform (default: mobile).

Attributes:
    platform (str): Target hardware platform.
    op_costs (dict): Predefined costs for each operation.

Methods:
    _init_op_costs():
        Initializes static cost values for supported operations.

    predict_costs(architecture, input_shape):
        Computes total costs for a network given its architecture and input shape.
        'architecture' should have a list of layers with 'op' and optional 'stride'.
        'input_shape' is (batch_size, channels, height, width).
"""

    def __init__(self, platform: str = "mobile"):
        self.platform = platform
        self.op_costs = self._init_op_costs()

    def _init_op_costs(self) -> Dict[str, Dict[str, float]]:
        # cost models (stated not learned from profiling)
        costs = {
        'mobile': {
            'conv_3x3': {'latency': 0.8, 'memory': 1.0, 'flops': 1.0, 'energy': 1.0},
            'sep_conv_3x3': {'latency': 0.3, 'memory': 0.4, 'flops': 0.3, 'energy': 0.4},
            'sep_conv_5x5': {'latency': 0.5, 'memory': 0.6, 'flops': 0.5, 'energy': 0.6},
            'dil_conv_3x3': {'latency': 0.4, 'memory': 0.5, 'flops': 0.4, 'energy': 0.5},
            'dil_conv_5x5': {'latency': 0.7, 'memory': 0.8, 'flops': 0.7, 'energy': 0.8},
            'avg_pool_3x3': {'latency': 0.1, 'memory': 0.1, 'flops': 0.05, 'energy': 0.1},
            'max_pool_3x3': {'latency': 0.1, 'memory': 0.1, 'flops': 0.05, 'energy': 0.1},
            'skip_connect': {'latency': 0.01, 'memory': 0.01, 'flops': 0.0, 'energy': 0.01},
            'none': {'latency': 0.0, 'memory': 0.0, 'flops': 0.0, 'energy': 0.0},
        }
    }
        return costs.get(self.platform, costs['mobile'])
    def predict_costs(self, architecture: Dict, input_shape: Tuple[int, int, int]) -> Dict[str, float]:
        """Hardware costs for an architecture"""
        batch_size, channels, height, width = input_shape
        total_costs = {'latency': 0.0, 'memory':0.0, 'flops': 0.0, 'energy':0.0}
        current_shape = (height, width, channels)

        for layer_config in architecture['layers']:
            op_name = layer_config['op']
            stride = layer_config.get('stride', 1)

            #operation specific costs
            h,w,c = current_shape
            op_costs = self.op_costs.get(op_name, self.op_costs['conv_3x3'])

            scale_factor = (h*2*c)/(32*32*16) #normalization

            for metric in total_costs: 
                total_costs[metric] += op_costs[metric]*scale_factor
            if stride > 1: 
                current_shape = (h//stride, w//stride, c)
        return total_costs
        
## src/hardware/predictor.py  

"""Usage example"""

# predictor = HardwarePredictor(platform="mobile")

# arch = {
#     'layers': [
#         {'op': 'conv_3x3', 'stride': 1},
#         {'op': 'sep_conv_3x3', 'stride': 2},
#         {'op': 'skip_connect'}
#     ]
# }

# input_shape = (1, 16, 32, 32)  # batch_size, channels, height, width
# costs = predictor.predict_costs(arch, input_shape)
# print(costs)

@dataclass
class HardwareConstraints:
    """Hardware constraints specifications"""
    max_latency: float=100.0
    max_memory: float=50.0
    max_flops: float=300.0
    max_energy: float=1000.0
    platform: str = 'mobile'










