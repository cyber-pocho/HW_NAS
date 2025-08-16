import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import yaml
from core.ops.operations import OPS
#src/core/search_space.py
class SearchSpace:
    """
    SearchSpace

    Defines a search space for neural network architectures, allowing random sampling
    of layer configurations for use in Neural Architecture Search (NAS) or DARTS.

    Attributes:
        config (Dict): Configuration dictionary for the search space.
        operations (List[str]): Candidate operations for each layer.
        num_layers (int): Number of layers in the architecture.
        channels (List[int]): Possible channel sizes for layers.
        reduce_layers (List[int]): Indices of layers that downsample the feature map.

    Methods:
        get_operations() -> List[str]:
            Returns the list of candidate operations.

        sample_architecture() -> Dict:
            Generates a random architecture by sampling operations, channels, and strides 
            according to the search space configuration. Returns a dictionary with 'layers'
            and 'num_classes'.
"""

    def __init__(self, config: Dict):
        self.config = config
        self.operations = config.get('operations', list(OPS.keys()))
        self.num_layers=config.get('num_layers',8)
        self.channels=config.get('channels', [16,32,64])
        self.reduce_layers=config.get('reduce_layers', [2,5])

    def get_operations(self)->List[str]:
        return self.operations
    def sample_architecture(self)->Dict:
        """Sample a random variable from search space"""
        layers=[]
        current_channels=self.channels[0]

        for i in range(self.num_layers):
            stride = 2 if i in self.reduce_layers else 1
            if stride == 2 and len([c for c in self.channels if c>current_channels])>0:
                current_channels = min([c for c in self.channels if c>current_channels])
            layer = {
                'op': np.random.choice(self.operations),
                'channels':current_channels,
                'stride':stride,
                'layer_id':1
            }
            layers.append(layer)
        return {'layers': layers, 'num_classes':self.config.get('num_classes', 10)}
    
"""EXAMPLE USAGE - TEST ONLY (uncomment to test)"""

# def sample_random_architecture():
#     OPS = {
#         'conv_3x3': lambda C, stride: None,
#         'sep_conv_3x3': lambda C, stride: None,
#         'max_pool_3x3': lambda C, stride: None
#     }

#     config = {
#         'operations': list(OPS.keys()),
#         'num_layers': 6,
#         'channels': [16, 32, 64],
#         'reduce_layers': [2, 4],
#         'num_classes': 10
#     }

#     space = SearchSpace(config)

#     arch = space.sample_architecture()
#     return arch

# # Example usage
# architecture = sample_random_architecture()
# print(architecture)
