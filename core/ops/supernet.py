#src/core/supernet.py 
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
from core.ops.search_space import SearchSpace
from core.ops.mixed_ops_darts import MixedOp

class SuperNet(nn.Module):
    """
    SuperNet

    A differentiable super-network for Neural Architecture Search (DARTS). 
    It contains all candidate operations (MixedOps) at each layer, with learnable 
    architecture parameters (alphas) to weight their contributions.

    Attributes:
        search_space (SearchSpace): Defines operations, channels, and layer structure.
        alphas (nn.ParameterList): Learnable weights for each MixedOp layer.
        layers (nn.ModuleList): Network layers including MixedOps and optional 1x1 convs.
        global_pooling (nn.AdaptiveAvgPool2d): Pools spatial dimensions before classifier.
        classifier (nn.Linear): Final linear layer for class prediction.

    Methods:
        _build_supernet(input_channels):
            Constructs the super-network by adding stem, MixedOps, and stride-adjustment layers.

        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the network, using softmax-weighted MixedOps.

        get_arch_weights() -> List[torch.Tensor]:
            Returns the current softmaxed architecture weights (alphas).

        sample_subnet() -> Dict:
            Samples a discrete subnet by choosing one operation per layer according to learned weights.
"""


    def __init__(self, search_space: SearchSpace, input_channels: int=3, num_classes: int=10):
        super().__init__()
        self.search_space=search_space
        self.num_classes=num_classes

        #Arch params
        self.alphas=nn.ParameterList()
        self.layers=nn.ModuleList()
        self._build_supernet(input_channels)
        #final classifier
        self.global_pooling=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Linear(search_space.channels[-1], num_classes)
    def _build_supernet(self, input_channels: int):
        """Build the supernet with mixed operations"""
        operations = self.search_space.get_operations()
        current_channels = self.search_space.channels[0]

        #stem layer
        self.stem=nn.Sequential(
            nn.Conv2d(input_channels, current_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True)
        )

        #mixed ops layers
        for i in range(self.search_space.num_layers):
            stride = 2 if i in self.search_space.reduce_layers else 1

            if stride == 2 and len([c for c in self.search_space.channels if c > current_channels]) > 0:
                next_channels = min([c for c in self.search_space.channels if c > current_channels])

                self.layers.append(nn.Conv2d(current_channels, next_channels, 1, stride=stride, bias=False))
                current_channels = next_channels
                stride = 1

            mixed_op = MixedOp(current_channels, stride, operations)
            self.layers.append(mixed_op)
            #arch params for this layer
            alpha=nn.Parameter(torch.randn(len(operations)))
            self.alphas.append(alpha)
       
        if current_channels != self.search_space.channels[-1]:
            self.layers.append(nn.Conv2d(current_channels, self.search_space.channels[-1], 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through supernet"""
        x=self.stem(x)

        alpha_idx = 0
        for layer in self.layers:
            if isinstance(layer, MixedOp): 
                weights = F.softmax(self.alphas[alpha_idx], dim=0)
                x=layer(x, weights)
                alpha_idx+=1
            else:
                x = layer(x)
        x=self.global_pooling(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
    def get_arch_weights(self) -> List[torch.Tensor]:
        """Get current arch weights (alpha)"""
        return [F.softmax(alpha, dim=0) for alpha in self.alphas]
    def sample_subnet(self)->Dict:
        layers=[]
        operations=self.search_space.get_operations()

        for i, alpha in enumerate(self.alphas):
            weights = F.softmax(alpha, dim=0)
            op_idx = torch.multinomial(weights, 1).item()

            layer = {
                'op': operations[op_idx],
                'layer_id': i,
                'weight': weights[op_idx].item()
            }
            layers.append(layer)
        return {'layers': layers}
    

"""To test, uncomment"""
# def test_supernet():
#     # Define a simple search space
#     search_space = SearchSpace({
#         'operations': ['conv_3x3', 'sep_conv_3x3', 'max_pool_3x3'],
#         'num_layers': 3,
#         'channels': [16, 32, 64],
#         'reduce_layers': [1],
#         'num_classes': 10
#     })

#     # Create SuperNet
#     supernet = SuperNet(search_space, input_channels=3, num_classes=10)

#     # Sample input
#     x = torch.randn(1, 3, 32, 32)

#     # Forward pass
#     print("Shape before classifier:", x.shape)
#     output = supernet(x)
#     print("Output shape:", output.shape)

#     # Get architecture weights
#     arch_weights = supernet.get_arch_weights()
#     print("Architecture weights:", arch_weights)

#     # Sample a discrete subnet
#     subnet = supernet.sample_subnet()
#     print("Sampled subnet:", subnet)

# # Run the test
# test_supernet()
