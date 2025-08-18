import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class ArchitectureBuilder:
    """Build deployable models from NAS search results"""

    def __init__(self, search_space_config: Dict):
        self.config = search_space_config
        self.operations = self._build_operation_map()
    def _build_operation_map(self) -> Dict:
        """Map operation names to actual PyTorch modules"""
        return {
            'sep_conv_3x3': lambda C: self._sep_conv_block(C, C, 3, 1, 1),
            'sep_conv_5x5': lambda C: self._sep_conv_block(C, C, 5, 1, 2),
            'dil_conv_3x3': lambda C: self._dil_conv_block(C, C, 3, 1, 2, 2),
            'dil_conv_5x5': lambda C: self._dil_conv_block(C, C, 5, 1, 4, 2),
            'avg_pool_3x3': lambda C: nn.AvgPool2d(3, 1, 1),
            'max_pool_3x3': lambda C: nn.MaxPool2d(3, 1, 1),
            'skip_connect': lambda C: nn.Identity(),
            'conv_1x1': lambda C: self._conv_block(C, C, 1, 1, 0),
            'conv_3x3': lambda C: self._conv_block(C, C, 3, 1, 1),
            'none': lambda C: self._zero_op()
        }
    def _sep_conv_block(self, C_in, C_out, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )
    def _dil_conv_block(self, C_in, C_out, kernel_size, stride, padding, dilation):
        return nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )
    
    def _conv_block(self, C_in, C_out, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True)
        )
    def _zero_op(self):
        return nn.Sequential()
    #having called the operations, let's build the model from the architecture description
    def build_model(self, architecture: Dict, input_channels: int=3, num_classes: int =10) -> nn.Module:
        """Build a complete model from the architecture description"""
        parent_builder  = self
        class FoundArchitecture(nn.Module):
            def __init__(self, arch_config, input_channels, num_classes):
                super().__init__()
                self.parent_builder = parent_builder
                self.layers = nn.ModuleList()
                
                # Stem
                current_channels = 16  # Starting channels
                self.stem = nn.Sequential(
                    nn.Conv2d(input_channels, current_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(current_channels),
                    nn.ReLU(inplace=True)
                )
                
                # Main layers
                for i, layer_config in enumerate(arch_config['layers']):
                    op_name = layer_config['op']
                    if op_name in self.parent_builder.operations:
                        layer = self.parent_builder.operations[op_name](current_channels)
                        self.layers.append(layer)

                    #channel changes at reduction points
                    if i in [2, 5] and current_channels < 64: 
                        current_channels *= 2 
                        self.layers.append(nn.Conv2d(current_channels//2, current_channels, 1, 2, bias=False))
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Linear(current_channels, num_classes)
            
            #forward pass
            def forward(self, x): 
                x = self.stem(x)
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                x=self.global_pool(x)
                x=x.view(x.size(0), -1)
                x=self.classifier(x)
                return x
        model = FoundArchitecture(architecture, input_channels, num_classes)
        return model
