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
from core.ops.supernet import SuperNet
from hardware.predictor import HardwarePredictor, HardwareConstraints
from core.ops.search_space import SearchSpace 


class DARTSSearcher:
    """Differentiable Architecture Search implementation"""
    
    def __init__(self, supernet: SuperNet, hardware_predictor: HardwarePredictor, 
                 constraints: HardwareConstraints, lambda_hw: float = 0.1):
        self.supernet = supernet
        self.hardware_predictor = hardware_predictor
        self.constraints = constraints
        self.lambda_hw = lambda_hw
        
        # Optimizers
        self.model_optimizer = torch.optim.SGD(
            supernet.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4
        )
        self.arch_optimizer = torch.optim.Adam(
            supernet.alphas, lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3
        )
        
    def search_step(self, train_data: torch.Tensor, train_targets: torch.Tensor,
                   valid_data: torch.Tensor, valid_targets: torch.Tensor) -> Dict[str, float]:
        """Single search step combining model and architecture optimization"""
        
        # Step 1: Update model parameters
        self.model_optimizer.zero_grad()
        logits = self.supernet(train_data)
        model_loss = F.cross_entropy(logits, train_targets)
        model_loss.backward()
        self.model_optimizer.step()
        
        # Step 2: Update architecture parameters
        self.arch_optimizer.zero_grad()
        logits = self.supernet(valid_data)
        accuracy_loss = F.cross_entropy(logits, valid_targets)
        
        # Hardware-aware loss
        hw_loss = self._compute_hardware_loss(valid_data.shape)
        total_loss = accuracy_loss + self.lambda_hw * hw_loss
        
        total_loss.backward()
        self.arch_optimizer.step()
        
        return {
            'model_loss': model_loss.item(),
            'accuracy_loss': accuracy_loss.item(),
            'hardware_loss': hw_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _compute_hardware_loss(self, input_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """Compute hardware constraint violation penalty"""
        subnet = self.supernet.sample_subnet()
        costs = self.hardware_predictor.predict_costs(subnet, input_shape)
        
        # Penalty for constraint violations
        penalties = torch.tensor(0.0, device=next(self.supernet.parameters()).device)
        
        if costs['latency'] > self.constraints.max_latency:
            penalties += (costs['latency'] - self.constraints.max_latency) ** 2
        if costs['memory'] > self.constraints.max_memory:
            penalties += (costs['memory'] - self.constraints.max_memory) ** 2
        if costs['flops'] > self.constraints.max_flops:
            penalties += (costs['flops'] - self.constraints.max_flops) ** 2
        if costs['energy'] > self.constraints.max_energy:
            penalties += (costs['energy'] - self.constraints.max_energy) ** 2
            
        return penalties
    
    def get_best_architecture(self) -> Dict:
        """Extract the best architecture from current weights"""
        operations = self.supernet.search_space.get_operations()
        best_arch = {'layers': []}
        
        for i, alpha in enumerate(self.supernet.alphas):
            weights = F.softmax(alpha, dim=0)
            best_op_idx = torch.argmax(weights).item()
            
            layer = {
                'op': operations[best_op_idx],
                'layer_id': i,
                'confidence': weights[best_op_idx].item()
            }
            best_arch['layers'].append(layer)
        
        return best_arch

def test():
    search_space = SearchSpace({
        'operations': ['conv3x3', 'conv5x5', 'maxpool3x3'],
        'num_layers': 4,
        'channels': [16, 32, 64],
        'reduce_layers': [2]
    })
    op_dict = {
        "conv3x3": lambda C_in, C_out, stride=1: nn.Conv2d(
            C_in, C_out, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        "conv5x5": lambda C_in, C_out, stride=1: nn.Conv2d(
            C_in, C_out, kernel_size=5, stride=stride, padding=2, bias=False
        ),
        "maxpool3x3": lambda C_in, C_out, stride=1: nn.MaxPool2d(
            kernel_size=3, stride=stride, padding=1
        ),
    }
    supernet = SuperNet(search_space, op_dict) 
    hardware_predictor = HardwarePredictor()
    constraints = HardwareConstraints(max_latency=50.0, max_memory=200.0, max_flops=1e8, max_energy=100.0)

    searcher = DARTSSearcher(supernet, hardware_predictor, constraints, lambda_hw=0.05)

    #validation data (not to be used)
    train_data = torch.randn(8,3,32,32)
    train_targets=torch.randint(0,10,(8,))
    valid_data=torch.randn(8,3,32,32)
    valid_targets=torch.randint(0,10,(8,))

    metrics = searcher.search_step(train_data, train_targets, valid_data, valid_targets)
    print("Search step metrics ", metrics)

    best_arch = searcher.get_best_architecture()
    print("Best architecture found", best_arch)
test()