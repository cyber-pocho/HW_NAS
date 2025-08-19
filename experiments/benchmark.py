import torch
import torch.nn as nn
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import psutil
import platform
from pathlib import Path

class BenchmarkResult: 
    """Container for benchmark results"""
    model_name : str
    platform : str
    inference_time_ms:float
    memory_usage_mb:float
    cp_usage_percent:float
    flops:int
    params:int
    accuracy:float = 0.0
    batch_size: int=1

class HardwareBenchmark: 
    """Comprehensive hardware benchmarking for deployed models"""
    def __init__(self, device: str = 'cpu', warmup_runs: int = 10, benchmark_runs: int = 100):
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, int, int, int], 
                        model_name: str = "unknown"):
        """Benchmarking model""" 
        model.eval()
        model = model.to(self.device)

        dummy_input = torch.randn(input_shape).to(self.device)

        #warm-up
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(dummy_input)
        
        #not using CUDA on this machina, but if src run in a machin e with CUDA, then
        if self.device == 'cuda':
            torch.cuda.synchronize()
        #memory measurement before
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
        else: 
            memory_before = psutil.Process().memory_info().rss
        times = []
        cpu_usage_samples = []

        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                cpu_before = psutil.cpu_percent()
                start_time = time.perf_counter()
                _ = model(dummy_input)

                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append((end_time - start_time)*1000) # convert to ms

                cpu_after = psutil.cpu_percent()
                cpu_usage_samples.append(cpu_after)
        
        if self.device == 'cuda':
            memory_after = torch.cuda.memory_allocated()
            memory_usage = (memory_after - memory_before) / (1024*1024) # mb
        else: 
            memory_after = psutil.Process().memory_info().rss
            memory_usage = (memory_after - memory_before) / (1024*1024)
        
        avg_inference_time = np.mean(times)
        avg_cpu_usage = np.mean(cpu_usage_samples)

        #calculation of flops and params
        flops = self._calculate_flops(model, dummy_input)
        params = sum(p.numel() for p in model.parameters())

        return BenchmarkResult(
            model_name=model_name,
            platform=platform.machine(),
            inference_time_ms=avg_inference_time,
            memory_usage_mb=max(0, memory_usage),
            cpu_usage_percent=avg_cpu_usage,
            flops=flops,
            params=params,
            batch_size=input_shape[0]
        )
    
    def _calculate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> int:
        """Simplified FLOP calculation"""
        total_flops = 0

        def flop_count_hook(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                output_elements = output.numel()
                kernel_flops = module.kernel_size[0] * module.kernel_size[1]*module.in_channels
                total_flops += module.in_feautures * model.out_features
            elif isinstance(module, nn.Linear):
                total_flops += module.in_features * module.out_features
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(flop_count_hook))
        with torch.no_grad():
            model(input_tensor)
        for hook in hooks: 
            hook.remove()
        return total_flops