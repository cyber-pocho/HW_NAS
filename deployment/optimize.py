import torch
import torch.nn as nn


class ModelOptimizer:
    """Post training optimization"""
    @staticmethod
    def quantize_dynamic(model: nn.Module) -> nn.Module:
        """Dynamic quantization"""
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        return quantized_model
    @staticmethod
    def prune_model(model: nn.Module, sparsity: float = 0.3) -> nn.Module:
        """Apply structured pruning"""
        import torch.nn.utils.prune as prune

        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        prune.global_unstructured(
            parameters_to_prune, 
            pruning_method=prune.L1Unstructured, 
            amount=sparsity
        )

        #pruning is now permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    


