import torch
import torch.nn as nn
from typing import Tuple, Dict, List
class ModelExporter:
    """Export model to various forms of deployement"""
    @staticmethod
    def export_to_onnx(model: nn.Module, input_shape:Tuple[int, int, int, int], 
                       output_path: str, opset_version: int = 11):
        """Export pytorch model to onnx format"""
        model.eval()
        dummy_input = torch.randn(input_shape)

        torch.onnx.export(
            model, 
            dummy_input, 
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input':{0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX: {output_path}")
    
    @staticmethod
    def export_to_torchscript(model: nn.Module, input_shape: Tuple[int, int, int, int], 
                              output_path: str, trace: bool = True):
        model.eval()
        if trace: 
            #tracing mode
            dummy_input = torch.randn(input_shape)
            traced_model = torch.jit.trace(model, dummy_input)
        else:
            traced_model = torch.jit.script(model)
        traced_model.save(output_path)
        print(f"model exported to torchscript: {output_path}")
        return traced_model
    
        