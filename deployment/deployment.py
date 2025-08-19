import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy
import pathlib
import json, time
from export import ModelExporter
from optimize import ModelOptimizer
import torch.optim as optim


class DeploymentManager: 
    def __init__(self, output_dir: str = "./deployment_outputs"):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    def prepare_for_deployment(self, model: nn.Module, architecture: Dict, 
                               input_shape: Tuple[int, int, int, int]=(1, 3, 32, 32), 
                               optimize: bool = True) -> Dict[str, str]:
        model_name = self._sanitize_model_name(architecture)
        outputs={}
        torch_path = self.output_dir / f"{model_name.pth}"
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': architecture,
            'input_shape': input_shape
        }, torch_path)
        outputs['pytorch']=str(torch_path)

        #ONNX report
        onnx_path = self.output_dir / f"{model_name}.onnx"
        ModelExporter.export_to_onnx(model, input_shape, str(onnx_path))
        outputs['onnx'] = str(onnx_path)
        #torch export
        torchscript_path = self.output_dir / f"{model_name}.py"
        ModelExporter.export_to_torchscript(model, input_shape, str(torchscript_path))
        outputs['torchscript'] = str(torchscript_path)

        if optimize:
            #quant version
            quantized_model = ModelOptimizer.quantize_dynamic(model)
            quantized_path = self.output_dir / f"{model_name}_quantized.pth"
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'architecture': architecture,
                'quantized': True
            }, quantized_path)
            outputs['quantized']=str(quantized_path)

            #pruned version
            pruned_model = ModelOptimizer.prune_model(model, sparsity=0.3)
            pruned_path = self.output_dir / f"{model_name}_pruned.pth"
            torch.save({
                'model_state_dict': pruned_model.state_dict(),
                'architecture': architecture,
                'prunned':True
            }, pruned_path)
            outputs['pruned']=str(pruned_path)
        self._generate_deployment_info(model_name, architecture, outputs, input_shape)
        return outputs
    def _sanitize_model_name(self, architecture: Dict)-> str: 
        """For our sanitie's sake, we'll process a safe filename from the architecture"""
        ops = [layer['op']for layer in architecture['layers'][:3]]
        name = "_".join(op.replace('_', '') for op in ops)
        return name[:50]
    def _generate_deployment_info(self, model_name:str, architecture: Dict, 
                                  output_files: Dict[str, str], input_shape: Tuple):
        info = {
            'model_name': model_name,
            'architecture': architecture,
            'input_shape': input_shape,
            'output_files': output_files,
            'deployment_instructions': {
                'pytorch': 'Load with torch.load() and model.load_state_dict()',
                'onnx': 'Use ONNX Runtime: ort.InferenceSession(model_path)',
                'torchscript': 'Load with torch.jit.load()',
                'quantized': 'Quantized PyTorch model for faster inference',
                'pruned': 'Pruned PyTorch model with reduced parameters'
            }
        }
        
        info_path = self.output_dir / f"{model_name}_deployment_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

