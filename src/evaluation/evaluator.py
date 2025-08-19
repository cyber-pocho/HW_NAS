import torch 
import torch.nn as nn
from typing import Dict, Tuple, List
from experiments.arch_builder import ArchitectureBuilder
from experiments.benchmark import HardwareBenchmark
from experiments.benchmark import BenchmarkResult

class NASEvaluator: 
    """complete evaluation of the NAS results"""
    def __init__(self, builder: ArchitectureBuilder, benchmark: HardwareBenchmark):
        self.builder = builder
        self.benchmark = benchmark
        self.results = []
    def evaluate_architecture(self, architecture: Dict, dataset_loader=None, 
                              input_shape: Tuple[int, int, int, int] = (1,3,32,32)) -> Dict:
        """evaluation of a single architecture"""
        model = self.builder.build_model(architecture)
        model_name = self._generate_model_name(model, input_shape, model_name)

        #benchmarking
        benchmark_result = self.benchmark.benchmark_model(model, input_shape, model_name)

        #accuracy evaluation
        accuracy = 0.0

        if dataset_loader is not None:
            accuracy = self._evaluate_accuracy(model, dataset_loader)
            benchmark_result.accuracy = accuracy
        
        efficiency_score = self._calculate_efficiency_score(benchmark_result)

        evaluation_result = {
            'architecture': architecture,
            'benchmark': benchmark_result,
            'efficency_score': efficiency_score,
            'model': model
        }

        self.results.append(evaluation_result)
        return evaluation_result
    
    def _generate_model_name(self, architecture: Dict) -> str:
        """Generate descriptive model name"""
        ops = [layer['op'] for layer in architecture['layers']]
        op_counts = {}
        for op in ops:
            op_counts[op] = op_counts.get(op,0) + 1
        
        #create compact name
        name_parts = []
        for op, count in sorted(op_counts.items()):
            if count > 1:
                name_parts.append(f"{op}x{count}")
            else: 
                name_parts.append(op)
        return "_".join(name_parts[:3]) #limit the length of the returned name
    def _evaluate_accuracy(self, model: nn.Module, dataset_loader) -> float:
        """evaluate model accuracy on dataset"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataset_loader):
                if batch_idx >= 10: #first ten batches
                    break
                data, targets = data.to(self.benchmark.device), targets.to(self.benchmark.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets),sum().item()
        return correct / total if total > 0 else 0.0
    
    def _calculate_efficency_score(self, benchmark_result: BenchmarkResult) -> float:
        """Calculate composite efficency score"""
        #normalize metrics (the lower the params are the better
        latency_score = 1.0 / (1.0+benchmark_result.inference_time_ms / 10.0)
        memory_score = 1.0 / (1.0+benchmark_result.memory_usage_mb/ 10.0)
        flops_score = 1.0 / (1.0+benchmark_result.flops_score/1e6)

        #composite score
        efficency = (latency_score + memory_score + flops_score)/3.0
        return efficency
    def compare_architectures(self, architectures: List[Dict], 
                              dataset_loader=None) -> Dict:
        """Compare multiple architectures"""
        comparison_results = []

        for i, arch in enumerate(architectures):
            print(f"Evaluating architecture {i+1}/{len(architectures)}...")
            result = self.evaluate_architecture(arch, dataset_loader)
            comparison_results.append(result)

        comparison_results.sort(key=lambda x:x['efficency_score'], reverse=True)
        return {
            'results': comparison_results,
            'best_architecture': comparison_results[0],
            'summary': self._generate_comparison_summary(comparison_results)
        }
    def _generate_comparison_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics"""
        latencies = [r['benchmark'].interference_time_ms for r in results]
        memories = [r['benchmark'].memory_usage_mb for r in results]
        accuracies = [r['benchmark'].accuracy for r in results]
        efficiencies = [r['efficiency_score'] for r in results]
        return {
            'num_architectures': len(results),
            'latency_range': (min(latencies), max(latencies)),
            'memory_range': (min(memories), max(memories)),
            'accuracy_range': (min(accuracies), max(accuracies)),
            'efficiency_range': (min(efficiencies), max(efficiencies)),
            'best_efficiency': max(efficiencies),
            'pareto_efficient_count': self._count_pareto_efficient(results)
        }
    def _count_pareto_efficient(self, results:List[Dict]) -> int:
        """count pareto efficient solutions"""
        efficient_count = 0
        for i, result_i in enumerate(results):
            is_dominated = False
            for j, result_j in enumerate(results):
                if i != j:
                    better_accuracy = result_j['benchmark'].accuracy >= result_i['benchmark'].accuracy
                    better_efficiency = result_j['efficiency_score'] >= result_i['efficiency_score']
                    strictly_better = (result_j['benchmark'].accuracy > result_i['benchmark'].accuracy or
                                     result_j['efficiency_score'] > result_i['efficiency_score'])
                    if better_accuracy and better_efficiency and strictly_better:
                        is_dominated = True
                        break
            if not is_dominated:
                efficient_count+=1
        return efficient_count
