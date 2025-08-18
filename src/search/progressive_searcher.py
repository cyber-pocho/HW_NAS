
from core.ops.search_space import SearchSpace
from hardware.predictor import HardwareConstraints, HardwarePredictor
from typing import Dict, Tuple, List
import numpy as np
from search.evolutionary_searcher import EvolutionarySearcher

class ProgressiveSearcher: 

    def __init__(self, search_space: SearchSpace, hardware_predictor: HardwarePredictor,
                 pruning_stages: int = 3, pruning_ratio: float = 0.3):
        self.search_space = search_space
        self.hardware_predictor = hardware_predictor
        self.pruning_stages = pruning_stages #new variable
        self.pruning_ratio = pruning_ratio #new variable
        self.operation_scores = {op: 0.0 for op in search_space.operations} #new variable
        self.evaluated_architectures = [] #new variable

    def evaluate_operations(self, num_samples: int=100) -> Dict[str, float]:
        """Evalueate individual operations to guide prunning"""
        operations_performance = {op: [] for op in self.search_space.operations}

        for _ in range(num_samples):
            arch = self.search_space.sample_architecture()
            costs = self.hardware_predictor.predict_costs(arch, (1,3,32,32))

            efficency = 1.0 / (1.0+costs['latency'] + costs['memory'] + costs['flops'])

            for layer in arch['layers']:
                operations_performance[layer['op']].append(efficency) 
        for op in operations_performance: 
            if operations_performance[op]:
                self.operation_scores[op] = np.mean(operations_performance[op])
        return self.operation_scores
    
    def prune_search_space(self) -> List[str]:
        """Prune least promosing operations"""
        sorted_ops = sorted(self.operation_scores.items(), key=lambda x:x[1], reverse=True)
        keep_count = max(2, int(len(sorted_ops)*(1 - self.pruning_ratio)))
        pruned_operations = [op for op, _ in sorted_ops[:keep_count]]

        print(f"Pruned search space: {len(pruned_operations)} operations remaining")
        print(f"Kept operations: {pruned_operations}")

        return pruned_operations
    
    def progressive_search(self, searcher_type: str = 'evolutionary', **kwargs) -> Dict:
        """Run progressive searc with multiple pruning stages"""
        results = []
        current_operations = self.search_space.operations.copy()
        for stage in range(self.pruning_stages):
            print(f"Progressive search stage {stage + 1}/{self.pruning_stages}")
            print(f"Current search space: {len(current_operations)} operations")

            # update search space
            stage_search_space = SearchSpace({
                **self.search_space.config,
                'operations': current_operations
            })

            #run search with current space
            if searcher_type == 'evolutionary':
                searcher = EvolutionarySearcher(
                    stage_search_space, self.hardware_predictor, 
                    HardwareConstraints(), **kwargs
                )
                stage_result = searcher.evolve(generations=50)
            else:
                raise NotImplementedError("DARTS progressive search not implemented yet")
            results.append(stage_result)

            if stage<self.pruning_stages - 1:
                self.evaluate_operations(num_samples=200)
                current_operations = self.prune_search_space()
        return {
            'stage_results': results,
            'final_architecture': results[-1]['best_architecture'],
            'operation_evolution': self.operation_scores
        }