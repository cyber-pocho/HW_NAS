
from hardware.predictor import HardwarePredictor
from typing import Dict, List
import numpy as np



class ParetoOptimizer: 
    """This class is essential. We use a pareto optimizer to not blindly pursue
    accuracy but rather have a balance with efficiency"""
    def __init__(self, hardware_predictor: HardwarePredictor): 
        self.hardware_predictor = hardware_predictor
        self.pareto_front = []
    def dominates(self, sol1: Dict, sol2: Dict ) -> bool: 
        """Check for pareto dominance"""

        better_in_accuracy = sol1['accuracy']>= sol2['accuracy']
        better_in_efficency = sol1['efficency']>= sol2['efficency']

        strictly_better = (sol1['accuracy'] > sol2['accuracy'] or
                           sol1['efficency'] > sol2['efficency'])
        return better_in_accuracy and better_in_efficency and strictly_better
    def calculate_efficency(self, architecture: Dict) -> float:
        """Calculate efficency score"""
        costs = self.hardware_predictor.predict_costs(architecture, (1,3,32,32))

        #normalization
        normalized_latency = costs['latency'] / 100.0
        normalized_memory = costs['memory'] / 100.0
        normalized_flops = costs['flops'] / 100.0

        #efficency is inverse of total_cost

        total_cost = normalized_flops + normalized_memory + normalized_latency
        efficency = 1.0 / (1.0 + total_cost)
        return efficency
    def update_pareto_front(self, architectures: List[Dict], accuracies: List[float]):
        """Update Pareto fron with new solutions"""

        solutions = []

        for arch, acc in zip(architectures, accuracies):
            efficency = self.calculate_efficency(arch)
            solution = {
                'architecture': arch,
                'accuracy': acc,
                'efficency': efficency
            }
            solutions.append(solution)
        all_solutions = self.pareto_front + solutions

        #finding new pareto front
        new_pareto_front = []

        for sol in all_solutions:
            is_dominated = False
            for other_sol in all_solutions:
                if sol != other_sol and self.dominates(other_sol, sol):
                    is_dominated = True
                    break
            if not is_dominated:
                new_pareto_front.append(sol)
        self.pareto_front = new_pareto_front
        return len(self.pareto_front)
    
    def get_diverse_solutions(self, num_solutions: int=5) -> List[Dict]:
        """Get diverse solutions from Pareto front"""
        if len(self.pareto_front) <= num_solutions:
            return self.pareto_front
        #we sort by accuracy
        sorted_front = sorted(self.pareto_front, key=lambda x: x['accuracy'])
        # select evenly spaced solutions
        indices = np.linspace(0, len(sorted_front)-1, num_solutions, dtype=int)
        diverse_solutions = [sorted_front[i] for i in indices]

        return diverse_solutions
    

# Dominated = another solution is better in both accuracy and efficiency.

# Pareto front = the best trade-offs between accuracy and efficiency.

        
