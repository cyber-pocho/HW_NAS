

#imports pending
from typing import Dict, Tuple
from core.ops.search_space import SearchSpace
from hardware.predictor import HardwareConstraints, HardwarePredictor
import json 
import numpy as np

class EvolutionarySearcher:
    """Evolutionary Neural Architecture Search"""
    
    
    def __init__(self, search_space: SearchSpace, hardware_predictor: HardwarePredictor,
                 constraints: HardwareConstraints, population_size: int = 50,
                 mutation_rate: float = 0.1):
        self.search_space = search_space
        self.hardware_predictor = hardware_predictor
        self.constraints = constraints
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_history = []
        
    def initialize_population(self):
        """Initialize random population"""
        self.population = []
        for _ in range(self.population_size):
            arch = self.search_space.sample_architecture()
            self.population.append(arch)
    
    def evaluate_fitness(self, architecture: Dict, accuracy: float = None) -> float:
        """Multi-objective fitness evaluation"""
        # Hardware costs
        costs = self.hardware_predictor.predict_costs(architecture, (1, 3, 32, 32))
        
        # Constraint violations
        violations = 0
        if costs['latency'] > self.constraints.max_latency:
            violations += (costs['latency'] - self.constraints.max_latency) / self.constraints.max_latency
        if costs['memory'] > self.constraints.max_memory:
            violations += (costs['memory'] - self.constraints.max_memory) / self.constraints.max_memory
        if costs['flops'] > self.constraints.max_flops:
            violations += (costs['flops'] - self.constraints.max_flops) / self.constraints.max_flops

        if accuracy is None:
            accuracy = self._estimate_accuracy(architecture)
        fitness = accuracy - 0.5*violations
        return max(0.0, fitness)
    def _estimate_accuracy(self, architecture: Dict) -> float: 

        score = 0.0
        for layer in architecture['layers']: 
            op = layer['op']
            if 'sep_conv' in op: 
                score += 0.15
            elif 'conv' in op: 
                score += 0.12
            elif 'pool' in op: 
                score += 0.05
            elif 'skip' in op: 
                score += 0.08
        return min(1.0, score / len(architecture['layers']))
    def mutate(self, architecture: Dict) -> Dict: 
        """Mutate an architecture"""
        mutated = json.loads(json.dumps(architecture)) # deep copy

        for layer in mutated['layers']: 
            if np.random.random() < self.mutation_rate:
                layer['op']=np.random.choice(self.search_space.operations)
        return mutated
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]: 
        child1 = json.loads(json.dumps(parent1))
        child2 = json.loads(json.dumps(parent2))

        crossover_point = np.random.randint(1, len(parent1['layers']))

        #swap layers
        for i in range(crossover_point, len(parent1['layers'])):
            child1['layers'][i]['op'] = parent2['layers'][i]['op']
            child2['layers'][i]['op'] = parent1['layers'][i]['op']
        return child1, child2
    def evolve(self, generations: int = 100) -> Dict:
        """Run evolutionary search"""
        self.initialize_population()
        
        for generation in range(generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for arch in self.population:
                fitness = self.evaluate_fitness(arch)
                fitness_scores.append(fitness)
            
            # Selection: tournament selection
            new_population = []
            for _ in range(self.population_size // 2):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
                winner_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
                parent1 = self.population[winner_idx]
                
                tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
                winner_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
                parent2 = self.population[winner_idx]
                
                # Crossover and mutation
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population
            
            # Track best fitness
            best_fitness = max(fitness_scores)
            self.fitness_history.append(best_fitness)
            
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Return best architecture
        final_fitness = [self.evaluate_fitness(arch) for arch in self.population]
        best_idx = np.argmax(final_fitness)
        
        return {
            'best_architecture': self.population[best_idx],
            'best_fitness': final_fitness[best_idx],
            'fitness_history': self.fitness_history
        }

