from src.core.ops.search_space import SearchSpace
from src.hardware.predictor import HardwareConstraints, HardwarePredictor
from src.search.search_test import create_default_config
from src.search.evolutionary_searcher import EvolutionarySearcher
from src.search.progressive_searcher import ProgressiveSearcher
from src.optimization.pareto import ParetoOptimizer

def run_complete_nas_pipeline(): 
    config = create_default_config()
    search_space = SearchSpace(config['search_space'])
    hardware_predictor = HardwarePredictor('mobile')
    constraints = HardwareConstraints(**config['hardware'])
    print("components initiliazed")

    print("Running evolutionary search...")
    evo_searcher = EvolutionarySearcher(search_space, hardware_predictor, constraints)
    evo_results = evo_searcher.evolve(generations=50)
    print(f" Best fitness: {evo_results['best_fitness']:.4f}")

    print("Runnung progressiver search")
    prog_searcher = ProgressiveSearcher(search_space, hardware_predictor)
    prog_results = prog_searcher.progressive_search('evolutionary', population_size = 30)

    print("Running Pareto optimization")
    pareto_optimizer = ParetoOptimizer(hardware_predictor)

    #simulate multiple archs with different accuracy/efficency trade-offs
    test_architectures = [evo_results['best_architecture'], prog_results['final_architecture']]
    test_accuracies = [0.85, 0.82]

    pareto_optimizer.update_pareto_front(test_architectures, test_accuracies)
    diverse_solutions = pareto_optimizer.get_diverse_solutions(3)

    print(f" Pareto front size: {pareto_optimizer.pareto_front}")

    #Summary
    print("Results Summary:")
    print("   Evolutionary Search Best:")
    for i, layer in enumerate(evo_results['best_architecture']['layers'][:3]):
        print(f"     Layer {i}: {layer['op']}")
    
    print("   Progressive Search Best:")
    for i, layer in enumerate(prog_results['final_architecture']['layers'][:3]):
        print(f"     Layer {i}: {layer['op']}")
    
    return {
        'evolutionary_results': evo_results,
        'progressive_results': prog_results,
        'pareto_solutions': diverse_solutions
    }

if __name__ == "__main__":
    # Run the complete pipeline
    results = run_complete_nas_pipeline()
