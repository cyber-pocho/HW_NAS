import yaml
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
from .src.search.search_space import SearchSpace
from .src.hardware.predictor import HardwarePredictor, HardwareConstraints
from .src.core.ops.supernet import SuperNet
from .src.search.darts_searcher import DARTSSearcher
from .src.search.evolutionary_searcher import EvolutionarySearcher 
from .src.training.trainer import NASTrainer
from .src.search.progressive_searcher import ProgressiveSearcher
from .src.optimization.pareto import ParetoOptimizer
from .experiments.arch_builder import ArchitectureBuilder
from .experiments.benchmark import HardwareBenchmark, BenchmarkResults
from .src.evaluation.evaluator import NASEvaluator
from .deployment.deployment import DeploymentManager
from .config.search_config import create_default_config, prepare_datasets, load_config, save_config
import time

class NASOrchestrator: 
    """Main orchestrator for NAS experiments"""
    def __init__(self, config: dict):
        self.config = config
        self.device = config['experiment']['device']
        self.output_dir = Path(config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        #random seeds
        torch.manual_seed(config['experiment']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['experiment']['seed'])
        #init components
        self.search_space = SearchSpace(config['search_space'])
        self.hardware_predictor = HardwarePredictor(config['hardware_constraints']['platform'])
        self.constraints = HardwareConstraints(**config['hardware_constraints'])
        
        self.search_results = {}
    
    def run_darts_search(self, train_loader, val_loader):
        """Run DARTS search algorithm"""
        if not self.config['search_algorithms']['darts']['enabled']:
            return None
        print("Running DARTS search...")
        darts_config = self.config['search_algorithms']['darts']
        
        supernet = SuperNet(
            self.search_space, 
            input_channels=self.config['dataset']['input_shape'][0],
            num_classes=self.config['search_space']['num_classes']
        )

        #create searcher
        searcher = DARTSSearcher(
            supernet, self.hardware_predictor, self.constraints, 
            lambda_hw=darts_config['lambda_hw']
        )

        trainer = NASTrainer(searcher, device=self.device)

        #Run and search

        start_time = time.time()
        darts_results = trainer.train(train_loader, val_loader, epochs=darts_config['epochs'])
        search_time = time.time() - start_time

        darts_results['search_time']=search_time
        darts_results['algorithm']='darts'

        print(f"DARTS search completed in {search_time:.2f} seconds")
        return darts_results
    def run_evolutionary_search(self):
        """Run evolutionary search algorithm"""
        if not self.config['search_algorithms']['evolutionary']:
            return None
        
        evo_config = self.config['search_algorithms']['evolutionary']

        searcher = EvolutionarySearcher(
            self.search_space, self.hardware_predictor, self.constraints, 
            population_size=evo_config['population_size'],
            mutation_rate=evo_config['mutation_rate']

        )

        start_time = time.time()
        evo_results = searcher.evolve(generations=evo_config['generations'])
        search_time = time.time() - start_time

        evo_results['search_time'] = search_time
        evo_results['algorithm'] = 'evolutionary'

        print(f" Evolutionary search completed in {search_time:.2f} seconds")
        return evo_results
    
    def run_progressive_search(self):
        """Run progressive search with pruning"""
        if not self.config['search_algorithsm']['progressive']['enabled']:
            return None
        
        prog_config = ProgressiveSearcher(
            self.search_space, self.hardware_predictor,
            pruning_stages=prog_config['stages'],
            pruning_ratio=prog_config['pruning_ratio']
        )
    def run_complete_search(self, train_loader, val_loader):
        """Run all enabled search algorithms"""
        print("=== Starting Complete NAS Pipeline ===")
        
        # Run DARTS
        darts_results = self.run_darts_search(train_loader, val_loader)
        if darts_results:
            self.search_results['darts'] = darts_results
        
        # Run Evolutionary
        evo_results = self.run_evolutionary_search()
        if evo_results:
            self.search_results['evolutionary'] = evo_results
        
        # Run Progressive
        prog_results = self.run_progressive_search()
        if prog_results:
            self.search_results['progressive'] = prog_results
        
        # Pareto analysis
        self.run_pareto_analysis()
        
        # Save results
        self.save_search_results()
        
        return self.search_results
    
    def run_pareto_analysis(self):
        """Analyze Pareto front from all search results"""
        print("\n=== Running Pareto Analysis ===")
        
        pareto_optimizer = ParetoOptimizer(self.hardware_predictor)
        all_architectures = []
        all_accuracies = []
        
        # Collect architectures from all search methods
        for method, results in self.search_results.items():
            if method == 'darts' and 'best_architecture' in results:
                all_architectures.append(results['best_architecture'])
                all_accuracies.append(0.85)  # Simulated accuracy
                
            elif method == 'evolutionary' and 'best_architecture' in results:
                all_architectures.append(results['best_architecture'])
                all_accuracies.append(0.83)  # Simulated accuracy
                
            elif method == 'progressive' and 'final_architecture' in results:
                all_architectures.append(results['final_architecture'])
                all_accuracies.append(0.81)  # Simulated accuracy
        
        if all_architectures:
            pareto_count = pareto_optimizer.update_pareto_front(all_architectures, all_accuracies)
            diverse_solutions = pareto_optimizer.get_diverse_solutions(min(5, pareto_count))
            
            self.search_results['pareto_analysis'] = {
                'pareto_front_size': pareto_count,
                'diverse_solutions': diverse_solutions,
                'total_evaluated': len(all_architectures)
            }
            
            print(f"Pareto front contains {pareto_count} non-dominated solutions")
    
    def save_search_results(self):
        """Save all search results to files"""
        results_file = self.output_dir / 'search_results.json'
        
        # Convert any non-serializable objects
        serializable_results = {}
        for method, results in self.search_results.items():
            serializable_results[method] = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Search results saved to {results_file}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

class DeploymentOrchestrator: 
    """Orchestrate evaluation and deployment of found architectures"""

    def __init__(self, config: dict, search_results: dict):
        self.config = config
        self.search_results = search_results
        self.output_dir = Path(config['experiment']['output_dir'])

        #initializing components
        self.builder = ArchitectureBuilder(config['search_space'])
        self.benchmark = HardwareBenchmark(
            device=config['experiment']['device'],
            benchmark_runs=config['evaluation']['benchmark_runs'],
            warmup_runs=config['evaluation']['warmup_runs']
        )
        self.evaluator=NASEvaluator(self.builder, self.benchmark)
        self.deployment_mgr=DeploymentManager(str(self.output_dir / 'deployed_models'))

    def evaluate_discovered_architectures(self, test_loader):
        architectures_to_evaluate = []
        architectures_sources = []

        for method, results in self.search_results.items():
            if method == 'darts' and 'best_architecture' in results:
                architectures_to_evaluate.append(results['best_architecture'])
                architectures_sources.append(f'darts_best')
            elif method == 'evolutionary' and 'best_architecture' in results:
                architectures_to_evaluate.append(results['best_architecture'])
                architectures_sources.append(f'evolutionary_best')
            elif method == 'progressive' and 'best_architecture' in results:
                architectures_to_evaluate.append(results['best_architecture'])
                architectures_sources.append(f'progressive_best')
            elif method == 'pareto_analysis' and 'diverse_solutions' in results:
                for i, solution in enumerate(results['diverse_solutions']):
                    if 'architecture' in solution:
                        architectures_to_evaluate.append(solution['architecture'])
                        architectures_sources.append(f'pareto_solution_{i}')
        
        if not architectures_to_evaluate:
            print("No architectures to evaluate")
            return {}
        print(f"evaluating {len(architectures_to_evaluate)} architectures")

        evaluation_results =[]
        for i, (arch, source) in enumerate(zip(architectures_to_evaluate, architectures_sources)):
            print(f" Evaluating {source} ({i+1}/{len(architectures_to_evaluate)})...")

            try: 
                result = self.evaluator.evaluate_architecture(
                    arch, 
                    dataset_loader=test_loader,
                    input_shape=(1, *self.config['dataset']['input_shape'])
                )
                result['source'] = source
                evaluation_results.append(result)
            except Exception as e:
                print(f" Error evaluating {source}: {e}")
                continue
        evaluation_results.sort(key=lambda x: x['efficiency_score'], reverse=True)

        return {
            'results': evaluation_results,
            'summary': self._generate_evaluation_summary(evaluation_results)
        }
    def deploy_top_architectures(self, evaluation_results: dict, top_k: int = 3):
        """Deploy top performing architectures"""
        print(f"\n=== Deploying Top {top_k} Architectures ===")
        
        if not evaluation_results or 'results' not in evaluation_results:
            print("No evaluation results available for deployment")
            return {}
        
        top_results = evaluation_results['results'][:top_k]
        deployed_models = []
        
        for i, result in enumerate(top_results):
            print(f"  Deploying architecture {i+1}/{len(top_results)} ({result['source']})...")
            
            try:
                deployment_files = self.deployment_mgr.prepare_for_deployment(
                    result['model'],
                    result['architecture'],
                    input_shape=(1, *self.config['dataset']['input_shape']),
                    optimize=self.config['deployment']['optimize']
                )
                
                deployed_models.append({
                    'rank': i + 1,
                    'source': result['source'],
                    'architecture': result['architecture'],
                    'benchmark': result['benchmark'],
                    'efficiency_score': result['efficiency_score'],
                    'deployment_files': deployment_files
                })
                
            except Exception as e:
                print(f"    Error deploying {result['source']}: {e}")
                continue
        
        return {
            'deployed_models': deployed_models,
            'deployment_summary': {
                'total_deployed': len(deployed_models),
                'deployment_formats': self.config['deployment']['export_formats'],
                'output_directory': str(self.deployment_mgr.output_dir)
            }
        }
    
    def _generate_evaluation_summary(self, results: list) -> dict:
        """Generate evaluation summary statistics"""
        if not results:
            return {}
        
        latencies = [r['benchmark'].inference_time_ms for r in results]
        memories = [r['benchmark'].memory_usage_mb for r in results]
        params = [r['benchmark'].params for r in results]
        flops = [r['benchmark'].flops for r in results]
        efficiencies = [r['efficiency_score'] for r in results]
        
        return {
            'num_architectures': len(results),
            'latency_stats': {
                'min': min(latencies), 'max': max(latencies), 'mean': np.mean(latencies)
            },
            'memory_stats': {
                'min': min(memories), 'max': max(memories), 'mean': np.mean(memories)
            },
            'params_stats': {
                'min': min(params), 'max': max(params), 'mean': np.mean(params)
            },
            'flops_stats': {
                'min': min(flops), 'max': max(flops), 'mean': np.mean(flops)
            },
            'efficiency_stats': {
                'min': min(efficiencies), 'max': max(efficiencies), 'mean': np.mean(efficiencies)
            },
            'best_architecture': {
                'source': results[0]['source'],
                'efficiency_score': results[0]['efficiency_score'],
                'latency_ms': results[0]['benchmark'].inference_time_ms,
                'memory_mb': results[0]['benchmark'].memory_usage_mb
            }
        }
    
#MAIN EXECUTION FUNCTION

def run_search_mode(config: dict):
    """Run NAS search mode"""
    print("=== NAS SEARCH MODE ===")
    
    # Prepare datasets
    train_loader, val_loader, test_loader = prepare_datasets(config)
    print(f"Dataset prepared: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Run search
    orchestrator = NASOrchestrator(config)
    search_results = orchestrator.run_complete_search(train_loader, val_loader)
    
    # Quick evaluation on test set
    deployment_orch = DeploymentOrchestrator(config, search_results)
    evaluation_results = deployment_orch.evaluate_discovered_architectures(test_loader)
    
    print("\n=== Search Complete ===")
    print("Results saved to:", orchestrator.output_dir)
    
    return {
        'search_results': search_results,
        'evaluation_results': evaluation_results
    }

def run_evaluate_mode(config: dict, results_file: str):
    """Run evaluation mode on existing results"""
    print("=== NAS EVALUATION MODE ===")
    
    # Load existing results
    with open(results_file, 'r') as f:
        search_results = json.load(f)
    
    # Prepare test dataset
    _, _, test_loader = prepare_datasets(config)
    
    # Run evaluation
    deployment_orch = DeploymentOrchestrator(config, search_results)
    evaluation_results = deployment_orch.evaluate_discovered_architectures(test_loader)
    
    print("Evaluation complete")
    return evaluation_results

def run_deploy_mode(config: dict, results_file: str):
    """Run deployment mode"""
    print("=== NAS DEPLOYMENT MODE ===")
    
    # Load existing results
    with open(results_file, 'r') as f:
        search_results = json.load(f)
    
    # Prepare test dataset for final evaluation
    _, _, test_loader = prepare_datasets(config)
    
    # Run complete deployment pipeline
    deployment_orch = DeploymentOrchestrator(config, search_results)
    evaluation_results = deployment_orch.evaluate_discovered_architectures(test_loader)
    deployment_results = deployment_orch.deploy_top_architectures(evaluation_results)
    
    print("Deployment complete")
    print("Models deployed to:", deployment_results['deployment_summary']['output_directory'])
    
    return deployment_results

def run_quick_demo():
    """Run quick demonstration of the complete pipeline"""
    print("=== QUICK DEMO MODE ===")
    
    # Create minimal config for fast demo
    config = create_default_config()
    config['dataset']['subset_size'] = 500  # Very small subset
    config['search_algorithms']['darts']['epochs'] = 5
    config['search_algorithms']['evolutionary']['generations'] = 20
    config['search_algorithms']['progressive']['stages'] = 2
    config['evaluation']['benchmark_runs'] = 10
    
    print("Running quick demo with minimal dataset and short search...")
    
    # Run search
    results = run_search_mode(config)
    
    # Deploy best models
    deployment_orch = DeploymentOrchestrator(config, results['search_results'])
    deployment_results = deployment_orch.deploy_top_architectures(
        results['evaluation_results'], top_k=2
    )
    
    print("\n=== Demo Complete ===")
    print("This demonstrates the complete NAS pipeline from search to deployment!")
    print("For full experiments, remove --quick-demo and use larger datasets.")
    
    return {
        'search_results': results['search_results'],
        'evaluation_results': results['evaluation_results'],
        'deployment_results': deployment_results
    }


def main():
    parser = argparse.ArgumentParser(description='Hardware-Aware Neural Architecture Search')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', choices=['search', 'evaluate', 'deploy'], 
                       default='search', help='Execution mode')
    parser.add_argument('--results', type=str, help='Results file for evaluate/deploy modes')
    parser.add_argument('--quick-demo', action='store_true', 
                       help='Run quick demonstration')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.quick_demo:
        return run_quick_demo()
    
    # Load configuration
    config = load_config(args.config)
    config['experiment']['output_dir'] = args.output_dir
    
    # Save config to output directory
    save_config(config, args.output_dir)
    
    # Run specified mode
    if args.mode == 'search':
        return run_search_mode(config)
    elif args.mode == 'evaluate':
        if not args.results:
            raise ValueError("--results file required for evaluate mode")
        return run_evaluate_mode(config, args.results)
    elif args.mode == 'deploy':
        if not args.results:
            raise ValueError("--results file required for deploy mode")
        return run_deploy_mode(config, args.results)

if __name__ == "__main__":
    # Example of programmatic usage
    print("Hardware-Aware Neural Architecture Search System")
    print("=" * 50)
    
    # You can run directly or use command line
    try:
        results = main()
        print("\n" + "=" * 50)
        print("Execution completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("\nFor help, run: python main_nas.py --help")
        
    # Or run quick demo directly
    # results = run_quick_demo()