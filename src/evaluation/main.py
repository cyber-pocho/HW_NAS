from typing import Dict, Tuple, List
from experiments.arch_builder import ArchitectureBuilder
from experiments.benchmark import BenchmarkResult, HardwareBenchmark
from evaluator import NASEvaluator
from deployment.deployment import DeploymentManager


def complete_evaluation_pipeline():
    print("COMPLETE EVALUATION PIPELINE")
    search_space_config = {
        'operations': ['sep_conv_3x3', 'sep_conv_5x5', 'avg_pool_3x3', 'skip_connect'],
        'num_layers': 6,
        'channels': [16, 32, 64],
        'num_classes': 10
    }
    builder=ArchitectureBuilder(search_space_config)
    benchmark=HardwareBenchmark(device='cpu', benchmark_runs=50)
    evaluator= NASEvaluator(builder, benchmark)
    deployment_manager = DeploymentManager()

    #sample architectures
    sample_architectures = [
        {
            'layers': [
                {'op': 'sep_conv_3x3'}, {'op': 'sep_conv_3x3'}, {'op': 'avg_pool_3x3'},
                {'op': 'sep_conv_5x5'}, {'op': 'skip_connect'}, {'op': 'sep_conv_3x3'}
            ]
        },
        {
            'layers': [
                {'op': 'skip_connect'}, {'op': 'sep_conv_3x3'}, {'op': 'sep_conv_5x5'},
                {'op': 'avg_pool_3x3'}, {'op': 'sep_conv_3x3'}, {'op': 'skip_connect'}
            ]
        },
        {
            'layers': [
                {'op': 'sep_conv_5x5'}, {'op': 'skip_connect'}, {'op': 'sep_conv_3x3'},
                {'op': 'sep_conv_3x3'}, {'op': 'avg_pool_3x3'}, {'op': 'sep_conv_5x5'}
            ]
        }
    ]
    print(f"Evaluating {len(sample_architectures)}architectures")

    comparison_results = evaluator.compare_architectures(sample_architectures)

    print("\n=== Evaluation Results ===")
    for i, result in enumerate(comparison_results['results']):
        benchmark = result['benchmark']
        print(f"\nArchitecture {i+1}:")
        print(f"  Model: {benchmark.model_name}")
        print(f"  Inference Time: {benchmark.inference_time_ms:.2f} ms")
        print(f"  Memory Usage: {benchmark.memory_usage_mb:.2f} MB")
        print(f"  Parameters: {benchmark.params:,}")
        print(f"  FLOPs: {benchmark.flops:,}")
        print(f"  Efficiency Score: {result['efficiency_score']:.4f}")
    
    # 5. Deploy best architecture
    best_result = comparison_results['best_architecture']
    best_model = best_result['model']
    best_arch = best_result['architecture']
    
    print(f"\n=== Deploying Best Architecture ===")
    print(f"Best model: {best_result['benchmark'].model_name}")
    
    deployment_files = deployment_manager.prepare_for_deployment(
        best_model, best_arch, input_shape=(1, 3, 32, 32)
    )
    
    print("Deployment files created:")
    for format_name, file_path in deployment_files.items():
        print(f"  {format_name}: {file_path}")
    
    # 6. Summary statistics
    summary = comparison_results['summary']
    print(f"\n=== Summary Statistics ===")
    print(f"Architectures evaluated: {summary['num_architectures']}")
    print(f"Latency range: {summary['latency_range'][0]:.2f} - {summary['latency_range'][1]:.2f} ms")
    print(f"Memory range: {summary['memory_range'][0]:.2f} - {summary['memory_range'][1]:.2f} MB")
    print(f"Best efficiency score: {summary['best_efficiency']:.4f}")
    print(f"Pareto efficient solutions: {summary['pareto_efficient_count']}")
    
    return {
        'evaluation_results': comparison_results,
        'deployment_files': deployment_files,
        'summary': summary
    }


def integration_NAS_search(nas_results:Dict) -> Dict: 
    architectures_to_evaluate = []
    if 'evolutionary_results' in nas_results:
        architectures_to_evaluate.append(nas_results['progressive_results']['best_architecture'])
    if 'progressive_results' in nas_results:
        architectures_to_evaluate.append(nas_results['progressive_results']['final_architecture'])
    
    if 'pareto_solutions' in nas_results:
        for solution in nas_results['pareto_solutions']:
            architectures_to_evaluate.append(solution['architecture'])
    
    search_space_config = {
        'operations': ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'avg_pool_3x3', 'skip_connect'],
        'num_layers': 8,
        'channels': [16, 32, 64],
        'num_classes': 10
    }

    builder = ArchitectureBuilder(search_space_config)
    benchmark = HardwareBenchmark(device='cpu')
    evaluator = NASEvaluator(builder, benchmark)
    deployment_mgr = DeploymentManager("./nas_deployment_outputs")
    print(f"\n === Evaluating {len(architectures_to_evaluate)}NAS-discovered architectures ====")
    comparison_results = evaluator.compare_architectures(architectures_to_evaluate)

    #deploy top 3 architectures
    deployed_models = []
    for i, result in enumerate(comparison_results['results'][:3]):
        print(f"\nDeploying architecture {i+1}/3...")
        deployment_files = deployment_mgr.prepare_for_deployment(
            result['model'], result['architecture']
        )
        deployed_models.append(
            {
                'rank': i+1,
                'architecture': result['architecture'],
                'benchmark': result['benchmark'],
                'deployment_files': deployment_files

            }
        )
    return {
        'evaluated_architectures': comparison_results,
        'deployed_models': deployed_models,
        'deployment_summary': {
            'total_evaluated': len(architectures_to_evaluate),
            'total_deployed': len(deployed_models),
            'best_efficiency': comparison_results['results'][0]['efficiency_score'],
            'deployment_formats': ['pytorch', 'onnx', 'torchscript', 'quantized', 'pruned']
        }
    }
if __name__ == "__main__":
    results = complete_evaluation_pipeline()
    print("\n=== Pipeline Complete ===")
    print("Check './deployment_outputs' directory for exported models")