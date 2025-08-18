from typing import Dict
from core.ops.search_space import SearchSpace
from hardware.predictor import HardwareConstraints, HardwarePredictor
from core.ops.supernet import SuperNet
from search.darts_searcher import DARTSSearcher
from training.trainer import NASTrainer

def create_default_config() -> Dict: 
    return {
        'search_space': {
            'operations': ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'avg_pool_3x3', 'skip_connect'],
            'num_layers': 8,
            'channels': [16, 32, 64],
            'reduce_layers': [2, 5],
            'num_classes': 10
        },
        'hardware': {
            'platform': 'mobile',
            'max_latency': 50.0,
            'max_memory': 25.0,
            'max_flops': 100.0,
            'max_energy': 500.0
        },
        'training': {
            'epochs': 50,
            'lambda_hw': 0.1
        }
    }
def main_example():
    """Example of how to run the NAS system"""
    config = create_default_config()
    #initialize components
    search_space = SearchSpace(config['search_space'])
    hardware_predictor = HardwarePredictor(config['hardware']['platform'])
    constraints = HardwareConstraints(**config['hardware'])
    #create supernet
    supernet = SuperNet(search_space, input_channels=3, num_classes=10)
    #initialize searcher
    searcher = DARTSSearcher(supernet, hardware_predictor, constraints, config['training']['lambda_hw'])
    #create trainer
    trainer = NASTrainer(searcher)

    print("NAS system initialized")
    print(f"search space: {len(search_space.operations)} operations, {search_space.num_layers} layers")
    print(f"hardware constraints: {constraints}")

    return trainer, searcher, supernet

if __name__ == "__main__": 
    trainer, searcher, supernet = main_example()

    sample_arch = supernet.sample_subnet()
    print(f"Sample architecture: {sample_arch}")
    #sample archs
    predictor = HardwarePredictor('mobile')
    costs = predictor.predict_costs(sample_arch, (1,3,32,32))
    print(f"hardware costs: {costs}")

# TO BE TESTED!
