import time
import json
import yaml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from  torch.utils.data import DataLoader, Subset
from pathlib import Path

def create_default_config():
    """Create comprehensive default configuration"""
    return {
        'experiment': {
            'name': 'hw_nas_experiment',
            'output_dir': './results',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'dataset': {
            'name': 'cifar10',
            'batch_size': 256,
            'num_workers': 4,
            'subset_size': 5000,  # For quick experiments
            'input_shape': [3, 32, 32]
        },
        'search_space': {
            'operations': [
                'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 
                'avg_pool_3x3', 'max_pool_3x3', 'skip_connect'
            ],
            'num_layers': 8,
            'channels': [16, 32, 64],
            'reduce_layers': [2, 5],
            'num_classes': 10
        },
        'hardware_constraints': {
            'platform': 'mobile',
            'max_latency': 50.0,  # ms
            'max_memory': 25.0,   # MB
            'max_flops': 100.0,   # MFLOPs
            'max_energy': 500.0   # mJ
        },
        'search_algorithms': {
            'darts': {
                'enabled': True,
                'epochs': 50,
                'lambda_hw': 0.1,
                'model_lr': 0.025,
                'arch_lr': 3e-4
            },
            'evolutionary': {
                'enabled': True,
                'generations': 100,
                'population_size': 50,
                'mutation_rate': 0.1
            },
            'progressive': {
                'enabled': True,
                'stages': 3,
                'pruning_ratio': 0.3
            }
        },
        'evaluation': {
            'benchmark_runs': 100,
            'warmup_runs': 10,
            'accuracy_samples': 1000
        },
        'deployment': {
            'export_formats': ['pytorch', 'onnx', 'torchscript'],
            'optimize': True,
            'quantize': True,
            'prune': True
        }
    }

def load_config(config_path: str = None):
    """Load configuration from file or create default"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        config = create_default_config()
        print("Using default configuration")
    
    return config

def save_config(config: dict, output_dir: str):
    """Save configuration to output directory"""
    output_path = Path(output_dir) / 'config.yaml'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def prepare_datasets(config: dict):
    """Prepare train and validation datasets"""
    dataset_config = config['dataset']
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load dataset
    if dataset_config['name'].lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_config['name']}")
    
    # Create subset for faster experimentation
    if dataset_config['subset_size'] > 0:
        subset_indices = torch.randperm(len(train_dataset))[:dataset_config['subset_size']]
        train_dataset = Subset(train_dataset, subset_indices)
        
        subset_indices = torch.randperm(len(test_dataset))[:dataset_config['subset_size']//5]
        test_dataset = Subset(test_dataset, subset_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=dataset_config['batch_size'],
        shuffle=True, 
        num_workers=dataset_config['num_workers'],
        pin_memory=True
    )
    
    # Split train into train/validation for DARTS
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_search_loader = DataLoader(
        train_subset,
        batch_size=dataset_config['batch_size'],
        shuffle=True,
        num_workers=dataset_config['num_workers']
    )
    
    val_search_loader = DataLoader(
        val_subset,
        batch_size=dataset_config['batch_size'],
        shuffle=False,
        num_workers=dataset_config['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=False,
        num_workers=dataset_config['num_workers']
    )
    
    return train_search_loader, val_search_loader, test_loader

