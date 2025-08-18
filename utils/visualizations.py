import matplotlib.pyplot as plt
from typing import List, Dict, Tuples

class NASVisualizer:
    """Visualization tools for NAS results"""
    
    @staticmethod
    def plot_search_progress(history: List[Dict], save_path: str = None):
        """Plot search progress over time"""
        epochs = [h['epoch'] for h in history]
        total_loss = [h['total_loss'] for h in history]
        accuracy_loss = [h['accuracy_loss'] for h in history]
        hardware_loss = [h['hardware_loss'] for h in history]
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs, total_loss)
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, accuracy_loss)
        plt.title('Accuracy Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 3)
        plt.plot(epochs, hardware_loss)
        plt.title('Hardware Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_pareto_front(pareto_solutions: List[Dict], save_path: str = None):
        """Plot Pareto front"""
        accuracies = [sol['accuracy'] for sol in pareto_solutions]
        efficiencies = [sol['efficiency'] for sol in pareto_solutions]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(efficiencies, accuracies, s=100, alpha=0.7)
        plt.xlabel('Efficiency')
        plt.ylabel('Accuracy')
        plt.title('Pareto Front: Accuracy vs Efficiency')
        plt.grid(True, alpha=0.3)
        
        # Connect points to show front
        sorted_points = sorted(zip(efficiencies, accuracies))
        x_sorted, y_sorted = zip(*sorted_points)
        plt.plot(x_sorted, y_sorted, 'r--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_architecture_distribution(architectures: List[Dict], save_path: str = None):
        """Plot distribution of operations in found architectures"""
        operation_counts = {}
        
        for arch in architectures:
            for layer in arch['layers']:
                op = layer['op']
                operation_counts[op] = operation_counts.get(op, 0) + 1
        
        operations = list(operation_counts.keys())
        counts = list(operation_counts.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(operations, counts)
        plt.title('Operation Distribution in Found Architectures')
        plt.xlabel('Operations')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()