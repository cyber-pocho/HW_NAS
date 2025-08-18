
from typing import Dict

class NASTrainer:
    def __init__(self, searcher, device: str = 'cuda'):
        self.searcher = searcher
        self.device = device
        self.history = []

    def train(self, train_loader, valid_loader, epochs: int = 50) -> Dict:
        self.searcher.supernet.to(self.device)

        for epoch in range(epochs):
            epoch_metrics = {}
            for batch_idx, ((train_data, train_targets), (valid_data, valid_targets)) in enumerate(zip(train_loader, valid_loader)):
                train_data, train_targets = train_data.to(self.device), train_targets.to(self.device)
                valid_data, valid_targets = valid_data.to(self.device), valid_targets.to(self.device)

                metrics = self.searcher.search_step(train_data, train_targets, valid_data, valid_targets)
                if batch_idx == 0:
                    epoch_metrics.update(metrics)

            self.history.append(epoch_metrics)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={epoch_metrics.get('total_loss', 0):.4f}, "
                      f"Acc Loss={epoch_metrics.get('accuracy_loss', 0):.4f}, "
                      f"HW Loss={epoch_metrics.get('hardware_loss', 0):.4f}")

        best_arch = self.searcher.get_best_architecture()
        return {
            'best_architecture': best_arch,
            'training_history': self.history,
            'final_supernet': self.searcher.supernet.state_dict()
        }