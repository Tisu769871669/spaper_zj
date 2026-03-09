"""
LSTM-based Intrusion Detection System Baseline.

This is a SUPERVISED LEARNING baseline using LSTM for sequence modeling.
It represents traditional deep learning approaches to IDS.

Key Difference from Bi-ARL:
    - LSTM-IDS: Supervised learning on labeled data
    - Bi-ARL: Reinforcement learning with adversarial training
    
Expected Result:
    LSTM should achieve competitive performance on clean data but may be
    vulnerable to adversarial attacks without explicit robustness training.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Tuple

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import Config
from src.utils.data_loader import build_data_loader


class LSTMIDS(nn.Module):
    """
    LSTM-based Intrusion Detection System.
    
    Architecture:
        Input (41 features) -> LSTM (2 layers, 64 hidden) -> FC -> Output (2 classes)
    """
    
    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2
    ):
        """
        Initialize LSTM-IDS.
        
        Args:
            input_size: Number of input features (41 for NSL-KDD)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (2: normal/attack)
        """
        super(LSTMIDS, self).__init__()
        
        input_size = input_size or Config.STATE_DIM
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
               For NSL-KDD: (batch, 1, 41) - treating each sample as seq_len=1
               
        Returns:
            Logits of shape (batch, num_classes)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected
        output = self.fc(last_hidden)  # (batch, num_classes)
        
        return output


class LSTMIDSTrainer:
    """Trainer for LSTM-IDS baseline."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize LSTM-IDS Trainer.
        
        Args:
            seed: Random seed
        """
        Config.set_seed(seed)
        self.seed = seed
        
        # Load data
        self.loader = build_data_loader(Config.DATASET_NAME)
        print(f"Loading dataset: {Config.DATASET_NAME}")
        self.X_train, self.y_train = self.loader.load_data(mode='train')
        self.X_test, self.y_test = self.loader.load_data(mode='test')
        
        # Initialize model
        self.model = LSTMIDS().to(Config.DEVICE)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        print(f"\n{'='*60}")
        print(f"LSTM-IDS Training")
        print(f"{'='*60}")
        print(f"Device: {Config.DEVICE}")
        print(f"Seed: {seed}")
        print(f"Train samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"{'='*60}\n")
    
    def train_epoch(self, batch_size: int = 128) -> float:
        """
        Train for one epoch.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(self.X_train))
        
        for i in range(0, len(self.X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = self.X_train[batch_indices]
            batch_y = self.y_train[batch_indices]
            
            # Convert to tensors
            # Add seq_len dimension: (batch, 41) -> (batch, 1, 41)
            x = torch.FloatTensor(batch_x).unsqueeze(1).to(Config.DEVICE)
            y = torch.LongTensor(batch_y).to(Config.DEVICE)
            
            # Forward
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate on test set.
        
        Returns:
            (accuracy, loss)
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            # Process in batches
            batch_size = 512
            for i in range(0, len(self.X_test), batch_size):
                batch_x = self.X_test[i:i+batch_size]
                batch_y = self.y_test[i:i+batch_size]
                
                x = torch.FloatTensor(batch_x).unsqueeze(1).to(Config.DEVICE)
                y = torch.LongTensor(batch_y).to(Config.DEVICE)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                total_loss += loss.item()
        
        accuracy = correct / total
        avg_loss = total_loss / (len(self.X_test) // batch_size + 1)
        
        return accuracy, avg_loss
    
    def train(self, num_epochs: int = 30) -> None:
        """
        Train LSTM-IDS.
        
        Args:
            num_epochs: Number of training epochs
        """
        print("Starting LSTM-IDS Training...\n")
        
        best_accuracy = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            if epoch % 5 == 0:
                test_acc, test_loss = self.evaluate()
                
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Test Loss: {test_loss:.4f}")
                print(f"  Test Accuracy: {test_acc:.4f}")
                
                # 保存最佳模型(使用Config路径)
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    save_path = Config.get_model_path("LSTM", self.seed, "model")
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  New best model saved (Acc: {best_accuracy:.4f})")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Test Accuracy: {best_accuracy:.4f}")
        print(f"{'='*60}\n")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM-IDS Baseline')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--dataset', type=str, default='nsl-kdd', help='Dataset: nsl-kdd / unsw-nb15')
    
    args = parser.parse_args()
    Config.configure_dataset(args.dataset)
    
    trainer = LSTMIDSTrainer(seed=args.seed)
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
