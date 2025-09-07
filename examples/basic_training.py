"""
Basic training example using Treadmill framework.

This example shows how to train a simple neural network on MNIST using Treadmill.
Features demonstrated:
- Basic model training with validation
- Automatic checkpointing
- Progress tracking and metrics logging
- Early stopping

This example is designed to run quickly for testing and validation purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import Treadmill components
from treadmill import Trainer, TrainingConfig, OptimizerConfig
from treadmill.metrics import StandardMetrics


class SimpleDNN(nn.Module):
    """Simple Deep Neural Network for MNIST classification."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleDNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        # Flatten the input (batch_size, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)


def prepare_data():
    """Prepare MNIST data loaders."""
    
    # Simple transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST datasets
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def main():
    """Main training function."""
    
    # Prepare data
    print("Preparing MNIST dataset...")
    train_loader, test_loader = prepare_data()
    
    # Create simple model
    model = SimpleDNN(input_size=784, hidden_size=128, num_classes=10)
    
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Define metrics
    metric_fns = {
        "accuracy": StandardMetrics.accuracy
    }
    
    # Configure training (simplified)
    config = TrainingConfig(
        epochs=5,  # Just 5 epochs for quick testing
        device="auto",
        
        # Experiment directory settings (NEW!)
        project_name="mnist_dnn",  # Optional: specify project name
        use_experiment_dir=True,  # Create unique experiment directories
        timezone="IST",  # Timezone for directory naming (IST, EST, PST, UTC, etc.)
        
        # Simple optimizer
        optimizer=OptimizerConfig(
            optimizer_class="Adam",
            lr=1e-3
        ),
        
        # Display settings
        print_every=100,  # Print every 100 batches
        progress_bar=True,
        
        # Basic settings
        validate_every=1,
        early_stopping_patience=3,
        save_best_model=True
    )
    
    # Create trainer (no custom callbacks for simplicity)
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        loss_fn=loss_fn,
        metric_fns=metric_fns
    )
    
    # Start training
    print("\nStarting basic training...")
    training_history = trainer.train()
    
    # Print results
    print("\n" + "="*50)
    print("✅ Basic training completed!")
    print(f"📊 Total epochs: {training_history['total_epochs']}")
    
    if training_history['best_metrics']:
        print("\n🏆 Best validation metrics:")
        for metric, value in training_history['best_metrics'].items():
            print(f"  • {metric}: {value:.4f}")


if __name__ == "__main__":
    main() 