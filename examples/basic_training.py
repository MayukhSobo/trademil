"""
Basic training example using Treadmill framework.
"""

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import click

# Import Treadmill components
from treadmill import OptimizerConfig, Trainer, TrainingConfig
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
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        # Flatten the input (batch_size, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)


def prepare_data():
    """Prepare MNIST data loaders."""

    # Simple transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    # Load MNIST datasets
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, val_loader


def run_training(epochs, checkpoint_dir, project_name, timezone, keep_all_checkpoints):
    """Main training function."""

    # Prepare data
    print("Preparing MNIST dataset...")
    train_loader, val_loader = prepare_data()

    # Create simple model
    model = SimpleDNN(input_size=784, hidden_size=128, num_classes=10)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define metrics
    metric_fns = {"accuracy": StandardMetrics.accuracy}

    # Configure training with proper checkpointing
    config = TrainingConfig(
        epochs=epochs,
        device="auto",
        # Checkpointing configuration
        checkpoint_dir=checkpoint_dir,
        keep_all_checkpoints=keep_all_checkpoints,
        project_name=project_name,
        timezone=timezone,
        # Simple optimizer
        optimizer=OptimizerConfig(optimizer_class="Adam", lr=1e-3),
        # Display settings
        print_every=100,  # Print every 100 batches
        progress_bar=True,
        # Training settings
        validate_every=1,
        early_stopping_patience=5,
        # Resume settings
        resume_training=False,  # This is initial training
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
    )

    # Start training
    print("\nStarting basic training...")
    training_history = trainer.train()

@click.command()
@click.option(
    "--epochs", 
    default=10, 
    help="Number of training epochs",
    type=int
)
@click.option(
    "--checkpoint-dir", 
    default="./checkpoints", 
    help="Directory to save checkpoints"
)
@click.option(
    "--project-name", 
    default="mnist", 
    help="Project name for experiment directory"
)
@click.option(
    "--timezone", 
    default="IST", 
    help="Timezone for directory naming"
)
@click.option(
    "--keep-all-checkpoints/--keep-best-only",
    default=True,
    help="Keep all checkpoints or only the best one"
)
def main(epochs, checkpoint_dir, project_name, timezone, keep_all_checkpoints):
    """Basic training example using Treadmill framework."""
    run_training(epochs, checkpoint_dir, project_name, timezone, keep_all_checkpoints)


if __name__ == "__main__":
    main()
