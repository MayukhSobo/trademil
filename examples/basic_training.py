"""
Basic training example using Trademil framework.

This example shows how to train a simple CNN on CIFAR-10 using the Trademil framework.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import Trademil components
from trademil import (
    Trainer, TrainingConfig, OptimizerConfig, SchedulerConfig,
    EarlyStopping, ModelCheckpoint
)
from trademil.metrics import StandardMetrics


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def prepare_data():
    """Prepare CIFAR-10 data loaders."""
    
    # Data transformations
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
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def main():
    """Main training function."""
    
    # Prepare data
    print("Preparing CIFAR-10 dataset...")
    train_loader, test_loader = prepare_data()
    
    # Create model
    model = SimpleCNN(num_classes=10)
    
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Define custom metrics
    metric_fns = {
        "accuracy": StandardMetrics.accuracy,
        "top5_acc": lambda p, t: StandardMetrics.top_k_accuracy(p, t, k=5)
    }
    
    # Configure training
    config = TrainingConfig(
        epochs=20,
        device="auto",  # Will use CUDA if available
        
        # Optimizer configuration
        optimizer=OptimizerConfig(
            optimizer_class="Adam",
            lr=1e-3,
            weight_decay=1e-4
        ),
        
        # Learning rate scheduler
        scheduler=SchedulerConfig(
            scheduler_class="StepLR",
            params={"step_size": 10, "gamma": 0.1}
        ),
        
        # Validation and early stopping
        validate_every=1,
        early_stopping_patience=5,
        
        # Display settings
        print_every=50,
        progress_bar=True,
        
        # Training optimizations
        grad_clip_norm=1.0,
        mixed_precision=True  # Use automatic mixed precision if available
    )
    
    # Create custom callbacks (optional)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=True
        ),
        ModelCheckpoint(
            filepath="./checkpoints/best_model_epoch_{epoch:03d}_acc_{val_accuracy:.4f}",
            monitor="val_accuracy",
            mode="max",  # Higher accuracy is better
            save_best_only=True,
            save_format="pt",  # Can also use "onnx", "safetensors", "pkl"
            verbose=True
        )
    ]
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        callbacks=callbacks
    )
    
    # Start training
    print("\nStarting training...")
    training_history = trainer.train()
    
    # Print final results
    print("\n" + "="*60)
    print("🎉 Training completed successfully!")
    print(f"📊 Total epochs: {training_history['total_epochs']}")
    
    if training_history['best_metrics']:
        print("\n🏆 Best validation metrics achieved:")
        for metric, value in training_history['best_metrics'].items():
            print(f"  • {metric}: {value:.4f}")
    
    print("\n💡 Try different save formats:")
    print("   - ONNX: save_format='onnx' for deployment")  
    print("   - SafeTensors: save_format='safetensors' for security")
    print("   - Pickle: save_format='pkl' for full checkpoints")


if __name__ == "__main__":
    main() 