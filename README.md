# 🏃‍♀️‍➡️ Treadmill 🏃‍♀️‍➡️

**A Clean and Modular PyTorch Training Framework**

Treadmill is a lightweight, modular training framework specifically designed for PyTorch. It provides clean, easy-to-understand training loops with beautiful output formatting while maintaining the power and flexibility of vanilla PyTorch.

## ✨ Features

- **🎯 Pure PyTorch**: Built specifically for PyTorch, no forced abstractions
- **🔧 Modular Design**: Easy to customize and extend with callback system  
- **📊 Beautiful Output**: Rich formatting with progress bars and metrics tables
- **⚡ Performance Optimizations**: Mixed precision, gradient accumulation, gradient clipping
- **🎛️ Flexible Configuration**: Dataclass-based configuration system
- **📈 Comprehensive Metrics**: Built-in metrics with support for custom metrics
- **💾 Smart Checkpointing**: Automatic model saving with customizable triggers
- **🛑 Early Stopping**: Configurable early stopping to prevent overfitting
- **🔄 Resumable Training**: Easy checkpoint loading and training resumption

## 🛠️ Installation

### From Source

```bash
git clone https://github.com/MayukhSobo/treadmill.git
cd treadmill
pip install -e .
```

### Install with Examples

```bash
pip install -e ".[examples]"  # Includes torchvision and additional dependencies
```

### Install Full Version

```bash
pip install -e ".[full]"  # Includes all optional dependencies
```

## 🚀 Quick Start

Here's a minimal example to get you started:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from treadmill import Trainer, TrainingConfig, OptimizerConfig
from treadmill.metrics import StandardMetrics

# Define your model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# Prepare your data (DataLoaders)
train_loader = DataLoader(...)  # Your training data
val_loader = DataLoader(...)    # Your validation data

# Configure training
config = TrainingConfig(
    epochs=10,
    optimizer=OptimizerConfig(optimizer_class="Adam", lr=1e-3),
    device="auto"  # Automatically uses GPU if available
)

# Create and run trainer
trainer = Trainer(
    model=SimpleNet(),
    config=config,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fns={"accuracy": StandardMetrics.accuracy}
)

# Start training
history = trainer.train()
```

## 📖 Core Components

### TrainingConfig

The main configuration class that controls all aspects of training:

```python
config = TrainingConfig(
    # Basic settings
    epochs=20,
    device="auto",  # "auto", "cpu", "cuda", or specific device
    
    # Optimizer configuration
    optimizer=OptimizerConfig(
        optimizer_class="Adam",  # Any PyTorch optimizer
        lr=1e-3,
        weight_decay=1e-4,
        params={"betas": (0.9, 0.999)}  # Additional optimizer parameters
    ),
    
    # Learning rate scheduler
    scheduler=SchedulerConfig(
        scheduler_class="StepLR",
        params={"step_size": 10, "gamma": 0.1}
    ),
    
    # Training optimizations
    mixed_precision=True,
    grad_clip_norm=1.0,
    accumulate_grad_batches=4,
    
    # Validation and early stopping
    validate_every=1,
    early_stopping_patience=5,
    
    # Display and logging
    print_every=50,
    progress_bar=True
)
```

### Callbacks System

Extend functionality with callbacks:

```python
from treadmill.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, verbose=True),
    ModelCheckpoint(
        filepath="./checkpoints/model_epoch_{epoch:03d}_{val_accuracy:.4f}.pt",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    ),
    LearningRateLogger(verbose=True)
]

trainer = Trainer(..., callbacks=callbacks)
```

### Custom Metrics

Define your own metrics or use built-in ones:

```python
from treadmill.metrics import StandardMetrics

# Built-in metrics
metric_fns = {
    "accuracy": StandardMetrics.accuracy,
    "top5_acc": lambda p, t: StandardMetrics.top_k_accuracy(p, t, k=5),
    "f1": StandardMetrics.f1_score
}

# Custom metrics
def custom_metric(predictions, targets):
    # Your custom metric calculation
    return some_value

metric_fns["custom"] = custom_metric
```

## 🔧 Advanced Usage

### Custom Forward/Backward Functions

For complex models with multiple components or special training procedures:

```python
def custom_forward_fn(model, batch):
    """Custom forward pass for complex models."""
    inputs, targets = batch
    
    # Your custom forward logic
    outputs = model(inputs)
    additional_outputs = model.some_other_forward(inputs)
    
    return (outputs, additional_outputs), targets

def custom_backward_fn(loss, model, optimizer):
    """Custom backward pass with special handling."""
    loss.backward()
    # Add any custom gradient processing here

config = TrainingConfig(
    custom_forward_fn=custom_forward_fn,
    custom_backward_fn=custom_backward_fn,
    # ... other config
)
```

### Model with Built-in Loss

Your model can implement its own loss computation:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ... model definition
    
    def forward(self, x):
        # ... forward pass
        return outputs
    
    def compute_loss(self, outputs, targets):
        """Custom loss computation."""
        return your_loss_calculation(outputs, targets)

# No need to provide loss_fn to trainer
trainer = Trainer(
    model=MyModel(),
    config=config,
    train_dataloader=train_loader,
    # loss_fn=None  # Will use model's compute_loss method
)
```

### Checkpointing and Resuming

```python
# Save checkpoint
trainer.save_checkpoint("my_checkpoint.pt")

# Load checkpoint
trainer.load_checkpoint("my_checkpoint.pt", resume_training=True)

# Or create new trainer and load
new_trainer = Trainer(...)
checkpoint = new_trainer.load_checkpoint("my_checkpoint.pt", resume_training=False)
```

## 📊 Output Examples

Treadmill provides beautiful, informative output during training:

```
============================================================
🚀 Starting Training with Treadmill
============================================================

┌─────────────────────────────────────────────────────────┐
│                       Model Info                        │
├─────────────────────────────────────────────────────────┤
│ Model: SimpleCNN                                        │
│ Total Parameters: 1.2M                                  │
│ Trainable Parameters: 1.2M                              │
└─────────────────────────────────────────────────────────┘

Epoch 1/20
────────────────────────────────────────
Batch   50/391 ( 12.8%) | loss: 2.1234 | accuracy: 0.2341
Batch  100/391 ( 25.6%) | loss: 1.8765 | accuracy: 0.3456
...

┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric         ┃ Train      ┃ Validation     ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Loss           │ 1.2345     │ 1.3456         │
│ Accuracy       │ 0.6789     │ 0.6234         │
│ Epoch Time     │ 2m 34.5s   │ 2m 34.5s       │
│ Total Time     │ 2m 34.5s   │ 2m 34.5s       │
└────────────────┴────────────┴────────────────┘
```

## 🎯 Examples

Check out the `/examples` directory for complete examples:

- **`basic_training.py`**: Simple CNN on CIFAR-10
- **`advanced_training.py`**: VAE with custom forward/backward functions

Run examples:

```bash
cd examples
python basic_training.py
python advanced_training.py
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the need for clean, modular PyTorch training
- Built with ❤️ for the PyTorch community
- Uses [Rich](https://github.com/Textualize/rich) for beautiful terminal output

---

**Happy Training with Treadmill! 🚀** 
