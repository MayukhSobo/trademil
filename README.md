# 🏃‍♀️‍➡️ Treadmill 🏃‍♀️‍➡️

<div align="center">
  <img src="https://raw.githubusercontent.com/MayukhSobo/treadmill/main/treadmill.png" alt="Treadmill Training Framework" width="300"/>
</div>

**A Clean and Modular PyTorch Training Framework**

Treadmill is a lightweight, modular training framework specifically designed for PyTorch. It provides clean, easy-to-understand training loops with beautiful output formatting while maintaining the power and flexibility of vanilla PyTorch.

## ✨ Features

- **🎯 Pure PyTorch**: Built specifically for PyTorch, no forced abstractions
- **🔧 Modular Design**: Easy to customize and extend with callback system  
- **📊 Beautiful Output**: Rich formatting with progress bars and metrics tables
- **📈 Comprehensive Training Reports**: Detailed reports with model info, hardware usage, and performance metrics
- **💻 Hardware Monitoring**: Real-time CPU, RAM, and GPU usage tracking during training
- **⚡ Performance Optimizations**: Mixed precision, gradient accumulation, gradient clipping
- **🎛️ Flexible Configuration**: Dataclass-based configuration system
- **📊 Built-in Metrics**: Comprehensive metrics with support for custom functions
- **💾 Smart Checkpointing**: Automatic model saving with customizable triggers
- **🛑 Early Stopping**: Configurable early stopping to prevent overfitting
- **🔄 Enhanced Resume Training**: Automatic epoch calculation and configuration consistency

## 🛠️ Installation

### From PyPI (Recommended)

```bash
pip install pytorch-treadmill
```

### Install with Optional Dependencies

```bash
# With examples dependencies (torchvision, scikit-learn)
pip install "pytorch-treadmill[examples]"

# With full dependencies (visualization tools, docs, hardware monitoring)
pip install "pytorch-treadmill[full]"

# For development
pip install "pytorch-treadmill[dev]"
```

### Hardware Monitoring Dependencies

For comprehensive hardware monitoring during training:

```bash
pip install psutil pynvml  # CPU, RAM, and GPU monitoring
```

These are automatically included with the full installation.

### From Source

For the latest development version or to contribute:

```bash
git clone https://github.com/MayukhSobo/treadmill.git
cd treadmill
pip install -e .
```

### Install with Examples (Development)

```bash
pip install -e ".[examples]"  # Includes torchvision and additional dependencies
```

### Install Full Version (Development)

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
class SimpleDNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes),
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        return self.network(x)

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
    model=SimpleDNN(),
    config=config,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fns={"accuracy": StandardMetrics.accuracy}
)

# Start training - comprehensive report displayed automatically
history = trainer.train()

# Access detailed training report
print(f"Training completed in {trainer.report.training_time:.1f}s")
print(f"Model parameters: {trainer.report.total_parameters:,}")
print(f"Best validation accuracy: {trainer.report.best_metrics.get('val_accuracy', 0):.4f}")
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
    
    # Checkpointing and resuming
    checkpoint_dir="./checkpoints",
    project_name="my_experiment",
    keep_all_checkpoints=False,  # Only keep best checkpoint
    resume_training=False,  # Set to True for resume training
    additional_epochs=None,  # For resume: specify additional epochs
    
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

### Comprehensive Training Reports

Access detailed training information programmatically:

```python
# After training completes
trainer.train()

# Access comprehensive report
report = trainer.report

print(f"Model: {report.model_name}")
print(f"Parameters: {report.total_parameters:,}")
print(f"Training time: {report.training_time:.1f}s")
print(f"Best accuracy: {report.best_metrics.get('val_accuracy', 0):.4f}")

# Hardware usage insights
if report.avg_cpu_percent:
    print(f"Average CPU usage: {report.avg_cpu_percent:.1f}%")
    print(f"Peak RAM usage: {report.max_ram_mb:.1f} MB")

# Serialize report for analysis/storage
report_dict = report.to_dict()
import json
with open("training_report.json", "w") as f:
    json.dump(report_dict, f, indent=2, default=str)
```

### Hardware Monitoring

Enable real-time hardware monitoring during training:

```python
# Hardware monitoring is automatic when dependencies are installed
pip install psutil pynvml

# Or disable by removing dependencies
# Hardware monitoring gracefully falls back to basic info
```

The framework automatically tracks:
- **CPU Usage**: Average and peak percentages during training
- **RAM Usage**: Memory consumption patterns
- **GPU Utilization**: GPU usage and memory (NVIDIA GPUs)
- **Training Efficiency**: Resource utilization insights

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

### Checkpointing and Resume Training

#### Automatic Checkpointing
```python
# Configure automatic checkpointing
config = TrainingConfig(
    epochs=10,
    checkpoint_dir="./checkpoints",
    project_name="my_experiment",
    keep_all_checkpoints=False  # Only keep best checkpoint
)
```

#### Manual Checkpointing
```python
# Save custom checkpoint
trainer.save_checkpoint("my_checkpoint.pt")

# Load checkpoint
trainer.load_checkpoint("my_checkpoint.pt", resume_training=True)
```

#### Resume Training Example
```python
# After initial training, resume with additional epochs
from treadmill import Trainer, TrainingConfig

# Simplified resume training - framework handles everything automatically
config = TrainingConfig(
    additional_epochs=5,  # Train for 5 more epochs
    checkpoint_dir="./checkpoints/my_experiment-15-09-2025-...",
    resume_training=True
)

trainer = Trainer(model=SimpleDNN(), config=config, train_dataloader=train_loader)
trainer.train()  # Automatically loads latest checkpoint and continues
```

## 📊 Output Examples

Treadmill provides beautiful, informative output during training:

### During Training
```
╭─────────────────────────────────────────────── Model Info ───────────────────────────────────────────────╮
│ Model: SimpleDNN                                                                                         │
│ Total Parameters: 109.4K                                                                                 │
│ Trainable Parameters: 109.4K                                                                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Epoch 1/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% 0:00:04
loss: 0.2093 | accuracy: 0.9339

                     Epoch 1 Summary                     
╭────────────┬────────┬────────────┬────────────────────╮
│ Metric     │  Train │ Validation │ Change (from prev) │
├────────────┼────────┼────────────┼────────────────────┤
│ Accuracy   │ 0.9339 │     0.9634 │                N/A │
│ Loss       │ 0.2093 │     0.1200 │                N/A │
│            │        │            │                    │
│ Epoch Time │     4s │         4s │                    │
│ Total Time │     4s │         4s │                    │
╰────────────┴────────┴────────────┴────────────────────╯
```

### Comprehensive Training Report
```
====================================================================================================
🎉 TRAINING COMPLETE! 🎉
====================================================================================================

                     📊 Training Summary                      
╭───────────────────────────┬────────────────────────────────╮
│ Metric                    │ Value                          │
├───────────────────────────┼────────────────────────────────┤
│ Total Epochs              │ 10                             │
│ Training Time             │ 42.3s                          │
│ Avg Time/Epoch            │ 4.2s                           │
│ Total Batches             │ 9,380                          │
│ Started At                │ 2025-09-16 00:48:27            │
│ Completed At              │ 2025-09-16 00:49:09            │
│ Early Stopping            │ ❌ No                          │
╰───────────────────────────┴────────────────────────────────╯

                     🏗️ Model Information                      
╭───────────────────────────┬────────────────────────────────╮
│ Property                  │ Value                          │
├───────────────────────────┼────────────────────────────────┤
│ Model Name                │ SimpleDNN                      │
│ Total Parameters          │ 109.4K                         │
│ Trainable Parameters      │ 109.4K                         │
│ Model Size                │ 0.4 MB                         │
│ Device                    │ cpu                            │
╰───────────────────────────┴────────────────────────────────╯

                      💻 Hardware Resources                      
╭───────────────────────────┬─────────────────┬─────────────────╮
│ Resource                  │         Average │            Peak │
├───────────────────────────┼─────────────────┼─────────────────┤
│ CPU Usage (%)             │            7.2% │           92.1% │
│ RAM Usage                 │         10.8 GB │         11.2 GB │
│ GPU Utilization (%)       │           45.3% │           89.7% │
│ GPU Memory                │  2.1 GB / 8.0GB │        (26.3%)  │
╰───────────────────────────┴─────────────────┴─────────────────╯

                            🏆 Performance Metrics                            
╭──────────────────────┬─────────────────┬─────────────────┬─────────────────╮
│ Metric               │      Best Value │     Final Value │     Improvement │
├──────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Train_Accuracy       │          0.9823 │          0.9823 │           +0.0% │
│ Train_Loss           │          0.0543 │          0.0543 │           -0.0% │
│ Val_Accuracy         │          0.9834 │          0.9834 │           +0.0% │
│ Val_Loss             │          0.0489 │          0.0489 │           -0.0% │
╰──────────────────────┴─────────────────┴─────────────────┴─────────────────╯

                       💾 Checkpoint Information                        
╭───────────────────────────┬──────────────────────────────────────────╮
│ Property                  │ Value                                    │
├───────────────────────────┼──────────────────────────────────────────┤
│ Total Checkpoints         │ 2                                        │
│ Best Checkpoint           │ checkpoint_010_0.0489.pt                 │
╰───────────────────────────┴──────────────────────────────────────────╯
```

## 🎯 Examples

Check out the `/examples` directory for complete examples:

- **`basic_training.py`**: Simple DNN on MNIST with comprehensive reports
- **`basic_training_resume.py`**: Resume training from checkpoints with automatic configuration
- **`advanced_training.py`**: Complex architectures with custom forward/backward functions

Run examples:

```bash
cd examples

# Basic training with hardware monitoring and comprehensive reports
python basic_training.py --epochs 10 --project-name "my_experiment"

# Resume training with additional epochs
python basic_training_resume.py --checkpoint-dir "./checkpoints/my_experiment-..." --epochs 5

# Advanced training patterns
python advanced_training.py
```

Both `basic_training.py` and `basic_training_resume.py` use click for consistent CLI:

```bash
# Get help for any example
python basic_training.py --help
python basic_training_resume.py --help
```

## 🤝 Contributing

I welcome contributions! Please see our contributing guidelines for more details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the need for clean, modular PyTorch training
- Built with ❤️ for the PyTorch community
- Uses [Rich](https://github.com/Textualize/rich) for beautiful terminal output

---

**Happy Training with Treadmill! 🚀** 
# Documentation will be available at: https://mayukhsobo.github.io/treadmill/
