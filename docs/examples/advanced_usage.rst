Advanced Usage Example
======================

This example demonstrates sophisticated training techniques using Treadmill's advanced features. Perfect for experienced practitioners who want to leverage all of Treadmill's capabilities.

Overview
--------

**Advanced techniques covered:**
- Custom callbacks and training hooks
- Learning rate scheduling and optimization
- Advanced data loading and augmentation
- Mixed precision training and gradient accumulation
- Custom metrics and monitoring
- Model checkpointing and resumption
- Multi-GPU training setup
- Advanced evaluation and analysis

**Estimated time:** 30-45 minutes

Prerequisites
-------------

.. code-block:: bash

    pip install -e ".[full]"
    pip install wandb tensorboard  # Optional: for advanced logging

Advanced Image Classification
-----------------------------

Let's build a sophisticated training pipeline for CIFAR-100 with all advanced features:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import os
    
    from treadmill import Trainer, TrainingConfig, OptimizerConfig, SchedulerConfig
    from treadmill.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
        Callback
    )
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

Step 1: Advanced Data Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class AdvancedDataPipeline:
        """Advanced data loading with sophisticated augmentation."""
        
        def __init__(self, data_dir='./data', batch_size=128, num_workers=4):
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.setup_transforms()
            self.load_data()
        
        def setup_transforms(self):
            """Create sophisticated augmentation pipelines."""
            
            # Advanced training augmentations
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                ),
                # Advanced augmentations
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
            ])
            
            # Test transform (no augmentation)
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
        
        def load_data(self):
            """Load CIFAR-100 with train/val split."""
            
            # Load full datasets
            train_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=True, download=True, 
                transform=self.train_transform
            )
            
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=False, download=True,
                transform=self.test_transform
            )
            
            # Create train/validation split
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Update validation dataset transform
            self.val_dataset.dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=True, download=False,
                transform=self.test_transform
            )
            
            self.test_dataset = test_dataset
            
            print(f"Dataset loaded:")
            print(f"  Training: {len(self.train_dataset)} samples")
            print(f"  Validation: {len(self.val_dataset)} samples") 
            print(f"  Test: {len(self.test_dataset)} samples")
            print(f"  Classes: 100")
        
        def get_loaders(self):
            """Get data loaders with advanced settings."""
            
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True  # For batch norm stability
            )
            
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            
            return train_loader, val_loader, test_loader
    
    # Create data pipeline
    data_pipeline = AdvancedDataPipeline(batch_size=128)
    train_loader, val_loader, test_loader = data_pipeline.get_loaders()

Step 2: Advanced Model Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class ResidualBlock(nn.Module):
        """Residual block with batch normalization."""
        
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            residual = self.shortcut(x)
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = F.relu(out)
            
            return out
    
    class AdvancedCNN(nn.Module):
        """Advanced CNN with residual connections and modern techniques."""
        
        def __init__(self, num_classes=100, dropout_rate=0.3):
            super().__init__()
            
            # Initial convolution
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
            
            # Residual blocks
            self.layer1 = self._make_layer(64, 64, 2, stride=1)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            
            # Global average pooling
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Classifier with dropout
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
            
            self._initialize_weights()
        
        def _make_layer(self, in_channels, out_channels, blocks, stride):
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            for _ in range(1, blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            
            return x
    
    # Create model
    model = AdvancedCNN(num_classes=100, dropout_rate=0.3)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {count_parameters(model):,} parameters")

Step 3: Custom Callbacks
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class CustomLoggingCallback(Callback):
        """Custom callback for advanced logging and monitoring."""
        
        def __init__(self, log_dir='./logs'):
            self.log_dir = log_dir
            self.metrics_history = defaultdict(list)
            os.makedirs(log_dir, exist_ok=True)
        
        def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
            # Log metrics
            for name, value in metrics.items():
                self.metrics_history[name].append(value)
            
            # Log to file
            with open(f"{self.log_dir}/training_log.txt", "a") as f:
                f.write(f"Epoch {epoch}: {metrics}\n")
            
            # Advanced logging every 10 epochs
            if epoch % 10 == 0:
                self._advanced_logging(trainer, epoch, metrics)
        
        def _advanced_logging(self, trainer, epoch, metrics):
            """Advanced logging with model analysis."""
            
            # Log gradient norms
            total_norm = 0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            print(f"  Gradient norm: {total_norm:.4f}")
            
            # Log learning rate
            if trainer.scheduler:
                current_lr = trainer.optimizer.param_groups[0]['lr']
                print(f"  Learning rate: {current_lr:.6f}")
    
    class WarmupCallback(Callback):
        """Learning rate warmup callback."""
        
        def __init__(self, warmup_epochs=5, base_lr=0.001):
            self.warmup_epochs = warmup_epochs
            self.base_lr = base_lr
        
        def on_epoch_start(self, trainer, epoch, **kwargs):
            if epoch < self.warmup_epochs:
                # Linear warmup
                lr = self.base_lr * (epoch + 1) / self.warmup_epochs
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = lr
                print(f"  Warmup LR: {lr:.6f}")

Step 4: Advanced Training Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Advanced optimizer configuration
    optimizer_config = OptimizerConfig(
        optimizer_class="AdamW",
        lr=0.001,
        weight_decay=0.01,
        params={
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "amsgrad": True  # Use AMSGrad variant
        }
    )
    
    # Learning rate scheduler configuration
    scheduler_config = SchedulerConfig(
        scheduler_class="CosineAnnealingLR",
        params={
            "T_max": 200,  # Maximum number of iterations
            "eta_min": 1e-6  # Minimum learning rate
        }
    )
    
    # Advanced training configuration
    config = TrainingConfig(
        # Training parameters
        epochs=200,
        device="auto",
        
        # Performance optimizations
        mixed_precision=True,
        gradient_accumulation_steps=2,  # Effective batch size = 128 * 2 = 256
        max_grad_norm=1.0,  # Gradient clipping
        
        # Validation and monitoring
        validation_frequency=1,
        log_frequency=50,
        
        # Early stopping (generous for long training)
        early_stopping_patience=30,
        early_stopping_min_delta=0.0001,
        
        # Checkpointing
        checkpoint_dir="./checkpoints/advanced_cifar100",
        save_best_model=True,
        save_last_model=True,
        checkpoint_frequency=10,  # Save every 10 epochs
        
        # Optimizer and scheduler
        optimizer=optimizer_config,
        scheduler=scheduler_config
    )

Step 5: Advanced Metrics
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class AdvancedMetrics:
        """Collection of advanced metrics."""
        
        @staticmethod
        def accuracy(predictions, targets):
            pred_classes = torch.argmax(predictions, dim=1)
            return (pred_classes == targets).float().mean().item()
        
        @staticmethod
        def top_k_accuracy(predictions, targets, k=5):
            _, top_k_preds = torch.topk(predictions, k, dim=1)
            targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
            correct = (top_k_preds == targets_expanded).any(dim=1)
            return correct.float().mean().item()
        
        @staticmethod
        def precision_at_k(predictions, targets, k=5):
            _, top_k_preds = torch.topk(predictions, k, dim=1)
            correct = (top_k_preds == targets.view(-1, 1)).float()
            return correct.sum(dim=1).mean().item() / k
        
        @staticmethod
        def confidence_score(predictions):
            probabilities = F.softmax(predictions, dim=1)
            max_probs = torch.max(probabilities, dim=1)[0]
            return max_probs.mean().item()
    
    # Create metrics dictionary
    custom_metrics = {
        'accuracy': AdvancedMetrics.accuracy,
        'top5_accuracy': lambda p, t: AdvancedMetrics.top_k_accuracy(p, t, k=5),
        'precision_at_5': lambda p, t: AdvancedMetrics.precision_at_k(p, t, k=5),
        'confidence': lambda p, t: AdvancedMetrics.confidence_score(p)
    }

Step 6: Advanced Callbacks Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create advanced callbacks
    callbacks = [
        # Learning rate warmup
        WarmupCallback(warmup_epochs=5, base_lr=0.001),
        
        # Custom logging
        CustomLoggingCallback(log_dir='./logs/advanced_training'),
        
        # Early stopping with validation loss
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            min_delta=0.0001,
            verbose=True,
            mode='min'
        ),
        
        # Model checkpointing
        ModelCheckpoint(
            filepath='./checkpoints/advanced_cifar100/best_model_{epoch:03d}_{val_acc:.4f}.pt',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=True,
            save_top_k=3  # Keep top 3 models
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=True
        )
    ]

Step 7: Advanced Loss Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class LabelSmoothingLoss(nn.Module):
        """Label smoothing loss for better generalization."""
        
        def __init__(self, num_classes, smoothing=0.1):
            super().__init__()
            self.num_classes = num_classes
            self.smoothing = smoothing
            self.confidence = 1.0 - smoothing
        
        def forward(self, predictions, targets):
            log_probs = F.log_softmax(predictions, dim=1)
            
            # Create smoothed targets
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            
            return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
    
    # Use label smoothing loss
    loss_fn = LabelSmoothingLoss(num_classes=100, smoothing=0.1)

Step 8: Training with All Advanced Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create advanced trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        metric_fns=custom_metrics,
        callbacks=callbacks
    )
    
    # Print training setup
    print("🚀 Advanced Training Setup:")
    print(f"  Model: {type(model).__name__} ({count_parameters(model):,} params)")
    print(f"  Optimizer: {config.optimizer.optimizer_class.__name__}")
    print(f"  Scheduler: {config.scheduler.scheduler_class.__name__}")
    print(f"  Mixed Precision: {config.mixed_precision}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"  Device: {trainer.device}")
    print(f"  Callbacks: {len(callbacks)}")
    print("-" * 80)
    
    # Start advanced training
    history = trainer.fit()
    print("✅ Advanced training completed!")

Step 9: Advanced Evaluation and Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def advanced_evaluation(trainer, test_loader, class_names=None):
        """Comprehensive model evaluation with advanced metrics."""
        
        print("📊 Comprehensive Model Evaluation")
        print("=" * 50)
        
        # Basic evaluation
        test_results = trainer.evaluate(test_loader)
        print(f"Test Results:")
        for metric_name, value in test_results.items():
            print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
        
        # Detailed predictions analysis
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        trainer.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(trainer.device), target.to(trainer.device)
                output = trainer.model(data)
                
                # Get predictions and confidence
                probs = F.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1)
                confidence = torch.max(probs, dim=1)[0]
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        confidences = np.array(all_confidences)
        
        # Detailed analysis
        print(f"\n🔍 Detailed Analysis:")
        print(f"  Average Confidence: {confidences.mean():.4f}")
        print(f"  Confidence Std: {confidences.std():.4f}")
        print(f"  Low Confidence Samples (<0.5): {(confidences < 0.5).sum()}")
        print(f"  High Confidence Samples (>0.9): {(confidences > 0.9).sum()}")
        
        # Per-class accuracy
        unique_classes = np.unique(targets)
        print(f"\n📈 Per-Class Performance (showing top 10 and bottom 10):")
        
        class_accuracies = []
        for cls in unique_classes:
            mask = targets == cls
            if mask.sum() > 0:
                acc = (predictions[mask] == targets[mask]).mean()
                class_accuracies.append((cls, acc))
        
        # Sort by accuracy
        class_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        print("  Best performing classes:")
        for cls, acc in class_accuracies[:10]:
            print(f"    Class {cls:2d}: {acc:.4f}")
        
        print("  Worst performing classes:")
        for cls, acc in class_accuracies[-10:]:
            print(f"    Class {cls:2d}: {acc:.4f}")
        
        return {
            'predictions': predictions,
            'targets': targets,
            'confidences': confidences,
            'class_accuracies': class_accuracies
        }
    
    # Run advanced evaluation
    eval_results = advanced_evaluation(trainer, test_loader)

Step 10: Model Optimization and Export
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def optimize_model_for_inference(model, example_input):
        """Optimize model for production inference."""
        
        # Set to evaluation mode
        model.eval()
        
        # Trace the model for optimization
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimize for inference
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
        return traced_model
    
    def export_model(model, filepath, example_input=None):
        """Export model in multiple formats."""
        
        print(f"📦 Exporting model to {filepath}")
        
        # Save complete checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_class': type(model).__name__,
            'model_config': {
                'num_classes': 100,
                'dropout_rate': 0.3
            },
            'training_history': history,
            'performance_metrics': eval_results
        }
        
        torch.save(checkpoint, f"{filepath}_complete.pt")
        
        # Save optimized model for inference
        if example_input is not None:
            optimized_model = optimize_model_for_inference(model, example_input)
            torch.jit.save(optimized_model, f"{filepath}_optimized.pt")
        
        # Save just the state dict (smaller file)
        torch.save(model.state_dict(), f"{filepath}_weights.pt")
        
        print("✅ Model export completed!")
    
    # Export the trained model
    example_input = torch.randn(1, 3, 32, 32).to(trainer.device)
    export_model(trainer.model, "./models/advanced_cifar100", example_input)

Advanced Training Summary
-------------------------

**🎯 Advanced Features Used:**

✅ **Data Pipeline:**
- Sophisticated augmentation strategies
- Efficient data loading with multiple workers
- Advanced normalization and preprocessing

✅ **Model Architecture:**
- Residual connections for better gradient flow
- Batch normalization for training stability
- Proper weight initialization
- Dropout for regularization

✅ **Training Optimization:**
- Mixed precision training (faster + less memory)
- Gradient accumulation (larger effective batch size)
- Gradient clipping (training stability)
- Label smoothing (better generalization)

✅ **Learning Rate Management:**
- Warmup for stable training start
- Cosine annealing scheduling
- Reduce on plateau for fine-tuning

✅ **Monitoring and Callbacks:**
- Custom logging and metrics tracking
- Advanced early stopping
- Multiple model checkpointing
- Performance monitoring

✅ **Evaluation and Analysis:**
- Comprehensive metrics (accuracy, top-k, confidence)
- Per-class performance analysis
- Model confidence analysis
- Production-ready export

**📊 Expected Results:**

With this advanced setup, you should achieve:
- **CIFAR-100 accuracy**: 70-75% (vs ~45% random)
- **Training stability**: Smooth convergence curves
- **Generalization**: Good test performance
- **Efficiency**: Fast training with mixed precision

**🚀 Production Readiness:**

The trained model is ready for production with:
- Optimized inference models
- Complete checkpoints for resuming
- Comprehensive performance metrics
- Export in multiple formats

This advanced example demonstrates how Treadmill scales from simple scripts to production-ready training pipelines while maintaining clean, readable code! 🏃‍♀️‍➡️

Next Steps
----------

- Explore multi-GPU training with DataParallel
- Try distributed training with DistributedDataParallel  
- Implement custom optimizers and schedulers
- Add Weights & Biases integration for experiment tracking
- Deploy models with TorchServe or TensorRT 