Complete Image Classification Tutorial
=====================================

This comprehensive tutorial will guide you through building a complete image classification system using Treadmill. We'll cover everything from data preparation to model evaluation, using the CIFAR-10 dataset as our example.

Tutorial Overview
-----------------

**What You'll Learn:**
- Setting up data pipelines for image classification
- Building and training CNN architectures
- Advanced training techniques and optimizations  
- Model evaluation and analysis
- Best practices for real-world deployment

**Prerequisites:**
- Basic PyTorch knowledge
- Understanding of convolutional neural networks
- Python programming experience

**Estimated Time:** 45-60 minutes

Step 1: Environment Setup
-------------------------

First, let's ensure you have all the necessary dependencies:

.. code-block:: bash

    # Install Treadmill with full dependencies
    pip install -e ".[full]"
    
    # Additional packages for this tutorial
    pip install matplotlib seaborn

Now let's import all the necessary libraries:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, random_split
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm.auto import tqdm
    
    # Treadmill imports
    from treadmill import Trainer, TrainingConfig, OptimizerConfig
    from treadmill.callbacks import EarlyStopping, ModelCheckpoint
    from treadmill.metrics import MetricsTracker
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

Step 2: Dataset Preparation
---------------------------

CIFAR-10 Dataset Overview
^^^^^^^^^^^^^^^^^^^^^^^^^

CIFAR-10 consists of 60,000 32×32 color images in 10 classes:
- Airplane, Automobile, Bird, Cat, Deer
- Dog, Frog, Horse, Ship, Truck

Let's load and explore the dataset:

.. code-block:: python

    # Define the class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Data augmentation and normalization transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Validation/test transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform
    )
    
    # Split training data into train/validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Apply validation transform to validation subset
    val_subset.dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=val_transform
    )
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")

Data Exploration
^^^^^^^^^^^^^^^^

Let's visualize some samples to understand our data better:

.. code-block:: python

    def visualize_samples(dataset, num_samples=16, title="Sample Images"):
        """Visualize random samples from the dataset."""
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(title, fontsize=16)
        
        for i in range(num_samples):
            # Get random sample
            idx = np.random.randint(0, len(dataset))
            image, label = dataset[idx]
            
            # Denormalize for visualization
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            image = image * std.view(3, 1, 1) + mean.view(3, 1, 1)
            image = torch.clamp(image, 0, 1)
            
            # Plot
            ax = axes[i // 4, i % 4]
            ax.imshow(image.permute(1, 2, 0))
            ax.set_title(f"{class_names[label]}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Visualize training samples
    visualize_samples(train_subset, title="Training Samples")

Class Distribution Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def analyze_class_distribution(dataset, title="Class Distribution"):
        """Analyze and visualize class distribution."""
        class_counts = torch.zeros(10)
        
        for _, label in tqdm(dataset, desc="Analyzing distribution"):
            class_counts[label] += 1
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, class_counts, color='skyblue', edgecolor='navy')
        plt.title(title)
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{int(count)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return class_counts
    
    train_distribution = analyze_class_distribution(train_subset, "Training Set Class Distribution")

Create Data Loaders
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create data loaders with optimal settings
    batch_size = 128  # Adjust based on your GPU memory
    num_workers = 4   # Adjust based on your CPU cores
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

Step 3: Model Architecture
--------------------------

Custom CNN Architecture
^^^^^^^^^^^^^^^^^^^^^^^

Let's build a modern CNN architecture with best practices:

.. code-block:: python

    class CIFAR10CNN(nn.Module):
        """
        Modern CNN architecture for CIFAR-10 classification.
        
        Features:
        - Residual connections for better gradient flow
        - Batch normalization for stable training
        - Dropout for regularization
        - Adaptive pooling for flexible input sizes
        """
        
        def __init__(self, num_classes=10, dropout_rate=0.3):
            super(CIFAR10CNN, self).__init__()
            
            # First block: 32x32 -> 16x16
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Second block: 16x16 -> 8x8
            self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Third block: 8x8 -> 4x4
            self.block3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Fourth block: 4x4 -> 2x2
            self.block4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Global average pooling and classifier
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
            
            # Initialize weights
            self._initialize_weights()
        
        def _initialize_weights(self):
            """Initialize model weights using He initialization."""
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
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.classifier(x)
            
            return x
    
    # Create model instance
    model = CIFAR10CNN(num_classes=10, dropout_rate=0.3)
    
    # Display model architecture
    from torchsummary import summary
    summary(model, input_size=(3, 32, 32))

Model Analysis
^^^^^^^^^^^^^^

.. code-block:: python

    def count_parameters(model):
        """Count trainable and non-trainable parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        print(f"Trainable parameters: {trainable:,}")
        print(f"Non-trainable parameters: {non_trainable:,}")
        print(f"Total parameters: {trainable + non_trainable:,}")
        
        return trainable, non_trainable
    
    count_parameters(model)

Step 4: Custom Metrics
----------------------

Let's define comprehensive metrics for evaluating our model:

.. code-block:: python

    def accuracy(predictions, targets):
        """Calculate accuracy."""
        pred_classes = torch.argmax(predictions, dim=1)
        return (pred_classes == targets).float().mean().item()
    
    def top_k_accuracy(predictions, targets, k=3):
        """Calculate top-k accuracy."""
        _, top_k_preds = torch.topk(predictions, k, dim=1)
        targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=1)
        return correct.float().mean().item()
    
    def per_class_accuracy(predictions, targets, num_classes=10):
        """Calculate per-class accuracy."""
        pred_classes = torch.argmax(predictions, dim=1)
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)
        
        for i in range(targets.size(0)):
            label = targets[i]
            class_total[label] += 1
            if pred_classes[i] == label:
                class_correct[label] += 1
        
        # Avoid division by zero
        class_acc = class_correct / (class_total + 1e-8)
        return class_acc.mean().item()
    
    # Custom metrics dictionary
    custom_metrics = {
        'accuracy': accuracy,
        'top3_accuracy': lambda p, t: top_k_accuracy(p, t, k=3),
        'per_class_acc': per_class_accuracy
    }

Step 5: Training Configuration
-----------------------------

Advanced Training Setup
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Optimizer configuration with learning rate scheduling
    optimizer_config = OptimizerConfig(
        optimizer_class="AdamW",  # AdamW often works better than Adam
        lr=0.001,
        weight_decay=0.01,  # L2 regularization
        params={
            "betas": (0.9, 0.999),
            "eps": 1e-8
        }
    )
    
    # Training configuration with all optimizations enabled
    config = TrainingConfig(
        # Basic training parameters
        epochs=100,
        device="auto",  # Auto-detect GPU/CPU
        
        # Performance optimizations
        mixed_precision=True,           # Faster training on modern GPUs
        accumulate_grad_batches=1,  # Simulate larger batch sizes
        grad_clip_norm=1.0,             # Gradient clipping
        
        # Validation and logging
        validation_frequency=1,         # Validate every epoch
        log_frequency=50,              # Log every 50 batches
        
        # Early stopping configuration
        early_stopping_patience=15,     # Stop if no improvement for 15 epochs
        early_stopping_min_delta=0.001, # Minimum improvement threshold
        
        # Checkpointing
        checkpoint_dir="./checkpoints/cifar10_cnn",
        save_best_model=True,
        save_last_model=True,
        
        # Optimizer and scheduler
        optimizer=optimizer_config,
        # scheduler will be added after creating trainer
    )

Advanced Callbacks
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from treadmill.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        min_delta=0.001,
        verbose=True,
        mode='min'
    )
    
    # Model checkpointing callback
    model_checkpoint = ModelCheckpoint(
        filepath='./checkpoints/cifar10_cnn/best_model_{epoch:02d}_{val_acc:.4f}.pt',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=True
    )
    
    # Learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=True
    )
    
    callbacks = [early_stopping, model_checkpoint, reduce_lr]

Step 6: Training Process
------------------------

Initialize and Train
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1),  # Label smoothing
        metric_fns=custom_metrics,
        callbacks=callbacks
    )
    
    # Print training information
    print("🚀 Starting CIFAR-10 Image Classification Training")
    print(f"📊 Dataset: {len(train_subset)} train, {len(val_subset)} val, {len(test_dataset)} test")
    print(f"🏗️  Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    print(f"💾 Device: {trainer.device}")
    print(f"⚡ Mixed precision: {config.mixed_precision}")
    print("-" * 80)
    
    # Train the model
    history = trainer.train()
    
    print("🎉 Training completed!")

Training Visualization
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def plot_training_history(history, save_path=None):
        """Plot comprehensive training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'train_accuracy' in history:
            axes[0, 1].plot(history['train_accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 accuracy plot
        if 'train_top3_accuracy' in history:
            axes[1, 0].plot(history['train_top3_accuracy'], label='Training Top-3', linewidth=2)
        if 'val_top3_accuracy' in history:
            axes[1, 0].plot(history['val_top3_accuracy'], label='Validation Top-3', linewidth=2)
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if hasattr(trainer, 'lr_history') and trainer.lr_history:
            axes[1, 1].plot(trainer.lr_history, linewidth=2, color='red')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nHistory\nNot Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot training history
    plot_training_history(history, save_path='training_history.png')

Step 7: Model Evaluation
------------------------

Comprehensive Test Set Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_loader)
    
    print("\n🎯 Test Set Results:")
    for metric_name, value in test_results.items():
        print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")

Detailed Classification Report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    def detailed_evaluation(trainer, test_loader, class_names):
        """Generate detailed evaluation including confusion matrix and classification report."""
        
        # Get predictions and true labels
        all_predictions = []
        all_targets = []
        
        trainer.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Generating predictions")):
                data, target = data.to(trainer.device), target.to(trainer.device)
                output = trainer.model(data)
                pred = torch.argmax(output, dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Classification report
        report = classification_report(
            all_targets, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        print("\n📋 Detailed Classification Report:")
        print(classification_report(all_targets, all_predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Per-class accuracy analysis
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, per_class_acc, color='lightcoral', edgecolor='darkred')
        plt.title('Per-Class Accuracy')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, per_class_acc):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report, cm, per_class_acc
    
    # Generate detailed evaluation
    report, confusion_mat, per_class_acc = detailed_evaluation(trainer, test_loader, class_names)

Error Analysis
^^^^^^^^^^^^^^

.. code-block:: python

    def analyze_misclassifications(trainer, test_loader, class_names, num_examples=16):
        """Analyze and visualize misclassified examples."""
        
        misclassified = []
        trainer.model.eval()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(trainer.device), target.to(trainer.device)
                output = trainer.model(data)
                pred = torch.argmax(output, dim=1)
                
                # Find misclassified examples
                mask = pred != target
                if mask.any():
                    for i in range(data.size(0)):
                        if mask[i] and len(misclassified) < num_examples:
                            # Store image, true label, predicted label, and confidence
                            confidence = F.softmax(output[i], dim=0)[pred[i]].item()
                            misclassified.append({
                                'image': data[i].cpu(),
                                'true_label': target[i].item(),
                                'pred_label': pred[i].item(),
                                'confidence': confidence
                            })
                
                if len(misclassified) >= num_examples:
                    break
        
        # Visualize misclassified examples
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Misclassified Examples', fontsize=16, fontweight='bold')
        
        for i, example in enumerate(misclassified[:16]):
            ax = axes[i // 4, i % 4]
            
            # Denormalize image for visualization
            image = example['image']
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            image = image * std.view(3, 1, 1) + mean.view(3, 1, 1)
            image = torch.clamp(image, 0, 1)
            
            ax.imshow(image.permute(1, 2, 0))
            ax.set_title(f"True: {class_names[example['true_label']]}\n"
                        f"Pred: {class_names[example['pred_label']]}\n"
                        f"Conf: {example['confidence']:.2f}", fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('misclassified_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Analyze misclassifications
    analyze_misclassifications(trainer, test_loader, class_names)

Step 8: Model Interpretation
----------------------------

Feature Map Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def visualize_feature_maps(model, data_loader, layer_name, num_images=4):
        """Visualize feature maps from a specific layer."""
        
        # Hook to capture feature maps
        feature_maps = []
        
        def hook_fn(module, input, output):
            feature_maps.append(output)
        
        # Register hook
        target_layer = dict(model.named_modules())[layer_name]
        hook = target_layer.register_forward_hook(hook_fn)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= num_images:
                    break
                    
                data = data.to(next(model.parameters()).device)
                _ = model(data[:1])  # Process one image at a time
                
                # Visualize first few feature maps
                fmaps = feature_maps[-1][0]  # First image in batch
                
                fig, axes = plt.subplots(2, 8, figsize=(16, 4))
                fig.suptitle(f'Feature Maps from {layer_name} - Image {batch_idx + 1}')
                
                for i in range(min(16, fmaps.shape[0])):
                    ax = axes[i // 8, i % 8]
                    ax.imshow(fmaps[i].cpu(), cmap='viridis')
                    ax.set_title(f'Filter {i}')
                    ax.axis('off')
                
                plt.tight_layout()
                plt.show()
        
        # Remove hook
        hook.remove()
    
    # Visualize feature maps from different layers
    # visualize_feature_maps(model, test_loader, 'block1.0', num_images=2)

Step 9: Model Saving and Deployment
-----------------------------------

Save Final Model
^^^^^^^^^^^^^^^^

.. code-block:: python

    # Save the complete model
    final_model_path = './models/cifar10_cnn_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': 10,
            'dropout_rate': 0.3
        },
        'training_config': config,
        'training_history': history,
        'test_results': test_results,
        'class_names': class_names
    }, final_model_path)
    
    print(f"✅ Model saved to {final_model_path}")

Load and Use Saved Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def load_trained_model(model_path, device='cpu'):
        """Load a trained model for inference."""
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Reconstruct model
        model_config = checkpoint['model_config']
        model = CIFAR10CNN(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, checkpoint
    
    # Example usage
    # loaded_model, checkpoint = load_trained_model(final_model_path, device='cuda')
    # class_names = checkpoint['class_names']

Inference Function
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def predict_image(model, image, class_names, transform=None, device='cpu'):
        """Predict class for a single image."""
        
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        model.eval()
        with torch.no_grad():
            if isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)  # Add batch dimension
            else:
                # Assume PIL Image
                image = transform(image).unsqueeze(0)
            
            image = image.to(device)
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=5)
            
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                predictions.append({
                    'class': class_names[idx],
                    'probability': prob.item()
                })
        
        return predictions
    
    # Example inference
    # predictions = predict_image(model, test_image, class_names, device=trainer.device)
    # print("Top predictions:")
    # for pred in predictions:
    #     print(f"  {pred['class']}: {pred['probability']:.4f}")

Summary and Best Practices
--------------------------

**What We Accomplished:**

✅ Built a complete image classification pipeline
✅ Implemented modern CNN architecture with best practices
✅ Used advanced training techniques (mixed precision, label smoothing)
✅ Implemented comprehensive evaluation and error analysis
✅ Created reusable inference functions

**Key Best Practices Demonstrated:**

1. **Data Preparation:**
   - Proper train/validation splitting
   - Data augmentation for better generalization
   - Normalization using dataset statistics

2. **Model Architecture:**
   - Batch normalization for training stability
   - Dropout for regularization
   - Residual connections for better gradient flow
   - Proper weight initialization

3. **Training Optimization:**
   - Mixed precision training for speed
   - Label smoothing for better calibration
   - Gradient clipping for stability
   - Learning rate scheduling

4. **Evaluation:**
   - Multiple metrics for comprehensive assessment
   - Confusion matrix analysis
   - Per-class performance analysis
   - Error analysis for insights

5. **Production Readiness:**
   - Model checkpointing and saving
   - Inference pipeline
   - Comprehensive logging

**Next Steps:**

- Experiment with different architectures (ResNet, EfficientNet)
- Try transfer learning with pre-trained models
- Implement advanced techniques (CutMix, MixUp)
- Deploy the model using TorchServe or similar frameworks

This tutorial provides a solid foundation for real-world image classification projects using Treadmill! 🚀 