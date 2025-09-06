MNIST Digit Classification
==========================

This example demonstrates how to build a complete MNIST digit classification system using Treadmill. We'll walk through every step from data loading to model evaluation.

Overview
--------

**What we'll build:**
- A convolutional neural network for handwritten digit recognition
- Complete training pipeline with validation and testing
- Comprehensive evaluation and visualization

**Key features demonstrated:**
- Custom model architecture
- Data augmentation techniques
- Training with callbacks
- Model evaluation and visualization

Prerequisites
-------------

.. code-block:: bash

    pip install -e ".[full]"
    pip install matplotlib seaborn

Complete Implementation
-----------------------

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, random_split
    import matplotlib.pyplot as plt
    import numpy as np

    from treadmill import Trainer, TrainingConfig, OptimizerConfig
    from treadmill.callbacks import EarlyStopping, ModelCheckpoint

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

Step 1: Data Preparation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),           # Random rotation ±10 degrees
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Small translations
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )

    # Split training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")

Step 2: Visualize Data
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def visualize_samples(dataset, num_samples=12):
        """Visualize random samples from the dataset."""
        fig, axes = plt.subplots(3, 4, figsize=(10, 8))
        fig.suptitle('MNIST Sample Images', fontsize=16)
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            image, label = dataset[idx]
            
            # Convert tensor to numpy and denormalize
            if isinstance(image, torch.Tensor):
                image_np = image.squeeze().numpy()
                image_np = image_np * 0.3081 + 0.1307  # Denormalize
            
            ax = axes[i // 4, i % 4]
            ax.imshow(image_np, cmap='gray')
            ax.set_title(f'Label: {label}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    # Visualize some training samples
    visualize_samples(train_subset)

Step 3: Define Model Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class MNISTNet(nn.Module):
        """
        Convolutional Neural Network for MNIST digit classification.
        
        Architecture:
        - Two convolutional layers with ReLU activation
        - Max pooling after each conv layer
        - Dropout for regularization
        - Two fully connected layers
        """
        
        def __init__(self, num_classes=10, dropout_rate=0.5):
            super(MNISTNet, self).__init__()
            
            # Convolutional layers
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            
            # Pooling layer
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Fully connected layers
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)
            
            # Dropout
            self.dropout = nn.Dropout(dropout_rate)
            
        def forward(self, x):
            # First conv block: 28x28 -> 14x14
            x = self.pool(F.relu(self.conv1(x)))
            
            # Second conv block: 14x14 -> 7x7
            x = self.pool(F.relu(self.conv2(x)))
            
            # Flatten: 7x7x64 -> 3136
            x = x.view(-1, 64 * 7 * 7)
            
            # Fully connected layers with dropout
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x

    # Create model instance
    model = MNISTNet(num_classes=10, dropout_rate=0.3)

    # Print model summary
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model has {count_parameters(model):,} trainable parameters")

Step 4: Define Custom Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def accuracy(predictions, targets):
        """Calculate accuracy."""
        pred_classes = torch.argmax(predictions, dim=1)
        return (pred_classes == targets).float().mean().item()

    def top_2_accuracy(predictions, targets):
        """Calculate top-2 accuracy."""
        _, top_2_preds = torch.topk(predictions, 2, dim=1)
        targets_expanded = targets.view(-1, 1).expand_as(top_2_preds)
        correct = (top_2_preds == targets_expanded).any(dim=1)
        return correct.float().mean().item()

    # Custom metrics dictionary
    custom_metrics = {
        'accuracy': accuracy,
        'top2_accuracy': top_2_accuracy
    }

Step 5: Training Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Optimizer configuration
    optimizer_config = OptimizerConfig(
        optimizer_class="Adam",
        lr=0.001,
        weight_decay=1e-4
    )

    # Training configuration
    config = TrainingConfig(
        epochs=20,
        device="auto",
        mixed_precision=True,
        
        # Validation settings
        validation_frequency=1,
        log_frequency=100,
        
        # Early stopping
        early_stopping_patience=5,
        early_stopping_min_delta=0.001,
        
        # Checkpointing
        checkpoint_dir="./checkpoints/mnist",
        save_best_model=True,
        save_last_model=True,
        
        # Optimizer
        optimizer=optimizer_config
    )

Step 6: Setup Callbacks
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from treadmill.callbacks import EarlyStopping, ModelCheckpoint

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        min_delta=0.0001,
        verbose=True,
        mode='min'
    )

    # Model checkpoint callback
    model_checkpoint = ModelCheckpoint(
        filepath='./checkpoints/mnist/best_model_{epoch:02d}_{val_acc:.4f}.pt',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=True
    )

    callbacks = [early_stopping, model_checkpoint]

Step 7: Training
^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        metric_fns=custom_metrics,
        callbacks=callbacks
    )

    # Display training info
    print("🚀 Starting MNIST Training")
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print("-" * 50)

    # Train the model
    history = trainer.fit()

    print("✅ Training completed!")

Step 8: Visualize Training Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def plot_training_history(history):
        """Plot training and validation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_accuracy'], label='Training Accuracy', color='blue')
        if 'val_accuracy' in history:
            axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-2 Accuracy plot
        axes[1, 0].plot(history['train_top2_accuracy'], label='Training Top-2', color='blue')
        if 'val_top2_accuracy' in history:
            axes[1, 0].plot(history['val_top2_accuracy'], label='Validation Top-2', color='red')
        axes[1, 0].set_title('Top-2 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training progress summary
        axes[1, 1].text(0.1, 0.9, f"Final Training Loss: {history['train_loss'][-1]:.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.8, f"Final Validation Loss: {history['val_loss'][-1]:.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"Best Training Accuracy: {max(history['train_accuracy']):.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Best Validation Accuracy: {max(history['val_accuracy']):.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

    # Plot the training history
    plot_training_history(history)

Step 9: Model Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_loader)

    print("\n🎯 Test Results:")
    for metric_name, value in test_results.items():
        print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")

Step 10: Confusion Matrix and Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    def detailed_evaluation(trainer, test_loader):
        """Generate detailed evaluation including confusion matrix."""
        
        all_predictions = []
        all_targets = []
        
        trainer.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(trainer.device)
                output = trainer.model(data)
                pred = torch.argmax(output, dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.numpy())
        
        # Classification report
        class_names = [str(i) for i in range(10)]
        print("\n📋 Classification Report:")
        print(classification_report(all_targets, all_predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        return cm

    # Generate detailed evaluation
    confusion_mat = detailed_evaluation(trainer, test_loader)

Step 11: Visualize Misclassified Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def visualize_mistakes(trainer, test_loader, num_examples=12):
        """Visualize misclassified examples."""
        
        mistakes = []
        trainer.model.eval()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(trainer.device), target.to(trainer.device)
                output = trainer.model(data)
                pred = torch.argmax(output, dim=1)
                probs = F.softmax(output, dim=1)
                
                # Find misclassified examples
                mask = pred != target
                if mask.any():
                    for i in range(data.size(0)):
                        if mask[i] and len(mistakes) < num_examples:
                            mistakes.append({
                                'image': data[i].cpu(),
                                'true': target[i].item(),
                                'pred': pred[i].item(),
                                'confidence': probs[i][pred[i]].item()
                            })
                
                if len(mistakes) >= num_examples:
                    break
        
        # Plot misclassified examples
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle('Misclassified Examples', fontsize=16, fontweight='bold')
        
        for i, mistake in enumerate(mistakes):
            ax = axes[i // 4, i % 4]
            
            # Denormalize image
            image = mistake['image'].squeeze()
            image = image * 0.3081 + 0.1307
            
            ax.imshow(image, cmap='gray')
            ax.set_title(f"True: {mistake['true']}, Pred: {mistake['pred']}\n"
                        f"Confidence: {mistake['confidence']:.2f}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    # Visualize mistakes
    visualize_mistakes(trainer, test_loader)

Step 12: Model Saving and Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Save the complete model
    model_path = './models/mnist_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {'num_classes': 10, 'dropout_rate': 0.3},
        'training_history': history,
        'test_results': test_results
    }, model_path)

    print(f"✅ Model saved to {model_path}")

    # Function to load and use the saved model
    def load_model(model_path, device='cpu'):
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location=device)
        
        model = MNISTNet(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, checkpoint

    # Example usage
    # loaded_model, checkpoint = load_model(model_path)

Step 13: Inference Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def predict_digit(model, image_tensor, device='cpu'):
        """Predict digit from image tensor."""
        model.eval()
        with torch.no_grad():
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            
            predicted_digit = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_digit].item()
            
            return predicted_digit, confidence, probabilities[0]

    # Example inference (using a test sample)
    # test_image, test_label = test_dataset[0]
    # predicted_digit, confidence, probs = predict_digit(model, test_image, trainer.device)
    # print(f"Predicted: {predicted_digit}, Actual: {test_label}, Confidence: {confidence:.4f}")

Summary
-------

**What we accomplished:**

✅ Built a complete MNIST digit classification system
✅ Implemented CNN with proper architecture
✅ Used data augmentation for better generalization
✅ Applied early stopping and model checkpointing
✅ Performed comprehensive evaluation with visualizations
✅ Created reusable inference functions

**Key Results:**

- **Test Accuracy**: ~98-99% (typical for MNIST)
- **Model Size**: ~100K parameters (lightweight and efficient)
- **Training Time**: ~5-10 minutes on GPU

**Best Practices Demonstrated:**

1. **Data Preparation**: Proper normalization and augmentation
2. **Model Architecture**: Simple but effective CNN design
3. **Training Process**: Early stopping, checkpointing, validation
4. **Evaluation**: Confusion matrix, error analysis, visualizations
5. **Deployment**: Model saving and inference pipeline

This example provides a solid foundation for digit recognition tasks and can be easily extended to other image classification problems! 🚀

Next Steps
----------

- Try different architectures (ResNet, DenseNet)
- Experiment with other datasets (Fashion-MNIST, EMNIST)
- Implement ensemble methods
- Deploy the model as a web service 