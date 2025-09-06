MNIST Convolutional Networks
============================

This example demonstrates how to use convolutional neural networks (CNNs) for image classification with Treadmill. We'll build a simple CNN for MNIST digit recognition, focusing on fundamental CNN concepts.

Overview
--------

**What you'll learn:**
- Basic convolutional neural network architecture
- Simple CNN layers (Conv2d, MaxPool2d)
- Image data handling with Treadmill
- Basic image classification workflow

**Estimated time:** 15 minutes

Prerequisites
-------------

.. code-block:: bash

    pip install -e ".[examples]"

Simple CNN for MNIST
---------------------

Let's build a basic CNN for handwritten digit recognition:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np

    from treadmill import Trainer, TrainingConfig
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

Step 1: Load MNIST Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Simple transforms for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

Step 2: Simple CNN Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class SimpleCNN(nn.Module):
        """Simple Convolutional Neural Network for MNIST."""
        
        def __init__(self):
            super(SimpleCNN, self).__init__()
            
            # Convolutional layers
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
            
            # Pooling layer
            self.pool = nn.MaxPool2d(2, 2)  # Halves the spatial dimensions
            
            # Fully connected layers
            self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 7x7 after two pooling operations
            self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9
            
            # Dropout for regularization
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            # First conv block: 28x28x1 -> 14x14x32
            x = self.pool(F.relu(self.conv1(x)))
            
            # Second conv block: 14x14x32 -> 7x7x64
            x = self.pool(F.relu(self.conv2(x)))
            
            # Flatten: 7x7x64 -> 3136
            x = x.view(-1, 64 * 7 * 7)
            
            # Fully connected layers
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x

    # Create model
    model = SimpleCNN()
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

Step 3: Visualize Sample Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def show_samples(dataset, num_samples=8):
        """Show sample images from the dataset."""
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        fig.suptitle('MNIST Sample Images')
        
        for i in range(num_samples):
            image, label = dataset[i]
            
            # Convert tensor to numpy and denormalize
            image_np = image.squeeze().numpy()
            image_np = image_np * 0.3081 + 0.1307  # Denormalize
            
            ax = axes[i // 4, i % 4]
            ax.imshow(image_np, cmap='gray')
            ax.set_title(f'Label: {label}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    # Show some samples
    show_samples(train_dataset)

Step 4: Define Simple Accuracy Metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def accuracy(predictions, targets):
        """Calculate classification accuracy."""
        pred_classes = torch.argmax(predictions, dim=1)
        return (pred_classes == targets).float().mean().item()

Step 5: Train the CNN
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Simple training configuration
    config = TrainingConfig(
        epochs=10,
        device="auto",
        early_stopping_patience=3
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        loss_fn=nn.CrossEntropyLoss(),
        metric_fns={'accuracy': accuracy}
    )

    # Train the model
    print("🚀 Training CNN on MNIST...")
    history = trainer.fit()

    # Evaluate on test set
    test_results = trainer.evaluate(test_loader)
    print(f"\n📊 Test Results:")
    print(f"  Test Loss: {test_results['loss']:.4f}")
    print(f"  Test Accuracy: {test_results['accuracy']:.4f}")

Step 6: Visualize Training Progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def plot_training_history(history):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        axes[0].plot(history['train_loss'], label='Training Loss', color='blue')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'train_accuracy' in history:
            axes[1].plot(history['train_accuracy'], label='Training Accuracy', color='blue')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    # Plot the training history
    plot_training_history(history)

Step 7: Test Individual Predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def test_predictions(model, test_dataset, num_samples=8):
        """Test model predictions on individual samples."""
        model.eval()
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle('Model Predictions')
        
        with torch.no_grad():
            for i in range(num_samples):
                # Get sample
                image, true_label = test_dataset[i]
                
                # Make prediction
                image_batch = image.unsqueeze(0)  # Add batch dimension
                output = model(image_batch)
                predicted_label = torch.argmax(output, dim=1).item()
                confidence = F.softmax(output, dim=1).max().item()
                
                # Plot
                ax = axes[i // 4, i % 4]
                
                # Denormalize image for display
                image_np = image.squeeze().numpy()
                image_np = image_np * 0.3081 + 0.1307
                
                ax.imshow(image_np, cmap='gray')
                
                # Color code: green if correct, red if wrong
                color = 'green' if predicted_label == true_label else 'red'
                ax.set_title(f'True: {true_label}, Pred: {predicted_label}\n'
                           f'Confidence: {confidence:.2f}', color=color)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    # Test some predictions
    test_predictions(model, test_dataset)

Understanding CNN Components
----------------------------

**🧠 What Each Layer Does:**

.. code-block:: python

    def explain_cnn_layers():
        """Explain CNN layer transformations."""
        print("CNN Layer Analysis:")
        print("==================")
        print("Input: 1 x 28 x 28 (1 channel, 28x28 pixels)")
        print()
        print("Conv1 + Pool1:")
        print("  Conv2d(1 → 32): 1x28x28 → 32x28x28")
        print("  MaxPool2d:       32x28x28 → 32x14x14")
        print()
        print("Conv2 + Pool2:")
        print("  Conv2d(32 → 64): 32x14x14 → 64x14x14") 
        print("  MaxPool2d:        64x14x14 → 64x7x7")
        print()
        print("Flatten:")
        print("  Reshape: 64x7x7 → 3136")
        print()
        print("Fully Connected:")
        print("  Linear: 3136 → 128 → 10")

    explain_cnn_layers()

**🎯 Key CNN Concepts:**

.. code-block:: python

    # Basic CNN building blocks
    """
    Convolution (nn.Conv2d):
    - Detects features like edges, shapes
    - Preserves spatial relationships
    - kernel_size: size of the filter
    - padding: adds zeros around input
    
    Pooling (nn.MaxPool2d):
    - Reduces spatial dimensions
    - Makes model translation invariant
    - Reduces computational cost
    
    Activation (F.relu):
    - Adds non-linearity
    - Allows learning complex patterns
    
    Fully Connected (nn.Linear):
    - Combines all features for classification
    - Maps to output classes
    """

Simple Model Variations
-----------------------

**🔧 Deeper CNN:**

.. code-block:: python

    class DeeperCNN(nn.Module):
        """Deeper CNN with more layers."""
        
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            
            self.fc1 = nn.Linear(64 * 3 * 3, 128)  # After 3 pooling: 28->14->7->3
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 28->14
            x = self.pool(F.relu(self.conv2(x)))  # 14->7
            x = self.pool(F.relu(self.conv3(x)))  # 7->3
            
            x = x.view(-1, 64 * 3 * 3)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

**🔧 CNN with Batch Normalization:**

.. code-block:: python

    class BatchNormCNN(nn.Module):
        """CNN with batch normalization for stable training."""
        
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            
            self.pool = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            
        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            
            x = x.view(-1, 64 * 7 * 7)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

Key Takeaways
-------------

**🎯 CNN Basics:**

✅ **Convolution**: Feature detection with learnable filters
✅ **Pooling**: Spatial dimension reduction and translation invariance  
✅ **Architecture**: Conv layers → Pooling → Fully connected
✅ **MNIST Performance**: Simple CNNs achieve ~98-99% accuracy

**📊 CNN vs Dense Networks:**

- **CNNs**: Better for images, preserve spatial relationships
- **Dense**: Better for tabular data, fully connected layers
- **Parameters**: CNNs usually have fewer parameters for images
- **Translation**: CNNs handle shifted/rotated images better

**⚙️ Training Tips:**

1. **Start Simple**: Begin with 2-3 conv layers
2. **Use Pooling**: Reduce dimensions progressively  
3. **Add Dropout**: Prevent overfitting in FC layers
4. **Normalize Data**: Always normalize input images
5. **Monitor Validation**: Watch for overfitting

This basic CNN example shows how Treadmill makes convolutional network training simple and straightforward! 🏃‍♀️‍➡️

Next Steps
----------

Ready for more advanced techniques? Check out:

- :doc:`advanced_usage` - Advanced CNN architectures and training techniques
- :doc:`../tutorials/image_classification` - Complete image classification project
- :doc:`encoder_decoder` - Different architecture patterns 