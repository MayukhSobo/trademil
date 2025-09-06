Basic Usage Example
===================

This example demonstrates the simplest way to get started with Treadmill. Perfect for beginners who want to understand the core concepts without complexity.

Overview
--------

**What you'll learn:**
- Minimal setup for training with Treadmill
- Basic configuration options
- Simple model training workflow
- Essential metrics and callbacks

**Estimated time:** 10 minutes

Prerequisites
-------------

.. code-block:: bash

    pip install -e ".[examples]"

Simple Linear Regression
-------------------------

Let's start with the most basic example - training a simple linear regression model:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib.pyplot as plt
    import numpy as np
    
    from treadmill import Trainer, TrainingConfig
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

Step 1: Generate Synthetic Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Generate synthetic linear regression data
    def generate_data(n_samples=1000):
        """Generate synthetic data for linear regression."""
        X = torch.randn(n_samples, 1)
        # y = 3*x + 2 + noise
        y = 3 * X.squeeze() + 2 + 0.1 * torch.randn(n_samples)
        return X, y.unsqueeze(1)
    
    # Generate training and test data
    train_X, train_y = generate_data(800)
    test_X, test_y = generate_data(200)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

Step 2: Define Simple Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class SimpleLinearModel(nn.Module):
        """Simple linear regression model."""
        
        def __init__(self, input_dim=1, output_dim=1):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create model
    model = SimpleLinearModel()
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

Step 3: Basic Training Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Minimal configuration - Treadmill handles the rest!
    config = TrainingConfig(
        epochs=50,
        device="auto",  # Automatically choose GPU if available
    )
    
    print(f"Training configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Device: {config.device}")

Step 4: Train with Minimal Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create trainer - this is all you need!
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        loss_fn=nn.MSELoss()  # Mean Squared Error for regression
    )
    
    # Train the model
    print("🚀 Starting training...")
    history = trainer.train()
    print("✅ Training completed!")

Step 5: Evaluate Results
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Evaluate on test set
    test_results = trainer.evaluate(test_loader)
    print(f"\n📊 Test Results:")
    print(f"  Test Loss: {test_results['loss']:.4f}")
    
    # Get model parameters to see what it learned
    learned_weight = model.linear.weight.item()
    learned_bias = model.linear.bias.item()
    
    print(f"\n🎯 Model learned:")
    print(f"  Weight (should be ~3.0): {learned_weight:.4f}")
    print(f"  Bias (should be ~2.0): {learned_bias:.4f}")
    print(f"  True equation: y = 3*x + 2")
    print(f"  Learned equation: y = {learned_weight:.2f}*x + {learned_bias:.2f}")

Step 6: Visualize Results
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def plot_results(model, train_X, train_y, test_X, test_y, history):
        """Plot training results and model predictions."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot training history
        axes[0].plot(history['train_loss'], color='blue', linewidth=2)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Plot data and predictions
        model.eval()
        with torch.no_grad():
            # Generate points for plotting the line
            x_plot = torch.linspace(-3, 3, 100).unsqueeze(1)
            y_pred_plot = model(x_plot)
            
            # Get test predictions
            test_pred = model(test_X)
        
        # Plot training data
        axes[1].scatter(train_X.numpy(), train_y.numpy(), 
                       alpha=0.5, color='blue', label='Training Data', s=20)
        
        # Plot test data  
        axes[1].scatter(test_X.numpy(), test_y.numpy(), 
                       alpha=0.7, color='red', label='Test Data', s=20)
        
        # Plot learned line
        axes[1].plot(x_plot.numpy(), y_pred_plot.numpy(), 
                    color='green', linewidth=3, label='Learned Line')
        
        # Plot true line
        true_y = 3 * x_plot.squeeze() + 2
        axes[1].plot(x_plot.numpy(), true_y.numpy(), 
                    color='orange', linewidth=2, linestyle='--', label='True Line')
        
        axes[1].set_title('Model Predictions')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Create the visualization
    plot_results(model, train_X, train_y, test_X, test_y, history)

Basic Classification Example
-----------------------------

Now let's see a basic classification example:

.. code-block:: python

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

Step 1: Generate Classification Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(torch.unique(y_train))}")
    print(f"Training samples: {len(train_dataset)}")

Step 2: Simple Classification Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class SimpleClassifier(nn.Module):
        """Simple neural network for classification."""
        
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, num_classes)
            )
        
        def forward(self, x):
            return self.classifier(x)
    
    # Create model
    model = SimpleClassifier(input_dim=10, num_classes=3)

Step 3: Add Custom Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def accuracy(predictions, targets):
        """Calculate classification accuracy."""
        pred_classes = torch.argmax(predictions, dim=1)
        return (pred_classes == targets).float().mean().item()
    
    # Custom metrics
    custom_metrics = {'accuracy': accuracy}

Step 4: Train Classification Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Configuration for classification
    config = TrainingConfig(
        epochs=30,
        device="auto",
        early_stopping_patience=5  # Stop if no improvement for 5 epochs
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=test_loader,  # Use test as validation for this example
        loss_fn=nn.CrossEntropyLoss(),
        metric_fns=custom_metrics
    )
    
    # Train
    print("🚀 Starting classification training...")
    history = trainer.train()
    
    # Evaluate
    test_results = trainer.evaluate(test_loader)
    print(f"\n📊 Final Results:")
    print(f"  Test Loss: {test_results['loss']:.4f}")
    print(f"  Test Accuracy: {test_results['accuracy']:.4f}")

Key Takeaways - Basic Usage
---------------------------

**🎯 Minimal Code Required:**

.. code-block:: python

    # This is literally all you need!
    from treadmill import Trainer, TrainingConfig
    
    trainer = Trainer(
        model=your_model,
        config=TrainingConfig(epochs=50),
        train_dataloader=train_loader,
        loss_fn=loss_function
    )
    
    history = trainer.train()

**🚀 What Treadmill Handles Automatically:**

- ✅ **Device management** (CPU/GPU detection)
- ✅ **Training loops** (forward pass, backward pass, optimization)
- ✅ **Progress tracking** (beautiful progress bars)
- ✅ **Metrics computation** (loss tracking)
- ✅ **Model evaluation** (validation loops)
- ✅ **History tracking** (training curves)

**📊 Default Features You Get:**

- **Automatic mixed precision** (if GPU available)
- **Progress bars** with ETA and metrics
- **Training history** for plotting
- **Model evaluation** methods
- **Checkpointing** capabilities
- **Early stopping** (if configured)

**⚙️ Common Configuration Options:**

.. code-block:: python

    config = TrainingConfig(
        epochs=100,                    # Number of training epochs
        device="auto",                 # "auto", "cpu", "cuda"
        validation_frequency=1,        # Validate every N epochs
        early_stopping_patience=10,    # Stop if no improvement
        checkpoint_dir="./models",     # Where to save models
        save_best_model=True          # Save best performing model
    )

**🔄 Typical Workflow:**

1. **Prepare data** → Create DataLoader
2. **Define model** → Standard PyTorch nn.Module
3. **Configure training** → TrainingConfig
4. **Create trainer** → Trainer class
5. **Train model** → trainer.train()
6. **Evaluate** → trainer.evaluate()

That's it! Treadmill makes PyTorch training as simple as possible while giving you all the power you need. 🏃‍♀️‍➡️

Next Steps
----------

Ready for more advanced features? Check out:

- :doc:`advanced_usage` - Advanced training techniques
- :doc:`simple_dnn` - Deep neural networks
- :doc:`encoder_decoder` - Sequence-to-sequence models
- :doc:`mnist` - Complete image classification example 