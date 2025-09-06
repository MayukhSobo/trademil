Simple DNN Usage Example
========================

This example demonstrates basic Deep Neural Network (DNN) usage with Treadmill. We'll cover fundamental multi-layer perceptrons for different tasks, focusing on core concepts and straightforward implementations.

Overview
--------

**What you'll learn:**
- Multi-Layer Perceptron (MLP) basics
- Simple deep networks for classification and regression
- Basic regularization techniques
- Essential DNN training patterns

**Use cases covered:**
- Tabular data classification
- Simple regression tasks
- Multi-class classification

**Estimated time:** 15-20 minutes

Prerequisites
-------------

.. code-block:: bash

    pip install -e ".[examples]"
    pip install scikit-learn

Basic Multi-Layer Perceptron
-----------------------------

Let's start with a simple DNN architecture:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    from treadmill import Trainer, TrainingConfig
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

Step 1: Simple DNN Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class SimpleDNN(nn.Module):
        """Basic Deep Neural Network with multiple hidden layers."""
        
        def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
            """
            Args:
                input_dim: Number of input features
                hidden_dims: List of hidden layer sizes [128, 64, 32]
                output_dim: Number of output classes/values
                dropout_rate: Dropout probability for regularization
            """
            super().__init__()
            
            # Build layers
            layers = []
            prev_dim = input_dim
            
            # Hidden layers
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim
            
            # Output layer
            layers.append(nn.Linear(prev_dim, output_dim))
            
            self.network = nn.Sequential(*layers)
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights properly."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            return self.network(x)

Step 2: Binary Classification with DNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create binary classification dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )
    
    # Split and standardize
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Binary Classification Dataset:")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

Step 3: Create and Train DNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create DNN model
    model = SimpleDNN(
        input_dim=20,
        hidden_dims=[128, 64, 32],  # Three hidden layers
        output_dim=2,               # Binary classification
        dropout_rate=0.3
    )
    
    # Simple accuracy metric
    def accuracy(predictions, targets):
        pred_classes = torch.argmax(predictions, dim=1)
        return (pred_classes == targets).float().mean().item()
    
    # Basic training configuration
    config = TrainingConfig(
        epochs=50,
        device="auto",
        early_stopping_patience=10
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
    print("🚀 Training Binary Classification DNN...")
    history = trainer.fit()
    
    # Evaluate
    results = trainer.evaluate(test_loader)
    print(f"\n📊 Results:")
    print(f"  Test Loss: {results['loss']:.4f}")
    print(f"  Test Accuracy: {results['accuracy']:.4f}")

Step 4: Multi-Class Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create multi-class dataset
    X_multi, y_multi = make_classification(
        n_samples=3000,
        n_features=25,
        n_informative=20,
        n_classes=5,  # 5-class classification
        random_state=42
    )
    
    # Prepare data
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    
    scaler_multi = StandardScaler()
    X_train_multi = scaler_multi.fit_transform(X_train_multi)
    X_test_multi = scaler_multi.transform(X_test_multi)
    
    # Create datasets
    train_multi_dataset = TensorDataset(
        torch.FloatTensor(X_train_multi), 
        torch.LongTensor(y_train_multi)
    )
    test_multi_dataset = TensorDataset(
        torch.FloatTensor(X_test_multi), 
        torch.LongTensor(y_test_multi)
    )
    
    train_multi_loader = DataLoader(train_multi_dataset, batch_size=64, shuffle=True)
    test_multi_loader = DataLoader(test_multi_dataset, batch_size=64, shuffle=False)
    
    print(f"\nMulti-Class Classification Dataset:")
    print(f"  Features: {X_train_multi.shape[1]}")
    print(f"  Classes: 5")
    print(f"  Training samples: {len(X_train_multi)}")

Step 5: Deeper Network for Multi-Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create deeper DNN for more complex task
    deeper_model = SimpleDNN(
        input_dim=25,
        hidden_dims=[256, 128, 64, 32],  # Four hidden layers
        output_dim=5,                    # 5 classes
        dropout_rate=0.4
    )
    
    # Training configuration with more epochs for deeper network
    deeper_config = TrainingConfig(
        epochs=80,
        device="auto",
        early_stopping_patience=15,
        validation_frequency=1
    )
    
    # Create trainer
    deeper_trainer = Trainer(
        model=deeper_model,
        config=deeper_config,
        train_dataloader=train_multi_loader,
        val_dataloader=test_multi_loader,
        loss_fn=nn.CrossEntropyLoss(),
        metric_fns={'accuracy': accuracy}
    )
    
    # Train deeper model
    print("🚀 Training Multi-Class DNN...")
    deeper_history = deeper_trainer.fit()
    
    # Evaluate deeper model
    deeper_results = deeper_trainer.evaluate(test_multi_loader)
    print(f"\n📊 Multi-Class Results:")
    print(f"  Test Loss: {deeper_results['loss']:.4f}")
    print(f"  Test Accuracy: {deeper_results['accuracy']:.4f}")

Step 6: Regression with DNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class RegressionDNN(nn.Module):
        """DNN optimized for regression tasks."""
        
        def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim
            
            # Output layer (no activation for regression)
            layers.append(nn.Linear(prev_dim, 1))
            
            self.network = nn.Sequential(*layers)
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            return self.network(x)
    
    # Create regression dataset
    X_reg, y_reg = make_regression(
        n_samples=2500,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    # Prepare regression data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    scaler_X_reg = StandardScaler()
    scaler_y_reg = StandardScaler()
    
    X_train_reg = scaler_X_reg.fit_transform(X_train_reg)
    X_test_reg = scaler_X_reg.transform(X_test_reg)
    
    y_train_reg = scaler_y_reg.fit_transform(y_train_reg.reshape(-1, 1)).flatten()
    y_test_reg = scaler_y_reg.transform(y_test_reg.reshape(-1, 1)).flatten()
    
    # Create regression datasets
    train_reg_dataset = TensorDataset(
        torch.FloatTensor(X_train_reg), 
        torch.FloatTensor(y_train_reg).unsqueeze(1)
    )
    test_reg_dataset = TensorDataset(
        torch.FloatTensor(X_test_reg), 
        torch.FloatTensor(y_test_reg).unsqueeze(1)
    )
    
    train_reg_loader = DataLoader(train_reg_dataset, batch_size=64, shuffle=True)
    test_reg_loader = DataLoader(test_reg_dataset, batch_size=64, shuffle=False)
    
    print(f"\nRegression Dataset:")
    print(f"  Features: {X_train_reg.shape[1]}")
    print(f"  Training samples: {len(X_train_reg)}")

Step 7: Train Regression DNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create regression model
    reg_model = RegressionDNN(
        input_dim=15,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3
    )
    
    # Regression metrics
    def mse(predictions, targets):
        return F.mse_loss(predictions, targets).item()
    
    def mae(predictions, targets):
        return F.l1_loss(predictions, targets).item()
    
    # Training configuration
    reg_config = TrainingConfig(
        epochs=60,
        device="auto",
        early_stopping_patience=12
    )
    
    # Create trainer
    reg_trainer = Trainer(
        model=reg_model,
        config=reg_config,
        train_dataloader=train_reg_loader,
        val_dataloader=test_reg_loader,
        loss_fn=nn.MSELoss(),
        metric_fns={'mse': mse, 'mae': mae}
    )
    
    # Train regression model
    print("🚀 Training Regression DNN...")
    reg_history = reg_trainer.fit()
    
    # Evaluate regression model
    reg_results = reg_trainer.evaluate(test_reg_loader)
    print(f"\n📊 Regression Results:")
    print(f"  Test MSE: {reg_results['mse']:.4f}")
    print(f"  Test MAE: {reg_results['mae']:.4f}")

Step 8: Understanding DNN Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def analyze_dnn_architecture(model, input_dim):
        """Analyze DNN architecture and parameters."""
        
        print(f"\n🔍 DNN Architecture Analysis:")
        print(f"  Input dimension: {input_dim}")
        
        total_params = 0
        layer_count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_count += 1
                params = sum(p.numel() for p in module.parameters())
                total_params += params
                print(f"  Layer {layer_count}: {module.in_features} → {module.out_features} ({params:,} params)")
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Total layers: {layer_count}")
        
        return total_params, layer_count
    
    # Analyze our models
    print("\n" + "="*50)
    print("Model Architecture Analysis")
    print("="*50)
    
    analyze_dnn_architecture(model, 20)
    analyze_dnn_architecture(deeper_model, 25)
    analyze_dnn_architecture(reg_model, 15)

Visualization and Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def plot_training_history(histories, titles):
        """Plot training histories for comparison."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['blue', 'red', 'green']
        
        # Plot losses
        for i, (history, title, color) in enumerate(zip(histories, titles, colors)):
            if 'train_loss' in history:
                axes[0].plot(history['train_loss'], 
                           label=f'{title} - Train', color=color, linestyle='-')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], 
                           label=f'{title} - Val', color=color, linestyle='--')
        
        axes[0].set_title('Training Loss Comparison')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracies (if available)
        for i, (history, title, color) in enumerate(zip(histories, titles, colors)):
            if 'train_accuracy' in history:
                axes[1].plot(history['train_accuracy'], 
                           label=f'{title} - Train', color=color, linestyle='-')
            if 'val_accuracy' in history:
                axes[1].plot(history['val_accuracy'], 
                           label=f'{title} - Val', color=color, linestyle='--')
        
        axes[1].set_title('Training Accuracy Comparison')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Compare training histories
    histories = [history, deeper_history]
    titles = ['Binary Classification', 'Multi-Class Classification']
    plot_training_history(histories, titles)

Key DNN Concepts Summary
------------------------

**🧠 What We Learned:**

✅ **Deep Neural Networks**: Multiple hidden layers for complex pattern learning
✅ **Architecture Design**: How to structure layers for different tasks
✅ **Regularization**: Using dropout to prevent overfitting  
✅ **Task Adaptation**: Different outputs for classification vs regression
✅ **Training Patterns**: Basic configurations for stable training

**📊 Architecture Guidelines:**

.. code-block:: python

    # Classification architecture pattern
    classification_dnn = SimpleDNN(
        input_dim=features,
        hidden_dims=[256, 128, 64],    # Decreasing sizes
        output_dim=num_classes,
        dropout_rate=0.3
    )
    
    # Regression architecture pattern  
    regression_dnn = RegressionDNN(
        input_dim=features,
        hidden_dims=[128, 64, 32],     # Smaller for regression
        dropout_rate=0.2               # Less aggressive dropout
    )

**⚙️ Training Best Practices:**

1. **Start Simple**: Begin with 2-3 hidden layers
2. **Scale Gradually**: Add layers/neurons based on complexity
3. **Use Dropout**: Prevent overfitting with 0.2-0.4 dropout
4. **Early Stopping**: Stop when validation performance plateaus
5. **Standardize Data**: Always normalize input features

**🎯 When to Use DNNs:**

- **Tabular Data**: Structured data with many features
- **Non-Linear Patterns**: Complex relationships in data
- **Medium Datasets**: 1K-100K samples (not too small/large)
- **Mixed Data Types**: Numerical features with interactions

This example shows how Treadmill makes deep neural network training straightforward while maintaining flexibility for different tasks! 🏃‍♀️‍➡️

Next Steps
----------

Ready for more? Check out:

- :doc:`advanced_usage` - Advanced training techniques and optimizations
- :doc:`mnist` - Convolutional networks for image data
- :doc:`encoder_decoder` - Sequence-to-sequence architectures 