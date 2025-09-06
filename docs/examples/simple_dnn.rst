Simple DNN Usage Example
========================

This example demonstrates how to build and train Deep Neural Networks (DNNs) for various tasks using Treadmill. We'll cover different architectures and use cases, from basic MLPs to more sophisticated deep networks.

Overview
--------

**What you'll learn:**
- Multi-Layer Perceptron (MLP) architectures
- Deep neural networks for different data types
- Regularization techniques (dropout, batch norm)
- Different activation functions and optimizers
- Handling various loss functions and metrics

**Use cases covered:**
- Tabular data classification and regression
- Feature learning and dimensionality reduction
- Multi-output and multi-task learning

**Estimated time:** 25-35 minutes

Prerequisites
-------------

.. code-block:: bash

    pip install -e ".[examples]"
    pip install scikit-learn pandas

Multi-Layer Perceptron (MLP) Basics
------------------------------------

Let's start with building various MLP architectures:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    from treadmill import Trainer, TrainingConfig, OptimizerConfig
    from treadmill.callbacks import EarlyStopping, ModelCheckpoint
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

Simple MLP Architecture
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class SimpleMLP(nn.Module):
        """Basic Multi-Layer Perceptron."""
        
        def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
            """
            Args:
                input_dim: Number of input features
                hidden_dims: List of hidden layer dimensions
                output_dim: Number of output classes/values
                dropout_rate: Dropout probability
            """
            super().__init__()
            
            # Build layers dynamically
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim
            
            # Output layer (no activation - will be handled by loss function)
            layers.append(nn.Linear(prev_dim, output_dim))
            
            self.network = nn.Sequential(*layers)
            self._initialize_weights()
        
        def _initialize_weights(self):
            """Initialize weights using Xavier/He initialization."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            return self.network(x)

Example 1: Binary Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def create_binary_classification_data():
        """Create synthetic binary classification dataset."""
        
        X, y = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            class_sep=1.5,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, scaler
    
    # Create data
    X_train, X_test, y_train, y_test, scaler = create_binary_classification_data()
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"Binary Classification Dataset:")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Class distribution: {np.bincount(y_train)}")

Model Training for Binary Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create model
    model = SimpleMLP(
        input_dim=20,
        hidden_dims=[128, 64, 32],  # 3 hidden layers
        output_dim=2,  # Binary classification
        dropout_rate=0.3
    )
    
    # Define metrics
    def accuracy(predictions, targets):
        pred_classes = torch.argmax(predictions, dim=1)
        return (pred_classes == targets).float().mean().item()
    
    def precision(predictions, targets):
        pred_classes = torch.argmax(predictions, dim=1)
        tp = ((pred_classes == 1) & (targets == 1)).float().sum()
        fp = ((pred_classes == 1) & (targets == 0)).float().sum()
        return (tp / (tp + fp + 1e-8)).item()
    
    def recall(predictions, targets):
        pred_classes = torch.argmax(predictions, dim=1)
        tp = ((pred_classes == 1) & (targets == 1)).float().sum()
        fn = ((pred_classes == 0) & (targets == 1)).float().sum()
        return (tp / (tp + fn + 1e-8)).item()
    
    custom_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
    
    # Training configuration
    config = TrainingConfig(
        epochs=100,
        device="auto",
        early_stopping_patience=15,
        validation_frequency=1
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        loss_fn=nn.CrossEntropyLoss(),
        metric_fns=custom_metrics
    )
    
    # Train
    print("🚀 Training Binary Classification MLP...")
    history = trainer.fit()
    
    # Evaluate
    test_results = trainer.evaluate(test_loader)
    print(f"\n📊 Binary Classification Results:")
    for metric, value in test_results.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

Example 2: Regression with Deep Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class RegressionMLP(nn.Module):
        """MLP optimized for regression tasks."""
        
        def __init__(self, input_dim, hidden_dims, output_dim=1, 
                     activation='relu', use_batch_norm=True, dropout_rate=0.2):
            super().__init__()
            
            # Activation function
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU(0.1)
            elif activation == 'elu':
                self.activation = nn.ELU()
            else:
                self.activation = nn.ReLU()
            
            # Build network
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                
                layers.append(self.activation)
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            
            # Output layer (no activation for regression)
            layers.append(nn.Linear(prev_dim, output_dim))
            
            self.network = nn.Sequential(*layers)
            self._initialize_weights()
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            return self.network(x)
    
    def create_regression_data():
        """Create synthetic regression dataset."""
        
        X, y = make_regression(
            n_samples=8000,
            n_features=15,
            n_informative=10,
            noise=0.1,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        
        # Reshape y for scaler
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        return X_train, X_test, y_train, y_test, scaler_X, scaler_y
    
    # Create regression data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, scaler_X_reg, scaler_y_reg = create_regression_data()
    
    # Convert to tensors
    X_train_reg_tensor = torch.FloatTensor(X_train_reg)
    X_test_reg_tensor = torch.FloatTensor(X_test_reg)
    y_train_reg_tensor = torch.FloatTensor(y_train_reg).unsqueeze(1)
    y_test_reg_tensor = torch.FloatTensor(y_test_reg).unsqueeze(1)
    
    # Create data loaders
    train_reg_dataset = TensorDataset(X_train_reg_tensor, y_train_reg_tensor)
    test_reg_dataset = TensorDataset(X_test_reg_tensor, y_test_reg_tensor)
    
    train_reg_loader = DataLoader(train_reg_dataset, batch_size=128, shuffle=True)
    test_reg_loader = DataLoader(test_reg_dataset, batch_size=128, shuffle=False)
    
    print(f"\nRegression Dataset:")
    print(f"  Features: {X_train_reg.shape[1]}")
    print(f"  Training samples: {len(X_train_reg)}")
    print(f"  Test samples: {len(X_test_reg)}")

Training Regression Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create regression model
    regression_model = RegressionMLP(
        input_dim=15,
        hidden_dims=[256, 128, 64, 32],  # Deeper network
        output_dim=1,
        activation='elu',  # ELU activation for regression
        use_batch_norm=True,
        dropout_rate=0.2
    )
    
    # Regression metrics
    def mse(predictions, targets):
        return F.mse_loss(predictions, targets).item()
    
    def mae(predictions, targets):
        return F.l1_loss(predictions, targets).item()
    
    def r2_score(predictions, targets):
        # Coefficient of determination
        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        return (1 - ss_res / (ss_tot + 1e-8)).item()
    
    regression_metrics = {
        'mse': mse,
        'mae': mae,
        'r2_score': r2_score
    }
    
    # Training configuration for regression
    reg_config = TrainingConfig(
        epochs=150,
        device="auto",
        early_stopping_patience=20,
        validation_frequency=1
    )
    
    # Create regression trainer
    reg_trainer = Trainer(
        model=regression_model,
        config=reg_config,
        train_dataloader=train_reg_loader,
        val_dataloader=test_reg_loader,
        loss_fn=nn.MSELoss(),
        metric_fns=regression_metrics
    )
    
    # Train regression model
    print("🚀 Training Regression MLP...")
    reg_history = reg_trainer.fit()
    
    # Evaluate regression model
    reg_test_results = reg_trainer.evaluate(test_reg_loader)
    print(f"\n📊 Regression Results:")
    for metric, value in reg_test_results.items():
        print(f"  {metric.upper()}: {value:.4f}")

Example 3: Multi-Class Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class AdvancedMLP(nn.Module):
        """Advanced MLP with various architectural improvements."""
        
        def __init__(self, input_dim, hidden_dims, output_dim, 
                     activation='relu', use_batch_norm=True, 
                     use_residual=True, dropout_rate=0.3):
            super().__init__()
            
            self.use_residual = use_residual and len(hidden_dims) > 1
            
            # Input layer
            self.input_layer = nn.Linear(input_dim, hidden_dims[0])
            self.input_bn = nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity()
            
            # Hidden layers
            self.hidden_layers = nn.ModuleList()
            self.hidden_bns = nn.ModuleList()
            
            for i in range(len(hidden_dims) - 1):
                self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                if use_batch_norm:
                    self.hidden_bns.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                else:
                    self.hidden_bns.append(nn.Identity())
            
            # Output layer
            self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
            self.dropout = nn.Dropout(dropout_rate)
            
            # Activation function
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU(0.1)
            elif activation == 'gelu':
                self.activation = nn.GELU()
            else:
                self.activation = nn.ReLU()
            
            self._initialize_weights()
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            # Input layer
            x = self.activation(self.input_bn(self.input_layer(x)))
            x = self.dropout(x)
            
            # Hidden layers with optional residual connections
            for i, (layer, bn) in enumerate(zip(self.hidden_layers, self.hidden_bns)):
                identity = x
                
                x = self.activation(bn(layer(x)))
                x = self.dropout(x)
                
                # Add residual connection if dimensions match
                if self.use_residual and x.shape == identity.shape:
                    x = x + identity
            
            # Output layer
            return self.output_layer(x)
    
    # Create multi-class classification data
    X_multi, y_multi = make_classification(
        n_samples=12000,
        n_features=30,
        n_informative=20,
        n_redundant=10,
        n_classes=10,  # 10-class classification
        class_sep=1.0,
        random_state=42
    )
    
    # Split and standardize
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    
    scaler_multi = StandardScaler()
    X_train_multi = scaler_multi.fit_transform(X_train_multi)
    X_test_multi = scaler_multi.transform(X_test_multi)
    
    # Convert to tensors and create loaders
    train_multi_dataset = TensorDataset(
        torch.FloatTensor(X_train_multi), 
        torch.LongTensor(y_train_multi)
    )
    test_multi_dataset = TensorDataset(
        torch.FloatTensor(X_test_multi), 
        torch.LongTensor(y_test_multi)
    )
    
    train_multi_loader = DataLoader(train_multi_dataset, batch_size=256, shuffle=True)
    test_multi_loader = DataLoader(test_multi_dataset, batch_size=256, shuffle=False)

Training Advanced Multi-Class Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create advanced model
    advanced_model = AdvancedMLP(
        input_dim=30,
        hidden_dims=[512, 256, 128, 64],  # Deep architecture
        output_dim=10,
        activation='gelu',  # GELU activation
        use_batch_norm=True,
        use_residual=True,
        dropout_rate=0.4
    )
    
    # Multi-class metrics
    def top_3_accuracy(predictions, targets):
        _, top_3_preds = torch.topk(predictions, 3, dim=1)
        targets_expanded = targets.view(-1, 1).expand_as(top_3_preds)
        correct = (top_3_preds == targets_expanded).any(dim=1)
        return correct.float().mean().item()
    
    def mean_class_accuracy(predictions, targets, num_classes=10):
        pred_classes = torch.argmax(predictions, dim=1)
        class_acc = []
        for c in range(num_classes):
            mask = targets == c
            if mask.sum() > 0:
                acc = (pred_classes[mask] == targets[mask]).float().mean().item()
                class_acc.append(acc)
        return np.mean(class_acc) if class_acc else 0.0
    
    multi_class_metrics = {
        'accuracy': accuracy,
        'top3_accuracy': top_3_accuracy,
        'mean_class_accuracy': lambda p, t: mean_class_accuracy(p, t, 10)
    }
    
    # Advanced optimizer configuration
    optimizer_config = OptimizerConfig(
        optimizer_class="AdamW",
        lr=0.001,
        weight_decay=0.01,
        params={'betas': (0.9, 0.999)}
    )
    
    # Training configuration
    multi_config = TrainingConfig(
        epochs=200,
        device="auto",
        mixed_precision=True,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        early_stopping_patience=25,
        validation_frequency=1,
        optimizer=optimizer_config
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=25,
            min_delta=0.001,
            mode='max',
            verbose=True
        ),
        ModelCheckpoint(
            filepath='./models/advanced_mlp_{epoch:03d}_{val_acc:.4f}.pt',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Create trainer
    multi_trainer = Trainer(
        model=advanced_model,
        config=multi_config,
        train_dataloader=train_multi_loader,
        val_dataloader=test_multi_loader,
        loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1),  # Label smoothing
        metric_fns=multi_class_metrics,
        callbacks=callbacks
    )
    
    # Train
    print("🚀 Training Advanced Multi-Class MLP...")
    multi_history = multi_trainer.fit()
    
    # Evaluate
    multi_test_results = multi_trainer.evaluate(test_multi_loader)
    print(f"\n📊 Multi-Class Classification Results:")
    for metric, value in multi_test_results.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

Example 4: Multi-Output Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class MultiOutputMLP(nn.Module):
        """MLP with multiple output heads for multi-task learning."""
        
        def __init__(self, input_dim, shared_dims, output_configs):
            """
            Args:
                input_dim: Input feature dimension
                shared_dims: Dimensions of shared hidden layers
                output_configs: Dict with output names and their dimensions
                               e.g., {'classification': 5, 'regression': 1}
            """
            super().__init__()
            
            # Shared layers
            shared_layers = []
            prev_dim = input_dim
            
            for dim in shared_dims:
                shared_layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                prev_dim = dim
            
            self.shared_network = nn.Sequential(*shared_layers)
            
            # Output heads
            self.output_heads = nn.ModuleDict()
            for name, output_dim in output_configs.items():
                self.output_heads[name] = nn.Sequential(
                    nn.Linear(prev_dim, prev_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(prev_dim // 2, output_dim)
                )
        
        def forward(self, x):
            # Shared feature extraction
            shared_features = self.shared_network(x)
            
            # Multiple outputs
            outputs = {}
            for name, head in self.output_heads.items():
                outputs[name] = head(shared_features)
            
            return outputs
    
    # Create multi-output data (simulating both classification and regression tasks)
    def create_multi_output_data():
        X, _ = make_classification(
            n_samples=8000,
            n_features=25,
            n_informative=15,
            random_state=42
        )
        
        # Create two different targets
        y_class = np.random.randint(0, 5, size=len(X))  # 5-class classification
        y_reg = X[:, :3].sum(axis=1) + np.random.normal(0, 0.1, len(X))  # Regression
        
        # Split data
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42
        )
        
        # Standardize
        scaler_X = StandardScaler()
        scaler_y_reg = StandardScaler()
        
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        
        y_reg_train = scaler_y_reg.fit_transform(y_reg_train.reshape(-1, 1)).flatten()
        y_reg_test = scaler_y_reg.transform(y_reg_test.reshape(-1, 1)).flatten()
        
        return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
    
    # Create multi-output model
    multi_output_model = MultiOutputMLP(
        input_dim=25,
        shared_dims=[256, 128, 64],
        output_configs={
            'classification': 5,  # 5 classes
            'regression': 1       # 1 continuous value
        }
    )
    
    print(f"\n🔀 Multi-Output Model Created:")
    print(f"  Parameters: {sum(p.numel() for p in multi_output_model.parameters()):,}")

Visualization and Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def plot_training_comparison(histories, titles):
        """Compare training histories across different models."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DNN Training Comparison', fontsize=16)
        
        colors = ['blue', 'red', 'green', 'orange']
        
        # Plot losses
        for i, (history, title, color) in enumerate(zip(histories, titles, colors)):
            if 'train_loss' in history:
                axes[0, 0].plot(history['train_loss'], 
                              label=f'{title} - Train', color=color, linestyle='-')
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], 
                              label=f'{title} - Val', color=color, linestyle='--')
        
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot accuracies (if available)
        for i, (history, title, color) in enumerate(zip(histories, titles, colors)):
            if 'train_accuracy' in history:
                axes[0, 1].plot(history['train_accuracy'], 
                              label=f'{title} - Train', color=color, linestyle='-')
            if 'val_accuracy' in history:
                axes[0, 1].plot(history['val_accuracy'], 
                              label=f'{title} - Val', color=color, linestyle='--')
        
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Model complexity comparison
        model_names = titles
        param_counts = []  # This would need to be calculated for each model
        
        # For demonstration, using dummy values
        param_counts = [50000, 75000, 150000, 200000]  # Replace with actual counts
        
        axes[1, 0].bar(model_names, param_counts, color=colors)
        axes[1, 0].set_title('Model Complexity (Parameters)')
        axes[1, 0].set_ylabel('Number of Parameters')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Performance comparison
        performance_data = {
            'Binary': 0.95,
            'Regression': 0.88,
            'Multi-Class': 0.82,
            'Multi-Output': 0.78
        }
        
        axes[1, 1].bar(performance_data.keys(), performance_data.values(), color=colors)
        axes[1, 1].set_title('Model Performance')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    # Compare all trained models
    histories = [history, reg_history, multi_history]
    titles = ['Binary Classification', 'Regression', 'Multi-Class']
    
    plot_training_comparison(histories, titles)

Best Practices for DNNs
-----------------------

**🎯 Architecture Design:**

.. code-block:: python

    # Good practices for DNN architecture
    def design_dnn_architecture(input_dim, output_dim, task_type='classification'):
        """Design DNN architecture based on best practices."""
        
        if task_type == 'classification':
            # Classification: Gradually decreasing hidden dimensions
            hidden_dims = [
                min(512, input_dim * 8),  # First layer: 8x input
                min(256, input_dim * 4),  # Second layer: 4x input
                min(128, input_dim * 2),  # Third layer: 2x input
                64                        # Final hidden layer
            ]
            dropout_rate = 0.3
            activation = 'relu'
            
        elif task_type == 'regression':
            # Regression: More conservative architecture
            hidden_dims = [
                min(256, input_dim * 4),
                min(128, input_dim * 2),
                64,
                32
            ]
            dropout_rate = 0.2
            activation = 'elu'  # ELU works well for regression
        
        return hidden_dims, dropout_rate, activation

**⚙️ Training Configuration:**

.. code-block:: python

    def get_optimal_config(task_type, dataset_size):
        """Get optimal training configuration based on task and data size."""
        
        base_lr = 0.001
        if dataset_size < 1000:
            batch_size = 32
            epochs = 200
        elif dataset_size < 10000:
            batch_size = 128
            epochs = 150
        else:
            batch_size = 256
            epochs = 100
        
        return TrainingConfig(
            epochs=epochs,
            device="auto",
            mixed_precision=True,
            early_stopping_patience=epochs // 5,  # 20% of total epochs
            validation_frequency=1
        )

**📊 Model Selection Guidelines:**

======================= ================== ==================== ===================
Use Case                Architecture       Loss Function        Key Considerations  
======================= ================== ==================== ===================
Binary Classification  2-4 hidden layers  CrossEntropyLoss     Balance classes
Multi-class (< 10)      3-5 hidden layers  CrossEntropyLoss     Use label smoothing
Multi-class (> 100)     5-8 hidden layers  CrossEntropyLoss     Consider hierarchical
Regression              3-6 hidden layers  MSELoss/L1Loss       Feature scaling crucial
Multi-output            Shared + heads     Combined loss        Balance task weights
High-dimensional        Wide first layers  Task-specific        Dimensionality reduction
======================= ================== ==================== ===================

**🔧 Hyperparameter Tuning:**

.. code-block:: python

    # Systematic hyperparameter search
    hyperparameters = {
        'learning_rates': [0.001, 0.003, 0.01],
        'hidden_dims': [
            [128, 64],
            [256, 128, 64],
            [512, 256, 128, 64]
        ],
        'dropout_rates': [0.1, 0.3, 0.5],
        'activations': ['relu', 'leaky_relu', 'gelu']
    }
    
    # Example of parameter search (simplified)
    def find_best_hyperparameters(X_train, y_train, X_val, y_val):
        best_score = 0
        best_params = {}
        
        for lr in hyperparameters['learning_rates']:
            for hidden_dims in hyperparameters['hidden_dims']:
                for dropout_rate in hyperparameters['dropout_rates']:
                    # Create and train model with these parameters
                    # Keep track of best performing combination
                    pass
        
        return best_params

Summary and Key Takeaways
--------------------------

**🎯 What We Learned:**

✅ **Basic MLPs**: Simple architectures for standard tasks
✅ **Advanced MLPs**: Batch norm, residual connections, better activations
✅ **Task-Specific Design**: Different architectures for different problems
✅ **Multi-Output Networks**: Shared representations for multiple tasks
✅ **Best Practices**: Architecture design, hyperparameter tuning

**📈 Performance Tips:**

1. **Start Simple**: Begin with basic architecture, add complexity gradually
2. **Regularization**: Use dropout, batch norm, early stopping
3. **Activation Functions**: ReLU for classification, ELU for regression
4. **Learning Rate**: Start with 0.001, adjust based on convergence
5. **Batch Size**: Larger batches for stable training, smaller for generalization

**🚀 Production Considerations:**

- **Model Size**: Balance performance vs inference speed
- **Inference Time**: Consider pruning for deployment
- **Memory Usage**: Optimize for target hardware
- **Robustness**: Test with out-of-distribution data

This comprehensive example demonstrates how Treadmill makes it easy to experiment with different DNN architectures while maintaining clean, readable code! 🏃‍♀️‍➡️

Next Steps
----------

- Explore convolutional architectures: :doc:`../tutorials/image_classification`
- Try sequence models: :doc:`encoder_decoder`
- Advanced techniques: :doc:`advanced_usage`
- Real-world applications: :doc:`mnist` 