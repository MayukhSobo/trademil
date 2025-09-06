Quick Start Guide
=================

Welcome to Treadmill! This guide will get you training your first model in minutes.

30-Second Overview
------------------

Treadmill simplifies PyTorch training into three simple steps:

1. **Configure**: Set up your training parameters
2. **Create**: Initialize the trainer with your model and data
3. **Train**: Start training with a single command

Let's see this in action!

Your First Model
----------------

Here's a complete example that trains a simple neural network:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from treadmill import Trainer, TrainingConfig

    # Step 1: Create your model (standard PyTorch)
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    # Step 2: Prepare your data (standard PyTorch)
    # For this example, we'll use dummy data
    train_X = torch.randn(1000, 784)
    train_y = torch.randint(0, 10, (1000,))
    val_X = torch.randn(200, 784)
    val_y = torch.randint(0, 10, (200,))

    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Step 3: Configure training
    config = TrainingConfig(
        epochs=10,
        device="auto",  # Automatically choose GPU if available
        mixed_precision=True,  # Speed up training
        early_stopping_patience=3
    )

    # Step 4: Create trainer and start training
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=nn.CrossEntropyLoss()
    )

    # Step 5: Train your model!
    trainer.train()

**That's it!** 🎉 Your model is now training with:
- Automatic device detection (GPU/CPU)
- Mixed precision training for speed
- Early stopping to prevent overfitting
- Beautiful progress bars and metrics
- Automatic checkpointing

Understanding the Components
----------------------------

Let's break down each component to understand what's happening.

TrainingConfig: Your Control Center
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``TrainingConfig`` class controls every aspect of training:

.. code-block:: python

    from treadmill import TrainingConfig

    config = TrainingConfig(
        # Basic training parameters
        epochs=20,
        device="auto",  # "auto", "cpu", "cuda", or specific device "cuda:0"
        
        # Performance optimizations
        mixed_precision=True,      # Use automatic mixed precision
        accumulate_grad_batches=1,  # Accumulate gradients
        grad_clip_norm=1.0,        # Gradient clipping
        
        # Validation and monitoring
        validation_frequency=1,    # Validate every N epochs
        log_frequency=10,         # Log every N batches
        
        # Early stopping
        early_stopping_patience=5,
        early_stopping_min_delta=0.001,
        
        # Checkpointing
        checkpoint_dir="./checkpoints",
        save_best_model=True,
        save_last_model=True
    )

Trainer: The Heart of Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Trainer`` class orchestrates the entire training process:

.. code-block:: python

    from treadmill import Trainer

    trainer = Trainer(
        model=model,                    # Your PyTorch model
        config=config,                  # Training configuration
        train_dataloader=train_loader,  # Training data
        val_dataloader=val_loader,      # Validation data (optional)
        loss_fn=nn.CrossEntropyLoss(),  # Loss function
        metric_fns={                    # Custom metrics (optional)
            'accuracy': accuracy_fn,
            'f1_score': f1_score_fn
        }
    )

Working with Different Data Types
---------------------------------

Image Classification
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torchvision
    import torchvision.transforms as transforms
    from treadmill import Trainer, TrainingConfig

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

    # Simple CNN model
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    config = TrainingConfig(epochs=20, mixed_precision=True)
    trainer = Trainer(model, config, train_loader)
    trainer.train()

Text Classification
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence

    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, (hidden, _) = self.lstm(embedded)
            # Use last hidden state
            output = self.classifier(hidden[-1])
            return output

    model = TextClassifier(vocab_size=10000, embed_dim=128, 
                          hidden_dim=64, num_classes=2)
    
    config = TrainingConfig(epochs=15, device="auto")
    trainer = Trainer(model, config, train_loader)
    trainer.train()

Customizing Your Training
-------------------------

Custom Loss Functions
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return focal_loss.mean()

    # Use custom loss
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        loss_fn=FocalLoss(alpha=1, gamma=2)
    )

Custom Metrics
^^^^^^^^^^^^^^

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

    # Use custom metrics
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        metric_fns={
            'accuracy': accuracy,
            'top3_accuracy': lambda p, t: top_k_accuracy(p, t, k=3)
        }
    )

Advanced Configuration Examples
-------------------------------

High-Performance Training
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Configuration for maximum performance
    config = TrainingConfig(
        epochs=100,
        device="cuda",
        
        # Performance optimizations
        mixed_precision=True,
        accumulate_grad_batches=4,  # Simulate larger batch size
        grad_clip_norm=1.0,              # Prevent exploding gradients
        
        # Efficient validation
        validation_frequency=5,          # Validate every 5 epochs
        
        # Aggressive early stopping
        early_stopping_patience=10,
        early_stopping_min_delta=0.0001,
        
        # Smart checkpointing
        save_best_model=True,
        save_last_model=False,           # Save space
        checkpoint_dir="./best_models"
    )

Research/Experimentation Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Configuration for research with extensive logging
    config = TrainingConfig(
        epochs=200,
        
        # Detailed monitoring
        validation_frequency=1,          # Validate every epoch
        log_frequency=5,                 # Log every 5 batches
        
        # Conservative early stopping
        early_stopping_patience=20,
        early_stopping_min_delta=1e-6,
        
        # Keep all checkpoints for analysis
        save_best_model=True,
        save_last_model=True,
        checkpoint_frequency=10,         # Save every 10 epochs
    )

Monitoring Training Progress
----------------------------

Treadmill provides beautiful, informative output during training:

.. code-block:: text

    Epoch 1/10 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05
    ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃      Phase    ┃      Loss     ┃   Accuracy    ┃      Time     ┃
    ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │     train     │    0.4523     │    0.8456     │   0:00:03     │
    │   validation  │    0.3876     │    0.8723     │   0:00:01     │
    └───────────────┴───────────────┴───────────────┴───────────────┘

Training Results Access
-----------------------

After training, access your results:

.. code-block:: python

    # Get training history
    history = trainer.get_history()
    print(f"Best validation accuracy: {max(history['val_accuracy'])}")

    # Get the trained model
    trained_model = trainer.model

    # Make predictions
    with torch.no_grad():
        predictions = trained_model(test_data)

Common Patterns and Tips
------------------------

**💡 Best Practices:**

1. **Start Simple**: Begin with basic config, then add optimizations
2. **Use Mixed Precision**: Almost always beneficial for modern GPUs
3. **Monitor Validation**: Always use validation data to prevent overfitting
4. **Save Checkpoints**: Enable checkpointing for long training runs
5. **Custom Metrics**: Add domain-specific metrics for better insights

**⚠️ Common Gotchas:**

1. **Device Mismatch**: Ensure model and data are on the same device
2. **Loss Function**: Match loss function to your task (CrossEntropy for classification)
3. **Learning Rate**: Start with default, then tune if needed
4. **Batch Size**: Larger isn't always better, find the sweet spot

Next Steps
----------

Now that you've got the basics down, explore:

📖 **Deep Dive Guides:**
- :doc:`configuration` - Complete configuration reference
- :doc:`callbacks` - Advanced training control with callbacks
- :doc:`metrics` - Custom metrics and monitoring

🎯 **Hands-on Tutorials:**
- :doc:`../tutorials/image_classification` - Complete image classification project
- :doc:`../tutorials/custom_callbacks` - Build your own callbacks
- :doc:`../tutorials/transfer_learning` - Transfer learning best practices

🚀 **Ready-to-Run Examples:**
- :doc:`../examples/mnist` - Classic MNIST digit recognition
- :doc:`../examples/cifar10` - CIFAR-10 image classification
- :doc:`../examples/nlp_sentiment` - Text sentiment analysis

Happy training! 🏃‍♀️‍➡️ 