Trainer API Reference
====================

The :class:`Trainer` class is the core component of Treadmill, orchestrating the entire training process with a clean, intuitive interface.

.. currentmodule:: treadmill

.. autoclass:: Trainer
   :members:
   :undoc-members:
   :show-inheritance:

Class Overview
--------------

The ``Trainer`` class provides a high-level interface for PyTorch model training with the following key features:

- **Automatic device management**: Handles GPU/CPU placement automatically
- **Mixed precision training**: Leverages automatic mixed precision for faster training
- **Flexible callbacks**: Extensible callback system for custom training logic
- **Rich metrics tracking**: Built-in and custom metrics with beautiful output
- **Smart checkpointing**: Automatic model saving and restoration
- **Early stopping**: Configurable early stopping to prevent overfitting

Constructor
-----------

.. automethod:: Trainer.__init__

**Parameters in Detail:**

- ``model`` (:class:`torch.nn.Module`): Your PyTorch model to train
- ``config`` (:class:`TrainingConfig`): Configuration object controlling all training parameters
- ``train_dataloader`` (:class:`torch.utils.data.DataLoader`): Training data loader
- ``val_dataloader`` (:class:`torch.utils.data.DataLoader`, optional): Validation data loader for monitoring
- ``loss_fn`` (callable, optional): Loss function. If None, attempts to infer from model
- ``metric_fns`` (dict, optional): Dictionary mapping metric names to functions
- ``callbacks`` (list, optional): List of callback objects for training hooks

**Example:**

.. code-block:: python

    import torch
    import torch.nn as nn
    from treadmill import Trainer, TrainingConfig
    
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    config = TrainingConfig(epochs=10, device="auto")
    
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        metric_fns={'accuracy': accuracy_function}
    )

Training Methods
----------------

fit()
^^^^^

.. automethod:: Trainer.fit

The primary method for training your model. Executes the complete training loop including:

- Model training over specified epochs
- Validation (if validation data provided)  
- Callback execution at appropriate hooks
- Metrics computation and logging
- Checkpoint saving
- Early stopping checks

**Returns:**
    Dictionary containing training history with keys:
    
    - ``train_loss``: List of training losses per epoch
    - ``val_loss``: List of validation losses per epoch (if validation enabled)  
    - ``train_{metric}``: Training metrics per epoch for each custom metric
    - ``val_{metric}``: Validation metrics per epoch for each custom metric

**Example:**

.. code-block:: python

    # Basic training
    history = trainer.train()
    
    # Access training history
    print(f"Final training loss: {history['train_loss'][-1]}")
    print(f"Best validation accuracy: {max(history.get('val_accuracy', [0]))}")

train_epoch()
^^^^^^^^^^^^^

.. automethod:: Trainer.train_epoch

Executes a single training epoch. Useful for custom training loops or debugging.

**Parameters:**
    - ``epoch`` (int): Current epoch number for logging and callbacks

**Returns:**
    Dictionary containing training metrics for the epoch

**Example:**

.. code-block:: python

    # Custom training loop
    for epoch in range(config.epochs):
        train_metrics = trainer.train_epoch(epoch)
        
        if epoch % 5 == 0:  # Custom validation frequency
            val_metrics = trainer.validate_epoch(epoch)
        
        # Custom logic here
        if some_condition:
            break

validate_epoch()
^^^^^^^^^^^^^^^^

.. automethod:: Trainer.validate_epoch

Executes validation for a single epoch. Only available if validation dataloader provided.

**Parameters:**
    - ``epoch`` (int): Current epoch number for logging and callbacks

**Returns:**
    Dictionary containing validation metrics for the epoch

**Example:**

.. code-block:: python

    # Manual validation
    val_metrics = trainer.validate_epoch(epoch=0)
    print(f"Validation loss: {val_metrics['loss']}")
    print(f"Validation accuracy: {val_metrics.get('accuracy', 'N/A')}")

State Management
----------------

save_checkpoint()
^^^^^^^^^^^^^^^^^

.. automethod:: Trainer.save_checkpoint

Saves the current training state including model weights, optimizer state, scheduler state, and training progress.

**Parameters:**
    - ``filepath`` (str): Path where to save the checkpoint
    - ``is_best`` (bool, optional): Whether this is the best model so far

**Example:**

.. code-block:: python

    # Save checkpoint manually
    trainer.save_checkpoint("model_epoch_10.pt")
    
    # Save as best model
    trainer.save_checkpoint("best_model.pt", is_best=True)

load_checkpoint()
^^^^^^^^^^^^^^^^^

.. automethod:: Trainer.load_checkpoint

Loads a previously saved checkpoint, restoring model weights, optimizer state, and training progress.

**Parameters:**
    - ``filepath`` (str): Path to the checkpoint file
    - ``map_location`` (str or torch.device, optional): Device to map loaded tensors to

**Returns:**
    Dictionary containing the loaded checkpoint data

**Example:**

.. code-block:: python

    # Resume training from checkpoint
    checkpoint_data = trainer.load_checkpoint("model_epoch_10.pt")
    print(f"Resuming from epoch {checkpoint_data.get('epoch', 0)}")
    
    # Continue training
    trainer.train()

get_history()
^^^^^^^^^^^^^

.. automethod:: Trainer.get_history

Returns the complete training history including losses and metrics for all epochs.

**Returns:**
    Dictionary containing training history with metric names as keys

**Example:**

.. code-block:: python

    history = trainer.get_history()
    
    # Plot training curves
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    if 'train_accuracy' in history:
        plt.plot(history['train_accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.show()

Prediction Methods
------------------

predict()
^^^^^^^^^

.. automethod:: Trainer.predict

Generates predictions for a given dataset using the trained model.

**Parameters:**
    - ``dataloader`` (:class:`torch.utils.data.DataLoader`): Data to generate predictions for

**Returns:**
    Tensor containing model predictions

**Example:**

.. code-block:: python

    # Generate predictions on test set
    test_predictions = trainer.predict(test_loader)
    
    # Convert to class predictions
    predicted_classes = torch.argmax(test_predictions, dim=1)
    
    # Calculate accuracy
    accuracy = (predicted_classes == test_labels).float().mean()
    print(f"Test accuracy: {accuracy:.4f}")

evaluate()
^^^^^^^^^^

.. automethod:: Trainer.evaluate

Evaluates the model on a given dataset, computing loss and all configured metrics.

**Parameters:**
    - ``dataloader`` (:class:`torch.utils.data.DataLoader`): Data to evaluate on

**Returns:**
    Dictionary containing evaluation metrics

**Example:**

.. code-block:: python

    # Evaluate on test set
    test_results = trainer.evaluate(test_loader)
    
    print(f"Test Results:")
    for metric_name, value in test_results.items():
        print(f"  {metric_name}: {value:.4f}")

Properties and Attributes
-------------------------

model
^^^^^

Access to the underlying PyTorch model.

.. code-block:: python

    # Access model for inference
    model = trainer.model
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor)

config
^^^^^^

The training configuration object.

.. code-block:: python

    # Check configuration
    print(f"Training for {trainer.config.epochs} epochs")
    print(f"Using device: {trainer.config.device}")

device
^^^^^^

The device (CPU/GPU) being used for training.

.. code-block:: python

    print(f"Training on: {trainer.device}")

metrics_tracker
^^^^^^^^^^^^^^^

The metrics tracking object that maintains training history.

.. code-block:: python

    # Access detailed metrics
    tracker = trainer.metrics_tracker
    latest_metrics = tracker.get_latest_metrics()

current_epoch
^^^^^^^^^^^^^

The current epoch number during training.

.. code-block:: python

    # Useful in callbacks
    print(f"Currently at epoch: {trainer.current_epoch}")

Advanced Usage Examples
-----------------------

Custom Training Loop
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    trainer = Trainer(model, config, train_loader, val_loader)
    
    for epoch in range(config.epochs):
        # Custom pre-epoch logic
        if epoch % 10 == 0:
            # Adjust learning rate or other parameters
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] *= 0.9
        
        # Train one epoch
        train_metrics = trainer.train_epoch(epoch)
        
        # Custom validation schedule
        if epoch % config.validation_frequency == 0:
            val_metrics = trainer.validate_epoch(epoch)
            
            # Custom early stopping logic
            if val_metrics['loss'] > previous_best_loss * 1.1:
                print("Custom early stopping triggered!")
                break
        
        # Custom checkpointing
        if epoch % 20 == 0:
            trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

Integration with Callbacks
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from treadmill.callbacks import EarlyStopping, ModelCheckpoint
    
    # Create custom callbacks
    early_stopping = EarlyStopping(
        patience=10, 
        min_delta=0.001, 
        verbose=True
    )
    
    checkpointing = ModelCheckpoint(
        filepath="models/best_model_{epoch:02d}_{val_loss:.4f}.pt",
        save_best_only=True,
        verbose=True
    )
    
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        callbacks=[early_stopping, checkpointing]
    )
    
    # Callbacks are automatically called during training
    history = trainer.train()

Multi-GPU Training
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    config = TrainingConfig(
        epochs=50,
        device="cuda",
        mixed_precision=True  # Works with DataParallel
    )
    
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()

See Also
--------

- :class:`TrainingConfig`: Configuration options for training
- :doc:`callbacks`: Available callbacks and custom callback creation
- :doc:`metrics`: Built-in metrics and custom metric functions
- :doc:`../tutorials/image_classification`: Complete training tutorial
- :doc:`../examples/advanced_training`: Advanced training examples 