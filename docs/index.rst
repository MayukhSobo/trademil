🏃‍♀️‍➡️ Treadmill Documentation 🏃‍♀️‍➡️
===============================================

.. image:: https://raw.githubusercontent.com/MayukhSobo/treadmill/main/treadmill.png
   :alt: Treadmill Training Framework
   :width: 300px
   :align: center

**A Clean and Modular PyTorch Training Framework**

Welcome to Treadmill, a lightweight, modular training framework specifically designed for PyTorch. 
Treadmill provides clean, easy-to-understand training loops with beautiful output formatting while 
maintaining the power and flexibility of vanilla PyTorch.

.. note::
   Treadmill is designed with simplicity and modularity in mind. Whether you're a beginner learning 
   PyTorch or an experienced practitioner, Treadmill helps you focus on your model and data rather 
   than boilerplate training code.

Features Overview
-----------------

✨ **Core Features**

- **🎯 Pure PyTorch**: Built specifically for PyTorch with no forced abstractions
- **🔧 Modular Design**: Easy to customize and extend with comprehensive callback system
- **📊 Beautiful Output**: Rich formatting with progress bars and detailed metrics tables
- **⚡ Performance Optimizations**: Mixed precision, gradient accumulation, gradient clipping
- **🎛️ Flexible Configuration**: Dataclass-based configuration system for easy parameter management

✨ **Advanced Features**

- **📈 Comprehensive Metrics**: Built-in metrics with support for custom metrics
- **💾 Smart Checkpointing**: Automatic model saving with customizable triggers
- **🛑 Early Stopping**: Configurable early stopping to prevent overfitting
- **🔄 Resumable Training**: Easy checkpoint loading and training resumption
- **🚀 Mixed Precision**: Automatic mixed precision for faster training
- **📝 Extensive Logging**: Rich logging with metrics tracking and visualization

Quick Start
-----------

Get started with Treadmill in just a few lines of code:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from treadmill import Trainer, TrainingConfig
    
    # Define your model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create training configuration
    config = TrainingConfig(
        epochs=10,
        device="auto",
        mixed_precision=True
    )
    
    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )
    
    # Train your model
    trainer.fit()

Installation
------------

**From Source (Recommended)**

.. code-block:: bash

    git clone https://github.com/MayukhSobo/treadmill.git
    cd treadmill
    pip install -e .

**Install with Examples**

.. code-block:: bash

    pip install -e ".[examples]"  # Includes additional dependencies for examples

**Install Full Version**

.. code-block:: bash

    pip install -e ".[full]"  # Includes all optional dependencies and development tools

Documentation Structure
-----------------------

This documentation is organized into several sections to help you get the most out of Treadmill:

📚 **Learning Path**

1. **Getting Started** - New to Treadmill? Start here!
2. **User Guide** - Comprehensive guides for common use cases
3. **Tutorials** - Step-by-step tutorials with real examples
4. **API Reference** - Detailed API documentation
5. **Examples** - Ready-to-run examples and best practices

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   user_guide/installation
   user_guide/quickstart
   user_guide/basic_concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/configuration
   user_guide/training
   user_guide/callbacks
   user_guide/metrics
   user_guide/checkpointing
   user_guide/advanced_features

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/image_classification
   tutorials/text_classification
   tutorials/custom_callbacks
   tutorials/custom_metrics
   tutorials/transfer_learning
   tutorials/distributed_training

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/mnist
   examples/cifar10
   examples/nlp_sentiment
   examples/custom_architectures
   examples/advanced_training

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/trainer
   api/config
   api/metrics
   api/callbacks
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   development/contributing
   development/changelog
   development/roadmap

Community and Support
---------------------

🤝 **Get Involved**

- **GitHub**: `Source code and issues <https://github.com/MayukhSobo/treadmill>`_
- **Discussions**: Join our community discussions
- **Contributing**: See our :doc:`development/contributing` guide

📊 **Project Status**

.. |build| image:: https://github.com/MayukhSobo/treadmill/workflows/CI/badge.svg
   :target: https://github.com/MayukhSobo/treadmill/actions

.. |coverage| image:: https://codecov.io/gh/MayukhSobo/treadmill/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/MayukhSobo/treadmill

.. |docs| image:: https://readthedocs.org/projects/treadmill/badge/?version=latest
   :target: https://treadmill.readthedocs.io/

|build| |coverage| |docs|

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 