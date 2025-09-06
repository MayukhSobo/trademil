Installation Guide
==================

This guide walks you through installing Treadmill and its dependencies on various platforms.

System Requirements
-------------------

**Minimum Requirements**

- Python 3.8 or higher
- PyTorch 1.12.0 or higher
- 4GB RAM (8GB recommended)
- CUDA-compatible GPU (optional but recommended for deep learning)

**Supported Platforms**

- ✅ Linux (Ubuntu 18.04+, CentOS 7+, etc.)
- ✅ macOS (10.15+)
- ✅ Windows 10/11
- ✅ Google Colab
- ✅ Kaggle Notebooks

Installation Methods
--------------------

Method 1: From PyPI (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install Treadmill is from PyPI:

.. code-block:: bash

    # Basic installation
    pip install pytorch-treadmill

**Advantages:**
- Quick and simple installation
- Stable, tested releases
- Automatic dependency management
- Perfect for most users

**With Optional Dependencies:**

.. code-block:: bash

    # With examples dependencies (torchvision, scikit-learn)
    pip install "pytorch-treadmill[examples]"
    
    # With full dependencies (visualization tools, docs, etc.)
    pip install "pytorch-treadmill[full]"
    
    # For development
    pip install "pytorch-treadmill[dev]"

Method 2: From Source (Development)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method gives you access to the latest features and allows easy contribution.

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/MayukhSobo/treadmill.git
    cd treadmill
    
    # Install in development mode
    pip install -e .

**Advantages:**
- Latest features and bug fixes
- Easy to modify and contribute
- Full access to examples and documentation

Method 3: Install with Optional Dependencies (Development)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For different use cases, you can install Treadmill with various optional dependencies:

**Basic Installation:**

.. code-block:: bash

    pip install -e .  # Core dependencies only

**With Examples:**

.. code-block:: bash

    pip install -e ".[examples]"  # Includes torchvision, scikit-learn

**With Full Features:**

.. code-block:: bash

    pip install -e ".[full]"  # All optional dependencies

**For Development:**

.. code-block:: bash

    pip install -e ".[dev]"  # Development tools (pytest, black, mypy, etc.)

Dependency Details
^^^^^^^^^^^^^^^^^^

**Core Dependencies (always installed):**

.. code-block:: text

    torch>=1.12.0          # PyTorch framework
    torchvision>=0.13.0    # Computer vision utilities
    numpy>=1.21.0          # Numerical computing
    rich>=12.0.0           # Beautiful terminal output
    torchinfo>=1.7.0       # Model summary information
    scikit-learn>=1.0.0    # Machine learning utilities

**Optional Dependencies:**

- ``examples``: Additional dependencies for running examples
- ``full``: Complete feature set including visualization tools
- ``dev``: Development and testing tools

Virtual Environment Setup
--------------------------

We highly recommend using a virtual environment to avoid dependency conflicts.

Using venv (Built-in)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Create virtual environment
    python -m venv treadmill_env
    
    # Activate (Linux/Mac)
    source treadmill_env/bin/activate
    
    # Activate (Windows)
    treadmill_env\Scripts\activate
    
    # Install Treadmill
    cd treadmill
    pip install -e .

Using conda
^^^^^^^^^^^^

.. code-block:: bash

    # Create conda environment
    conda create -n treadmill python=3.9
    conda activate treadmill
    
    # Install PyTorch (recommended to use conda for PyTorch)
    conda install pytorch torchvision torchaudio -c pytorch
    
    # Install Treadmill
    cd treadmill
    pip install -e .

GPU Support Setup
-----------------

For optimal performance, especially with large models, GPU support is highly recommended.

CUDA Installation
^^^^^^^^^^^^^^^^^

**Step 1: Check GPU Compatibility**

.. code-block:: bash

    # Check if CUDA is available
    nvidia-smi

**Step 2: Install CUDA-enabled PyTorch**

Visit `PyTorch website <https://pytorch.org/get-started/locally/>`_ for the latest installation commands.

.. code-block:: bash

    # Example for CUDA 11.8 (check website for latest)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Step 3: Verify Installation**

.. code-block:: python

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.get_device_name()}")

Verification
------------

After installation, verify that everything works correctly:

**Quick PyPI Installation Test:**

.. code-block:: bash

    # Install from PyPI
    pip install pytorch-treadmill
    
    # Test basic import
    python -c "import treadmill; print(f'Treadmill {treadmill.__version__} installed successfully!')"

**Basic Verification:**

.. code-block:: python

    import treadmill
    print(f"Treadmill version: {treadmill.__version__}")
    
    # Test basic functionality
    from treadmill import TrainingConfig, Trainer
    print("✅ Import successful!")

**Complete Test:**

.. code-block:: python

    import torch
    import torch.nn as nn
    from treadmill import Trainer, TrainingConfig
    
    # Create a simple test model
    model = nn.Linear(10, 1)
    
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Test configuration
    config = TrainingConfig(epochs=1, device="cpu")
    
    # Test trainer creation
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=dataloader
    )
    print("✅ Trainer creation successful!")

Troubleshooting
---------------

Common Installation Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue 1: PyTorch Version Compatibility**

.. code-block:: text

    ERROR: No matching distribution found for torch>=1.12.0

**Solution:**

.. code-block:: bash

    # Update pip first
    pip install --upgrade pip
    
    # Install specific PyTorch version
    pip install torch==1.12.0 torchvision==0.13.0

**Issue 2: CUDA Version Mismatch**

.. code-block:: text

    UserWarning: CUDA initialization: Found no NVIDIA driver

**Solution:**

1. Check CUDA driver installation: ``nvidia-smi``
2. Install matching CUDA toolkit version
3. Reinstall PyTorch with correct CUDA version

**Issue 3: Permission Denied (Linux/Mac)**

.. code-block:: text

    PermissionError: [Errno 13] Permission denied

**Solution:**

.. code-block:: bash

    # Use --user flag
    pip install --user -e .
    
    # Or fix permissions
    sudo chown -R $USER ~/.local/

Platform-Specific Notes
-----------------------

Windows
^^^^^^^

- Use Command Prompt or PowerShell as Administrator
- Consider using Windows Subsystem for Linux (WSL2)
- Visual Studio Build Tools may be required for some packages

.. code-block:: bash

    # Install Visual Studio Build Tools if needed
    # Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio

macOS
^^^^^

- Xcode Command Line Tools required
- Consider using Homebrew for Python installation

.. code-block:: bash

    # Install Xcode Command Line Tools
    xcode-select --install
    
    # Install Python via Homebrew (optional)
    brew install python@3.9

Google Colab
^^^^^^^^^^^^

Treadmill works out of the box on Google Colab:

.. code-block:: python

    # In a Colab cell (PyPI installation - recommended)
    !pip install pytorch-treadmill
    
    # Or from source for latest features
    !git clone https://github.com/MayukhSobo/treadmill.git
    %cd treadmill
    !pip install -e .

Docker Installation
-------------------

For containerized environments, we provide Docker support:

.. code-block:: bash

    # Pull the Docker image (when available)
    docker pull treadmill/treadmill:latest
    
    # Or build from source
    git clone https://github.com/MayukhSobo/treadmill.git
    cd treadmill
    docker build -t treadmill .

Next Steps
----------

After successful installation:

1. 📖 Read the :doc:`quickstart` guide
2. 🏃‍♀️ Try the :doc:`../tutorials/image_classification` tutorial
3. 🔍 Explore the :doc:`../examples/mnist` example
4. 📚 Check the :doc:`../api/trainer` API reference

If you encounter any issues not covered here, please:

- Check our `GitHub Issues <https://github.com/MayukhSobo/treadmill/issues>`_
- Create a new issue with your system details and error messages
- Join our community discussions for help 