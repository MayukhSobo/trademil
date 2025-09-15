"""
Unit tests for configuration module.
"""

import pytest
import torch
from treadmill.config import TrainingConfig, OptimizerConfig, SchedulerConfig


class TestOptimizerConfig:
    """Test OptimizerConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = OptimizerConfig()
        
        assert config.optimizer_class == torch.optim.Adam
        assert config.lr == 1e-3
        assert config.weight_decay == 0.0
        assert config.params == {}
    
    def test_string_optimizer_conversion(self):
        """Test string optimizer name conversion."""
        config = OptimizerConfig(optimizer_class="SGD")
        assert config.optimizer_class == torch.optim.SGD
    
    def test_custom_parameters(self):
        """Test custom optimizer parameters."""
        config = OptimizerConfig(
            optimizer_class="Adam",
            lr=1e-4,
            weight_decay=1e-3,
            params={"betas": (0.9, 0.999)}
        )
        
        assert config.optimizer_class == torch.optim.Adam
        assert config.lr == 1e-4
        assert config.weight_decay == 1e-3
        assert config.params["betas"] == (0.9, 0.999)


class TestSchedulerConfig:
    """Test SchedulerConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SchedulerConfig()
        
        assert config.scheduler_class is None
        assert config.params == {}
    
    def test_string_scheduler_conversion(self):
        """Test string scheduler name conversion."""
        config = SchedulerConfig(scheduler_class="StepLR")
        assert config.scheduler_class == torch.optim.lr_scheduler.StepLR


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()
        
        assert config.epochs == 10
        assert config.device in ["cpu", "cuda"]  # Auto-detected
        assert isinstance(config.optimizer, OptimizerConfig)
        assert config.scheduler is None
    
    def test_auto_device_detection(self):
        """Test automatic device detection."""
        config = TrainingConfig(device="auto")
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected_device
    
    def test_custom_optimizer(self):
        """Test custom optimizer configuration."""
        optimizer_config = OptimizerConfig(optimizer_class="SGD", lr=1e-2)
        config = TrainingConfig(optimizer=optimizer_config)
        
        assert config.optimizer.optimizer_class == torch.optim.SGD
        assert config.optimizer.lr == 1e-2
    
    def test_dict_optimizer_conversion(self):
        """Test dictionary to OptimizerConfig conversion."""
        config = TrainingConfig(
            optimizer={"optimizer_class": "SGD", "lr": 1e-2, "momentum": 0.9}
        )
        
        assert isinstance(config.optimizer, OptimizerConfig)
        assert config.optimizer.optimizer_class == torch.optim.SGD
        assert config.optimizer.lr == 1e-2
        assert config.optimizer.params["momentum"] == 0.9
    
    def test_resume_training_config(self):
        """Test resume training configuration."""
        config = TrainingConfig(
            resume_training=False,  # Not testing actual resume, just config
            additional_epochs=5,
            checkpoint_dir="./test_checkpoints"
        )
        
        assert config.resume_training is False
        assert config.additional_epochs == 5
        assert "./test_checkpoints" in config.checkpoint_dir  # Directory path is expanded
    
    def test_checkpoint_directory_config(self):
        """Test checkpoint directory and project name configuration."""
        config = TrainingConfig(
            checkpoint_dir="./checkpoints",
            project_name="test_project",
            timezone="UTC",
            keep_all_checkpoints=False
        )
        
        # Checkpoint dir gets expanded with experiment name
        assert "./checkpoints" in config.checkpoint_dir
        assert "test_project-experiment" in config.checkpoint_dir
        assert config.project_name == "test_project"
        assert config.timezone == "UTC"
        assert config.keep_all_checkpoints is False 