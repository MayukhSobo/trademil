"""
Integration tests for Trainer class.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from treadmill import Trainer, TrainingConfig, OptimizerConfig
from treadmill.metrics import StandardMetrics


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    
    X_val = torch.randn(50, 10)
    y_val = torch.randint(0, 2, (50,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader


@pytest.fixture
def basic_config():
    """Create basic training configuration."""
    return TrainingConfig(
        epochs=2,  # Small for testing
        device="cpu",
        optimizer=OptimizerConfig(optimizer_class="Adam", lr=1e-2),
        validate_every=1,
        print_every=5,
        progress_bar=False  # Disable for testing
    )


class TestTrainer:
    """Test Trainer class."""
    
    def test_trainer_initialization(self, dummy_data, basic_config):
        """Test trainer initialization."""
        train_loader, val_loader = dummy_data
        model = SimpleModel()
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            config=basic_config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_fn=loss_fn
        )
        
        assert trainer.model is model
        assert trainer.config is basic_config
        assert trainer.train_dataloader is train_loader
        assert trainer.val_dataloader is val_loader
        assert trainer.loss_fn is loss_fn
        assert trainer.device.type == "cpu"
        assert trainer.optimizer is not None
    
    def test_basic_training(self, dummy_data, basic_config):
        """Test basic training functionality."""
        train_loader, val_loader = dummy_data
        model = SimpleModel()
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            config=basic_config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_fn=loss_fn,
            metric_fns={"accuracy": StandardMetrics.accuracy}
        )
        
        history = trainer.train()
        
        # Check training completed
        assert history["total_epochs"] == 2
        assert len(history["train_metrics"]["loss"]) == 2
        assert len(history["val_metrics"]["loss"]) == 2
        
        # Check metrics were computed
        assert "accuracy" in history["train_metrics"]
        assert "accuracy" in history["val_metrics"]
    
    def test_training_without_validation(self, dummy_data, basic_config):
        """Test training without validation data."""
        train_loader, _ = dummy_data
        model = SimpleModel()
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            config=basic_config,
            train_dataloader=train_loader,
            val_dataloader=None,  # No validation
            loss_fn=loss_fn
        )
        
        history = trainer.train()
        
        assert history["total_epochs"] == 2
        assert len(history["train_metrics"]["loss"]) == 2
        assert len(history["val_metrics"]) == 0  # No validation metrics
    
    def test_model_with_compute_loss(self, dummy_data, basic_config):
        """Test model with built-in compute_loss method."""
        
        class ModelWithLoss(SimpleModel):
            def compute_loss(self, outputs, targets):
                return nn.functional.cross_entropy(outputs, targets)
        
        train_loader, val_loader = dummy_data
        model = ModelWithLoss()
        
        trainer = Trainer(
            model=model,
            config=basic_config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_fn=None  # Should use model's compute_loss
        )
        
        history = trainer.train()
        assert history["total_epochs"] == 2
    
    def test_checkpoint_save_load(self, dummy_data, basic_config, tmp_path):
        """Test checkpoint saving and loading."""
        train_loader, val_loader = dummy_data
        model = SimpleModel()
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            config=basic_config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_fn=loss_fn
        )
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        new_model = SimpleModel()
        new_trainer = Trainer(
            model=new_model,
            config=basic_config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_fn=loss_fn
        )
        
        checkpoint = new_trainer.load_checkpoint(str(checkpoint_path), resume_training=False)
        
        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
    
    def test_training_report_generation(self, dummy_data, basic_config):
        """Test training report generation after training."""
        train_loader, val_loader = dummy_data
        model = SimpleModel()
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            config=basic_config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_fn=loss_fn,
            metric_fns={"accuracy": StandardMetrics.accuracy}
        )
        
        # Run training
        history = trainer.train()
        
        # Check that training report was generated
        assert trainer.training_report is not None
        assert trainer.report is not None  # Test property access
        
        # Check report contents
        report = trainer.report
        assert report.total_epochs == basic_config.epochs
        assert report.model_name == "SimpleModel"
        assert report.total_parameters > 0
        assert report.trainable_parameters > 0
        assert report.model_size_mb > 0
        assert report.device == "cpu"
        assert report.batch_size == 16  # From dummy data fixture
        
        # Check that metrics are recorded
        assert len(report.best_metrics) > 0
        assert len(report.final_metrics) > 0
        
        # Test serialization
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert "total_epochs" in report_dict
        assert "model_name" in report_dict
    
    def test_hardware_monitor_initialization(self, dummy_data, basic_config):
        """Test that hardware monitor is properly initialized."""
        train_loader, val_loader = dummy_data
        model = SimpleModel()
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            config=basic_config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_fn=loss_fn
        )
        
        # Check hardware monitor exists
        assert hasattr(trainer, 'hardware_monitor')
        assert trainer.hardware_monitor is not None
        
        # Test sampling (should not raise errors)
        trainer.hardware_monitor.sample()
        
        # Test getting summary
        summary = trainer.hardware_monitor.get_summary()
        assert isinstance(summary, dict)
        assert 'avg_cpu_percent' in summary
        assert 'num_gpus' in summary 