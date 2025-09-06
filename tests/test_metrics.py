"""
Unit tests for metrics module.
"""

import pytest
import torch
import numpy as np
from treadmill.metrics import MetricsTracker, StandardMetrics, compute_metrics


class TestStandardMetrics:
    """Test StandardMetrics class."""
    
    def test_accuracy_classification(self):
        """Test accuracy for classification."""
        predictions = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        targets = torch.tensor([0, 1, 0])
        
        accuracy = StandardMetrics.accuracy(predictions, targets)
        assert accuracy == 1.0  # All correct predictions
    
    def test_accuracy_binary(self):
        """Test accuracy for binary classification."""
        predictions = torch.tensor([0.8, 0.3, 0.9])
        targets = torch.tensor([1, 0, 1])
        
        accuracy = StandardMetrics.accuracy(predictions, targets)
        # With threshold 0.5: [0.8>0.5, 0.3>0.5, 0.9>0.5] = [1, 0, 1] 
        # Targets are [1, 0, 1], so all 3 are correct
        expected = 1.0  # 3 out of 3 correct
        assert abs(accuracy - expected) < 1e-6
    
    def test_top_k_accuracy(self):
        """Test top-k accuracy."""
        predictions = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        targets = torch.tensor([2, 0])  # Correct classes are 2 and 0
        
        top2_acc = StandardMetrics.top_k_accuracy(predictions, targets, k=2)
        assert top2_acc == 1.0  # Both should be in top-2
    
    def test_mean_squared_error(self):
        """Test MSE for regression."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.2])
        
        mse = StandardMetrics.mean_squared_error(predictions, targets)
        expected = ((0.1)**2 + (0.1)**2 + (-0.2)**2) / 3
        assert abs(mse - expected) < 1e-6
    
    def test_mean_absolute_error(self):
        """Test MAE for regression."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.2])
        
        mae = StandardMetrics.mean_absolute_error(predictions, targets)
        expected = (0.1 + 0.1 + 0.2) / 3
        assert abs(mae - expected) < 1e-6


class TestMetricsTracker:
    """Test MetricsTracker class."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = MetricsTracker(window_size=50)
        assert tracker.window_size == 50
        assert len(tracker.epoch_metrics) == 0
        assert len(tracker.batch_metrics) == 0
        assert len(tracker.best_metrics) == 0
    
    def test_update_metrics(self):
        """Test updating metrics."""
        tracker = MetricsTracker()
        
        tracker.update({"loss": 0.5, "accuracy": 0.8}, mode="train")
        tracker.update({"loss": 0.4, "accuracy": 0.85}, mode="train")
        
        current_metrics = tracker.get_current_metrics("train")
        assert "loss" in current_metrics
        assert "accuracy" in current_metrics
        assert current_metrics["loss"] == 0.45  # Average of 0.5 and 0.4
        assert current_metrics["accuracy"] == 0.825  # Average of 0.8 and 0.85
    
    def test_end_epoch(self):
        """Test end epoch calculation."""
        tracker = MetricsTracker()
        
        # Add some batch metrics
        tracker.update({"loss": 0.6, "accuracy": 0.7}, mode="train")
        tracker.update({"loss": 0.4, "accuracy": 0.9}, mode="train")
        
        # End epoch
        epoch_summary = tracker.end_epoch()
        
        assert "train_loss" in epoch_summary
        assert "train_accuracy" in epoch_summary
        assert epoch_summary["train_loss"] == 0.5
        assert epoch_summary["train_accuracy"] == 0.8
    
    def test_best_metrics_tracking(self):
        """Test tracking of best metrics."""
        tracker = MetricsTracker()
        
        # First epoch
        tracker.update({"loss": 0.6, "accuracy": 0.7}, mode="val")
        tracker.end_epoch()
        
        # Second epoch (better)
        tracker.update({"loss": 0.4, "accuracy": 0.9}, mode="val")
        tracker.end_epoch()
        
        best_metrics = tracker.get_best_metrics("val")
        assert best_metrics["loss"] == 0.4  # Lower is better for loss
        assert best_metrics["accuracy"] == 0.9  # Higher is better for accuracy


class TestComputeMetrics:
    """Test compute_metrics function."""
    
    def test_classification_default_metrics(self):
        """Test default metrics for classification."""
        predictions = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([0, 1])
        
        metrics = compute_metrics(predictions, targets)
        
        assert "accuracy" in metrics
        assert "top5_acc" in metrics
        assert metrics["accuracy"] == 1.0
    
    def test_regression_default_metrics(self):
        """Test default metrics for regression."""
        predictions = torch.tensor([1.0, 2.0])
        targets = torch.tensor([1.1, 1.9])
        
        metrics = compute_metrics(predictions, targets)
        
        assert "mse" in metrics
        assert "mae" in metrics
    
    def test_custom_metrics(self):
        """Test custom metric functions."""
        predictions = torch.tensor([1.0, 2.0])
        targets = torch.tensor([1.0, 2.0])
        
        def custom_metric(pred, targ):
            return torch.mean((pred - targ) ** 2).item()
        
        metrics = compute_metrics(predictions, targets, {"custom": custom_metric})
        
        assert "custom" in metrics
        assert metrics["custom"] == 0.0  # Perfect predictions 