"""
Metrics tracking system for Treadmill.
"""

from typing import Dict, List, Optional, Any, Callable
import torch
import numpy as np
from collections import defaultdict, deque


class MetricsTracker:
    """Track and compute metrics during training."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of the sliding window for moving averages
        """
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        """Reset all tracked metrics."""
        self.epoch_metrics = defaultdict(list)
        self.batch_metrics = defaultdict(lambda: deque(maxlen=self.window_size))
        self.best_metrics = {}
        
    def update(self, metrics: Dict[str, float], mode: str = "train"):
        """
        Update metrics for current batch.
        
        Args:
            metrics: Dictionary of metric name -> value
            mode: "train" or "val" 
        """
        for name, value in metrics.items():
            metric_key = f"{mode}_{name}"
            self.batch_metrics[metric_key].append(float(value))
    
    def end_epoch(self):
        """Compute epoch-level metrics from batch metrics."""
        epoch_summary = {}
        
        for metric_key, values in self.batch_metrics.items():
            if values:
                epoch_summary[metric_key] = np.mean(values)
                self.epoch_metrics[metric_key].append(epoch_summary[metric_key])
        
        # Update best metrics
        for metric_key, value in epoch_summary.items():
            if metric_key not in self.best_metrics:
                self.best_metrics[metric_key] = value
            else:
                # For loss, lower is better; for accuracy, higher is better
                if "loss" in metric_key.lower():
                    if value < self.best_metrics[metric_key]:
                        self.best_metrics[metric_key] = value
                else:
                    if value > self.best_metrics[metric_key]:
                        self.best_metrics[metric_key] = value
        
        # Clear batch metrics for next epoch
        for metric_key in self.batch_metrics:
            self.batch_metrics[metric_key].clear()
        
        return epoch_summary
    
    def get_current_metrics(self, mode: str = "train") -> Dict[str, float]:
        """Get current metrics for the specified mode."""
        current = {}
        for metric_key, values in self.batch_metrics.items():
            if metric_key.startswith(f"{mode}_") and values:
                current[metric_key.replace(f"{mode}_", "")] = np.mean(values)
        return current
    
    def get_epoch_metrics(self, mode: str = "train") -> Dict[str, List[float]]:
        """Get all epoch metrics for the specified mode."""
        epoch_data = {}
        for metric_key, values in self.epoch_metrics.items():
            if metric_key.startswith(f"{mode}_"):
                epoch_data[metric_key.replace(f"{mode}_", "")] = values
        return epoch_data
    
    def get_best_metrics(self, mode: str = "train") -> Dict[str, float]:
        """Get best metrics for the specified mode."""
        best = {}
        for metric_key, value in self.best_metrics.items():
            if metric_key.startswith(f"{mode}_"):
                best[metric_key.replace(f"{mode}_", "")] = value
        return best
    
    def is_best_epoch(self, metric_name: str, mode: str = "val") -> bool:
        """Check if current epoch achieved best metric value."""
        metric_key = f"{mode}_{metric_name}"
        if metric_key not in self.best_metrics or metric_key not in self.epoch_metrics:
            return False
            
        current_value = self.epoch_metrics[metric_key][-1]  # Latest epoch value
        return abs(current_value - self.best_metrics[metric_key]) < 1e-8


class StandardMetrics:
    """Collection of standard metrics for common tasks."""
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute accuracy for classification tasks."""
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # Multi-class classification
            predictions = torch.argmax(predictions, dim=1)
        else:
            # Binary classification - apply threshold
            predictions = (predictions > 0.5).float()
        correct = (predictions == targets).float()
        return correct.mean().item()
    
    @staticmethod
    def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Compute top-k accuracy."""
        _, top_k_preds = torch.topk(predictions, k, dim=1)
        targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).float()
        return correct.sum(dim=1).clamp(max=1).mean().item()
    
    @staticmethod
    def f1_score(predictions: torch.Tensor, targets: torch.Tensor, 
                 average: str = "macro") -> float:
        """Compute F1 score."""
        try:
            from sklearn.metrics import f1_score as sklearn_f1
            
            if predictions.dim() > 1 and predictions.size(1) > 1:
                predictions = torch.argmax(predictions, dim=1)
                
            return sklearn_f1(targets.cpu().numpy(), predictions.cpu().numpy(), 
                            average=average, zero_division=0)
        except ImportError:
            # Fallback to binary F1 if sklearn not available
            if predictions.dim() > 1 and predictions.size(1) > 1:
                predictions = torch.argmax(predictions, dim=1)
            
            tp = ((predictions == 1) & (targets == 1)).float().sum()
            fp = ((predictions == 1) & (targets == 0)).float().sum() 
            fn = ((predictions == 0) & (targets == 1)).float().sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            return f1.item()
    
    @staticmethod
    def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute mean squared error for regression tasks."""
        return torch.mean((predictions - targets) ** 2).item()
    
    @staticmethod
    def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute mean absolute error for regression tasks."""
        return torch.mean(torch.abs(predictions - targets)).item()


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                   metric_fns: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
    """
    Compute multiple metrics for given predictions and targets.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets  
        metric_fns: Dictionary of metric name -> metric function
        
    Returns:
        Dictionary of computed metrics
    """
    if metric_fns is None:
        # Default metrics based on task type
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # Classification task
            metric_fns = {
                "accuracy": StandardMetrics.accuracy,
                "top5_acc": lambda p, t: StandardMetrics.top_k_accuracy(p, t, k=5)
            }
        else:
            # Regression task
            metric_fns = {
                "mse": StandardMetrics.mean_squared_error,
                "mae": StandardMetrics.mean_absolute_error
            }
    
    results = {}
    for name, fn in metric_fns.items():
        try:
            results[name] = fn(predictions, targets)
        except Exception as e:
            print(f"Warning: Could not compute metric '{name}': {e}")
            results[name] = 0.0
    
    return results 