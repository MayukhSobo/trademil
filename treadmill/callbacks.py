"""
Callback system for Treadmill training framework.
"""

import os
import torch
from abc import ABC
from typing import Dict, Optional, List
from pathlib import Path
import numpy as np


class Callback(ABC):
    """Base class for all callbacks."""
    
    def on_train_start(self, trainer, **kwargs):
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training."""
        pass
        
    def on_epoch_start(self, trainer, epoch: int, **kwargs):
        """Called at the start of each epoch."""
        pass
        
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of each epoch."""
        pass
        
    def on_batch_start(self, trainer, batch_idx: int, **kwargs):
        """Called at the start of each batch."""
        pass
        
    def on_batch_end(self, trainer, batch_idx: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of each batch."""
        pass
        
    def on_validation_start(self, trainer, **kwargs):
        """Called at the start of validation."""
        pass
        
    def on_validation_end(self, trainer, metrics: Dict[str, float], **kwargs):
        """Called at the end of validation."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, monitor: str = "val_loss", patience: int = 10, 
                 mode: str = "min", min_delta: float = 0.0, verbose: bool = True):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor for early stopping
            patience: Number of epochs with no improvement to wait
            mode: "min" for metrics that should decrease, "max" for those that should increase
            min_delta: Minimum change to qualify as an improvement
            verbose: Whether to print early stopping messages
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0
        
        if mode == "min":
            self.monitor_op = lambda current, best: current < (best - min_delta)
            self.best_value = np.inf
        else:
            self.monitor_op = lambda current, best: current > (best + min_delta)  
            self.best_value = -np.inf
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Check for early stopping condition."""
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            if self.verbose:
                print(f"Warning: Early stopping metric '{self.monitor}' not found in metrics")
            return
        
        if self.monitor_op(current_value, self.best_value):
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                
                if self.verbose:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"Best {self.monitor}: {self.best_value:.6f}")
    
    # // TODO: Is it really needed!
    def on_train_end(self, trainer, **kwargs):
        """Print early stopping summary."""
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Training stopped early at epoch {self.stopped_epoch + 1}")


class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""
    
    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min",
                 save_best_only: bool = True, save_freq: int = 1, verbose: bool = True,
                 save_format: str = "pt"):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints (can include epoch/metric placeholders)
            monitor: Metric to monitor for best model selection
            mode: "min" or "max" for the monitored metric
            save_best_only: Whether to only save the best model
            save_freq: Frequency of saving (every N epochs)
            verbose: Whether to print checkpoint messages
            save_format: Format to save model ("pt", "pth", "pkl", "safetensors", "onnx")
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose
        self.save_format = save_format
        
        self.best_value = np.inf if mode == "min" else -np.inf
        self.epochs_since_last_save = 0
        self.best_model_path = None  # Track best model path for cleanup
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Save checkpoint if conditions are met."""
        self.epochs_since_last_save += 1
        
        current_value = metrics.get(self.monitor, 0.0)
        should_save = False
        
        if self.save_best_only:
            if self.mode == "min":
                should_save = current_value < self.best_value
            else:
                should_save = current_value > self.best_value
                
            if should_save:
                self.best_value = current_value
        else:
            should_save = self.epochs_since_last_save >= self.save_freq
        
        if should_save:
            self.epochs_since_last_save = 0
            self._save_checkpoint(trainer, epoch, metrics)
    
    def _save_checkpoint(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Save the actual checkpoint."""
        # Format filepath with epoch and metrics (using 1-based epoch numbering)
        filepath = self.filepath.format(epoch=epoch, epoch_1based=epoch+1, **metrics)
        
        # Ensure correct file extension
        if not filepath.endswith(f".{self.save_format}"):
            filepath = f"{filepath.rsplit('.', 1)[0]}.{self.save_format}"
        
        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metrics": metrics,
            "config": trainer.config.__dict__ if hasattr(trainer.config, "__dict__") else trainer.config
        }
        
        if trainer.scheduler:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()
        
        # Save checkpoint based on format
        if self.save_format in ["pt", "pth"]:
            torch.save(checkpoint, filepath)
        elif self.save_format == "pkl":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)
        elif self.save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                # Convert state dicts to flat dict for safetensors
                flat_dict = {}
                for key, value in checkpoint["model_state_dict"].items():
                    flat_dict[f"model.{key}"] = value
                for key, value in checkpoint["optimizer_state_dict"].items():
                    if isinstance(value, torch.Tensor):
                        flat_dict[f"optimizer.{key}"] = value
                save_file(flat_dict, filepath)
            except ImportError:
                print("Warning: safetensors not available, falling back to .pt format")
                filepath = filepath.replace('.safetensors', '.pt')
                torch.save(checkpoint, filepath)
        elif self.save_format == "onnx":
            try:
                # Export model to ONNX format
                trainer.model.eval()
                onnx_filepath = filepath.replace('.onnx', '_model.onnx')
                
                # Need dummy input to export ONNX - try to get from dataloader
                dummy_input = None
                try:
                    for batch in trainer.train_dataloader:
                        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                            dummy_input = batch[0][:1]  # Take first sample
                        else:
                            dummy_input = batch[:1]
                        break
                except:
                    print("Warning: Could not get sample input for ONNX export")
                
                if dummy_input is not None:
                    dummy_input = dummy_input.to(trainer.device)
                    torch.onnx.export(
                        trainer.model,
                        dummy_input,
                        onnx_filepath,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                    )
                    
                    # Also save training checkpoint separately
                    checkpoint_filepath = filepath.replace('.onnx', '_checkpoint.pt')
                    torch.save(checkpoint, checkpoint_filepath)
                    
                    if self.verbose:
                        print(f"ONNX model saved: {onnx_filepath}")
                        print(f"Training checkpoint saved: {checkpoint_filepath}")
                else:
                    print("Warning: ONNX export failed - no valid input found, falling back to .pt")
                    filepath = filepath.replace('.onnx', '.pt')
                    torch.save(checkpoint, filepath)
                    
            except ImportError:
                print("Warning: ONNX not available, falling back to .pt format")
                filepath = filepath.replace('.onnx', '.pt')
                torch.save(checkpoint, filepath)
        
        # Clean up previous best model if we're only keeping the best
        if self.save_best_only and self.best_model_path and self.best_model_path != filepath:
            try:
                if os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                    if self.verbose:
                        print(f"Removed previous checkpoint: {self.best_model_path}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not remove old checkpoint: {e}")
        
        self.best_model_path = filepath
        
        if self.verbose:
            print(f"Checkpoint saved: {filepath}")


class LearningRateLogger(Callback):
    """Log learning rate changes during training."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.lr_history = []
    
    def on_epoch_start(self, trainer, epoch: int, **kwargs):
        """Log current learning rate."""
        if trainer.optimizer:
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            self.lr_history.append(current_lr)
            
            if self.verbose and len(self.lr_history) > 1:
                prev_lr = self.lr_history[-2]
                if abs(current_lr - prev_lr) > 1e-8:
                    print(f"Learning rate changed: {prev_lr:.2e} → {current_lr:.2e} at epoch {epoch}")


class GradientClipping(Callback):
    """Apply gradient clipping during training."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Initialize gradient clipping callback.
        
        Args:
            max_norm: Maximum norm of gradients
            norm_type: Type of norm to use (2.0 for L2 norm)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def on_batch_end(self, trainer, batch_idx: int, metrics: Dict[str, float], **kwargs):
        """Apply gradient clipping after backward pass."""
        if trainer.model.training:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 
                                         self.max_norm, self.norm_type)


class MetricsLogger(Callback):
    """Log metrics to various destinations."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize metrics logger.
        
        Args:
            log_file: Optional file to log metrics to
        """
        self.log_file = log_file
        self.metrics_history = []
        
        if self.log_file:
            # Write header
            with open(self.log_file, "w") as f:
                f.write("epoch,")
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Log epoch metrics."""
        # Add to history
        epoch_data = {"epoch": epoch, **metrics}
        self.metrics_history.append(epoch_data)
        
        # Log to file if specified
        if self.log_file:
            if len(self.metrics_history) == 1:
                # Write header on first epoch
                with open(self.log_file, "w") as f:
                    header = ",".join(epoch_data.keys())
                    f.write(header + "\n")
            
            # Write metrics
            with open(self.log_file, "a") as f:
                values = ",".join([str(v) for v in epoch_data.values()])
                f.write(values + "\n") 

class WandbLogger(Callback):
    """Log metrics to W&B Server"""

    def __init__(self, project_name: str, 
                 epoch_metrics: List[str], batch_metrics: List[str], 
                 local_dir: Path, experiment_name_prefix: str):
        pass