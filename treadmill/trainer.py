"""
Main Trainer class for Treadmill framework.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any, Union
import os

from .config import TrainingConfig
from .metrics import MetricsTracker, compute_metrics
from .utils import ProgressTracker, print_model_summary, create_experiment_dir
from .callbacks import Callback, EarlyStopping, ModelCheckpoint


class Trainer:
    """
    Main training class that orchestrates the entire training process.
    
    This class provides a clean, modular interface for PyTorch model training
    with support for validation, callbacks, metrics tracking, and more.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 loss_fn: Optional[Callable] = None,
                 metric_fns: Optional[Dict[str, Callable]] = None,
                 callbacks: Optional[List[Callback]] = None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            loss_fn: Loss function (if None, will try to infer from model)
            metric_fns: Dictionary of metric functions
            callbacks: List of callbacks for training hooks
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns or {}
        
        # Setup device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_callbacks(callbacks)
        
        # Training state
        self.metrics_tracker = MetricsTracker()
        self.progress_tracker = ProgressTracker()
        self.current_epoch = 0
        self.stop_training = False
        
        # Mixed precision setup
        self.scaler = None
        if config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _setup_optimizer(self):
        """Setup optimizer from config."""
        optimizer_class = self.config.optimizer.optimizer_class
        optimizer_params = {
            "lr": self.config.optimizer.lr,
            "weight_decay": self.config.optimizer.weight_decay,
            **self.config.optimizer.params
        }
        
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler from config."""
        self.scheduler = None
        if self.config.scheduler and self.config.scheduler.scheduler_class:
            scheduler_class = self.config.scheduler.scheduler_class
            scheduler_params = self.config.scheduler.params
            self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
    
    def _setup_callbacks(self, callbacks: Optional[List[Callback]]):
        """Setup callbacks with default ones if needed."""
        self.callbacks = callbacks or []
        
        # Add default early stopping if configured
        if self.config.early_stopping_patience:
            early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                verbose=True
            )
            self.callbacks.append(early_stopping)
        
        # Add default model checkpointing if configured
        if self.config.save_best_model:
            # Create unique experiment directory for this training run
            experiment_dir = create_experiment_dir(self.config.checkpoint_dir)
            
            # Store the experiment directory for potential later use
            self.experiment_dir = experiment_dir
            
            # Inform user about the experiment directory
            print(f"🔄 Experiment directory created: {experiment_dir}")
            
            # Choose monitor metric based on whether validation data is available
            if self.val_dataloader:
                monitor_metric = "val_loss"
                checkpoint_path = os.path.join(
                    experiment_dir, 
                    "best_model_epoch_{epoch:03d}_{val_loss:.4f}.pt"
                )
            else:
                monitor_metric = "loss"
                checkpoint_path = os.path.join(
                    experiment_dir, 
                    "best_model_epoch_{epoch:03d}_{loss:.4f}.pt"
                )
            
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor_metric,
                save_best_only=True,
                verbose=True
            )
            self.callbacks.append(checkpoint)
    
    def _call_callbacks(self, event: str, **kwargs):
        """Call all callbacks for a specific event."""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                method(self, **kwargs)
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the complete training loop.
        
        Returns:
            Dictionary containing training history and final metrics
        """
        # Print model summary
        print_model_summary(self.model)
        
        # Initialize progress tracking
        self.progress_tracker.start_training(
            total_epochs=self.config.epochs,
            total_batches_per_epoch=len(self.train_dataloader)
        )
        
        # Start training callbacks
        self._call_callbacks("on_train_start")
        
        try:
            for epoch in range(self.config.epochs):
                if self.stop_training:
                    break
                    
                self.current_epoch = epoch
                self.progress_tracker.start_epoch(epoch, len(self.train_dataloader), self.config.epochs, self.config.progress_bar)
                # Only print epoch header if Rich Live display is not being used
                if not self.config.progress_bar:
                    self.progress_tracker.print_epoch_header(epoch, self.config.epochs)
                
                # Epoch start callbacks
                self._call_callbacks("on_epoch_start", epoch=epoch)
                
                # Training phase
                train_metrics = self._train_epoch(epoch)
                
                # End the Rich Live display for this epoch
                self.progress_tracker.end_epoch_display()
                
                # Validation phase
                val_metrics = None
                if (self.val_dataloader and 
                    (epoch + 1) % self.config.validate_every == 0):
                    val_metrics = self._validate_epoch(epoch)
                
                # Combine metrics and update tracker
                epoch_metrics = {**train_metrics}
                if val_metrics:
                    epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                
                # Update learning rate scheduler
                if self.scheduler:
                    if hasattr(self.scheduler, 'step'):
                        if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                            # ReduceLROnPlateau needs a metric
                            monitor_metric = epoch_metrics.get("val_loss", epoch_metrics.get("loss", 0))
                            self.scheduler.step(monitor_metric)
                        else:
                            self.scheduler.step()
                
                # Check if early stopping is active
                has_early_stopping = any(
                    callback.__class__.__name__ == 'EarlyStopping' 
                    for callback in self.callbacks
                )
                
                # Print epoch summary
                self.progress_tracker.print_epoch_summary(
                    epoch, train_metrics, val_metrics, self.config, has_early_stopping
                )
                
                # Epoch end callbacks
                self._call_callbacks("on_epoch_end", epoch=epoch, metrics=epoch_metrics)
                
        except KeyboardInterrupt:
            from rich.text import Text
            from treadmill.utils import COLORS, console
            interrupt_text = Text("Training interrupted by user", style=f"bold {COLORS['warning']}")
            console.print(f"\n{interrupt_text}")
            
        finally:
            # Training end callbacks
            self._call_callbacks("on_train_end")
            self.progress_tracker.finish_training()
        
        # Return training history
        return {
            "train_metrics": self.metrics_tracker.get_epoch_metrics("train"),
            "val_metrics": self.metrics_tracker.get_epoch_metrics("val"),
            "best_metrics": self.metrics_tracker.get_best_metrics("val"),
            "total_epochs": self.current_epoch + 1
        }
    
    def fit(self) -> Dict[str, Any]:
        """
        Alias for train() method for sklearn-style compatibility.
        
        Many users expect a fit() method from sklearn/other ML libraries.
        This method simply calls train() for compatibility.
        
        Returns:
            Dictionary containing training history and final metrics
        """
        return self.train()
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Batch start callbacks
            self._call_callbacks("on_batch_start", batch_idx=batch_idx)
            
            # Process batch
            batch_metrics = self._train_step(batch, batch_idx)
            
            # Update metrics tracker
            self.metrics_tracker.update(batch_metrics, mode="train")
            
            # Print progress
            if self.config.progress_bar:
                self.progress_tracker.print_batch_progress(
                    batch_idx, len(self.train_dataloader), 
                    batch_metrics, self.config.print_every
                )
            
            # Batch end callbacks
            self._call_callbacks("on_batch_end", batch_idx=batch_idx, metrics=batch_metrics)
        
        # Compute epoch metrics
        epoch_metrics = self.metrics_tracker.end_epoch()
        train_metrics = {k.replace("train_", ""): v for k, v in epoch_metrics.items() 
                        if k.startswith("train_")}
        
        return train_metrics
    
    def _train_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        """Execute one training step."""
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = [item.to(self.device) if hasattr(item, 'to') else item for item in batch]
        else:
            batch = batch.to(self.device) if hasattr(batch, 'to') else batch
        
        # Zero gradients
        if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
            self.optimizer.zero_grad()
        
        # Forward pass
        device_type = 'cuda' if torch.cuda.is_available() and self.device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type, enabled=self.config.mixed_precision):
            if self.config.custom_forward_fn:
                outputs, targets = self.config.custom_forward_fn(self.model, batch)
            else:
                # Default forward pass assumes batch is (inputs, targets)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                else:
                    raise ValueError("Please provide custom_forward_fn or ensure batch format is (inputs, targets)")
            
            # Compute loss
            if self.loss_fn:
                loss = self.loss_fn(outputs, targets)
            else:
                # Try to get loss from model
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(outputs, targets)
                else:
                    raise ValueError("Please provide loss_fn or implement compute_loss method in model")
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.accumulate_grad_batches
        
        # Backward pass
        if self.config.custom_backward_fn:
            self.config.custom_backward_fn(loss, self.model, self.optimizer)
        else:
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Update parameters
        if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
            # Gradient clipping
            if self.config.grad_clip_norm:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = {"loss": loss.item() * self.config.accumulate_grad_batches}
            
            # Add custom metrics
            if self.metric_fns:
                custom_metrics = compute_metrics(outputs.detach(), targets.detach(), self.metric_fns)
                metrics.update(custom_metrics)
        
        return metrics
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute validation for one epoch."""
        if not self.val_dataloader:
            return {}
        
        self._call_callbacks("on_validation_start")
        
        self.model.eval()
        val_metrics_list = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch_metrics = self._validate_step(batch)
                val_metrics_list.append(batch_metrics)
                
                self.metrics_tracker.update(batch_metrics, mode="val")
        
        # Compute validation metrics
        val_epoch_metrics = self.metrics_tracker.end_epoch()
        val_metrics = {k.replace("val_", ""): v for k, v in val_epoch_metrics.items() 
                      if k.startswith("val_")}
        
        self._call_callbacks("on_validation_end", metrics=val_metrics)
        
        return val_metrics
    
    def _validate_step(self, batch: Any) -> Dict[str, float]:
        """Execute one validation step."""
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            batch = [item.to(self.device) if hasattr(item, 'to') else item for item in batch]
        else:
            batch = batch.to(self.device) if hasattr(batch, 'to') else batch
        
        # Forward pass
        device_type = 'cuda' if torch.cuda.is_available() and self.device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type, enabled=self.config.mixed_precision):
            if self.config.custom_forward_fn:
                outputs, targets = self.config.custom_forward_fn(self.model, batch)
            else:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                else:
                    raise ValueError("Please provide custom_forward_fn or ensure batch format is (inputs, targets)")
            
            # Compute loss
            if self.loss_fn:
                loss = self.loss_fn(outputs, targets)
            else:
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(outputs, targets)
                else:
                    raise ValueError("Please provide loss_fn or implement compute_loss method in model")
        
        # Compute metrics
        metrics = {"loss": loss.item()}
        
        # Add custom metrics
        if self.metric_fns:
            custom_metrics = compute_metrics(outputs.detach(), targets.detach(), self.metric_fns)
            metrics.update(custom_metrics)
        
        return metrics
    
    def save_checkpoint(self, filepath: str, additional_info: Optional[Dict] = None):
        """Save a training checkpoint."""
        # If experiment directory exists and filepath is just a filename, save to experiment directory
        if hasattr(self, 'experiment_dir') and os.path.dirname(filepath) == "":
            filepath = os.path.join(self.experiment_dir, filepath)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else self.config,
            "metrics_history": self.metrics_tracker.epoch_metrics
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, resume_training: bool = True):
        """Load a training checkpoint."""
        # Load with weights_only=False for full checkpoint compatibility
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if resume_training:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_epoch = checkpoint.get("epoch", 0)
            
            if self.scheduler and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint 