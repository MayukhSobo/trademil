"""
Main Trainer class for Treadmill framework.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
import os
import glob
import re

from .config import TrainingConfig
from .metrics import MetricsTracker, compute_metrics
from .utils import ProgressTracker, print_model_summary
from .callbacks import Callback, EarlyStopping, ModelCheckpoint
from .report import TrainingReport, create_training_report_from_trainer, display_training_report, HardwareMonitor

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
        self.start_epoch = 0  # For resume training
        self.stop_training = False
        self._history = None  # Store training history for later access
        self.training_report = None  # Store comprehensive training report
        
        # Initialize hardware monitoring
        self.hardware_monitor = HardwareMonitor()
        
        # Mixed precision setup
        self.scaler = None
        if config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Handle resume training if configured
        if self.config.resume_training:
            self._resume_from_checkpoint()
    
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
        
        # Add model checkpointing automatically when checkpoint_dir is provided
        if self.config.checkpoint_dir is not None:
            # Use the checkpoint directory from config
            experiment_dir = self.config.checkpoint_dir
            
            # Store the experiment directory for potential later use
            self.experiment_dir = experiment_dir
            
            # Choose monitor metric based on whether validation data is available
            if self.val_dataloader:
                monitor_metric = "val_loss"
                checkpoint_path = os.path.join(
                    experiment_dir, 
                    "checkpoint_{epoch_1based:03d}_{val_loss:.4f}.pt"
                )
            else:
                monitor_metric = "loss"
                checkpoint_path = os.path.join(
                    experiment_dir, 
                    "checkpoint_{epoch_1based:03d}_{loss:.4f}.pt"
                )
            
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor_metric,
                save_best_only=not self.config.keep_all_checkpoints,  # Only save best if keep_all_checkpoints=False
                verbose=True
            )
            
            # For resume training, initialize the checkpoint callback with existing best model info
            if self.config.resume_training:
                self._initialize_checkpoint_for_resume(checkpoint, monitor_metric)
            
            self.callbacks.append(checkpoint)
    
    def _resume_from_checkpoint(self):
        """Resume training from the latest checkpoint in the checkpoint directory."""
        if not os.path.exists(self.config.checkpoint_dir):
            raise ValueError(f"Checkpoint directory does not exist: {self.config.checkpoint_dir}")
        
        # Find all checkpoint files (both best model and training checkpoints)
        checkpoint_pattern = os.path.join(self.config.checkpoint_dir, "*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {self.config.checkpoint_dir}")
        
        training_checkpoints = [f for f in checkpoint_files if "training_checkpoint_epoch_" in os.path.basename(f)]
        
        latest_checkpoint = None
        
        # PRIORITY 1: Use the most recent training checkpoint (represents actual training state)
        if training_checkpoints:
            latest_checkpoint = self._find_latest_training_checkpoint(training_checkpoints)
        
        # PRIORITY 2: If no training checkpoint found, use the most recently modified file
        if not latest_checkpoint:
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        print(f"🔄 Resuming training from checkpoint: {os.path.basename(latest_checkpoint)}")
        
        # Load the checkpoint
        checkpoint_data = self.load_checkpoint(latest_checkpoint, resume_training=True)
        
        # Set starting epoch for the training loop (start from NEXT epoch after checkpoint)
        self.start_epoch = self.current_epoch + 1
        
        # Restore metrics history if available
        if "metrics_history" in checkpoint_data:
            self.metrics_tracker.epoch_metrics = checkpoint_data["metrics_history"]
        
        # Extract previous epoch metrics from checkpoint filename for "Change (from prev)" display
        prev_metrics = self._extract_metrics_from_checkpoint_filename(latest_checkpoint)
        if prev_metrics:
            self.progress_tracker.prev_epoch_metrics = prev_metrics

        # Provide clarity about checkpoint types
        print(f"💡 Note: Training checkpoints represent actual progress; normal checkpoint represents best performance")
    
    def _initialize_checkpoint_for_resume(self, checkpoint_callback, monitor_metric):
        """Initialize ModelCheckpoint callback with existing best checkpoint info for resume training."""
        import glob
        import re
        import os
        
        # Find all checkpoint files in the directory
        checkpoint_pattern = os.path.join(self.experiment_dir, "checkpoint_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return  # No existing checkpoints to initialize from
        
        best_checkpoint_path = None
        best_value = float('inf') if checkpoint_callback.mode == "min" else float('-inf')
        
        # Find the best checkpoint based on the monitored metric
        for checkpoint_file in checkpoint_files:
            filename = os.path.basename(checkpoint_file)
            
            # Extract metric value from checkpoint filename
            # Pattern: checkpoint_XXX_METRIC_VALUE.pt
            if monitor_metric == "val_loss":
                match = re.search(r'checkpoint_(\d+)_(\d+\.?\d*)\.pt$', filename)
            else:  # monitor_metric == "loss"
                match = re.search(r'checkpoint_(\d+)_(\d+\.?\d*)\.pt$', filename)
            
            if match:
                try:
                    metric_value = float(match.group(2))
                    
                    # Check if this is better than current best
                    if checkpoint_callback.mode == "min":
                        is_better = metric_value < best_value
                    else:
                        is_better = metric_value > best_value
                    
                    if is_better:
                        best_value = metric_value
                        best_checkpoint_path = checkpoint_file
                        
                except ValueError:
                    continue  # Skip if we can't parse the metric value
        
        # Initialize the callback with the existing best checkpoint info
        if best_checkpoint_path:
            checkpoint_callback.best_model_path = best_checkpoint_path
            checkpoint_callback.best_value = best_value

    def _extract_metrics_from_checkpoint_filename(self, checkpoint_path):
        """Extract metrics from checkpoint filename for resume training comparison."""
        import re
        import os
        
        filename = os.path.basename(checkpoint_path)
        
        # Try to extract from training checkpoint filename: training_checkpoint_epoch_003_0.0686.pt
        training_match = re.search(r'training_checkpoint_epoch_(\d+)_(\d+\.?\d*)\.pt$', filename)
        if training_match:
            loss_value = float(training_match.group(2))
            # Training checkpoints typically store validation loss if validation is enabled
            if self.val_dataloader:
                # Return as validation metrics format (what gets stored as val_metrics)
                return {"loss": loss_value}  # This will be used as val_metrics for comparison
            else:
                # Return as training metrics format
                return {"loss": loss_value}
        
        # Try to extract from best model checkpoint filename: checkpoint_003_0.0664.pt
        best_match = re.search(r'checkpoint_(\d+)_(\d+\.?\d*)\.pt$', filename)
        if best_match:
            loss_value = float(best_match.group(2))
            # Best model checkpoints are saved based on the monitored metric
            if self.val_dataloader:
                # This represents validation loss (monitored metric for best model)
                return {"loss": loss_value}  # This will be used as val_metrics for comparison
            else:
                # This represents training loss (monitored metric when no validation)
                return {"loss": loss_value}
        
        return None
    
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
        from rich.text import Text
        from treadmill.utils import COLORS, console
        if self.config.checkpoint_dir is None:
            console.print("\n🚨 Checkpoint directory not configured. No checkpoints will be saved. 🚨", 
                          style=f"bold {COLORS['warning']}", justify="center")
        
        # Print model summary
        print_model_summary(self.model)
        
        # Calculate remaining epochs
        remaining_epochs = self.config.epochs - self.start_epoch
        total_epochs_to_run = remaining_epochs
        
        # Initialize progress tracking
        self.progress_tracker.start_training(
            total_epochs=total_epochs_to_run,
            total_batches_per_epoch=len(self.train_dataloader)
        )
        
        # Start training callbacks
        self._call_callbacks("on_train_start")
        
        # Initial hardware sample
        self.hardware_monitor.sample()
        
        try:
            # Start from the correct epoch (either 0 for new training or start_epoch for resume)
            for epoch in range(self.start_epoch, self.config.epochs):
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
                
                # Save training checkpoint every 2 epochs for better resume capability
                if (self.config.checkpoint_dir is not None and 
                    ((epoch + 1) % 2 == 0 or epoch == self.config.epochs - 1)):
                    # Use validation loss if available, otherwise training loss
                    loss_for_checkpoint = epoch_metrics.get("val_loss", epoch_metrics.get("loss", 0.0))
                    self.save_training_checkpoint(epoch, loss_for_checkpoint)
                
        except KeyboardInterrupt:
            # Save checkpoint when training is interrupted to preserve progress
            if self.config.checkpoint_dir is not None:
                print("\n💾 Saving checkpoint due to interruption...")
                # Use the last known loss value
                last_loss = getattr(self, '_last_loss', 0.0)
                self.save_training_checkpoint(self.current_epoch, last_loss)
            
            from treadmill.utils import COLORS, console
            from rich.text import Text
            interrupt_text = Text("Training interrupted by user", style=f"bold {COLORS['warning']}")
            console.print(f"\n{interrupt_text}")
            console.print(f"🔄 You can resume training from epoch {self.current_epoch + 1}")
            
        finally:
            # Training end callbacks
            self._call_callbacks("on_train_end")
            
            # Calculate total training time
            self.progress_tracker.finish_training()
            
            # Generate and display comprehensive training report
            self.training_report = create_training_report_from_trainer(self)
            display_training_report(self.training_report)
        
        # Return training history
        history = {
            "train_metrics": self.metrics_tracker.get_epoch_metrics("train"),
            "val_metrics": self.metrics_tracker.get_epoch_metrics("val"),
            "best_metrics": self.metrics_tracker.get_best_metrics("val"),
            "total_epochs": self.current_epoch + 1
        }
        
        # Store history for later access via .history property
        self._history = history
        return history
    
    def fit(self) -> Dict[str, Any]:
        """
        Alias for train() method for sklearn-style compatibility.
        
        Many users expect a fit() method from sklearn/other ML libraries.
        This method simply calls train() for compatibility.
        
        Returns:
            Dictionary containing training history and final metrics
        """
        return self.train()
    
    @property
    def history(self) -> Optional[Dict[str, Any]]:
        """
        Access training history after training has completed.
        
        This property allows access to training results even if the return value
        from train() or fit() was not stored in a variable.
        
        Returns:
            Dictionary containing training history and final metrics, or None if training hasn't run yet
            
        Example:
            trainer = Trainer(...)
            trainer.fit()  # Don't store the result
            
            # Access history later
            print(f"Total epochs: {trainer.history['total_epochs']}")
            print(f"Best accuracy: {trainer.history['best_metrics']['accuracy']:.4f}")
        """
        return self._history
    
    @property
    def report(self) -> Optional[TrainingReport]:
        """
        Access comprehensive training report after training has completed.
        
        This property provides detailed information about the training session
        including model info, configuration, metrics, timing, and more.
        
        Returns:
            TrainingReport object with comprehensive training information, or None if training hasn't completed yet
            
        Example:
            trainer = Trainer(...)
            trainer.fit()
            
            # Access detailed report
            print(f"Model: {trainer.report.model_name}")
            print(f"Parameters: {trainer.report.total_parameters:,}")
            print(f"Training time: {trainer.report.training_time:.1f}s")
            print(f"Best loss: {trainer.report.best_metrics['val_loss']:.4f}")
            
            # Convert to dictionary for serialization
            report_dict = trainer.report.to_dict()
        """
        return self.training_report
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Batch start callbacks
            self._call_callbacks("on_batch_start", batch_idx=batch_idx)
            
            # Process batch
            batch_metrics = self._train_step(batch, batch_idx)
            
            # Sample hardware usage (every 10 batches to avoid overhead)
            if batch_idx % 10 == 0:
                self.hardware_monitor.sample()
            
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
        
        # Store last loss for interruption handling
        self._last_loss = train_metrics.get("loss", 0.0)
        
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
                
                # Sample hardware usage during validation (every 20 batches to reduce overhead)
                if batch_idx % 20 == 0:
                    self.hardware_monitor.sample()
                
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
        if hasattr(self, 'experiment_dir') and self.experiment_dir and os.path.dirname(filepath) == "":
            filepath = os.path.join(self.experiment_dir, filepath)
        elif not hasattr(self, 'experiment_dir') or not self.experiment_dir:
            # No experiment directory configured
            if os.path.dirname(filepath) == "":
                print("⚠️  Warning: No checkpoint directory configured. Please provide a full file path or configure checkpoint_dir.")
                return
        
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
    
    def save_training_checkpoint(self, epoch: int, loss_value: float):
        """Save a comprehensive training checkpoint for resume capability."""
        if not hasattr(self, 'experiment_dir') or not self.experiment_dir:
            return
        
        # Create filename with epoch number and loss value
        checkpoint_filename = f"training_checkpoint_epoch_{epoch + 1:03d}_{loss_value:.4f}.pt"
        checkpoint_path = os.path.join(self.experiment_dir, checkpoint_filename)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "metrics_history": self.metrics_tracker.epoch_metrics,
            "current_epoch": self.current_epoch,
            "start_epoch": self.start_epoch,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save additional training state
        checkpoint.update({
            "device": str(self.device),
            "stop_training": self.stop_training,
        })
        
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only the latest training checkpoint
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the latest training checkpoint to ensure maximum 2 files total."""
        if not hasattr(self, 'experiment_dir') or not self.experiment_dir:
            return
        
        checkpoint_pattern = os.path.join(self.experiment_dir, "training_checkpoint_epoch_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= 1:
            return
        
        # Sort by epoch number (extract from filename)
        def get_epoch_from_filename(filename):
            match = re.search(r'epoch_(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else 0
        
        checkpoint_files.sort(key=get_epoch_from_filename)
        
        # Remove all but the latest training checkpoint
        for old_checkpoint in checkpoint_files[:-1]:
            try:
                os.remove(old_checkpoint)
            except OSError:
                pass  # Ignore if file doesn't exist or can't be removed
    
    def _find_latest_training_checkpoint(self, training_checkpoints):
        """Find the most recent training checkpoint based on epoch number."""
        latest_checkpoint = None
        latest_epoch = -1
        latest_loss = None
        
        for checkpoint_file in training_checkpoints:
            filename = os.path.basename(checkpoint_file)
            
            # Extract epoch number and loss value from filename (e.g., "training_checkpoint_epoch_010_1.5678.pt")
            match = re.search(r'training_checkpoint_epoch_(\d+)_(\d+\.?\d*)\.pt$', filename)
            if match:
                epoch_num = int(match.group(1))
                loss_value = float(match.group(2))
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_loss = loss_value
                    latest_checkpoint = checkpoint_file
        
        # if latest_checkpoint and latest_loss is not None:
        #     print(f"📈 Selected training checkpoint of epoch {latest_epoch} with loss: {latest_loss:.4f}")
        
        return latest_checkpoint
    
    # // TODO: Investigate if we really need the `resume_training` param here
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
            
            # Load scaler state for mixed precision training
            if self.scaler and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            # Load additional training state if available
            if "start_epoch" in checkpoint:
                self.start_epoch = checkpoint["start_epoch"]
            
            if "stop_training" in checkpoint:
                self.stop_training = checkpoint["stop_training"]
        return checkpoint 