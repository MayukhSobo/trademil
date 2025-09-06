"""
Configuration classes for Treadmill training framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Callable
import torch.optim as optim


@dataclass
class OptimizerConfig:
    """Configuration for optimizer setup."""
    
    optimizer_class: Union[str, type] = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert string optimizer names to classes."""
        if isinstance(self.optimizer_class, str):
            self.optimizer_class = getattr(optim, self.optimizer_class)


@dataclass  
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    
    scheduler_class: Optional[Union[str, type]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert string scheduler names to classes."""
        if isinstance(self.scheduler_class, str) and self.scheduler_class:
            self.scheduler_class = getattr(optim.lr_scheduler, self.scheduler_class)


@dataclass
class TrainingConfig:
    """Main training configuration."""
    
    # Training parameters
    epochs: int = 10
    device: str = "auto"  # "auto", "cpu", "cuda", or specific device
    
    # Optimizer and scheduler
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = None
    
    # Validation settings
    validate_every: int = 1  # Validate every N epochs
    early_stopping_patience: Optional[int] = None
    
    # Checkpointing
    save_best_model: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Display settings
    print_every: int = 10  # Print progress every N batches
    progress_bar: bool = True
    
    # Custom forward/backward functions
    custom_forward_fn: Optional[Callable] = None
    custom_backward_fn: Optional[Callable] = None
    
    # Additional settings
    grad_clip_norm: Optional[float] = None
    accumulate_grad_batches: int = 1
    mixed_precision: bool = False
    
    # Overfitting detection
    overfit_threshold: float = 0.1  # Threshold for overfitting warning
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if isinstance(self.optimizer, dict):
            # Separate known OptimizerConfig parameters from optimizer-specific ones
            optimizer_dict = self.optimizer.copy()
            optimizer_params = {}
            
            # Extract parameters that don't belong to OptimizerConfig
            for key in ['momentum', 'nesterov', 'betas', 'eps', 'amsgrad']:
                if key in optimizer_dict:
                    optimizer_params[key] = optimizer_dict.pop(key)
            
            # Add any remaining params to the params dict
            if 'params' in optimizer_dict:
                optimizer_params.update(optimizer_dict.pop('params'))
            
            optimizer_dict['params'] = optimizer_params
            self.optimizer = OptimizerConfig(**optimizer_dict)
            
        if self.scheduler and isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler) 