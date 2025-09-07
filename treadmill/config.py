"""
Configuration classes for Treadmill training framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Callable
import torch.optim as optim
from datetime import datetime
import pytz
import os


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
    project_name: Optional[str] = None  # Project name for experiment directory
    use_experiment_dir: bool = True  # Create unique experiment directories
    timezone: str = "IST"  # Default timezone for experiment naming
    
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
        
        # Create experiment directory if enabled
        if self.use_experiment_dir:
            self.checkpoint_dir = self._create_experiment_dir()
            
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
    
    def _create_experiment_dir(self) -> str:
        """Create a unique experiment directory with timestamp."""
        # Get current time in specified timezone
        if self.timezone == "IST":
            tz = pytz.timezone('Asia/Kolkata')
        elif self.timezone == "EST":
            tz = pytz.timezone('US/Eastern')
        elif self.timezone == "PST":
            tz = pytz.timezone('US/Pacific')
        elif self.timezone == "UTC":
            tz = pytz.UTC
        else:
            # Try to use the provided timezone string directly
            try:
                tz = pytz.timezone(self.timezone)
            except:
                # Fallback to UTC if timezone is invalid
                tz = pytz.UTC
                self.timezone = "UTC"
        
        now = datetime.now(tz)
        
        # Format date as DD-MM-YYYY
        date_str = now.strftime("%d-%m-%Y")
        
        # Format time as HH:MM:SSam/pm (e.g., 01:45:30pm)
        time_str = now.strftime("%I:%M:%S%p").lower()
        
        # Create project name if not provided
        if not self.project_name:
            # Try to infer project name from parent directory or use default
            try:
                import os
                current_dir = os.getcwd()
                parent_dir = os.path.basename(current_dir)
                # Use parent directory name if it's not a generic name
                if parent_dir and parent_dir not in ['src', 'lib', 'app', 'project', 'code']:
                    self.project_name = parent_dir
                else:
                    self.project_name = "experiment"
            except:
                self.project_name = "experiment"
        
        # Clean project name (remove special characters)
        project_name_clean = "".join(c if c.isalnum() or c in ['-', '_'] else '_' 
                                    for c in self.project_name)
        
        # Create experiment directory name
        exp_dir_name = f"{project_name_clean}-experiment-{date_str}-{time_str}-{self.timezone}"
        
        # Create full path
        exp_dir_path = os.path.join(self.checkpoint_dir, exp_dir_name)
        
        # Create directory if it doesn't exist
        os.makedirs(exp_dir_path, exist_ok=True)
        
        # Print the created directory for user awareness
        print(f"📁 Experiment directory created: {exp_dir_path}")
        
        return exp_dir_path 