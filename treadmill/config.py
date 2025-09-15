"""
Configuration classes for Treadmill training framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Callable
import torch.optim as optim
from datetime import datetime
import pytz
import os
import re
from rich.console import Console

# Rich console for warnings
console = Console()


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
    additional_epochs: Optional[int] = None  # For resume training: specify additional epochs instead of total
    device: str = "auto"  # "auto", "cpu", "cuda", or specific device
    
    # Optimizer and scheduler
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = None
    
    # Validation settings
    validate_every: int = 1  # Good practice to validate every 1 epoch
    early_stopping_patience: Optional[int] = None
    
    # Checkpointing settings
    checkpoint_dir: Optional[str] = None
    project_name: Optional[str] = None
    resume_training: bool = False
    keep_all_checkpoints: bool = False  # if False, only keep the best
    timezone: str = "UTC"
    
    # Display settings
    print_every: int = 10  # Print metrics every 10 batches
    progress_bar: bool = True
    
    # Custom forward/backward functions for complex architectures
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
        
        # Validate resume training requirements
        if self.resume_training and self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir cannot be None when resume_training is True. "
                           "Please provide the path to the existing checkpoint directory.")
        
        # Handle resume training: extract project_name and timezone from checkpoint directory
        if self.resume_training:
            self._handle_resume_training_config()
        
        # Handle automatic epoch calculation for resume training
        if self.resume_training and self.additional_epochs is not None:
            # Automatically calculate total epochs from checkpoint + additional epochs
            last_epoch = self._get_last_completed_epoch()
            if last_epoch > 0:
                self.epochs = last_epoch + self.additional_epochs
            else:
                raise ValueError("No valid checkpoints found for resume training")
        
        # Handle experiment directory setup
        if self.resume_training:
            # Resume training: validate existing directory
            self._validate_resume_directory()
        elif self.checkpoint_dir is not None:
            # Not resuming but checkpoint_dir provided: create new experiment dir within it
            base_dir = self.checkpoint_dir
            self.checkpoint_dir = self._create_experiment_dir(base_dir)
        # If checkpoint_dir is None and not resuming: no checkpointing (do nothing)
            
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
    
    def _create_experiment_dir(self, base_dir: str = "./checkpoints") -> str:
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
            except (pytz.UnknownTimeZoneError, AttributeError):
                # Fallback to UTC if timezone is invalid
                tz = pytz.UTC
                self.timezone = "UTC"
        
        now = datetime.now(tz)
        
        # Format date as DD-MM-YYYY
        date_str = now.strftime("%d-%m-%Y")
        
        # Format time as HH-MM-SSam/pm (e.g., 01-45-30pm) - Windows compatible
        time_str = now.strftime("%I:%M:%S%p").lower().replace(":", "-")
        
        # Create project name if not provided
        if not self.project_name:
            # Try to infer project name from parent directory or use default
            try:
                current_dir = os.getcwd()
                parent_dir = os.path.basename(current_dir)
                # Use parent directory name if it's not a generic name
                if parent_dir and parent_dir not in ['src', 'lib', 'app', 'project', 'code']:
                    self.project_name = parent_dir
                else:
                    self.project_name = "experiment"
            except (OSError, AttributeError):
                self.project_name = "experiment"
        
        # Clean project name (remove special characters)
        project_name_clean = "".join(c if c.isalnum() or c in ['-', '_'] else '_'
                                      for c in self.project_name)
        
        # Create experiment directory name
        exp_dir_name = f"{project_name_clean}-experiment-{date_str}-{time_str}-{self.timezone}"
        
        # Create full path
        exp_dir_path = os.path.join(base_dir, exp_dir_name)
        
        # Create directory if it doesn't exist
        os.makedirs(exp_dir_path, exist_ok=True)
        
        # Print the created directory for user awareness
        print(f"📁 Experiment directory created: {exp_dir_path}")
        
        return exp_dir_path
    
    def _validate_resume_directory(self) -> None:
        """Validate that the resume directory exists and contains checkpoints."""
        if not os.path.exists(self.checkpoint_dir):
            raise ValueError(f"Resume directory does not exist: {self.checkpoint_dir}")
        
        if not os.path.isdir(self.checkpoint_dir):
            raise ValueError(f"Resume path is not a directory: {self.checkpoint_dir}")
        
        # Print resume message
        print(f"🔍 Looking for checkpoints in: {self.checkpoint_dir}") 
    
    def _handle_resume_training_config(self):
        """Extract project_name and timezone from checkpoint directory when resuming."""
        extracted_project_name, extracted_timezone = self._extract_project_info_from_checkpoint_dir()

        if extracted_project_name and extracted_timezone:
            user_provided_different_values = (
                (self.project_name is not None and self.project_name != extracted_project_name) or
                (self.timezone != "UTC" and self.timezone != extracted_timezone)
            )

            if user_provided_different_values:
                console.print("[yellow]⚠️  Resume Training: Ignoring user-provided project-name and timezone[/yellow]")
                console.print("[yellow]   Using values from checkpoint directory for consistency:[/yellow]")
                console.print(f"[yellow]   • Project name: {extracted_project_name}[/yellow]")
                console.print(f"[yellow]   • Timezone: {extracted_timezone}[/yellow]")
                console.print()

            self.project_name = extracted_project_name
            self.timezone = extracted_timezone
        else:
            console.print("[yellow]⚠️  Warning: Could not extract project info from checkpoint directory path[/yellow]")
            console.print(f"[yellow]   Using provided values: project_name='{self.project_name}', timezone='{self.timezone}'[/yellow]")
            console.print()
    
    def _extract_project_info_from_checkpoint_dir(self):
        """Extract project name and timezone from checkpoint directory path.
        
        Expected format: {project_name}-experiment-{date}-{time}-{timezone}
        Example: mnist-experiment-15-09-2025-10:31:10pm-IST
        
        Returns:
            tuple: (project_name, timezone) or (None, None) if parsing fails
        """
        # Get the directory name (last part of path)
        dir_name = os.path.basename(self.checkpoint_dir.rstrip('/'))
        
        # Pattern to match: project_name-experiment-DD-MM-YYYY-HH:MM:SSam/pm-TIMEZONE
        pattern = r'^(.+?)-experiment-\d{2}-\d{2}-\d{4}-\d{1,2}:\d{2}:\d{2}[ap]m-(.+)$'
        match = re.match(pattern, dir_name)
        
        if match:
            project_name = match.group(1)
            timezone = match.group(2)
            return project_name, timezone
        
        return None, None

    def _get_last_completed_epoch(self):
        """Extract the last completed epoch from checkpoint files."""
        import os
        import glob
        import re
        
        if not os.path.exists(self.checkpoint_dir):
            return 0
        
        # Find all checkpoint files (both types)
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return 0
        
        max_epoch = 0
        
        # Check training checkpoint files first (these have the actual training progress)
        for checkpoint_file in checkpoint_files:
            filename = os.path.basename(checkpoint_file)
            
            # Extract epoch from training checkpoint filename (1-based): training_checkpoint_epoch_016_0.0516.pt
            training_match = re.search(r'training_checkpoint_epoch_(\d+)_(\d+\.?\d*)\.pt$', filename)
            if training_match:
                epoch_num = int(training_match.group(1))  # Already 1-based
                max_epoch = max(max_epoch, epoch_num)
                continue
            
            # Extract epoch from best model checkpoint filename (now 1-based): checkpoint_016_0.0516.pt  
            best_match = re.search(r'checkpoint_(\d+)_(\d+\.?\d*)\.pt$', filename)
            if best_match:
                epoch_num = int(best_match.group(1))  # Now 1-based after our fix
                max_epoch = max(max_epoch, epoch_num)
        
        return max_epoch 