"""
Utility functions for Treadmill framework.
"""

import time
import os
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box
from rich.live import Live
from rich.text import Text
from rich.console import Group
from rich.align import Align


console = Console()


def create_experiment_dir(base_dir: str = "./checkpoints") -> str:
    """
    Create a unique experiment directory with format:
    [PROJECT_NAME]-experiment-[date]-[time]-[TIMEZONE]
    
    Args:
        base_dir: Base directory where experiment folders will be created
        
    Returns:
        str: Path to the created experiment directory
        
    Example:
        "trademil-experiment-25-12-2024-2-30-00pm-UTC"
    """
    # Try to get project name from different sources
    project_name = _get_project_name()
    
    # Get current datetime with timezone
    now = datetime.now()
    
    # Format date as DD-MM-YYYY
    date_str = now.strftime("%d-%m-%Y")
    
    # Format time as 1-45-30pm (12-hour with seconds and am/pm) - Windows compatible
    time_str = now.strftime("%I:%M:%S%p").lower().replace(":", "-")
    
    # Get timezone (simplified)
    try:
        import time
        timezone = time.tzname[time.daylight] if time.daylight else time.tzname[0]
        # Fallback to UTC if timezone is empty/None
        if not timezone:
            timezone = "UTC"
    except:
        timezone = "UTC"
    
    # Create experiment directory name
    experiment_name = f"{project_name}-experiment-{date_str}-{time_str}-{timezone}"
    experiment_path = os.path.join(base_dir, experiment_name)
    
    # Create the directory
    os.makedirs(experiment_path, exist_ok=True)
    
    return experiment_path


def _get_project_name() -> str:
    """
    Try to detect project name from various sources.
    
    Returns:
        str: Project name or 'treadmill' as default
    """
    # Try 1: Git repository name
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            git_root = result.stdout.strip()
            project_name = os.path.basename(git_root)
            if project_name and project_name != ".":
                return project_name.lower().replace(" ", "-")
    except:
        pass
    
    # Try 2: Current working directory name
    try:
        cwd = os.getcwd()
        dir_name = os.path.basename(cwd)
        if dir_name and dir_name != ".":
            return dir_name.lower().replace(" ", "-")
    except:
        pass
    
    # Try 3: Check for common project files (pyproject.toml, setup.py, package.json)
    try:
        if os.path.exists("pyproject.toml"):
            # Try to parse project name from pyproject.toml
            with open("pyproject.toml", "r") as f:
                content = f.read()
                import re
                match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1).lower().replace(" ", "-")
        
        if os.path.exists("setup.py"):
            # Try to parse project name from setup.py
            with open("setup.py", "r") as f:
                content = f.read()
                import re
                match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1).lower().replace(" ", "-")
    except:
        pass
    
    # Default fallback
    return "treadmill"


def get_universal_colors():
    """
    Universal colors using hex codes that work well across all environments and backgrounds.
    These hex colors provide consistent appearance and good contrast on both light and dark backgrounds.
    """
    return {
        'header': '#0066FF',           # Professional blue - great contrast everywhere
        'epoch': '#9933CC',            # Purple/magenta - excellent visibility 
        'train': '#228B22',            # Forest green - clear positive indicator
        'val': '#1E90FF',              # Dodger blue - distinct from train
        'metric': 'default',           # Default terminal color (adapts automatically)
        'improvement': '#228B22',      # Same green as train - consistency
        'regression': '#DC143C',       # Crimson red - clear negative indicator
        'warning': '#FF8C00',          # Dark orange - visible warning color
        'success': '#228B22',          # Same green - success indicator
        'info': '#20B2AA'              # Light sea green - info color
    }


# Universal colors for all environments
COLORS = get_universal_colors()


def set_color_theme(color_dict: Optional[Dict[str, str]] = None):
    """
    Override color theme with custom colors.
    
    Args:
        color_dict: Custom color dictionary to use
    """
    global COLORS
    
    if color_dict:
        COLORS.update(color_dict)
    else:
        COLORS = get_universal_colors()


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format."""
    if seconds < 60:
        return f"{round(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = round(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = round(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """Format metrics dictionary for display."""
    return {key: f"{value:.{precision}f}" for key, value in metrics.items()}


def format_number(num: float, precision: int = 4) -> str:
    """Format a number for display."""
    if abs(num) >= 1e6:
        return f"{num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.{precision}f}"


class ProgressTracker:
    """Enhanced progress tracking with rich formatting."""
    
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
        self.live_display = None
        self.current_batch = 0
        self.total_time = 0.0  # Store total training time
        # Store previous epoch metrics for change calculation
        self.prev_epoch_metrics = None
        # Rich Live display components
        self.progress = None
        self.task_id = None
        self.title = None
        self.metrics_text = None
    
    def start_training(self, total_epochs: int, total_batches_per_epoch: int):
        """Initialize training progress tracking."""
        self.start_time = time.time()
        self.total_epochs = total_epochs
        self.total_batches_per_epoch = total_batches_per_epoch
        
        # Print training header
        # header_line = "_"*60
        # console.print(f"\n{header_line}", justify="center")
        # console.print(f"[bold {COLORS['header']}]🚀 Starting Training with Treadmill[/bold {COLORS['header']}]", justify="center")
        # console.print(header_line, justify="center")
    
    def start_epoch(self, epoch: int, total_batches: int, total_epochs: int = None, progress_bar: bool = True):
        """Start epoch tracking with Rich Live display or simple mode."""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        
        if progress_bar:
            # Setup Rich Live display components with Epoch X/Y format
            if total_epochs:
                epoch_title = f"\nEpoch {epoch + 1}/{total_epochs}"
            else:
                epoch_title = f"\nEpoch {epoch + 1}"
            self.title = Text(epoch_title, justify="left", style=f"bold {COLORS['epoch']}")
            self.metrics_text = Text("", justify="left", style=COLORS['info'])
            self.progress = Progress(
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                TimeElapsedColumn(),
            )
            
            # Create the layout group
            layout = Group(
                self.title,
                self.progress,
                self.metrics_text,
            )
            
            # Add task to progress bar
            self.task_id = self.progress.add_task("", total=total_batches)
            
            # Initialize Live display
            self.live_display = Live(layout, refresh_per_second=10)
            self.live_display.start()
        else:
            # Simple mode - no Rich Live display
            self.live_display = None
            self.progress = None
            self.task_id = None
    
    def print_epoch_header(self, epoch: int, total_epochs: int):
        """Print nice epoch header when Rich Live display is not used."""
        console.print(f"\n[bold {COLORS['epoch']}]Epoch {epoch+1}/{total_epochs}[/bold {COLORS['epoch']}]")
        console.print("-" * 40)
    
    def print_batch_progress(self, batch_idx: int, total_batches: int, 
                           metrics: Dict[str, float], print_every: int = 10):
        """Update batch progress using Rich Live display or simple print."""
        if self.live_display and self.progress and self.task_id is not None:
            # Rich Live display mode
            # Format metrics string (configurable based on provided metrics)
            if metrics:
                metrics_str = " | ".join([f"{k.replace('train_', '')}: {format_number(v)}" for k, v in metrics.items()])
                self.metrics_text.plain = metrics_str
            
            # Update progress bar
            # Calculate how much to advance (since we may not update every batch)
            if (batch_idx + 1) % print_every == 0 or batch_idx == total_batches - 1:
                current_completed = batch_idx + 1
                advance_amount = current_completed - self.progress.tasks[self.task_id].completed
                if advance_amount > 0:
                    self.progress.update(self.task_id, advance=advance_amount)
        else:
            # Simple print mode (when progress_bar is disabled)
            if (batch_idx + 1) % print_every == 0 or batch_idx == total_batches - 1:
                progress_pct = (batch_idx + 1) / total_batches * 100
                
                # Format metrics for display
                metric_strs = []
                for key, value in metrics.items():
                    clean_key = key.replace("train_", "")
                    metric_strs.append(f"{clean_key}: {format_number(value)}")
                
                progress_text = f"Batch {batch_idx + 1}/{total_batches} ({progress_pct:.1f}%) | " + " | ".join(metric_strs)
                console.print(f"  {progress_text}")

    def end_epoch_display(self):
        """Stop the Live display at the end of epoch."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
    
    def print_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], 
                          val_metrics: Optional[Dict[str, float]] = None,
                          config=None, has_early_stopping: bool = False):
        """Print epoch summary in a nice table format."""
        
        # Calculate epoch time
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Create summary table with enhanced styling
        table = Table(
            show_header=True, 
            header_style=f"bold {COLORS['metric']}", 
            box=box.ROUNDED,
            title=f"\n[bold {COLORS['epoch']}]Epoch {epoch + 1} Summary[/bold {COLORS['epoch']}]",
            title_style=f"bold {COLORS['epoch']}"
        )
        table.add_column("Metric", style=COLORS['metric'], no_wrap=True)
        table.add_column("Train", style=COLORS['train'], justify="right")
        if val_metrics:
            table.add_column("Validation", style=COLORS['val'], justify="right")
            table.add_column("Change (from prev)", style=COLORS['improvement'], justify="right")
        else:
            table.add_column("Change (from prev)", style=COLORS['improvement'], justify="right")
        
        # Add metrics to table
        all_metrics = set(train_metrics.keys())
        if val_metrics:
            all_metrics.update(val_metrics.keys())
        
        # Determine which metrics to use for change calculation
        # If validation data is available, use validation metrics; otherwise use training metrics
        current_metrics_for_change = val_metrics if val_metrics else train_metrics
            
        for metric in sorted(all_metrics):
            train_val = format_number(train_metrics.get(metric, 0.0))
            if val_metrics:
                val_val = format_number(val_metrics.get(metric, 0.0))
                
                # Calculate change from previous epoch
                change_text = "N/A"
                if self.prev_epoch_metrics is not None:
                    # Use validation metrics for change calculation since they exist
                    prev_val = self.prev_epoch_metrics.get(metric, 0.0) if val_metrics else self.prev_epoch_metrics.get(metric, 0.0)
                    current_val = val_metrics.get(metric, 0.0)
                    
                    if prev_val != 0:
                        pct_change = ((current_val - prev_val) / abs(prev_val)) * 100
                    else:
                        pct_change = 0
                    
                    # Determine if lower is better for this metric
                    lower_is_better_metrics = ["loss", "error", "mae", "mse", "rmse", "mape", "perplexity"]
                    metric_lower = metric.lower()
                    is_lower_better = any(keyword in metric_lower for keyword in lower_is_better_metrics)
                    
                    if is_lower_better:
                        # For metrics where lower is better (loss, error, etc.)
                        is_improvement = current_val < prev_val
                        arrow = "↓" if is_improvement else "↑"
                    else:
                        # For metrics where higher is better (accuracy, precision, recall, f1, etc.)
                        is_improvement = current_val > prev_val
                        arrow = "↑" if is_improvement else "↓"
                    
                    change_color = COLORS['improvement'] if is_improvement else COLORS['regression']
                    change_text = f"[{change_color}]{arrow} {abs(pct_change):.1f}%[/{change_color}]"
                
                table.add_row(metric.capitalize(), train_val, val_val, change_text)
            else:
                # No validation metrics - show change for training metrics
                change_text = "N/A"
                if self.prev_epoch_metrics is not None:
                    prev_val = self.prev_epoch_metrics.get(metric, 0.0)
                    current_val = train_metrics.get(metric, 0.0)
                    
                    if prev_val != 0:
                        pct_change = ((current_val - prev_val) / abs(prev_val)) * 100
                    else:
                        pct_change = 0
                    
                    # Determine if lower is better for this metric
                    lower_is_better_metrics = ["loss", "error", "mae", "mse", "rmse", "mape", "perplexity"]
                    metric_lower = metric.lower()
                    is_lower_better = any(keyword in metric_lower for keyword in lower_is_better_metrics)
                    
                    if is_lower_better:
                        # For metrics where lower is better (loss, error, etc.)
                        is_improvement = current_val < prev_val
                        arrow = "↓" if is_improvement else "↑"
                    else:
                        # For metrics where higher is better (accuracy, precision, recall, f1, etc.)
                        is_improvement = current_val > prev_val
                        arrow = "↑" if is_improvement else "↓"
                    
                    change_color = COLORS['improvement'] if is_improvement else COLORS['regression']
                    change_text = f"[{change_color}]{arrow} {abs(pct_change):.1f}%[/{change_color}]"
                
                table.add_row(metric.capitalize(), train_val, change_text)
        
        # Add separator
        if val_metrics:
            table.add_row("", "", "", "")
        else:
            table.add_row("", "", "")
        
        # Add timing information
        if val_metrics:
            table.add_row(f"[bold {COLORS['info']}]Epoch Time[/bold {COLORS['info']}]", 
                         f"[{COLORS['info']}]{format_time(epoch_time)}[/{COLORS['info']}]", 
                         f"[{COLORS['info']}]{format_time(epoch_time)}[/{COLORS['info']}]",
                         "")
            table.add_row(f"[bold {COLORS['info']}]Total Time[/bold {COLORS['info']}]", 
                         f"[{COLORS['info']}]{format_time(total_time)}[/{COLORS['info']}]",
                         f"[{COLORS['info']}]{format_time(total_time)}[/{COLORS['info']}]",
                         "")
        else:
            table.add_row(f"[bold {COLORS['info']}]Epoch Time[/bold {COLORS['info']}]", 
                         f"[{COLORS['info']}]{format_time(epoch_time)}[/{COLORS['info']}]",
                         "")
            table.add_row(f"[bold {COLORS['info']}]Total Time[/bold {COLORS['info']}]", 
                         f"[{COLORS['info']}]{format_time(total_time)}[/{COLORS['info']}]",
                         "")
        
        console.print(table)
        
        # Store current epoch metrics for next epoch's change calculation
        self.prev_epoch_metrics = current_metrics_for_change.copy()
        
        # Check for overfitting if early stopping is not active
        if (config and val_metrics and not has_early_stopping and 
            "loss" in train_metrics and "loss" in val_metrics):
            
            train_loss = train_metrics["loss"]
            val_loss = val_metrics["loss"]
            
            if train_loss + config.overfit_threshold < val_loss:
                console.print(f"\n[{COLORS['warning']}]⚠️  Model is potentially overfitting "
                             f"(val_loss: {val_loss:.4f} > train_loss + threshold: "
                             f"{train_loss + config.overfit_threshold:.4f})[/{COLORS['warning']}]")
        
        console.print()
        
    def finish_training(self):
        """Calculate and store total training time."""
        if self.start_time:
            self.total_time = time.time() - self.start_time


def print_model_summary(model, sample_input_shape: Optional[tuple] = None):
    """Print a summary of the model."""
    try:
        import torch
        from torchinfo import summary as torch_summary
        
        if sample_input_shape:
            console.print("\n[bold blue]Model Summary:[/bold blue]")
            summary_str = str(torch_summary(model, input_size=sample_input_shape, verbose=0))
            console.print(Panel(summary_str, title="Model Architecture", border_style="blue"))
        else:
            # Basic parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            info_text = f"""
Model: {model.__class__.__name__}
Total Parameters: {format_number(total_params, 0)}
Trainable Parameters: {format_number(trainable_params, 0)}
            """.strip()
            
            console.print(Panel(info_text, title="Model Info", border_style="blue"))
            
    except ImportError:
        # Fallback if torchinfo not available
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info_text = f"""
Model: {model.__class__.__name__}
Total Parameters: {format_number(total_params, 0)}
Trainable Parameters: {format_number(trainable_params, 0)}
        """.strip()
        
        console.print(Panel(info_text, title="Model Info", border_style="blue")) 