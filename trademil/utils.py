"""
Utility functions for Trademil framework.
"""

import time
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich import box


console = Console()


def detect_terminal_theme() -> str:
    """Detect if terminal has dark or light theme."""
    try:
        # Try to detect terminal background
        import os
        
        # Check common environment variables
        term_program = os.environ.get('TERM_PROGRAM', '').lower()
        colorterm = os.environ.get('COLORTERM', '').lower()
        term = os.environ.get('TERM', '').lower()
        
        # Some heuristics for light themes
        light_indicators = [
            'light' in term,
            'bright' in term,
            os.environ.get('TERMINAL_THEME', '').lower() == 'light'
        ]
        
        if any(light_indicators):
            return 'light'
        
        # Default to dark theme (most common for developers)
        return 'dark'
        
    except:
        return 'dark'  # Safe fallback


def get_theme_colors():
    """Get colors based on terminal theme."""
    theme = detect_terminal_theme()
    
    if theme == 'light':
        return {
            'header': 'blue',
            'epoch': 'blue', 
            'train': 'dark_green',
            'val': 'dark_orange3',
            'metric': 'black',
            'improvement': 'dark_green',
            'regression': 'red3',
            'warning': 'red3',
            'success': 'dark_green',
            'info': 'blue'
        }
    else:  # dark theme
        return {
            'header': 'bright_blue',
            'epoch': 'bright_cyan',
            'train': 'bright_green', 
            'val': 'bright_yellow',
            'metric': 'cyan',
            'improvement': 'bright_green',
            'regression': 'bright_red',
            'warning': 'bright_red',
            'success': 'bright_green',
            'info': 'bright_blue'
        }


# Get theme colors
COLORS = get_theme_colors()


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
        
    def start_training(self, total_epochs: int, total_batches_per_epoch: int):
        """Initialize training progress tracking."""
        self.start_time = time.time()
        self.total_epochs = total_epochs
        self.total_batches_per_epoch = total_batches_per_epoch
        
        # Print training header
        header_line = "_"*60
        console.print(f"\n{header_line}", justify="center")
        console.print(f"[bold {COLORS['header']}]🚀 Starting Training with Trademil[/bold {COLORS['header']}]", justify="center")
        console.print(header_line, justify="center")
        
    def start_epoch(self, epoch: int):
        """Start epoch tracking."""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        
    def print_epoch_header(self, epoch: int, total_epochs: int):
        """Print nice epoch header."""
        console.print(f"\n[bold {COLORS['epoch']}]Epoch {epoch+1}/{total_epochs}[/bold {COLORS['epoch']}]")
        console.print("-" * 40)
        
    def print_batch_progress(self, batch_idx: int, total_batches: int, 
                           metrics: Dict[str, float], print_every: int = 10):
        """Print batch progress."""
        if (batch_idx + 1) % print_every == 0 or batch_idx == total_batches - 1:
            progress = (batch_idx + 1) / total_batches * 100
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            
            console.print(f"[{COLORS['train']}]Batch {batch_idx+1:4d}/{total_batches} "
                         f"({progress:5.1f}%)[/{COLORS['train']}] | {metrics_str}")
    
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
            title=f"[bold {COLORS['epoch']}]Epoch {epoch + 1} Summary[/bold {COLORS['epoch']}]",
            title_style=f"bold {COLORS['epoch']}"
        )
        table.add_column("Metric", style=COLORS['metric'], no_wrap=True)
        table.add_column("Train", style=COLORS['train'], justify="right")
        if val_metrics:
            table.add_column("Validation", style=COLORS['val'], justify="right")
            table.add_column("Change", style=COLORS['improvement'], justify="right")
        
        # Add metrics to table
        all_metrics = set(train_metrics.keys())
        if val_metrics:
            all_metrics.update(val_metrics.keys())
            
        for metric in sorted(all_metrics):
            train_val = format_number(train_metrics.get(metric, 0.0))
            if val_metrics:
                val_val = format_number(val_metrics.get(metric, 0.0))
                
                # Calculate change (improvement indicator with percentage)
                train_num = train_metrics.get(metric, 0.0)
                val_num = val_metrics.get(metric, 0.0)
                
                # Calculate percentage change
                if train_num != 0:
                    pct_change = ((val_num - train_num) / abs(train_num)) * 100
                else:
                    pct_change = 0
                
                if "loss" in metric.lower():
                    # For loss, lower is better
                    is_improvement = val_num < train_num
                    arrow = "↓" if is_improvement else "↑"
                else:
                    # For other metrics, higher is usually better
                    is_improvement = val_num > train_num
                    arrow = "↑" if is_improvement else "↓"
                
                change_color = COLORS['improvement'] if is_improvement else COLORS['regression']
                change_text = f"[{change_color}]{arrow} {abs(pct_change):.1f}%[/{change_color}]"
                table.add_row(metric.capitalize(), train_val, val_val, change_text)
            else:
                table.add_row(metric.capitalize(), train_val)
        
        # Add separator
        if val_metrics:
            table.add_row("", "", "", "")
        else:
            table.add_row("", "")
        
        # Add timing information
        table.add_row("[bold]Epoch Time[/bold]", 
                     f"[dim]{format_time(epoch_time)}[/dim]", 
                     f"[dim]{format_time(epoch_time)}[/dim]" if val_metrics else None,
                     "" if val_metrics else None)
        table.add_row("[bold]Total Time[/bold]", 
                     f"[dim]{format_time(total_time)}[/dim]",
                     f"[dim]{format_time(total_time)}[/dim]" if val_metrics else None,
                     "" if val_metrics else None)
        
        console.print(table)
        
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
        """Print training completion message."""
        if self.start_time:
            total_time = time.time() - self.start_time
            
            console.print("\n" + "_"*60, justify="center")
            console.print(f"[bold {COLORS['success']}]✅ Training Complete![/bold {COLORS['success']}]", justify="center")
            console.print(f"[{COLORS['success']}]Total training time: {format_time(total_time)}[/{COLORS['success']}]", justify="center")
            console.print("_"*60, justify="center")


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