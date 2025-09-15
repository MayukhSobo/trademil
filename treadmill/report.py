"""
Training Report System for Treadmill Framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
import os


console = Console()


@dataclass
class TrainingReport:
    """Comprehensive training report data structure."""

    # Basic training info
    total_epochs: int = 0
    training_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Model information
    model_name: str = "Unknown"
    total_parameters: int = 0
    trainable_parameters: int = 0
    model_size_mb: float = 0.0

    # Training configuration
    device: str = "Unknown"
    batch_size: int = 0
    learning_rate: float = 0.0
    optimizer: str = "Unknown"
    scheduler: Optional[str] = None

    # Performance metrics
    best_metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)

    # Checkpoint information
    experiment_dir: Optional[str] = None
    total_checkpoints: int = 0
    best_checkpoint: Optional[str] = None

    # Training details
    total_batches: int = 0
    avg_epoch_time: float = 0.0
    early_stopped: bool = False
    early_stop_epoch: Optional[int] = None

    # Hardware resource usage
    avg_cpu_percent: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    avg_ram_mb: Optional[float] = None
    max_ram_mb: Optional[float] = None
    avg_gpu_utilization: Optional[float] = None
    max_gpu_utilization: Optional[float] = None
    num_gpus: int = 0
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            'total_epochs': self.total_epochs,
            'training_time': self.training_time,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'model_name': self.model_name,
            'total_parameters': self.total_parameters,
            'trainable_parameters': self.trainable_parameters,
            'model_size_mb': self.model_size_mb,
            'device': self.device,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'best_metrics': self.best_metrics,
            'final_metrics': self.final_metrics,
            'metrics_history': self.metrics_history,
            'experiment_dir': self.experiment_dir,
            'total_checkpoints': self.total_checkpoints,
            'best_checkpoint': self.best_checkpoint,
            'total_batches': self.total_batches,
            'avg_epoch_time': self.avg_epoch_time,
            'early_stopped': self.early_stopped,
            'early_stop_epoch': self.early_stop_epoch,
            'avg_cpu_percent': self.avg_cpu_percent,
            'max_cpu_percent': self.max_cpu_percent,
            'avg_ram_mb': self.avg_ram_mb,
            'max_ram_mb': self.max_ram_mb,
            'avg_gpu_utilization': self.avg_gpu_utilization,
            'max_gpu_utilization': self.max_gpu_utilization,
            'num_gpus': self.num_gpus,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'gpu_memory_total_mb': self.gpu_memory_total_mb,
        }


def format_time_duration(seconds: float) -> str:
    """Format time duration in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def format_number(num: float, precision: int = 2) -> str:
    """Format number with appropriate units (K, M, B)."""
    if abs(num) < 1000:
        return f"{num:.{precision}f}"
    elif abs(num) < 1_000_000:
        return f"{num/1000:.{precision-1}f}K"
    elif abs(num) < 1_000_000_000:
        return f"{num/1_000_000:.{precision-1}f}M"
    else:
        return f"{num/1_000_000_000:.{precision-1}f}B"


def format_memory(mb: float) -> str:
    """Format memory in MB/GB."""
    if mb < 1024:
        return f"{mb:.1f} MB"
    else:
        return f"{mb/1024:.1f} GB"


def create_training_summary_table(report: TrainingReport) -> Table:
    """Create training summary table."""
    table = Table(title="📊 Training Summary", box=box.ROUNDED,
                  show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True, width=25)
    table.add_column("Value", style="green", width=30)

    # Training details
    table.add_row("Total Epochs", str(report.total_epochs))
    table.add_row("Training Time", format_time_duration(report.training_time))
    table.add_row("Avg Time/Epoch", format_time_duration(report.avg_epoch_time))
    table.add_row("Total Batches", f"{report.total_batches:,}")

    # Timing information
    if report.start_time:
        table.add_row("Started At", report.start_time.strftime("%Y-%m-%d %H:%M:%S"))
    if report.end_time:
        table.add_row("Completed At", report.end_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Early stopping info
    if report.early_stopped:
        table.add_row("Early Stopping", f"✅ Yes (Epoch {report.early_stop_epoch})",
                      style="yellow")
    else:
        table.add_row("Early Stopping", "❌ No")

    return table


def create_model_info_table(report: TrainingReport) -> Table:
    """Create model information table."""
    table = Table(title="🏗️ Model Information", box=box.ROUNDED,
                  show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", no_wrap=True, width=25)
    table.add_column("Value", style="green", width=30)

    table.add_row("Model Name", report.model_name)
    table.add_row("Total Parameters", format_number(report.total_parameters))
    table.add_row("Trainable Parameters", format_number(report.trainable_parameters))
    table.add_row("Model Size", format_memory(report.model_size_mb))
    table.add_row("Device", report.device)

    return table


def create_hardware_info_table(report: TrainingReport) -> Table:
    """Create hardware resource usage table."""
    table = Table(title="💻 Hardware Resources", box=box.ROUNDED,
                  show_header=True, header_style="bold magenta")
    table.add_column("Resource", style="cyan", no_wrap=True, width=25)
    table.add_column("Average", style="green", width=15, justify="right")
    table.add_column("Peak", style="yellow", width=15, justify="right")

    # CPU Usage
    if report.avg_cpu_percent is not None:
        avg_cpu = f"{report.avg_cpu_percent:.1f}%" if report.avg_cpu_percent is not None else "N/A"
        max_cpu = f"{report.max_cpu_percent:.1f}%" if report.max_cpu_percent is not None else "N/A"
        table.add_row("CPU Usage (%)", avg_cpu, max_cpu)

    # RAM Usage
    if report.avg_ram_mb is not None:
        avg_ram = format_memory(report.avg_ram_mb) if report.avg_ram_mb is not None else "N/A"
        max_ram = format_memory(report.max_ram_mb) if report.max_ram_mb is not None else "N/A"
        table.add_row("RAM Usage", avg_ram, max_ram)

    # GPU Information
    if report.num_gpus > 0:
        table.add_row("Number of GPUs", str(report.num_gpus), "")

        if report.avg_gpu_utilization is not None:
            avg_gpu = f"{report.avg_gpu_utilization:.1f}%" if report.avg_gpu_utilization is not None else "N/A"
            max_gpu = f"{report.max_gpu_utilization:.1f}%" if report.max_gpu_utilization is not None else "N/A"
            table.add_row("GPU Utilization (%)", avg_gpu, max_gpu)

        if report.gpu_memory_used_mb is not None and report.gpu_memory_total_mb is not None:
            usage_percent = (report.gpu_memory_used_mb / report.gpu_memory_total_mb) * 100
            gpu_mem_str = f"{format_memory(report.gpu_memory_used_mb)} / {format_memory(report.gpu_memory_total_mb)}"
            table.add_row("GPU Memory", gpu_mem_str, f"({usage_percent:.1f}%)")

    # Show message if no hardware data available
    if (report.avg_cpu_percent is None and report.avg_ram_mb is None and
            report.num_gpus == 0 and report.avg_gpu_utilization is None):
        table.add_row("Status", "Hardware monitoring disabled", "")

    return table


def create_metrics_table(report: TrainingReport) -> Table:
    """Create performance metrics table."""
    table = Table(title="🏆 Performance Metrics", box=box.ROUNDED,
                  show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True, width=20)
    table.add_column("Best Value", style="green", width=15, justify="right")
    table.add_column("Final Value", style="yellow", width=15, justify="right")
    table.add_column("Improvement", style="blue", width=15, justify="right")

    for metric in sorted(set(report.best_metrics.keys()) | set(report.final_metrics.keys())):
        best_val = report.best_metrics.get(metric, 0.0)
        final_val = report.final_metrics.get(metric, 0.0)

        # Calculate improvement
        if metric in report.best_metrics and metric in report.final_metrics:
            if abs(best_val) > 1e-8:  # Avoid division by zero
                improvement = ((final_val - best_val) / abs(best_val)) * 100
                if "loss" in metric.lower() or "error" in metric.lower():
                    improvement = -improvement  # Invert for loss metrics
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
        else:
            improvement_str = "N/A"

        table.add_row(
            metric.title(),
            f"{best_val:.4f}" if metric in report.best_metrics else "N/A",
            f"{final_val:.4f}" if metric in report.final_metrics else "N/A",
            improvement_str
        )

    return table


def create_checkpoint_info_table(report: TrainingReport) -> Table:
    """Create checkpoint information table."""
    table = Table(title="💾 Checkpoint Information", box=box.ROUNDED,
                  show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", no_wrap=True, width=25)
    table.add_column("Value", style="green", width=40)

    table.add_row("Total Checkpoints", str(report.total_checkpoints))

    if report.best_checkpoint:
        table.add_row("Best Checkpoint", os.path.basename(report.best_checkpoint))

    return table


def display_training_report(report: TrainingReport) -> None:
    """Display comprehensive training report using Rich formatting."""

    # Create main title
    title = Text("🎉 TRAINING COMPLETE! 🎉", style="bold green", justify="center")

    # Create tables
    summary_table = create_training_summary_table(report)
    model_table = create_model_info_table(report)
    hardware_table = create_hardware_info_table(report)
    metrics_table = create_metrics_table(report)
    checkpoint_table = create_checkpoint_info_table(report)

    # Display with proper spacing
    console.print()
    console.print("=" * 100, style="green")
    console.print(title)
    console.print("=" * 100, style="green")
    console.print()

    # Display all tables separately
    console.print(summary_table)
    console.print()
    console.print(model_table)
    console.print()
    console.print(hardware_table)
    console.print()
    console.print(metrics_table)
    console.print()
    console.print(checkpoint_table)
    console.print()
    console.print("=" * 100, style="green")
    console.print()


def save_report_to_file(report: TrainingReport, filepath: str) -> None:
    """Save training report to JSON file."""
    import json

    with open(filepath, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    console.print(f"📄 Training report saved to: {filepath}")


class HardwareMonitor:
    """Hardware resource monitoring during training."""

    def __init__(self):
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_samples = []
        self.gpu_memory_samples = []
        self.monitoring_enabled = False

        # Try to import monitoring libraries
        try:
            import psutil
            self.psutil = psutil
            self.monitoring_enabled = True
        except ImportError:
            self.psutil = None

        try:
            import pynvml
            self.pynvml = pynvml
            self.pynvml.nvmlInit()
            self.gpu_available = True
        except (ImportError, Exception):
            self.pynvml = None
            self.gpu_available = False

    def sample(self):
        """Take a sample of current resource usage."""
        if not self.monitoring_enabled:
            return

        try:
            # CPU and RAM monitoring
            if self.psutil:
                cpu_percent = self.psutil.cpu_percent()
                memory = self.psutil.virtual_memory()

                self.cpu_samples.append(cpu_percent)
                self.ram_samples.append(memory.used / 1024 / 1024)  # Convert to MB

            # GPU monitoring
            if self.gpu_available and self.pynvml:
                try:
                    device_count = self.pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        # Monitor first GPU for simplicity
                        handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                        utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)

                        self.gpu_samples.append(utilization.gpu)
                        self.gpu_memory_samples.append({
                            'used': memory_info.used / 1024 / 1024,  # Convert to MB
                            'total': memory_info.total / 1024 / 1024
                        })
                except Exception:
                    pass  # GPU monitoring failed, continue without it
        except Exception:
            pass  # Monitoring failed, continue without it

    def get_summary(self):
        """Get hardware usage summary."""
        summary = {
            'avg_cpu_percent': None,
            'max_cpu_percent': None,
            'avg_ram_mb': None,
            'max_ram_mb': None,
            'avg_gpu_utilization': None,
            'max_gpu_utilization': None,
            'num_gpus': 0,
            'gpu_memory_used_mb': None,
            'gpu_memory_total_mb': None,
        }

        # CPU statistics
        if self.cpu_samples:
            summary['avg_cpu_percent'] = sum(self.cpu_samples) / len(self.cpu_samples)
            summary['max_cpu_percent'] = max(self.cpu_samples)

        # RAM statistics
        if self.ram_samples:
            summary['avg_ram_mb'] = sum(self.ram_samples) / len(self.ram_samples)
            summary['max_ram_mb'] = max(self.ram_samples)

        # GPU statistics
        if self.gpu_samples:
            summary['avg_gpu_utilization'] = sum(self.gpu_samples) / len(self.gpu_samples)
            summary['max_gpu_utilization'] = max(self.gpu_samples)

        # GPU count
        if self.gpu_available and self.pynvml:
            try:
                summary['num_gpus'] = self.pynvml.nvmlDeviceGetCount()
            except Exception:
                pass

        # GPU memory (use last sample)
        if self.gpu_memory_samples:
            last_sample = self.gpu_memory_samples[-1]
            summary['gpu_memory_used_mb'] = last_sample['used']
            summary['gpu_memory_total_mb'] = last_sample['total']

        return summary


def create_training_report_from_trainer(trainer) -> TrainingReport:
    """Create a training report from a Trainer instance."""
    from datetime import datetime
    import torch

    report = TrainingReport()

    # Basic training info
    if hasattr(trainer, 'progress_tracker') and trainer.progress_tracker.start_time:
        report.training_time = trainer.progress_tracker.total_time if hasattr(trainer.progress_tracker, 'total_time') else 0.0
        report.start_time = datetime.fromtimestamp(trainer.progress_tracker.start_time)
        report.end_time = datetime.now()

    report.total_epochs = trainer.current_epoch + 1 if hasattr(trainer, 'current_epoch') else 0

    if report.total_epochs > 0 and report.training_time > 0:
        report.avg_epoch_time = report.training_time / report.total_epochs

    # Model information
    if trainer.model:
        report.model_name = trainer.model.__class__.__name__
        report.total_parameters = sum(p.numel() for p in trainer.model.parameters())
        report.trainable_parameters = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

        # Estimate model size (rough approximation)
        param_size = sum(p.nelement() * p.element_size() for p in trainer.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in trainer.model.buffers())
        report.model_size_mb = (param_size + buffer_size) / 1024 / 1024

    # Training configuration
    if hasattr(trainer, 'config'):
        config = trainer.config
        report.device = str(trainer.device) if hasattr(trainer, 'device') else config.device
        report.learning_rate = config.optimizer.lr if hasattr(config.optimizer, 'lr') else 0.0
        opt_class_name = getattr(config.optimizer, 'optimizer_class', None)
        if opt_class_name:
            report.optimizer = opt_class_name.__name__ if hasattr(opt_class_name, '__name__') else "Unknown"
        if config.scheduler:
            sched_class_name = getattr(config.scheduler, 'scheduler_class', None)
            if sched_class_name:
                report.scheduler = sched_class_name.__name__ if hasattr(sched_class_name, '__name__') else "Unknown"

    # Batch size from dataloader
    if hasattr(trainer, 'train_dataloader') and trainer.train_dataloader:
        report.batch_size = trainer.train_dataloader.batch_size
        report.total_batches = len(trainer.train_dataloader) * report.total_epochs

    # Performance metrics
    if hasattr(trainer, 'metrics_tracker'):
        metrics_tracker = trainer.metrics_tracker
        if hasattr(metrics_tracker, 'best_metrics'):
            report.best_metrics = dict(metrics_tracker.best_metrics)
        if hasattr(metrics_tracker, 'epoch_metrics'):
            report.metrics_history = dict(metrics_tracker.epoch_metrics)
            # Get final metrics (last values from each metric)
            for metric, values in report.metrics_history.items():
                if values:
                    report.final_metrics[metric] = values[-1]

    # Checkpoint information
    if hasattr(trainer, 'experiment_dir'):
        report.experiment_dir = trainer.experiment_dir
        if os.path.exists(trainer.experiment_dir):
            import glob
            checkpoints = glob.glob(os.path.join(trainer.experiment_dir, "*.pt"))
            report.total_checkpoints = len(checkpoints)

            # Find best checkpoint (if it exists)
            best_checkpoints = [f for f in checkpoints if "checkpoint_" in os.path.basename(f) and "training_checkpoint" not in os.path.basename(f)]
            if best_checkpoints:
                # Get the one with the best loss (lowest number in filename)
                best_checkpoint = min(best_checkpoints, key=lambda x: float(os.path.basename(x).split('_')[-1].replace('.pt', '')))
                report.best_checkpoint = best_checkpoint

    # Early stopping detection
    if hasattr(trainer, 'stop_training') and trainer.stop_training:
        if hasattr(trainer, 'config') and trainer.config.early_stopping_patience:
            report.early_stopped = True
            report.early_stop_epoch = trainer.current_epoch + 1

    # Hardware monitoring data (if available)
    if hasattr(trainer, 'hardware_monitor'):
        hardware_summary = trainer.hardware_monitor.get_summary()
        report.avg_cpu_percent = hardware_summary['avg_cpu_percent']
        report.max_cpu_percent = hardware_summary['max_cpu_percent']
        report.avg_ram_mb = hardware_summary['avg_ram_mb']
        report.max_ram_mb = hardware_summary['max_ram_mb']
        report.avg_gpu_utilization = hardware_summary['avg_gpu_utilization']
        report.max_gpu_utilization = hardware_summary['max_gpu_utilization']
        report.num_gpus = hardware_summary['num_gpus']
        report.gpu_memory_used_mb = hardware_summary['gpu_memory_used_mb']
        report.gpu_memory_total_mb = hardware_summary['gpu_memory_total_mb']
    else:
        # Basic GPU count detection if hardware monitor not available
        try:
            if torch.cuda.is_available():
                report.num_gpus = torch.cuda.device_count()
        except Exception:
            pass

    return report 