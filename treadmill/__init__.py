"""
Treadmill - A Clean and Modular PyTorch Training Framework

A training framework designed specifically for PyTorch that provides clean,
easy-to-understand training loops with modular components.
"""

__version__ = "0.6.5"
__author__ = "Mayukh Sarkar"

from .trainer import Trainer
from .config import TrainingConfig, OptimizerConfig, SchedulerConfig
from .metrics import MetricsTracker
from .callbacks import Callback, EarlyStopping, ModelCheckpoint
from .utils import format_time, format_metrics, set_color_theme

__all__ = [
    "Trainer",
    "TrainingConfig", 
    "OptimizerConfig",
    "SchedulerConfig",
    "MetricsTracker",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "format_time",
    "format_metrics",
    "set_color_theme"
] 