"""
Resume Training Example using Treadmill framework.
"""
import torch.nn as nn
import os
import click

# Import Treadmill components
from treadmill import OptimizerConfig, Trainer, TrainingConfig
from treadmill.metrics import StandardMetrics

# Import model and data preparation from basic_training.py
from examples.basic_training import SimpleDNN, prepare_data


def resume_training(checkpoint_dir, epochs, project_name, timezone, keep_all_checkpoints):
    """Resume training from existing checkpoint.
    
    The Treadmill framework automatically handles:
    - Loading model weights
    - Restoring optimizer state  
    - Restoring scheduler state
    - Restoring training progress (current epoch, metrics history)
    - Finding the best checkpoint to resume from
    - Calculating total epochs (last completed epoch + additional epochs)
    - Extracting project_name and timezone from checkpoint directory (ignoring CLI values)
    """
    
    print("🔄 Resume Training with Treadmill Framework")
    print("=" * 60)
    
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        click.echo(click.style("⚠️  Expected workflow:", fg="yellow"))
        click.echo("   1. Run: python basic_training.py        (creates checkpoints)")
        click.echo("   2. Run: python basic_training_resume.py (imports & resumes)")
        click.echo(click.style(f"\n❌ Checkpoint directory not found: {checkpoint_dir}", fg="red"))
        return False
    
    # 1. Create model architecture (imported from basic_training.py)
    model = SimpleDNN()
    
    # 2. Prepare data (imported from basic_training.py)
    train_loader, val_loader = prepare_data()
    
    # 3. Define loss and metrics (same as basic_training.py)
    loss_fn = nn.CrossEntropyLoss()
    metric_fns = {"accuracy": StandardMetrics.accuracy}
    
    # 4. Configure for RESUME - Framework handles everything automatically!
    config = TrainingConfig(
        # Just specify epochs - framework treats them as additional epochs for resume!
        additional_epochs=epochs,  # Framework calculates: last_epoch + additional_epochs
        device="auto",
        
        # RESUME SETTINGS - Framework does all the work!
        resume_training=True,           # Enables automatic checkpoint loading & epoch calculation
        checkpoint_dir=checkpoint_dir,  # Directory containing checkpoints
        
        # Framework automatically extracts project_name and timezone from checkpoint directory
        # CLI values are ignored to maintain consistency with original training
        keep_all_checkpoints=keep_all_checkpoints,
        project_name=project_name,      # Will be overridden by framework
        timezone=timezone,              # Will be overridden by framework
        optimizer=OptimizerConfig(optimizer_class="Adam", lr=1e-3),
        validate_every=1,
        early_stopping_patience=5,
        print_every=100,
        progress_bar=True,
    )
    
    # 5. Create trainer - Framework automatically handles everything!
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
    )
    
    # 6. Continue training - Framework knows exactly where to start and stop!
    print(f"🚀 Starting resume training...")
    history = trainer.train()
    
    # The comprehensive training report is now displayed automatically by the framework!
    # You can access detailed information via trainer.report property for interactive sessions
    
    return history


@click.command()
@click.option(
    "--checkpoint-dir", 
    default="./checkpoints", 
    help="Path to checkpoint directory"
)
@click.option(
    "--epochs", 
    default=5, 
    type=int,
    help="Number of epochs to train (framework automatically adds to previous progress)"
)
@click.option(
    "--project-name", 
    default="mnist", 
    help="Project name (ignored during resume - extracted from checkpoint path)"
)
@click.option(
    "--timezone", 
    default="IST", 
    help="Timezone (ignored during resume - extracted from checkpoint path)"
)
@click.option(
    "--keep-all-checkpoints/--keep-best-only",
    default=True,
    help="Keep all checkpoints or only the best one"
)

def main(checkpoint_dir, epochs, project_name, timezone, keep_all_checkpoints):
    """Resume training from basic_training.py checkpoints.
    """
    click.echo(click.style("🎯 Treadmill Resume Training Example", fg="cyan", bold=True))
    click.echo("🤖 Framework automatically handles everything!\n")
    
    result = resume_training(
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        project_name=project_name,
        timezone=timezone,
        keep_all_checkpoints=keep_all_checkpoints
    )
    if result is False:
        click.get_current_context().exit(1)


if __name__ == "__main__":
    main() 