"""
Advanced training example using Treadmill framework.

This example demonstrates more advanced features including:
- Custom forward and backward functions
- Multiple models (encoder-decoder architecture)
- Custom callbacks
- Complex metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Import Treadmill components
from treadmill import (
    Trainer, TrainingConfig, OptimizerConfig, SchedulerConfig,
    Callback, ModelCheckpoint
)
from treadmill.metrics import StandardMetrics


class Encoder(nn.Module):
    """Simple encoder network."""
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and log_var
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        mean, log_var = encoded.chunk(2, dim=-1)
        return mean, log_var


class Decoder(nn.Module):
    """Simple decoder network."""
    
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=784):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    """Variational Autoencoder combining encoder and decoder."""
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def reparameterize(self, mean, log_var):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decoder(z)
        return reconstructed, mean, log_var
    
    def compute_loss(self, outputs, targets):
        """Custom loss function for VAE."""
        reconstructed, mean, log_var = outputs
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, targets.view(targets.size(0), -1), reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + 0.1 * kl_loss
        
        return total_loss / targets.size(0)  # Average over batch


class DummyDataset(Dataset):
    """Dummy dataset for demonstration."""
    
    def __init__(self, size=1000, input_dim=784):
        self.size = size
        self.data = torch.randn(size, input_dim)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Autoencoder: input = target


class ReconstructionMetric:
    """Custom metric for reconstruction quality."""
    
    @staticmethod
    def reconstruction_loss(predictions, targets):
        """Compute reconstruction loss."""
        reconstructed, _, _ = predictions
        targets = targets.view(targets.size(0), -1)
        return F.mse_loss(reconstructed, targets).item()
    
    @staticmethod
    def kl_divergence(predictions, targets):
        """Compute KL divergence."""
        _, mean, log_var = predictions
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (kl_div / targets.size(0)).item()


class LossComponentLogger(Callback):
    """Custom callback to log individual loss components."""
    
    def __init__(self):
        self.loss_history = {
            "reconstruction": [],
            "kl_divergence": []
        }
    
    def on_batch_end(self, trainer, batch_idx: int, metrics: dict, **kwargs):
        """Log loss components after each batch."""
        if hasattr(trainer.model, 'last_outputs'):
            reconstructed, mean, log_var = trainer.model.last_outputs
            targets = kwargs.get('targets')
            
            if targets is not None:
                # Compute individual loss components
                recon_loss = F.mse_loss(
                    reconstructed, targets.view(targets.size(0), -1), reduction='mean'
                ).item()
                
                kl_loss = (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / targets.size(0)).item()
                
                self.loss_history["reconstruction"].append(recon_loss)
                self.loss_history["kl_divergence"].append(kl_loss)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: dict, **kwargs):
        """Print average loss components for the epoch."""
        if self.loss_history["reconstruction"]:
            avg_recon = np.mean(self.loss_history["reconstruction"][-100:])  # Last 100 batches
            avg_kl = np.mean(self.loss_history["kl_divergence"][-100:])
            
            print(f"Average Reconstruction Loss: {avg_recon:.4f}")
            print(f"Average KL Divergence: {avg_kl:.4f}")


def custom_forward_fn(model, batch):
    """Custom forward function for VAE."""
    inputs, targets = batch
    outputs = model(inputs)
    
    # Store outputs for callback access
    model.last_outputs = outputs
    
    return outputs, targets


def custom_backward_fn(loss, model, optimizer):
    """Custom backward function (optional customization)."""
    # Standard backward pass - could be customized for specific needs
    loss.backward()


def main():
    """Main training function for advanced example."""
    
    # Create dummy dataset
    train_dataset = DummyDataset(size=5000, input_dim=784)
    val_dataset = DummyDataset(size=1000, input_dim=784)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create VAE model
    model = VAE(input_dim=784, hidden_dim=256, latent_dim=32)
    
    # Define custom metrics
    metric_fns = {
        "recon_loss": ReconstructionMetric.reconstruction_loss,
        "kl_div": ReconstructionMetric.kl_divergence
    }
    
    # Configure training with advanced options
    config = TrainingConfig(
        epochs=15,
        device="auto",
        
        # Optimizer with custom parameters
        optimizer=OptimizerConfig(
            optimizer_class="AdamW",
            lr=1e-3,
            weight_decay=1e-5,
            params={"betas": (0.9, 0.999), "eps": 1e-8}
        ),
        
        # Advanced scheduler
        scheduler=SchedulerConfig(
            scheduler_class="CosineAnnealingLR",
            params={"T_max": 15, "eta_min": 1e-6}
        ),
        
        # Custom forward and backward functions
        custom_forward_fn=custom_forward_fn,
        custom_backward_fn=custom_backward_fn,
        
        # Training optimizations
        grad_clip_norm=0.5,
        accumulate_grad_batches=2,  # Gradient accumulation
        mixed_precision=False,  # Disable for this example
        
        # Validation settings
        validate_every=2,
        early_stopping_patience=10,
        
        # Display settings
        print_every=20,
        progress_bar=True
    )
    
    # Create custom callbacks
    callbacks = [
        LossComponentLogger(),  # Custom callback to track loss components
        ModelCheckpoint(
            filepath="./checkpoints/vae_model_epoch_{epoch:03d}",
            monitor="val_recon_loss",
            mode="min",
            save_best_only=True,
            save_format="safetensors",  # Demo safetensors format
            verbose=True
        )
    ]
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=None,  # Will use model's compute_loss method
        metric_fns=metric_fns,
        callbacks=callbacks
    )
    
    # Start training
    print("Starting advanced VAE training...")
    training_history = trainer.train()
    
    # Demonstrate checkpoint saving/loading
    checkpoint_path = "vae_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")
    
    # Show how to load checkpoint
    new_model = VAE(input_dim=784, hidden_dim=256, latent_dim=32)
    new_trainer = Trainer(
        model=new_model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )
    
    checkpoint_data = new_trainer.load_checkpoint(checkpoint_path, resume_training=False)
    print("Checkpoint loaded successfully!")
    
    # Print training summary
    print("\n" + "="*60)
    print("🚀 Advanced Training completed!")
    print(f"📈 Total epochs: {training_history['total_epochs']}")
    
    if training_history['best_metrics']:
        print("\n🏆 Best validation metrics achieved:")
        for metric, value in training_history['best_metrics'].items():
            print(f"  • {metric}: {value:.6f}")
    
    print("\n✨ Advanced features demonstrated:")
    print("   ✅ Custom forward/backward functions")
    print("   ✅ Multi-model architecture (VAE)")
    print("   ✅ Custom metrics and callbacks")  
    print("   ✅ SafeTensors format saving")
    print("   ✅ Enhanced epoch summaries with improvement indicators")


if __name__ == "__main__":
    main() 