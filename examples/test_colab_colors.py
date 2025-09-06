"""
Test script to demonstrate the new adaptive color system in different environments.
This is especially useful for testing Google Colab compatibility.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from treadmill import Trainer, TrainingConfig, OptimizerConfig, set_color_theme

print("🎨 Testing Treadmill's Adaptive Color System")
print("=" * 50)

# Simple model for testing
class ColorTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)
    
    def forward(self, x):
        return self.linear(x)

# Create minimal data
X = torch.randn(50, 5)
y = torch.randn(50, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10)

# Test 1: Default adaptive colors
print("\n🧪 Test 1: Default Adaptive Colors")
print("Colors automatically adapt to your environment (Colab, Jupyter, Terminal)")

trainer1 = Trainer(
    model=ColorTestModel(),
    config=TrainingConfig(epochs=1, device="cpu", print_every=5),
    train_dataloader=dataloader,
    loss_fn=nn.MSELoss()
)

print("Running with auto-detected colors...")
trainer1.train()

# Test 2: Force Colab-optimized colors
print("\n🧪 Test 2: Force Colab-Optimized Colors")
set_color_theme(environment='colab')

trainer2 = Trainer(
    model=ColorTestModel(),
    config=TrainingConfig(epochs=1, device="cpu", print_every=5),
    train_dataloader=dataloader,
    loss_fn=nn.MSELoss()
)

print("Running with Colab-optimized colors...")
trainer2.train()

# Test 3: Custom color override
print("\n🧪 Test 3: Custom Color Override")
custom_colors = {
    'header': 'bold white',
    'epoch': 'bold yellow',
    'train': 'green',
    'success': 'bold cyan'
}
set_color_theme(color_dict=custom_colors)

trainer3 = Trainer(
    model=ColorTestModel(),
    config=TrainingConfig(epochs=1, device="cpu", print_every=5),
    train_dataloader=dataloader,
    loss_fn=nn.MSELoss()
)

print("Running with custom colors...")
trainer3.train()

print("\n✅ Color System Test Complete!")
print("\n💡 Tips for Google Colab users:")
print("   • Colors now automatically adapt to Colab's theme")
print("   • Both light and dark themes are supported")
print("   • Use set_color_theme(environment='colab') to force optimization")
print("   • Use set_color_theme(color_dict={...}) for custom colors") 