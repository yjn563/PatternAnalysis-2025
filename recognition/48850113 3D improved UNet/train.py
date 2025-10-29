import torch
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducability
torch.manual_seed(42) # Seed for PyTorch operations
np.random.seed(42) # Seed for NumPy operations
random.seed(42) # Seed for python's built in random module

