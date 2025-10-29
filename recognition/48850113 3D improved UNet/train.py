import torch
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducability
torch.manual_seed(42) # Seed for PyTorch operations
np.random.seed(42) # Seed for NumPy operations
random.seed(42) # Seed for python's built in random module

def train_validate_3d(
    model, train_loader, optimizer, criterion,
    epochs=100
):
    """
    Train a 3D segmentation model on the provided dataset.

    This function performs the training loop over the specified number of epochs.

    Parameters:
    model: The 3D segmentation model to train
    train_loader: Dataloader for the training dataset
    optimizer: Optimizer used for model parameter updates
    criterion: Loss function used to compute training loss
    epochs: Number of training epochs
    """
    # Move model to selected device
    model.to(device)

    # Keep track of average training losses per epoch
    train_losses = []

    # Learning rate scheduler that reduces learning rate when training loss has stopped improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop
    for epoch in range(epochs):
        model.train() # Set model to training mode
        epoch_loss = 0

        # Iterate over batches from the training
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad() # Reset gradients from previous step
            predictions = model(images) # Forward pass (compuate predictions)
            loss = criterion(predictions, masks) # Compute loss
            loss.backward() # Backward pass (compute gradients)
            optimizer.step() # Update model weights
            epoch_loss += loss.item() # Accumulate batch loss

        # Compute average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
