import torch
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from modules import dice_score_per_class, visualise_volume_prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducability
torch.manual_seed(42) # Seed for PyTorch operations
np.random.seed(42) # Seed for NumPy operations
random.seed(42) # Seed for python's built in random module

def train_validate_3d(
    model, train_loader, val_dataset, optimizer, criterion,
    epochs=100, visualise_every=25
):
    """
    Train a 3D segmentation model on the provided dataset.

    This function performs the training loop over the specified number of epochs. 
    It calculates and average training loss and dice scores per epoch, adjusts the learning rate 
    dynamically. and visualises the model predictions every few epochs.

    Parameters:
    model: The 3D segmentation model to train
    train_loader: Dataloader for the training dataset
    val_dataset: Dataset used for validation
    optimizer: Optimizer used for model parameter updates
    criterion: Loss function used to compute training loss
    epochs: Number of training epochs
    visualise_every: Freqquency which model predictions are visualised

    Returns:
    List: List of mean dice score for each epoch.
    """
    # Move model to selected device
    model.to(device)

    # Keep track of average training losses per epoch
    train_losses = []
    val_mean_dice_history = []

    # Learning rate scheduler that reduces learning rate when training loss has stopped improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop
    for epoch in range(epochs):
        model.train() # Set model to training mode
        epoch_loss = 0

        # Iterate over batches from the training data loader
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

        # Validation phase
        model.eval() # Swicth model to evaluation mode
        with torch.no_grad():
            # Stores dice scores for all validation samples
            val_dice_scores = []

            # Loop through the validation dataset
            for i in range(len(val_dataset)):
                # Retrieve one validation image and its ground truth mask
                image, mask = val_dataset[i]

                # Add batch dimention
                image = image.unsqueeze(0).to(device)
                mask = mask.unsqueeze(0).to(device)

                # Forward pass (generate model prediction)
                outputs = model(image)

                # Calculate dice score per class for this prediction
                dice_scores = dice_score_per_class(outputs, mask)

                val_dice_scores.append(dice_scores)

            # Calculate mean dice score for each class across all validation samples
            mean_dice_per_class = np.mean(val_dice_scores, axis=0)

            # Calculate overall mean dice across all classes
            mean_dice = np.mean(mean_dice_per_class)

            val_mean_dice_history.append(mean_dice)

        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Mean Dice={mean_dice:.4f}")

        # Update learning rate based on training loss
        scheduler.step(avg_loss)

        # Visualise model predictions every few epochs
        if (epoch + 1) % visualise_every == 0:
            save_path = f"prediction_epoch_{epoch+1}.png"
            print(f"Visualising predictions at Epoch {epoch+1} ...")

            # Generate and save a visualisation
            visualise_volume_prediction(model, val_dataset, idx=0, save_path=save_path)

    print("✅ Training complete!")

    return val_mean_dice_history
