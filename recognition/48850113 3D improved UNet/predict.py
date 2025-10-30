"""
This file contains an example usage of the trained 3D U-Net model
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from dataset import Prostate3DDataset
from modules import ImprovedUNet3D, dice_score_per_class, visualise_volume_prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path):
    """
    Load the pre-trained model weights from the saved checkpoint.

    This function loads the model architecture and the pre trained weights from a specified file path.
    The model is then set to evaluation mode to be ready for inference.

    Parameters:
    model_path: Path to the saved model file.

    Returns:
    model: Loaded model with weights.
    """
    # Instantise the model architecture
    model = ImprovedUNet3D(in_channels=1, n_classes=6).to(device)

    # Load the saved model weights from the speicifed file path into the model
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()
    return model

def make_predictions(model, test_loader):
    """
    Make predictions on the test set using the trained model.

    Parameters:
    model: The trained model to use for inference
    test_loader: DataLoader for the test set

    Returns:
    List: List of predictions on the test set
    """
    predictions = []

    # Disable gradient computation
    with torch.no_grad():
        # Iterate over the test data loader in batches
        for images, _ in test_loader:
            images = images.to(device)

            # Perform forward pass to get predictions
            output = model(images)

            # Move back to CPU and append to the list
            predictions.append(output.cpu())

    return predictions

def visualise_volume_prediction(model, dataset, idx=0, device="cpu", save_path="prediction.png"):
    """
    Visualise the segmentation prediction compared to the ground truth.

    This function takes a 3D image and its corresponding segmentation mask, generates 
    a prediction from the model, and displays the middle slice of the volume alongside 
    the ground truth for comparison.

    Parameters:
    model: The segmentation model used to generate predictions.
    dataset: The dataset containing input 3D images and one-hot encoded masks.
    idx: Index of the sample in the dataset to visualise.
    device: device to run the model
    save_path: File path to save the generated visualisation image.
    """
    model.eval()

    # Retrieve one sample (image and its one-hot encoded mask)
    image, mask_onehot = dataset[idx]

    # Disable gradient computation
    with torch.no_grad():
        # Forward pass (predict the segmentation mask)
        predictions = model(image.unsqueeze(0).to(device))

        # Convert model output probabilities to class labels
        prediction_label = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()

    # Convert one-hot encoded ground truth mask to label format
    mask_label = torch.argmax(mask_onehot, dim=0).numpy()

    # Choose the middle slice along the depth axis for visualization
    mid_slice = image.shape[1] // 2

    # Create a figure with 3 panels (input MRI, ground truth, and prediction)
    plt.figure(figsize=(12,4))

    # Panel 1: MRI slice
    plt.subplot(1,3,1)
    plt.imshow(image[0, mid_slice].cpu(), cmap="gray")
    plt.title("MRI Slice")

    # Panel 2: Ground truth segmentation mask
    plt.subplot(1,3,2)
    plt.imshow(mask_label[mid_slice], cmap="jet", vmin=0, vmax=5)
    plt.title("Ground Truth")

    # Panel 3: Model prediction
    plt.subplot(1,3,3)
    plt.imshow(prediction_label[mid_slice], cmap="jet", vmin=0, vmax=5)
    plt.title("Prediction")

    plt.suptitle("3D Improved UNet Segmentation")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved prediction visualisation to {save_path}")

def predict():
    """
    Load a pre trained model and make predictions on a test dataset. Then visualise the results
    """
    # Path to the trained model
    model_path = "trained_model.pth"
    
    # Load the trained model
    model = load_model(model_path)

    # Define test dataset
    root = "/home/groups/comp3710/HipMRI_Study_open"
    image_dir = os.path.join(root, "semantic_MRs")
    mask_dir = os.path.join(root, "semantic_labels_only")

    # Get total number of samples in dataset
    dataset = Prostate3DDataset(image_dir, mask_dir)
    num_samples = len(dataset)

    # Create a test dataset using last 3 samples from dataset
    test_ds = Prostate3DDataset(image_dir, mask_dir, subset_start=num_samples - 3, subset_end=num_samples)  
    
    # Create data loader for test dataset
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Get predictions on the test set
    predictions = make_predictions(model, test_loader)

    # Store per class dice scores for each test sample
    val_dice_scores = []

    for idx, (images, masks) in enumerate(test_loader):
        # Move to the correct device
        images = images.to(device)
        masks = masks.to(device)

        # Get model predictions
        predictions = model(images) 

        # Calculate dice scores per class for this batch
        dice_scores = dice_score_per_class(predictions, masks)

        # Append the per class dice scores to the list
        val_dice_scores.append(dice_scores)

        # Save visualisation for each sample in the batch
        save_path = f"prediction_{idx}.png"
        visualise_volume_prediction(model, dataset, idx, device, save_path)

    # Calculate the mean dice score for each class across all test samples
    val_dice_scores = np.array(val_dice_scores)

    # Calculate mean per class dice scores
    mean_dice_per_class = np.mean(val_dice_scores, axis=0)
    print(f"Per-class Dice: {mean_dice_per_class}")

    # Print the per class dice scores for each test sample
    for idx, dice_scores in enumerate(val_dice_scores):
        print(f"Sample {idx} Per-class Dice Scores: {dice_scores}")

if __name__ == "__main__":
    predict()