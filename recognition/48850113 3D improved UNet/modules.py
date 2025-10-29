import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class ConvBlock3D(nn.Module):
    """
    3D convolutional block for U-Net architecture.
    Consists of two 3D convolutional layers, each followed by instance normalisation and 
    LeakyReLU activation, along with a residual skip connection for stable gradient flow.

    Used in both encoder and decoder paths of the U-Net model.
    """
    def __init__(self, in_channels, out_channels, dropout=0.2):
        """
        Initialise the 3D convolutional block.

        Parameters:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dropout: Dropout rate applied after convolutions to reduce overfitting.
        """
        super().__init__()

        # Convolutional path
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout)
        )

        # Shortcut connection that allows the input to skip over the convolutional layers
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the convolutional block.

        Parameters:
        x: Input feature map with shape 
        [Batch size, In channel number, Depth, Height, Width]

        Returns:
        Tensor: Output feature map with shape 
        [Batch size, Out channel number, Depth, Height, Width]
        """
        return self.conv(x) + self.residual(x)
    
class ImprovedUNet3D(nn.Module):
    """
    Improved 3D U-Net architecture for 3D image segmentation.

    This model takes 3d images and predicts a segmentation mask.
    It uses an encoder-decoder structure. The encoder compresses the images and 
    learns features. The decoder upsamples and reconstructs the segmentation map.
    Skip connections are used to combine details from the encoder with 
    features in the decoder.

    Each block has small residual connections inside to help training.
    Instance normalisation keeps training stable, and trilinear upsampling is 
    used for smooth decoding.
    """
    def __init__(self, in_channels=1, n_classes=6, base_channels=32):
        """
        Initialise the 3D U-Net model

        Parameters: 
        in_channels: Number of input channels (e.g. 1 for grayscale)
        n_classes: Number of segmentation output classes
        base_channels: Base number of feature channels (doubles with each encoder level)
        """
        super().__init__()

        # Encoder (Downampling)
        self.encoder1 = ConvBlock3D(in_channels, base_channels)
        self.encoder2 = ConvBlock3D(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.encoder4 = ConvBlock3D(base_channels * 4, base_channels * 8)

        # Pooling and upsampling layers
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # Decoder (Upsampling)
        self.decoder4 = ConvBlock3D(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.decoder3 = ConvBlock3D(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.decoder2 = ConvBlock3D(base_channels * 2 + base_channels, base_channels)

        # Final output layer
        self.final = nn.Conv3d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the full U-Net model

        Parameters:
        x: Input 3D image with shape 
        [Batch size, In channel number, Depth, Height, Width]

        Returns:
        Tensor: Output feature map with shape 
        [Batch size, Out channel number, Depth, Height, Width]
        """
        # Encoder path
        encoder1_output = self.encoder1(x)
        encoder2_output = self.encoder2(self.pool(encoder1_output))
        encoder3_output = self.encoder3(self.pool(encoder2_output))
        encoder4_output = self.encoder4(self.pool(encoder3_output))

        # Decoder path
        # Level 4 decoding (combine encoder3 skip)
        decoder4_input = self.upsample(encoder4_output)
        decoder4_output = self.decoder4(torch.cat([decoder4_input, encoder3_output], dim=1))

        # Level 3 decoding (combine encoder2 skip)
        decoder3_input = self.upsample(decoder4_output)
        decoder3_output = self.decoder3(torch.cat([decoder3_input, encoder2_output], dim=1))

        # Level 2 decoding (combine encoder1 skip)
        decoder2_input = self.upsample(decoder3_output)
        decoder2_output = self.decoder2(torch.cat([decoder2_input, encoder1_output], dim=1))

        # Final output layer
        output = self.final(decoder2_output)
        return output
    
class DiceLoss3D(nn.Module):
    """
    Dice loss for segmentation

    This loss measures the overlap between predicted segmentation maps and the ground truth masks. 
    It is bassed on the dicce simimlarity coefficient.
    """
    def __init__(self, smooth=1e-5):
        """
        Initialise the dice loss class

        Parameters:
        smooth: constant added to numerator and denominator to avoid division by 
        zero when masks are empty
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        """
        Compute dice loss between predictions and targets.

        Parameters:
        outputs: Raw model outputs with shape 
        [Batch size, Number of classes, Depth, Height, Width]
        targets: One-hot encoded ground truth with same shape as preds

        Returns:
        Tensor: Scalar dice loss value
        """
        # Convert model outputs to probabilities using softmax across classes
        outputs = torch.softmax(outputs, dim=1)

        # Find the overlap between prediction and target for each class
        intersection = (outputs * targets).sum(dim=(2, 3, 4))

        # Compute dice score for each class in each batch
        dice_score = (2. * intersection + self.smooth) / (
            outputs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4)) + self.smooth
        )

        dice_loss = 1 - dice_score.mean()
        return dice_loss
    
class DiceCELoss(nn.Module):
    """
    Combines dice loss and cross entropy loss.

    Dice loss measures how much the predicted and true masks overlap.
    Cross entropy loss measures how well each 3D pixel is classified.
    Using both helps the model learn both shape and detail accuracy.
    """
    def __init__(self, smooth=1e-5, ce_weight=0.5):
        """
        Parameters:
        smooth: constant added to numerator and denominator to avoid division by 
        zero when masks are empty
        ce_weight: Weight factor for the cross entropy part
        """
        super().__init__()
        self.dice = DiceLoss3D(smooth)
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, outputs, targets_one_hot):
        """
        Calculate the combined dice and cross entropy loss

        Parameters:
        outputs: Raw model outputs with shape 
        [Batch size, Number of classes, Depth, Height, Width]
        targets_one_hot: Ground truth masks with same shape as outputs

        Returns:
        Tensor: Combined dice + cross entropy loss value
        """
        # Dice loss
        # Measures overlap between predicted and true masks
        dice_loss = self.dice(outputs, targets_one_hot)

        # Cross entropy loss
        # Converts one-hot masks into class labels for cross entropy 
        targets_labels = torch.argmax(targets_one_hot, dim=1)
        ce_loss = self.ce(outputs, targets_labels)

        # Combine both losses
        return dice_loss + self.ce_weight * ce_loss
    
def dice_score_per_class(predictions, targets_one_hot, smooth=1e-5):
    """
    Compute the dice score for each class.

    Dice score measures the overlap between the predicted segmentation and the 
    ground truth mask for each class.

    Parameters:
    predictions: Model outputs with shape [Batch, Number of classes, Depth, Height, Width]
    targets_one_hot: One-hot encoded ground truth masks with same shape as predictions
    smooth: Small constant to avoid division by zero when masks are empty

    Returns:
    List: Dice score for each class
    """
    # Convert to probabilities using softmax
    probabilities = torch.softmax(predictions, dim=1)

    # Convert probabilities to predicted class labels
    predicted_labels = torch.argmax(probabilities, dim=1)

    # Convert one-hot ground truth to class labels
    targets_labels = torch.argmax(targets_one_hot, dim=1)

    dice_scores = []
    num_classes = predictions.shape[1]

    # Calculate dice score for each class
    for cls in range(num_classes):
        # Binary mask for current class
        predicted_mask = (predicted_labels == cls).float()
        true_mask = (targets_labels == cls).float()

        # Compute intersection and dice score
        intersection = (predicted_mask * true_mask).sum()
        dice = (2 * intersection + smooth) / (predicted_mask.sum() + true_mask.sum() + smooth)
        
        dice_scores.append(dice.item())
        
    return dice_scores
    
def visualise_volume_prediction(model, dataset, idx=0, device="cpu", save_path="prediction.png"):
    """
    Visulise the segmentation prediction compared to the ground truth.

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
    # Set model to evaluation movde
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

    # Choose the middle slice along the depth axis for visualisation
    mid_slice = image.shape[1] // 2

    # Create a figure will 3 panels (input MRI, ground truth, and prediction)
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
    print(f"Saved prediction visualization to {save_path}")