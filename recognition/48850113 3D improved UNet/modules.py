import torch
import torch.nn as nn

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