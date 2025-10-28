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