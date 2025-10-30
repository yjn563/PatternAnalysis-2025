import torch
from train import ImprovedUNet3D

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