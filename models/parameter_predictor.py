import torch
import torch.nn as nn

class ParameterPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        params = self.fc(x)
        return {
            "num_superpixels": params[:, 0],
            "compactness": params[:, 1]
        }

if __name__ == '__main__':
    # Example usage (optional, for testing)
    predictor = ParameterPredictor()
    dummy_input = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256 image
    output = predictor(dummy_input)
    print(output)
    # Expected output shape for num_superpixels: torch.Size([1])
    # Expected output shape for compactness: torch.Size([1])
    print(output["num_superpixels"].shape)
    print(output["compactness"].shape)
