# Added for Adaptive Superpixels feature
import torch
import torch.nn as nn

class ParameterPredictor(nn.Module):
    """
    Predicts SLIC parameters K (number of superpixels) and m (compactness) 
    for an input image. The network uses a series of convolutional blocks 
    followed by an MLP head.
    """
    def __init__(self, k_min=50, k_max=400, m_min=1, m_max=30):
        """
        Initializes the ParameterPredictor.

        Args:
            k_min (int): Minimum value for the predicted K (number of superpixels).
            k_max (int): Maximum value for the predicted K.
            m_min (int): Minimum value for the predicted m (compactness).
            m_max (int): Maximum value for the predicted m.
        """
        super().__init__()

        # Store min/max values for scaling the output
        self.k_min = k_min
        self.k_max = k_max
        self.m_min = m_min
        self.m_max = m_max

        # Feature extraction backbone: Convolutional blocks
        # Takes (N, 3, H, W) input
        self.conv_blocks = nn.Sequential(
            # Block 1: Reduces spatial dimensions by a factor of 4 (2 from stride, 2 from pool)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # (N, 32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (N, 32, H/4, W/4)

            # Block 2: Reduces spatial dimensions by a factor of 4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (N, 64, H/8, W/8)
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (N, 64, H/16, W/16)

            # Block 3: Reduces spatial dimensions by a factor of 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (N, 128, H/32, W/32)
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Global feature aggregation
        # Reduces spatial dimensions to (1, 1) per channel
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # (N, 128, 1, 1)

        # Prediction head: MLP
        self.mlp_head = nn.Sequential(
            nn.Flatten(), # (N, 128)
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 2)  # Outputs two raw values: one for K, one for m
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict K and m for a batch of images.

        Args:
            x (torch.Tensor): Input batch of images (N, C, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - predicted_k (torch.Tensor): Predicted K values, scaled to [k_min, k_max]. Shape (N,).
                - predicted_m (torch.Tensor): Predicted m values, scaled to [m_min, m_max]. Shape (N,).
        """
        # Pass input through convolutional blocks for feature extraction
        x = self.conv_blocks(x)

        # Reduce spatial dimensions to a single feature vector per image
        x = self.avgpool(x)

        # Pass features through MLP head to get raw predictions for K and m
        x = self.mlp_head(x)

        # Split the 2-dim output into raw K and m predictions
        predicted_k_raw = x[:, 0] # (N,)
        predicted_m_raw = x[:, 1] # (N,)

        # Apply sigmoid to constrain outputs to the range (0, 1)
        # This is a common technique before scaling to a specific range.
        sigmoid_k = torch.sigmoid(predicted_k_raw)
        sigmoid_m = torch.sigmoid(predicted_m_raw)

        # Scale the sigmoid outputs to the desired [min, max] ranges
        # Formula: scaled_value = min_val + sigmoid_output * (max_val - min_val)
        predicted_k = self.k_min + sigmoid_k * (self.k_max - self.k_min)
        predicted_m = self.m_min + sigmoid_m * (self.m_max - self.m_min)

        return predicted_k, predicted_m

if __name__ == '__main__':
    # Example usage:
    # Create a dummy input tensor (batch size 4, 3 channels, height 256, width 256)
    dummy_input = torch.randn(4, 3, 256, 256)

    # Instantiate the predictor
    predictor = ParameterPredictor()

    # Get predictions
    k_preds, m_preds = predictor(dummy_input)

    print("Predicted K:", k_preds)
    print("Predicted m:", m_preds)
    print("K shape:", k_preds.shape)
    print("m shape:", m_preds.shape)

    # Example with different ranges
    custom_predictor = ParameterPredictor(k_min=10, k_max=100, m_min=0.5, m_max=5)
    k_custom, m_custom = custom_predictor(dummy_input)
    print("\nCustom Predicted K:", k_custom)
    print("Custom Predicted m:", m_custom)
