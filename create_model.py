
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()

        # Define the EfficientNet backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b5')

        # Replace the classifier with a custom fully connected layer
        num_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Reshape the input tensor to match the expected shape
        # The expected shape for EfficientNet is [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)  # Reshape from [batch_size, height, width, channels] to [batch_size, channels, height, width]

        # Pass the input through the EfficientNet backbone
        x = self.backbone(x)

        return x
