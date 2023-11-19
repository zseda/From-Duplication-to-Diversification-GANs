import torch
import torch.nn as nn
import timm


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Base model for feature extraction
        self.base_model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=1
        )

    def forward(self, x):
        x = self.base_model(x)
        x = torch.sigmoid(x)
        return x
