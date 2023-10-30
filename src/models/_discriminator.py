import torch.nn as nn
import torch.nn.functional as F
import timm


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Base model for feature extraction
        self.base_model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            out_indices=[3],
        )

        # Classifier head
        self.head = nn.Sequential(
            # Flatten the output from base model
            nn.Flatten(),
            # Lazy fully connected layer
            nn.LazyLinear(out_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            # Output layer for binary classification (real vs fake)
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.base_model(x)[0]
        x = self.head(x)
        return x
