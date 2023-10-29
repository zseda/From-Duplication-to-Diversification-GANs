import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            # TODO: test diffrent output indices for feature extraction
            out_indices=[4],
        )

    def forward(self, img, noise):
        # extract features from image
        features = self.feature_extractor(img)
        # TODO: test the need of subnetwork for noise processing
        # merge noise with features
        features = torch.cat((features, noise), dim=1)

        return features
