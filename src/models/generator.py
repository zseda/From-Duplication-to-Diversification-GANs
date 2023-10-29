import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from functools import partial


class CustomLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer=nn.InstanceNorm2d,
        activation_function=nn.LeakyReLU,
    ):
        self.sequential = nn.Sequential(
            # 1st layer
            nn.Conv2d(
                in_channels=in_channels,
                # options:encoding, decoding,keep the same
                # decide the output channel based on your goal
                # decoding might more sense since we want diversity in the output
                out_channels=16,
                kernel_size=3,
                padding="same",
            ),
            # TODO: Adaptive Instance Normalization
            norm_layer(num_features=16),
            activation_function(),
        )

    def forward(self, x):
        return self.sequential(x)


class DecodingModule(nn.Module):
    """Decoding module that grows to twice its input channel size"""

    def __init__(
        self,
        in_channels: int,
        norm_layer=nn.InstanceNorm2d,
        activation_function=nn.LeakyReLU,
    ):
        super().__init__()

        Layer = partial(
            CustomLayer, norm_layer=norm_layer, activation_function=activation_function
        )

        self.sequential = nn.Sequential(
            Layer(in_channels=in_channels, out_channels=int(in_channels * 1.5)),
            Layer(
                in_channels=int(in_channels * 1.5), out_channels=int(in_channels * 1.5)
            ),
            Layer(
                in_channels=int(in_channels * 1.5), out_channels=int(in_channels * 2)
            ),
        )

        self.shortcut = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * 2),
            kernel_size=1,
            padding="same",
        )

    def forward(self, x):
        # feed input to sequential module => compute output features
        sequential_out = self.sequential(x)
        # transform input to match feature match size of output layer of sequential module
        transformed_input = self.shortcut(x)

        # residual output
        return transformed_input + sequential_out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        CustomDecodeModule = partial(
            DecodingModule,
            norm_layer=nn.InstanceNorm2d,
            activation_function=nn.LeakyReLU,
        )

        # feature extractor for processing input images
        self.feature_extractor = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            # TODO: test diffrent output indices for feature extraction
            out_indices=[3],
        )

        # generative module
        self.generative = nn.Sequential(
            # TODO: input resolution: ???
            # TODO: figure out if upsampling or downsampling is needed (e.g.) timm output is too large or too small
            # *LAZY* conv2d layer which automatically calculates number of in_channels
            # from merged and outputs the specified channel
            nn.LazyConv2d(out_channels=16, kernel_size=1, padding="same"),
            CustomDecodeModule(in_channels=16),
            CustomDecodeModule(in_channels=32),
            CustomDecodeModule(in_channels=64),
            CustomLayer(
                in_channels=64,
                out_channels=3,
                norm_layer=nn.InstanceNorm2d,
                activation_function=nn.Identity,
            ),
        )

    def forward(self, img, noise):
        # extract features from image
        features = self.feature_extractor(img)
        # TODO: test the need of subnetwork for noise processing

        # merge noise with features
        # => noise and features need to have *same* dimensions if concatenated
        # => merging at dim=1 means concat at channel dim => (b, c, h, w)
        # TODO: check out adaptive instance normalization
        merged = torch.cat((features, noise), dim=1)

        # compute output image
        output_img = self.generative(merged)

        return output_img
