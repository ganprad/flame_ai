import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size=3):
        super(ResNetBlock, self).__init__()
        self.resnet_block = torch.nn.Sequential(
            *[
                nn.Conv2d(num_filters, num_filters, kernel_size, padding="same"),
                nn.Conv2d(num_filters, num_filters, kernel_size, padding="same"),
            ]
        )
        self.input = nn.Sequential()

    def forward(self, x):
        inp = self.input(x)
        x = self.resnet_block(x)
        return x + inp


class Model(nn.Module):
    def __init__(
        self, in_channels=4, factor=2, scale=3, num_of_residual_blocks=16, num_filters=64, kernel_size=3, **kwargs
    ):
        super().__init__()
        self.num_of_residual_blocks = num_of_residual_blocks
        self.scale = scale
        self.factor = factor
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.res_blocks = nn.Sequential(
            *[
                ResNetBlock(
                    in_channels=in_channels,
                    num_filters=num_filters,
                    kernel_size=kernel_size,
                )
            ]
            * num_of_residual_blocks
        )
        # Upsampling (factor ** 2) ** scale times : (2**2)**3 : 16*16 -> 128 * 128
        self.upsample = nn.Sequential(
            *[
                nn.Conv2d(num_filters, num_filters * (factor**2), kernel_size, padding="same", **kwargs),
                nn.PixelShuffle(upscale_factor=factor),
            ]
            * scale
        )
        self.resnet_input = nn.Conv2d(in_channels, num_filters, 1, padding="same")
        self.output_layer = nn.Conv2d(num_filters, in_channels, 3, padding="same")
        self.resnet_out = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, padding="same")

    def forward(self, x):
        x = self.resnet_input(x)
        x_res = self.res_blocks(x)
        x_res = self.resnet_out(x_res)
        out = x + x_res
        out = self.upsample(out)
        return self.output_layer(out)
