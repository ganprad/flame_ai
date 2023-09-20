import torch
import torch.nn as nn


def massflux(ff, flatten=True):
    """
    Assuming flowfield ff => [rho, ux, uy, uz]
    """
    if flatten:
        ff = ff.squeeze()
        flux = ff[..., 0].unsqueeze(-1) * ff[..., 1:]
        return flux / flux.norm()
    flux = ff[:, 0, ...] * ff[:, 1:, ...].squeeze(0)
    return flux / flux.norm()


def continuity_loss(preds, targets):
    preds = massflux(preds)
    targets = massflux(targets)
    return nn.MSELoss()(preds, targets)
class ResNetBlock(nn.Module):
    def __init__(self, num_filters, kernel_size=3):
        super(ResNetBlock, self).__init__()
        self.resnet_block = torch.nn.Sequential(
            *[
                nn.Conv2d(num_filters, num_filters, kernel_size, padding="same"),
                nn.Softplus(),
                nn.Conv2d(num_filters, num_filters, kernel_size, padding="same"),
                nn.Softplus(),
            ]
        )
        self.input = nn.Sequential()

    def forward(self, x):
        inp = self.input(x)
        x = self.resnet_block(x)
        return x + inp


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        factor,
        scale,
        num_of_residual_blocks,
        num_filters,
        kernel_size,
        do_upsample=True,
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
                    num_filters=num_filters,
                    kernel_size=kernel_size,
                )
            ]
            * num_of_residual_blocks
        )
        self.do_upsample = do_upsample
        if self.do_upsample:
            # Upsampling (factor ** 2) ** scale times : (2**(2**3)) : 16*16 -> 128 * 128
            self.upsample = nn.Sequential(
                *[
                    nn.Conv2d(num_filters, num_filters * (factor**2), kernel_size, padding="same"),
                    nn.PixelShuffle(upscale_factor=factor),
                ]
                * scale
            )
            self.lr = nn.Sequential(
                *[
                    nn.Conv2d(num_filters, num_filters, kernel_size, padding="same"),
                    nn.PixelShuffle(upscale_factor=1),
                ]
                * scale
            )
            self.output_layer_lr = nn.Conv2d(num_filters, in_channels, 3, padding="same")
        self.input_layer = nn.Conv2d(in_channels, num_filters, 3, padding="same")
        self.output_layer = nn.Conv2d(num_filters, in_channels, 3, padding="same")

    def forward(self, x):
        x = nn.Softplus()(self.input_layer(x))
        x_res = self.res_blocks(x)
        out = x + x_res
        if self.do_upsample:
            up = self.upsample(out)
            return self.output_layer_lr(self.lr(out)), self.output_layer(up)
        return self.output_layer(out)
