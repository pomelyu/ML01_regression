import functools
from typing import Optional, Union

import torch.nn.functional as F
from torch import nn

from .activations import Activation
from .normalizations import Normalization2d


# pylint: disable=arguments-differ

class Conv2dBlock(nn.Sequential):
    def __init__(self, in_nc: int, out_nc: int, kernel_size: int, stride: int = 1, padding: int = 0,
                    norm: Optional[Union[str, nn.Module]] = None,
                    actv: Optional[Union[str, nn.Module]] = None, **kwargs):

        super().__init__()

        self.add_module("conv", nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding, **kwargs))

        if isinstance(norm, str):
            self.add_module("norm", Normalization2d(norm, out_nc))
        elif isinstance(norm, nn.Module):
            self.add_module("norm", norm)

        if isinstance(actv, str):
            self.add_module("actv", Activation(str))
        elif isinstance(actv, nn.Module):
            self.add_module("actv", actv)


class DeConv2dBlock(nn.Sequential):
    def __init__(self, in_nc: int, out_nc: int, method: str, kernel_size: int, stride: int = 1, padding: int = 0,
                    norm: Optional[Union[str, nn.Module]] = None,
                    actv: Optional[Union[str, nn.Module]] = None,
                    mode: str = "nearest", scale: int = 2, **kwargs):

        super().__init__()

        if method == "conv_trans":
            self.add_module("deconv", nn.ConvTranspose2d(in_nc, out_nc, kernel_size, stride, padding, **kwargs))
        elif method == "deconv":
            align_corners = None if mode == "nearest" else False
            self.add_module("deconv", DeConv2d(in_nc, out_nc, kernel_size, scale=scale, mode=mode,
                                align_corners=align_corners, stride=stride, padding=padding, **kwargs))
        elif method == "pixel_shuffle":
            self.add_module("deconv", PixelShuffle2d(in_nc, out_nc, kernel_size, scale=scale, stride=stride,
                                                        padding=padding, **kwargs))
        else:
            raise NameError("Unknown method: " + method)

        if isinstance(norm, str):
            self.add_module("norm", Normalization2d(norm, out_nc))
        elif isinstance(norm, nn.Module):
            self.add_module("norm", norm)

        if isinstance(actv, str):
            self.add_module("actv", Activation(str))
        elif isinstance(actv, nn.Module):
            self.add_module("actv", actv)


class DeConv2d(nn.Conv2d):
    def __init__(self, in_nc: int, out_nc: int, kernel_size: int,
                    scale: int = 2, mode: str = "nearest", align_corners: Optional[bool] = None, **kwargs):
        super().__init__(in_nc, out_nc, kernel_size, **kwargs)

        self.up = functools.partial(F.interpolate, scale_factor=scale, mode=mode, align_corners=align_corners)

    def forward(self, x):
        x = self.up(x)
        x = super().forward(x)
        return x


class PixelShuffle2d(nn.Conv2d):
    def __init__(self, in_nc, out_nc, kernel_size: int, scale: int = 2, **kwargs):
        super().__init__(in_nc, out_nc * scale * scale, kernel_size, **kwargs)

        self.up = functools.partial(F.pixel_shuffle, upscale_factor=scale)

    def forward(self, x):
        x = super().forward(x)
        x = self.up(x)
        return x
