import mlconfig
import torch
from torch import nn

from .components.conv2d import DeConv2dBlock, Conv2dBlock

# pylint: disable=arguments-differ

@mlconfig.register()
class DCGAN_G(nn.Sequential):
    def __init__(self, latent_size=128, out_nc=3, nd=64, num_layers=5, bias=False):
        super().__init__()

        assert num_layers >= 3

        up_nds = [latent_size] + [nd * (2 ** i) for i in list(range(3)) + [3] * (num_layers - 4)]

        for i, (in_nd, out_nd) in enumerate(zip(up_nds[:-1], up_nds[1:])):
            if i == 0:
                m = DeConv2dBlock(in_nd, out_nd, "conv_trans",
                                    kernel_size=4, stride=1, bias=bias,
                                    norm="instance", actv=nn.LeakyReLU(inplace=True))
            else:
                m = DeConv2dBlock(in_nd, out_nd, "deconv",
                                    kernel_size=3, stride=1, padding=1, bias=bias,
                                    mode="bilinear", scale=2,
                                    norm="instance", actv=nn.LeakyReLU(inplace=True))
            self.add_module(f"up{i+1}", m)

        m = DeConv2dBlock(up_nds[-1], out_nc, "deconv", kernel_size=3, stride=1, padding=1,
                            bias=bias, mode="bilinear", scale=2, norm=None, actv=nn.Sigmoid())
        self.add_module(f"up{num_layers}", m)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = super().forward(x)
        return x


@mlconfig.register()
class DCGAN_D(nn.Sequential):
    def __init__(self, in_nc, nd=128, num_layers=5, bias=False):
        super().__init__()

        assert num_layers >= 3

        down_nds = [in_nc] + [nd * (2 ** i) for i in list(range(3)) + [3] * (num_layers - 4)]
        for i, (in_nd, out_nd) in enumerate(zip(down_nds[:-1], down_nds[1:])):
            m = Conv2dBlock(in_nd, out_nd, kernel_size=4, stride=2, padding=1, bias=bias,
                    norm="instance", actv=nn.LeakyReLU(inplace=True))
            self.add_module(f"down{i+1}", m)

        m = nn.Conv2d(down_nds[-1], 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.add_module("out", m)
