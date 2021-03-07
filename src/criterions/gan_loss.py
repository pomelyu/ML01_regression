import mlconfig
import torch
from torch import nn

# pylint: disable=arguments-differ

@mlconfig.register()
class GANLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def loss_g(self, net_D: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        logit_fake = net_D(fake)
        loss_real = self.criterion(logit_fake, torch.ones_like(logit_fake))
        loss_fake = torch.zeros(1, device=fake.device)

        return loss_real, loss_fake

    def loss_d(self, net_D: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        logit_real = net_D(real)
        loss_real = self.criterion(logit_real, torch.ones_like(logit_real))

        logit_fake = net_D(fake.detach())
        loss_fake = self.criterion(logit_fake, torch.zeros_like(logit_fake))

        return loss_real, loss_fake

    def forward(self, x):
        raise NotImplementedError()


@mlconfig.register()
class LSGANLoss(GANLoss):
    def __init__(self, reduction="mean"):
        super().__init__()

        self.criterion = nn.MSELoss(reduction=reduction)
