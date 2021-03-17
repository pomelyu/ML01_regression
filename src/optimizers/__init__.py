import mlconfig

from ..utils.mlconfig_torch import register_torch_optimizers
from .radam import RAdam

register_torch_optimizers()

mlconfig.register(RAdam)
