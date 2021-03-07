from inspect import getmembers, isclass
from typing import Dict

from torch import nn


def register_torch_normalization() -> Dict[str, nn.Module]:
    _norm_dict = {
        "batch1d": nn.BatchNorm1d,
        "batch2d": nn.BatchNorm2d,
        "batch3d": nn.BatchNorm3d,
        "instance1d": nn.InstanceNorm1d,
        "instance2d": nn.InstanceNorm2d,
        "instance3d": nn.InstanceNorm3d,
    }

    return _norm_dict

norm_dict = register_torch_normalization()

def Normalization1d(name: str, num_features: int, **kwargs) -> nn.Module:
    name = f"{name}1d"

    name = name.lower()
    if name not in norm_dict:
        raise NameError(f"Unknow normalization: {name}")

    return norm_dict[name](num_features, **kwargs)

def Normalization2d(name: str, num_features: int, **kwargs) -> nn.Module:
    name = f"{name}2d"

    name = name.lower()
    if name not in norm_dict:
        raise NameError(f"Unknow normalization: {name}")

    return norm_dict[name](num_features, **kwargs)

def Normalization3d(name: str, num_features: int, **kwargs) -> nn.Module:
    name = f"{name}3d"

    name = name.lower()
    if name not in norm_dict:
        raise NameError(f"Unknow normalization: {name}")

    return norm_dict[name](num_features, **kwargs)
