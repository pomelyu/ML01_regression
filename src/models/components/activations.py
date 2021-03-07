from inspect import getmembers, isclass
from typing import Dict

from torch import nn


def register_torch_activation() -> Dict[str, nn.Module]:
    _actv_dict = {
        name.lower(): actv
        for name, actv in getmembers(nn.modules.activation)
        if isclass(actv)
    }

    return _actv_dict

actv_dict = register_torch_activation()

def Activation(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in actv_dict:
        raise NameError(f"Unknow activation: {name}")

    return actv_dict[name](**kwargs)
