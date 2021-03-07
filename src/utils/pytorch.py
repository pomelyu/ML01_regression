from torch.nn import Module


def get_state_dict(model: Module):
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def set_state_dict(model: Module, state_dict: dict):
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def set_requires_grad(models: Module, requires_grad=False):
    if not isinstance(models, list):
        models = [models]
    for m in models:
        if m is not None:
            for param in m.parameters():
                param.requires_grad = requires_grad
