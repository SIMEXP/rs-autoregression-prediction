import torch


def load_model_from_ckpt(
    ckpt_path: str, net: torch.nn.Module
) -> torch.nn.Module:
    """Load a pytorch model with lightning checkpoint file"""
    ckpt = torch.load(ckpt_path)
    model_weights = ckpt["state_dict"]
    for key in list(model_weights):  # hack to use the model directly
        model_weights[key.replace("net.", "")] = model_weights.pop(key)
    net.load_state_dict(model_weights)
    return net
