from typing import Dict, Any
from torch.optim import Optimizer, AdamW, SGD


def create_optimizer(opt: Dict[str, Any], parameters) -> Optimizer:
    if isinstance(opt, str):
        opt = {"type": opt}
    name = opt.pop("type")

    if name == "sgd":
        return SGD(parameters, **opt)
    if name == "adam":
        return AdamW(parameters, **opt)

    raise ValueError(f"Unknown optimizer {name}")
