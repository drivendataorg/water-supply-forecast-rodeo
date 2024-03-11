from typing import Dict, Any, Union

import torch
from torch.nn import Module

from libs.models.mlp import MLP, ConcatMLP, MLPWithShiftOutput, ConcatMLPWithShiftOutput, MLPSumRes, MLPEmbedSumRes


def create_model(opt: Union[str, Dict[str, Any]]) -> Module:
    if isinstance(opt, str):
        opt = {"type": opt}
    name = opt.pop("type")
    if name == "mlp":
        model = MLP(opt)
    elif name == "mlp_shift":
        model = MLPWithShiftOutput(opt)
    elif name == "mlp_sum_res":
        model = MLPSumRes(opt)
    elif name == "mlp_embed_sum_res":
        model = MLPEmbedSumRes(opt)
    elif name == "concat_mlp":
        model = ConcatMLP(opt)
    elif name == "concat_mlp_shift":
        model = ConcatMLPWithShiftOutput(opt)
    else:
        raise ValueError(f"Unknown model {name}.")

    weights = opt.pop("weights", None)
    if weights is not None:
        model.load_state_dict(torch.load(weights))
        print(f"Load weights from {weights}")
    return model
