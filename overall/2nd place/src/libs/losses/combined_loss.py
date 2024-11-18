from torch import nn
import torch
from torch.nn.functional import mse_loss

from typing import Union, Dict, List, Tuple, Any


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = nn.ModuleList()
        self.losses_details = []

    def add_loss(self, output_name: str, loss: nn.Module, weight: float = 1, name: str = None):
        self.losses.append(loss)
        self.losses_details.append((output_name, weight, name if name is not None else output_name))

    def forward(self,
                output_dict: Dict[str, torch.Tensor],
                target_dict: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        result_loss = 0
        logs = {}

        for loss, details in zip(self.losses, self.losses_details):
            out_name, weight, loss_name = details
            if out_name not in output_dict:
                raise ValueError(f"No output {out_name}, which is needed by loss.")

            if out_name in target_dict:
                l = loss(output_dict[out_name], target_dict[out_name])
            else:
                l = loss(output_dict[out_name])

            logs[loss_name] = l.item()
            result_loss = result_loss + l * weight

        return result_loss, logs


class RMSE(nn.Module):
    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        return mse_loss(predicted, target).sqrt()


class PercentileLoss(nn.Module):
    def _percentile(self, predicted: torch.Tensor, target: torch.Tensor, percentile: float):
        return 2*(percentile*torch.clamp(target-predicted, min=0.0) + (1-percentile)*torch.clamp(predicted-target, min=0.0)).mean()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor):
        return (
            self._percentile(predicted[:, 0], target, 0.5) +
            self._percentile(predicted[:, 1], target, 0.1) +
            self._percentile(predicted[:, 2], target, 0.9)
        )/3.


class PercentileSquareLoss(PercentileLoss):
    def _percentile(self, predicted: torch.Tensor, target: torch.Tensor, percentile: float):
        return 2*(percentile*torch.square(torch.clamp(target-predicted, min=0.0)) + (1-percentile)*torch.square(torch.clamp(predicted-target, min=0.0))).mean()


class ReluRegularization(nn.Module):
    def forward(self, predicted: torch.Tensor):
        return nn.functional.relu(predicted).sum()


def _loss_factory(name: str) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "l2" or name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=2)
    if name == "rmse":
        return RMSE()
    if name == "percentile":
        return PercentileLoss()
    if name == "percentile_square":
        return PercentileSquareLoss()
    if name == "relu_regularization":
        return ReluRegularization()
    raise ValueError(f"Unknown loss {name}")


def create_loss(opt: List[Union[Any, List[Any]]]) -> nn.Module:
    if not isinstance(opt, list):
        raise ValueError("Losses should be configured as the list of 3 values or list of 3-lists")
    if not isinstance(opt[0], list):
        opt = [opt]

    loss = CombinedLoss()
    used_names = {}
    for loss_conf in opt:
        if len(loss_conf) == 2:
            loss_conf.append(1)
        name = loss_conf[1]
        if name in used_names:
            name = loss_conf[0]+"_"+loss_conf[1]
        if name in used_names:
            id = 1
            while f"{name}_{id}" in used_names:
                id += 1
            name = f"{name}_{id}"
        used_names[name] = 1
        loss.add_loss(
            loss_conf[0],
            _loss_factory(loss_conf[1]),
            loss_conf[2],
            name
        )

    return loss
