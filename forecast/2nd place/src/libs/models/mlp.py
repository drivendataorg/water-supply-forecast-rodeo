import torch
from torch import nn

from typing import Dict, Any
from copy import deepcopy


class MLP(nn.Module):
    def __init__(self, opt: Dict[str, Any]):
        super().__init__()

        in_f = opt["in_features"]
        activation = nn.ReLU    # nn.GELU  # nn.ReLU

        mlp_dropout = opt.get("mlp_dropout", None)
        mlp_last_dropout = opt.get("mlp_last_dropout", mlp_dropout)
        blocks = []
        layers = opt["layers"]
        for i, f in enumerate(layers):
            dropout = mlp_last_dropout if i == len(layers)-1 else mlp_dropout
            if dropout and i > 0:
                blocks.append(nn.Dropout(dropout))
            blocks.append(nn.Linear(in_f, f))
            blocks.append(activation())
            in_f = f
        blocks = blocks[:-1]
        self.mlp = nn.Sequential(*blocks)

    def forward(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        y = self.mlp(x)
        return {"out": y.squeeze()}


class MLPWithShiftOutput(MLP):
     def forward(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        y = super().forward(x)["out"]
        return {
            "out": y,
            "out_shift": x[:, 28].unsqueeze(-1)-y
        }


class MLPSumRes(MLP):
    def forward(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        y = super().forward(x)["out"]
        # y = nn.functional.relu(y) + x[:, 28].unsqueeze(-1)
        y = y + x[:, 28].unsqueeze(-1)
        return {"out": y}


class MLPEmbedSumRes(MLP):
    def __init__(self, opt: Dict[str, Any]):

        self.embed_cnt = opt["embed_cnt"]
        self.embed_dim = opt["embed_dim"]
        opt = deepcopy(opt)
        opt["in_features"] = opt["in_features"] - self.embed_cnt + self.embed_dim
        super().__init__(opt)

        self.embed = nn.Parameter(torch.randn((self.embed_cnt, self.embed_dim)), requires_grad=True)

    def forward(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        monthly_sum = x[:, 28].unsqueeze(-1)
        embed = x[:, :self.embed_cnt] @ self.embed
        x = torch.cat([x[:, self.embed_cnt:], embed], dim=-1)
        y = super().forward(x)["out"]
        y = nn.functional.relu(y) + monthly_sum
        return {"out": y}


class ConcatMLP(nn.Module):
    def __init__(self, opt: Dict[str, Any]):
        super().__init__()

        in_f = opt["in_features"]
        first_f = in_f
        activation = nn.ReLU    # nn.GELU  # nn.ReLU

        mlp_dropout = opt.get("mlp_dropout", None)
        mlp_last_dropout = opt.get("mlp_last_dropout", mlp_dropout)
        blocks = []
        layers = opt["layers"]
        for i, f in enumerate(layers):
            dropout = mlp_last_dropout if i == len(layers)-1 else mlp_dropout
            if dropout and i > 0:
                blocks.append(nn.Dropout(dropout))
            blocks.append(nn.Linear(in_f + (0 if i==0 else first_f), f))
            blocks.append(activation())
            in_f = f
        blocks = blocks[:-1]
        self.blocks = nn.ModuleList(blocks)

    def forward(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        in_x = x
        for i, block in enumerate(self.blocks):
            if i>0 and isinstance(block, nn.Linear):
                x = torch.concatenate([x, in_x], dim=-1)
            x = block(x)
        return {"out": x.squeeze()}


class ConcatMLPWithShiftOutput(ConcatMLP):
    def forward(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        y = super().forward(x)["out"]
        return {
            "out": y,
            "out_shift": x[:, 28].unsqueeze(-1) - y
        }
