import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Optimizer


class LightningModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 loss: nn.Module,
                 optimizer: Optimizer
                 ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = optimizer["optimizer"].param_groups[0]["lr"]
        self.optimizer = optimizer

    def forward(self, x) -> torch.Tensor:
        return self.model(*x)

    def configure_optimizers(self) -> Optimizer:
        self.optimizer["optimizer"].param_groups[0]["lr"] = self.lr
        return self.optimizer

    def _step(self, batch, log_prefix=""):
        x = batch["X"]
        target_dict = {"out": batch["target"]}
        scale = batch["volume_scale_factor"]
        outputs = self.model(x)
        # pad_mask = target_dict["pad_mask"]
        # outputs = self.model(x, common, pad_mask)

        loss, logs = self.loss(
            {k: v * scale.unsqueeze(-1) for k, v in outputs.items()},
            {k: v * scale for k, v in target_dict.items()}
        )
        for n, v in logs.items():
            self.log(log_prefix + "scaled_" + n, v, prog_bar=True)

        # _, logs = self.loss(outputs, target_dict)
        # for n, v in logs.items():
        #     self.log(log_prefix+n, v, prog_bar=True)
        if log_prefix != "":
            self.log(log_prefix+"loss", loss.item(), prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_")
