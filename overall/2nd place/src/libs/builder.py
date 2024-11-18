from enum import Enum
import yaml
import os
from copy import deepcopy
from typing import Optional, List, Tuple, Dict, Any, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateFinder
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from libs.losses.combined_loss import create_loss
from libs.optimizers.optimizer_factory import create_optimizer
from libs.models.model_factory import create_model
from libs.lightning_module import LightningModule
from libs.data.datasets import create_dataloaders


class BuildMode(Enum):
    LIGHTNING = 0
    INFERENCE = 1


class BuildObjects(Enum):
    MODEL = 0
    CALLBACKS = 1
    CHECKPOINT_CALLBACK = 2
    DATALOADERS = 3
    TRAINER = 4
    WEIGHT_PATH = 5
    LOGGER = 6
    OPTIONS = 7


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        opt = yaml.safe_load(f)
    return opt


def _build_callbacks(train_opt: Dict[str, Any]) -> List[Callback]:
    res = []
    if "early_stop" in train_opt:
        res.append(EarlyStopping(
            monitor=train_opt.get("checkpoint_monitor", "val_loss"),
            patience=train_opt["early_stop"],
            verbose=True
        ))
    if "checkpoints" in train_opt:
        path = train_opt["checkpoints"]
        os.makedirs(path, exist_ok=True)
        res.append(ModelCheckpoint(
            dirpath=path,
            monitor=train_opt.get("checkpoint_monitor", "val_loss"),
        ))
    if "lr_finder" in train_opt and train_opt["lr_finder"]:
        res.append(LearningRateFinder())

    return res


def _build_lr_scheduler(optimizer, train_opt: Dict[str, Any]) -> Optional[object]:
    if "reduce_lr" in train_opt:
        d = train_opt["reduce_lr"]
        return ReduceLROnPlateau(
            optimizer,
            factor=d.get("factor", 0.1),
            patience=d.get("patience", 5),
            verbose=True
        )
    if "cosine" in train_opt:
        d = train_opt["cosine"]
        return CosineAnnealingWarmRestarts(
            optimizer,
            **d
        )

    return None


def _build_logger(train_opt: Dict[str, Any]) -> Optional[TensorBoardLogger]:
    if "log_path" in train_opt:
        os.makedirs(train_opt["log_path"], exist_ok=True)
        return TensorBoardLogger(
            save_dir=train_opt["log_path"],
            name=train_opt.get("log_name", "default")
        )

    return None


def _save_config_to_log(logger: TensorBoardLogger, opt: Dict[str, Any]):
    logger.experiment.add_text("config", str(opt))

    def make_dict_hparams(opt: Dict[str, Any], prefix: str="") -> Dict[str, str]:
        res = {}
        for k, v in opt.items():
            if isinstance(v, dict):
                res.update(make_dict_hparams(v, f"{prefix}{k}/"))
            else:
                res[f"{prefix}{k}"] = str(v)
        return res

    # logger.experiment.add_hparams(hparam_dict=make_dict_hparams(opt), metric_dict={})


def _build_trainer(train_opt: Dict[str, Any]) -> Tuple[pl.Trainer, List[Callback], TensorBoardLogger]:
    callbacks = _build_callbacks(train_opt)
    logger = _build_logger(train_opt)
    return pl.Trainer(
        # gpus=-1,
        # precision=16 if train_opt.get("fp16", False) else 32,
        # accelerator="ddp",
        accumulate_grad_batches=train_opt.get("grad_accum", 1),
        max_epochs=train_opt.get("epochs", 20),
        default_root_dir=train_opt.get("root_dir", None),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=train_opt.get("grad_clip", None),
        # track_grad_norm=True,
        gradient_clip_algorithm="value",
        num_sanity_val_steps=0
    ), callbacks, logger


def build_all(
        mode: BuildMode,
        opt: Union[str, Dict[str, Any]]
) -> Dict[BuildObjects, Any]:
    if isinstance(opt, str):
        opt = _load_yaml(opt)

    res = {}
    res[BuildObjects.OPTIONS] = opt

    model = create_model(deepcopy(opt["model"]))
    if mode == BuildMode.LIGHTNING:
        train_opt = deepcopy(opt["train"])
        loss = create_loss(train_opt["losses"])
        optimizer = create_optimizer(train_opt["optimizer"], model.parameters())
        scheduler = _build_lr_scheduler(optimizer, train_opt)
        if scheduler is not None:
            optimizer = {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        model = LightningModule(
            model=model,
            loss=loss,
            optimizer=optimizer
        )

        trainer, callbacks, logger = _build_trainer(train_opt)
        res[BuildObjects.TRAINER] = trainer
        res[BuildObjects.CALLBACKS] = callbacks
        res[BuildObjects.LOGGER] = logger

        if logger is not None:
            # refactor later
            # this is better done after training with the saving of metrics
            _save_config_to_log(logger, opt)

        for c in callbacks:
            if isinstance(c, ModelCheckpoint):
                res[BuildObjects.CHECKPOINT_CALLBACK] = c

        if "weights_path" in train_opt:
            res[BuildObjects.WEIGHT_PATH] = train_opt["weights_path"]

    res[BuildObjects.MODEL] = model

    dls = create_dataloaders(opt)
    res[BuildObjects.DATALOADERS] = dls

    return res
