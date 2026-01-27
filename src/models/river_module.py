from typing import Any, Dict, Tuple
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.segmentation import DiceScore
from torch.cuda.amp import GradScaler, autocast

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms.transform import Transform
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch

from torch_ema import ExponentialMovingAverage as EMA

import numpy as np
import time
import os
import shutil

from functools import partial
import wandb
import inspect 
from copy import copy, deepcopy
from contextlib import contextmanager

import rootutils
rootutils.setup_root(__file__, indicator="setup.py", pythonpath=True)
from src.utils.ema import LitEma

def structured_loss(logits, tgt, sequential=False):
    pred = F.sigmoid(logits).squeeze(1)
    reduce_dim = list(range(2 if sequential is True else 1, len(pred.shape)))
    intersection = torch.sum(pred * tgt, dim=reduce_dim)
    summation = torch.sum(pred + tgt, dim=reduce_dim)
    dice_loss = 1 - 2 * intersection / summation
    bce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), tgt,
                                                reduction='none').mean(dim=reduce_dim)
    print(dice_loss, bce_loss)
    return 0.5 * dice_loss + 0.5 * bce_loss

def semi_loss(logits, teacher_logits, tgt, labels, sequential=False, weight = 0.1):
    sup_loss = structured_loss(logits, tgt, sequential=sequential)
    if weight == 1:
        if labels.mean() == 0:
            return (sup_loss * 0).mean()
        return (sup_loss * labels).mean() / labels.mean()
    reduce_dim = list(range(2 if sequential is True else 1, len(logits.shape)))
    # print(reduce_dim)
    # print(teacher_logits.shape, logits.shape)
    unsup_loss = F.mse_loss(logits, teacher_logits, reduction='none').mean(dim=reduce_dim)
    # print(sup_loss.shape, unsup_loss.shape)
    return torch.mean(weight * sup_loss * labels + (1 - labels) * (1 - weight) * unsup_loss)

def dice_score(prob, tgt, threshold=0.5, sequential=False, jaccard=False):
    reduce_dim = list(range(2 if sequential else 1, len(pred.shape)))
    pred = torch.where(prob > threshold, 1, 0).squeeze_()
    intersection = torch.sum(pred * tgt, dim=reduce_dim)
    union = torch.sum(pred + tgt, dim=reduce_dim)
    return 2 * intersection / union

class RiverLitModule(LightningModule):
    """

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        sw_batch_size = 4,
        roi = [256, 256],
        infer_overlap = 0.5,
        threshold=0.6,
        temporal=False,
        semi_epoch=5,
        semi_weight=0.3,
        ema: LitEma | EMA | Callable | None = None,
    ) -> None:
        """Initialize a `SpiderLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['criterion', 'net'])

        self.net = net
        
        self.ema = ema(self.net) if ema is not None else None

        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=roi,
            sw_batch_size=sw_batch_size,
            predictor=net,
            overlap=infer_overlap,
            mode='gaussian',
            sigma_scale=0.25,
            threshold=threshold,
        )
        self.criterion = partial( semi_loss,
                                    sequential=temporal)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        self.metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

        # counter for semi-supervised
        self.epoch = 0

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_dice = MeanMetric()
        self.datamodule = None

    def train_on_device(model):
        super().train_on_device(model)
        if self.ema is not None:
            self.ema.to(self.device)

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema is not None:
            self.ema.store(self.net.parameters())
            self.ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.ema is not None:
                self.ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_dice.reset()
        self.val_acc_best.reset()

    def on_train_epoch_start(self) -> None:
        self.net.train()
        self.train_loss.reset()
    
    def on_train_epoch_end(self) -> None:
        self.log("train/loss", self.train_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.epoch += 1
        for param in self.net.parameters():
            param.grad = None

        if self.ema is not None: 
            self.ema(self.net)
        self.train_loss.reset()

        if self.datamodule:
            self.datamodule.data_train.data, _ = self.datamodule.pool.augment_crop(skip_val=True)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if isinstance(batch, list):
            data, target = batch
        else:
            data, hard_data, target = batch["image"], batch['hard'], batch["mask"]
        # print(data.shape, hard_data.shape)
        with autocast(enabled=False):
            
            
            if self.epoch < self.hparams.semi_epoch:
                weight = 1
                teacher_logits = {'logit': 0}
            else: 
                weight = self.hparams.semi_weight
                with self.ema_scope():
                    with torch.no_grad():
                        teacher_logits = self.net(data)
            logits = self.net(hard_data)
            # Only take first argument of the function
            # logits, teacher_logits, tgt, labels, sequential=False, weight = 0.1):
            if 'cls_logit' in logits.keys():
                cls_loss = self.bce(logits['cls_logit'], batch['label']).mean()
            else:
                cls_loss = 0
            loss = self.criterion(  logits=logits['logit'],  
                                    teacher_logits=teacher_logits['logit'], 
                                    tgt=target, 
                                    labels=batch['label'],
                                    weight=weight)
        # *Place holder for archived code id 1*
        return cls_loss + loss, logits, target
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
    
        loss, logits, targets = self.model_step(batch)
        
        # update and log metrics
        if torch.any(torch.isnan(loss)): 
            print(logits.shape, targets.shape)
        self.train_loss.update(loss)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss
        
    @torch.no_grad()
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
    
        data, target = batch["image"], batch["mask"]
        if isinstance(self.ema, LitEma):
            with self.ema_scope():
                with autocast(enabled=False):
                    logits = self.model_inferer(data) ## why does it require [b, 4, w, h, d]?????
        else: 
            with autocast(enabled=False):
                logits = self.model_inferer(data)
        
        # Inference
        val_labels_list = decollate_batch(target) ## Optimal to use decollate_batch, we can choose to use it or not
        val_outputs_list = decollate_batch(logits) ## Optimal to use decollate_batch, we can choose to use it or not
        val_output_convert = [torch.where(F.sigmoid(val_pred_tensor) >= self.hparams.threshold, 1, 0) for val_pred_tensor in val_outputs_list]
        # print(len(val_output_convert), [tensor.shape for tensor in val_output_convert])
        # Metric computations
        # print(logits.shape, target.shape)
        for pred, tgt in zip(val_output_convert, val_labels_list):
            print(pred.shape, tgt.shape)
        metrics = [self.metric(pred, tgt.int().unsqueeze_(0)) for pred, tgt in zip(val_output_convert, val_labels_list)]
        metrics = torch.where(torch.isnan(metrics[0]), 0, metrics)
        loss = structured_loss(logits, target, sequential=self.hparams.temporal)
        print(loss, metrics, torch.unique(val_labels_list[0]))
        self.val_loss.update(loss, data.size(0))
        self.val_dice.update(metrics, data.size(0))
        self.log("val/loss", loss.mean(), on_step=False, on_epoch=True, prog_bar=True)
        for metric in metrics:
            self.log("val/dice", metric, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'pred': val_output_convert, 'target': target}

    @torch.no_grad()
    def on_validation_epoch_start(self) -> None:
        print("Evaluating !!!")
        self.net.eval()
        self.val_dice.reset()
        self.val_loss.reset()
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        dice = self.val_dice.compute()  # get current val acc
        self.val_acc_best(dice)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        # self.post_metric.log('val-post', labels=self.name)

    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        data, target = batch["image"], batch["mask"]
        if isinstance(self.ema, LitEma):
            # print("Using Ema")
            with self.ema.average_parameters():
                with autocast(enabled=False):
                    logits = self.model_inferer(data) ## logits shape = [B, in_channel, D, W, H]
        else: 
            with autocast(enabled=False):
                logits = self.model_inferer(data) ## logits shape = [B, in_channel, D, W, H]

        test_labels_list = decollate_batch(target) ## Optimal to use decollate_batch, we can choose to use it or not
        test_outputs_list = decollate_batch(logits) ## Optimal to use decollate_batch, we can choose to use it or not
        
        test_output_convert = [self.post_pred(self.post_activation(test_pred_tensor)) for test_pred_tensor in test_outputs_list]
        test_output_post = [self.post_proc(sample) for sample in test_output_convert]
        
        self.metric(test_output_post, test_labels_list, 'test', labels=self.name, on_step=True)
        # print(self.dice_acc(y_pred=val_output_convert, y=val_labels_list))

        loss = self.val_criterion(logits, target)

        self.val_loss.update(loss, data.size(0))
    
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'pred': test_output_convert, 'target': target}

    
    def on_test_epoch_start(self,) -> None:
        self.net.eval()
        self.val_loss.reset()
        self.val_dice.reset()
        self.metric.reset()
        pass
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.metric.log('test', labels = self.name)
        # self.post_metric.log('test-post', labels= self.name)
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/dice", ##val/loss
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def save_checkpoint(self, model, epoch, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
        with self.ema_scope():
            state_dict = model.state_dict()
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        
        if optimizer is not None:
            save_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()
        filename = os.path.join(self.trainer.log_dir, filename)
        torch.save(save_dict, filename)
        print("Saving checkpoint", filename)
    

if __name__ == "__main__":
    import rootutils
    from src.data.components.river_dataset import *
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    from omegaconf import DictConfig
    import hydra
    @hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
    def test(cfg: DictConfig):
        model = hydra.utils.instantiate(cfg.model)
        model.epoch = 1000
        root = "/work/hpc/potato/airc/data"
        temporal = False
        data_path = os.path.join(root, "v2")
        suffix = "/norm" if not temporal else ""
        train_path = os.path.join(root, "dataset/v2/train/norm/sup")
        val_path = os.path.join(root, "dataset/v2/val") + suffix
        temp = RiverDataset(train_path, temporal=temporal)
        dataset = TransformedRiverDataset(temp, None)
        print("Data:", temp.data[0])
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=4,
            num_workers=4,
            pin_memory=False,
            shuffle=True,
        )
        loader = iter(dataloader)
        batch = next(loader)
        print(batch['filename'])
        model.net.eval()
        loss = model.training_step(batch, 1)
        print(loss)
    test()