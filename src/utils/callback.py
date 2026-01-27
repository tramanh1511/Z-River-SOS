import os
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import lightning as pl
import torch

from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid
from torchvision.transforms import Resize
from torchvision.transforms import functional as F

from scipy.ndimage import zoom
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import wandb


class WandbCallback(Callback):
    def __init__(self, mean, var, log_spatial_size = [480, 640]):
        self.images = []
        self.captions = []
        self.transform = Resize(size=log_spatial_size)
        self.mean = torch.Tensor(mean)[:, None, None]
        self.var = torch.Tensor(var)[:, None, None]
        
    def setup(self, trainer, pl_module, stage):
        self.logger = trainer.logger
        print("NOTICEEEEEE!!!!!!!")
        self.save_folder = os.path.join(trainer.logger.save_dir, "outputs")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
    

    def visualize(self, img, pred, label, img_name=None):
        if self.mean.device != img.device:
            self.mean = self.mean.to(img.device)
            self.var = self.var.to(img.device)
        # c, h, w
        img = torch.flip(torch.clamp(img * self.var + self.mean, 0, 1), dims=[0])
        img[0] = torch.where(pred * (1 - label) > 0.5, 1, img[0])
        img[2] = torch.where((1 - pred) * label > 0.5, 1, img[2])
        img[1] = torch.where(pred * label > 0.5, 1, img[1])
        img = self.transform(img)
        img = (img - img.min()) / (img.max() - img.min())
        image = F.to_pil_image(img)

        save_path = os.path.join(self.save_folder, img_name if img_name else f"{len(self.images)}.png")
        image.save(save_path)
        self.images.append(image)
        self.captions.append(img_name)

        # self.logger.log_image(key='Visualize', images=[image_all], caption=[img_name])
    def on_validation_batch_end(self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        
        images = batch["image"] ## (B, Channel, Slice, W, H)
        labels = batch["mask"]  ## (B, Class, Slice, W, H)
        pred = outputs["pred"] ## (B, Class, Slice, W, H)

        for i in range(len(pred)):
            img_name = batch["filename"][i]
            print("Inference on case {}".format(img_name))

            self.visualize(images[i], pred[i], labels[i], img_name=img_name)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.logger.log_image(key='Visualize', images=self.images, caption=self.captions)
        self.images = []
        self.captions = []

    
    # def on_test_batch_end(self,
    #                         trainer: pl.Trainer,
    #                         pl_module: pl.LightningModule,
    #                         outputs,
    #                         batch: Any,
    #                         batch_idx: int,
    #                         dataloader_idx: int = 0,
    #                     ) -> None:
        
    
    # def on_test_epoch_end(self, trainer, pl_module):
    #     wandb.log({"Test Table": self.table})