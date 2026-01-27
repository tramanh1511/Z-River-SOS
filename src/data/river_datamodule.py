from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import rootutils
rootutils.setup_root(__file__, indicator="setup.py", pythonpath=True)
from src.data.components.river_dataset import *

class RiverDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        pool: RiverDataPool,
        transform: Compose | None = None,
        hard_transform: Compose | None = None,
        seed: int = 200,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ) if not transform else transform
        self.hard_transform = self.transform if not hard_transform else hard_transform
        self.pool = pool
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # load and split datasets only if not loaded already
        train_cfg, val_cfg = self.pool.augment_crop()
        if not self.data_val or not self.data_test or not self.data_train:
            self.data_train = RiverDataset(cfg=train_cfg, temporal=self.pool.temporal)
            self.data_val = RiverDataset(cfg=val_cfg, temporal=self.pool.temporal)
            self.data_test = RiverDataset(cfg=val_cfg, temporal=self.pool.temporal)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        # self.data_train.data, _ = self.pool.augment_crop(skip_val=True)
        dataset = TransformedRiverDataset(self.data_train, transform=self.transform, hard_transform=self.hard_transform)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        dataset = TransformedRiverDataset(self.data_val, transform=None, hard_transform=None)
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.
        :return: The test dataloader.
        """
        dataset = TransformedRiverDataset(self.data_val, transform=None)
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    import omegaconf
    from omegaconf import DictConfig
    import hydra
    
    @hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
    def main(cfg: DictConfig) -> Optional[float]:
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()
        loader = datamodule.train_dataloader()
        loader = iter(loader)
        batch = next(loader)
        print(batch.keys(), 
                batch['hard'].shape,
                batch['image'].shape,
                batch['mask'].shape,
                batch['label'].shape)
        mean
    
    main()
