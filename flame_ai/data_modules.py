import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import Normalize, Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip

from flame_ai.modules import get_mgrid


def minmax(x):
    return x - x.min() / (x.max() - x.min())


class FlowFieldDataset(Dataset):
    def __init__(self, input_path, mode):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.csv_file = pd.read_csv(input_path + f"{mode}.csv")
        if mode == "test":
            self.csv_file = pd.read_csv(input_path + f"{mode}.csv")
        self.LR_path = input_path + "flowfields/LR/" + mode
        self.HR_path = input_path + "flowfields/HR/" + mode

        self.mean = np.array([0.24, 28.0, 28.0, 28.0])
        self.std = np.array([0.068, 48.0, 48.0, 48.0])

    def transform(self, x, mode):
        if mode == "test":
            return Compose([ToTensor(), Normalize(self.mean, self.std, inplace=True)])(x)
        return Compose(
            [
                ToTensor(),
                Normalize(self.mean, self.std, inplace=True),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ]
        )(x)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # input
        if self.mode == "test":
            id = self.csv_file["id"][idx]
            rho_i = np.fromfile(self.LR_path + "/" + self.csv_file["rho_filename"][idx], dtype="<f4").reshape(16, 16)
            ux_i = np.fromfile(self.LR_path + "/" + self.csv_file["ux_filename"][idx], dtype="<f4").reshape(16, 16)
            uy_i = np.fromfile(self.LR_path + "/" + self.csv_file["uy_filename"][idx], dtype="<f4").reshape(16, 16)
            uz_i = np.fromfile(self.LR_path + "/" + self.csv_file["uz_filename"][idx], dtype="<f4").reshape(16, 16)
            X = np.stack([rho_i, ux_i, uy_i, uz_i], axis=2)
            return id, self.transform(X, mode=self.mode)

        rho_i = np.fromfile(self.LR_path + "/" + self.csv_file["rho_filename"][idx], dtype="<f4").reshape(16, 16)
        ux_i = np.fromfile(self.LR_path + "/" + self.csv_file["ux_filename"][idx], dtype="<f4").reshape(16, 16)
        uy_i = np.fromfile(self.LR_path + "/" + self.csv_file["uy_filename"][idx], dtype="<f4").reshape(16, 16)
        uz_i = np.fromfile(self.LR_path + "/" + self.csv_file["uz_filename"][idx], dtype="<f4").reshape(16, 16)
        # output
        rho_o = np.fromfile(self.HR_path + "/" + self.csv_file["rho_filename"][idx], dtype="<f4").reshape(128, 128)
        ux_o = np.fromfile(self.HR_path + "/" + self.csv_file["ux_filename"][idx], dtype="<f4").reshape(128, 128)
        uy_o = np.fromfile(self.HR_path + "/" + self.csv_file["uy_filename"][idx], dtype="<f4").reshape(128, 128)
        uz_o = np.fromfile(self.HR_path + "/" + self.csv_file["uz_filename"][idx], dtype="<f4").reshape(128, 128)
        X = np.stack([rho_i, ux_i, uy_i, uz_i], axis=2)
        Y = np.stack([rho_o, ux_o, uy_o, uz_o], axis=2)
        return self.transform(X, mode=self.mode), self.transform(Y, mode=self.mode)


class FlameAIDataModule(LightningDataModule):
    def __init__(
        self,
        input_path,
        batch_size,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        drop_last=False,
        **kwargs,
    ):

        super().__init__()
        self._init_datasets(input_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def _init_datasets(self, input_path):
        self.train_dataset = ConcatDataset(
            [FlowFieldDataset(input_path=input_path, mode="train"), FlowFieldDataset(input_path=input_path, mode="val")]
        )
        self.val_dataset = FlowFieldDataset(input_path=input_path, mode="val")
        self.test_dataset = FlowFieldDataset(input_path=input_path, mode="test")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


class FlowFieldMetaDataset(Dataset):
    def __init__(self, input_path, mode, flatten):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.csv_file = pd.read_csv(input_path + f"{mode}.csv")
        self.LR_path = input_path + "flowfields/LR/" + mode
        self.HR_path = input_path + "flowfields/HR/" + mode

        self.mean = np.array([0.24, 28.0, 28.0, 28.0])
        self.std = np.array([0.068, 48.0, 48.0, 48.0])
        self.coords = get_mgrid(16, 2)
        self.flatten = flatten

    def transform(self, x, mode):
        if mode == "test":
            return Compose([ToTensor(), Normalize(self.mean, self.std, inplace=True)])(x)
        return Compose(
            [
                ToTensor(),
                Normalize(self.mean, self.std, inplace=True),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ]
        )(x)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # input
        if self.mode == "test":
            id = self.csv_file["id"][idx]
            rho_i = np.fromfile(self.LR_path + "/" + self.csv_file["rho_filename"][idx], dtype="<f4").reshape(16, 16)
            ux_i = np.fromfile(self.LR_path + "/" + self.csv_file["ux_filename"][idx], dtype="<f4").reshape(16, 16)
            uy_i = np.fromfile(self.LR_path + "/" + self.csv_file["uy_filename"][idx], dtype="<f4").reshape(16, 16)
            uz_i = np.fromfile(self.LR_path + "/" + self.csv_file["uz_filename"][idx], dtype="<f4").reshape(16, 16)
            X = np.stack([rho_i, ux_i, uy_i, uz_i], axis=2)
            if self.flatten:
                return id, self.transform(X, mode=self.mode).permute(1, 2, 0).view(-1, 4), self.coords
            else:
                return id, self.transform(X, mode=self.mode)

        rho_i = np.fromfile(self.LR_path + "/" + self.csv_file["rho_filename"][idx], dtype="<f4").reshape(16, 16)
        ux_i = np.fromfile(self.LR_path + "/" + self.csv_file["ux_filename"][idx], dtype="<f4").reshape(16, 16)
        uy_i = np.fromfile(self.LR_path + "/" + self.csv_file["uy_filename"][idx], dtype="<f4").reshape(16, 16)
        uz_i = np.fromfile(self.LR_path + "/" + self.csv_file["uz_filename"][idx], dtype="<f4").reshape(16, 16)
        # output
        rho_o = np.fromfile(self.HR_path + "/" + self.csv_file["rho_filename"][idx], dtype="<f4").reshape(128, 128)
        ux_o = np.fromfile(self.HR_path + "/" + self.csv_file["ux_filename"][idx], dtype="<f4").reshape(128, 128)
        uy_o = np.fromfile(self.HR_path + "/" + self.csv_file["uy_filename"][idx], dtype="<f4").reshape(128, 128)
        uz_o = np.fromfile(self.HR_path + "/" + self.csv_file["uz_filename"][idx], dtype="<f4").reshape(128, 128)
        X = np.stack([rho_i, ux_i, uy_i, uz_i], axis=2)
        Y = np.stack([rho_o, ux_o, uy_o, uz_o], axis=2)
        if self.flatten:
            return (
                self.transform(X, mode=self.mode).permute(1, 2, 0).view(-1, 4),
                self.transform(Y, mode=self.mode),
                get_mgrid(16, 2),
            )
        else:
            return (
                self.transform(X, mode=self.mode),
                self.transform(Y, mode=self.mode),
            )


class MetaModelDataModule(LightningDataModule):
    def __init__(
        self,
        input_path,
        batch_size,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        drop_last=False,
        flatten=True,
        **kwargs,
    ):

        super().__init__()
        self._init_datasets(input_path, flatten)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.flatten = flatten

    def _init_datasets(self, input_path, flatten):
        self.train_dataset = FlowFieldMetaDataset(input_path=input_path, mode="train", flatten=flatten)
        self.val_dataset = FlowFieldMetaDataset(input_path=input_path, mode="val", flatten=flatten)
        self.test_dataset = FlowFieldMetaDataset(input_path=input_path, mode="test", flatten=flatten)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
        )
