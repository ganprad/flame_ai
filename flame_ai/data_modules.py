import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FlowFieldDataset(Dataset):
    def __init__(self, input_path, mode):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.csv_file = pd.read_csv(input_path + f"{mode}.csv").iloc[:100]
        if mode == "test":
            self.csv_file = pd.read_csv(input_path + f"{mode}.csv").reset_index().to_dict(orient="list")
        self.LR_path = input_path + "flowfields/LR/" + mode
        self.HR_path = input_path + "flowfields/HR/" + mode

        self.mean = np.array([0.24, 28.0, 28.0, 28.0])
        self.std = np.array([0.068, 48.0, 48.0, 48.0])

    def normalize(self, x):
        return (x - self.mean) / self.std

    def __len__(self):
        return 100  # len(self.csv_file)

    def __getitem__(self, idx):
        # input
        if self.mode == "test":
            id = self.csv_file["id"][idx]
            rho_i = np.fromfile(self.LR_path + "/" + self.csv_file["rho_filename"][idx], dtype="<f4").reshape(16, 16)
            ux_i = np.fromfile(self.LR_path + "/" + self.csv_file["ux_filename"][idx], dtype="<f4").reshape(16, 16)
            uy_i = np.fromfile(self.LR_path + "/" + self.csv_file["uy_filename"][idx], dtype="<f4").reshape(16, 16)
            uz_i = np.fromfile(self.LR_path + "/" + self.csv_file["uz_filename"][idx], dtype="<f4").reshape(16, 16)
            X = np.stack([rho_i, ux_i, uy_i, uz_i], axis=2)
            X = self.normalize(X)
            # X = (X - X.min()) / (X.max() - X.min())
            return id, torch.tensor(X).movedim((0, 2), (2, 0)).type(torch.float32)

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
        # [(X[:, :, i].min(), X[:, :, i].max()) for i in range(X.shape[-1])]
        # X = (X - X.min()) / (X.max() - X.min())
        # Y = (Y - Y.min()) / (Y.max() - Y.min())
        # X = (X - X.min(axis=(0, 1))) / (X.max(axis=(0, 1)) - X.min(axis=(0, 1)))
        # Y = (Y - Y.min(axis=(0, 1))) / (Y.max(axis=(0, 1)) - Y.min(axis=(0, 1)))
        X = self.normalize(X)
        Y = self.normalize(Y)
        X = torch.tensor(X).movedim((0, 2), (2, 0)).type(torch.float32)
        Y = torch.tensor(Y).movedim((0, 2), (2, 0)).type(torch.float32)
        return X, Y
