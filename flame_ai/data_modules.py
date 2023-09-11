import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose, ToTensor


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
        self.normalize = Normalize(self.mean, self.std, inplace=True)

    def transform(self, x):
        return Compose([ToTensor(), self.normalize])(x)

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
            return id, self.transform(X)

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
        return self.transform(X), self.transform(Y)
