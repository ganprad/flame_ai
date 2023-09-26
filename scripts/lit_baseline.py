import os
from typing import Optional, Any, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from names_generator import generate_name
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from flame_ai.data_modules import FlameAIDataModule
from flame_ai.modules import Model


class FlameAIModel(LightningModule):
    def __init__(
        self,
        in_channels,
        num_filters,
        num_of_residual_blocks,
        factor,
        scale,
        kernel_size,
        optimizer,
        learning_rate,
        l1_strength=0,
        l2_strength=0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = getattr(optim, optimizer)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau
        self.metric = torchmetrics.PeakSignalNoiseRatio()
        self.model = Model(
            in_channels=in_channels,
            num_filters=num_filters,
            num_of_residual_blocks=num_of_residual_blocks,
            scale=scale,
            factor=factor,
            kernel_size=kernel_size,
            do_upsample=True,
        )
        self.output_layer_lr = nn.Conv2d(in_channels, in_channels, 3, padding="same")
        self.output_layer_hr = nn.Conv2d(in_channels - 1, in_channels, 3, padding="same")
        self.loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), lr=self.hparams.learning_rate)
        sch = self.scheduler(opt, mode="max")
        return {"optimizer": opt, "scheduler": sch, "metric": self.metric}

    def forward(self, inputs):
        _, outputs = self.model(inputs)
        return outputs

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.hparams.l2_strength * l2_reg
        loss /= inputs.size(0)
        psnr = self.metric(outputs, targets)
        logs = {"loss": loss.detach(), "psnr": psnr}
        self.log_dict(dictionary={f"train_{k}": v for k, v in logs.items()}, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)

        loss /= inputs.size(0)

        psnr = self.metric(outputs, targets)

        logs = {"loss": loss.detach(), "psnr": psnr}
        self.log_dict(dictionary={f"val_{k}": v for k, v in logs.items()}, on_epoch=True, prog_bar=True)
        return {"loss": loss.detach()}

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        id, inputs = batch
        _, outputs, _, _ = self(inputs)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.cpu().detach().numpy().flatten(order="C").astype(np.float32)
        id = id.cpu().detach().numpy()[0]
        return {"id": id, "predictions": outputs}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        id, inputs = batch
        _, outputs, _, _ = self(inputs)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.cpu().detach().numpy().flatten(order="C").astype(np.float32)
        id = id.cpu().detach().numpy()[0]
        return {"id": id, "predictions": outputs}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss_epoch", train_loss, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss_epoch", val_loss, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        test_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("test_loss_epoch", test_loss, on_epoch=True, prog_bar=False)

    def on_predict_epoch_end(self, results: List[Any]) -> pd.DataFrame:
        print("Generating predictions")
        df = pd.DataFrame.from_records(results).T
        return df


if __name__ == "__main__":

    base_path = "../"
    INPUT_PATH = base_path + "dataset/"
    OUTPUT_PATH = base_path + "outputs/"

    # create directories for checkpoints and logs
    LOG_DIR = OUTPUT_PATH + "logs/"
    CHECKPOINT_DIR = OUTPUT_PATH + "ckpt/"
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    NUM_EPOCHS = 100
    IN_CHANNELS = 4
    FACTOR = 2
    NUM_FILTERS = 64
    NUM_OF_RESIDUAL_BLOCKS = 16
    BATCH_SIZE = 32
    SCALE = 3
    KERNEL_SIZE = 3
    OPTIMIZER = "Adam"
    LEARNING_RATE = 2e-4

    ckpt_name = f"{generate_name()}"
    loaders = FlameAIDataModule(input_path=INPUT_PATH, batch_size=BATCH_SIZE)
    logger = MLFlowLogger(
        experiment_name="flame_ai",
        run_name=ckpt_name,
        save_dir=LOG_DIR + ckpt_name,
        artifact_location="mlruns",
    )
    ckpt_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=ckpt_name,
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    trainer = Trainer(
        val_check_interval=1,
        logger=logger,
        max_epochs=NUM_EPOCHS,
        devices=1,
        accelerator="gpu",
        auto_lr_find=True,
        callbacks=[ckpt_callback, early_stopping],
    )
    model = FlameAIModel(
        in_channels=IN_CHANNELS,
        num_filters=NUM_FILTERS,
        num_of_residual_blocks=NUM_OF_RESIDUAL_BLOCKS,
        factor=FACTOR,
        scale=SCALE,
        kernel_size=KERNEL_SIZE,
        optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE,
    )
    model.name = ckpt_name
    trainer.fit(model, train_dataloaders=loaders.train_dataloader(), val_dataloaders=loaders.val_dataloader())
    predictions = trainer.predict(model, dataloaders=loaders.test_dataloader(), return_predictions=True)
    df = pd.DataFrame.from_records([v["predictions"] for i, v in enumerate(predictions)])
    df["id"] = [v["id"] for i, v in enumerate(predictions)]
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    # reset index
    df = df.reset_index(drop=True)
    df.to_csv(f"{OUTPUT_PATH}{ckpt_name}.csv", index=False)
    print(ckpt_name)
