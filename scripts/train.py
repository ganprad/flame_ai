import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch.nn.functional as F
from accelerate import Accelerator
from names_generator import generate_name
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from flame_ai.data_modules import FlowFieldDataset
from flame_ai.modules import Model


def train():
    base_path = "../"
    input_path = base_path + "dataset/"
    output_path = base_path + "outputs/"

    # create directories for checkpoints and logs
    log_dir = output_path + "logs/"
    checkpoint_dir = output_path + "ckpt/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    NUM_EPOCHS = 100
    IN_CHANNELS = 4
    FACTOR = 2
    NUM_FILTERS = 64
    NUM_OF_RESIDUAL_BLOCKS = 16
    BATCH_SIZE = 32
    SCALE = 3
    KERNEL_SIZE = 3

    train_dataset = FlowFieldDataset(input_path=input_path, mode="train")
    val_dataset = FlowFieldDataset(input_path=input_path, mode="val")
    test_dataset = FlowFieldDataset(input_path=input_path, mode="test")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, pin_memory=False)

    ckpt_name = f"{generate_name()}"
    learning_rate = 1e-3

    model = Model(
        in_channels=IN_CHANNELS,
        num_filters=NUM_FILTERS,
        num_of_residual_blocks=NUM_OF_RESIDUAL_BLOCKS,
        factor=FACTOR,
        scale=SCALE,
        kernel_size=KERNEL_SIZE,
    )
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=optimizer)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, test_dataloader, scheduler
    )

    # Register the LR scheduler
    accelerator.register_for_checkpointing(scheduler)
    # Save the starting state

    accelerator.save_state(output_dir=checkpoint_dir + ckpt_name)
    progress_bar = tqdm(range(NUM_EPOCHS))
    for epoch in range(NUM_EPOCHS):
        model.train()
        for step, batch in enumerate(train_dataloader):
            inputs, targets = batch
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            progress_bar.set_description(f"epoch : {epoch} | batch {step} | loss : {loss.detach().cpu()}")

        scheduler.step(loss.detach().cpu())

        model.eval()
        for step, batch in enumerate(val_dataloader):
            inputs, targets = batch
            outputs = model(inputs)
            val_loss = F.mse_loss(outputs, targets)
            progress_bar.set_description(f"epoch : {epoch} | batch {step} | val_loss : {val_loss.detach().cpu()}")
        progress_bar.update(1)

    progress_bar = tqdm(range(len(test_dataloader)))
    predictions = {}
    ids = []
    for idx, batch in enumerate(test_dataloader):
        id, inputs = batch
        outputs = model(inputs)
        outputs = outputs.permute(0, 2, 3, 1)
        predictions[idx] = outputs.cpu().detach().numpy().flatten(order="C").astype(np.float32)
        ids.append(id.cpu().detach().numpy()[0])
        progress_bar.set_description(f"test prediction: {idx}")
        progress_bar.update(1)
    progress_bar.close()

    print("Generating predictions")
    df = pd.DataFrame.from_dict(predictions).T
    df["id"] = ids
    # move id to first column
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    # reset index
    df = df.reset_index(drop=True)
    df.to_csv(f"{output_path}{ckpt_name}.csv", index=False)
    accelerator.load_state(checkpoint_dir + ckpt_name)


if __name__ == "__main__":
    train()
