import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

from types import SimpleNamespace

from utils.config import args
from utils.dataset import *
from utils.unitraj import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main(config, logger):

    # Create the model
    model = UniTraj(
        # trajectory_length=200,
        trajectory_length=config.data.traj_length,
        patch_size=1,
        # embedding_dim=128,
        embedding_dim=config.data.emb_dim,
        encoder_layers=8,
        encoder_heads=4,
        decoder_layers=4,
        decoder_heads=4,
        mask_ratio=0.5,
    )

    # 模型放置到gpu上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    print(next(model.parameters()).device)


    # 读取训练集和验证集
    # train/val path can be a .pkl file, a directory of shard .pkl files, or a glob pattern.
    train_file_path = config.data.train_file_path
    val_file_path = config.data.val_file_path
    normalize_transform = Normalize()
    train_set = TrajectoryDataset(
        data_path=train_file_path, max_len=200, transform=normalize_transform
    )
    val_set = TrajectoryDataset(
        data_path=val_file_path, max_len=200, transform=normalize_transform
    )

    num_workers = max(0, int(config.data.num_workers))
    pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataloader_val = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)  # Optimizer
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience = 20
    trigger_times = 0
    for epoch in range(0, config.training.n_epochs + 1):
        model.train()
        train_losses = []  # Store losses 
        logger.info("<----- Epoch {} Training ---->".format(epoch))
        for batch_idx, batch in enumerate(dataloader):
            traj, atten_mask = batch["trajectory"], batch["attention_mask"]
            interval, indices = batch["intervals"], batch["indices"]
            interval = interval.to(device)
            traj = traj.to(device)
            atten_mask = atten_mask.to(device)
            atten_mask = atten_mask.unsqueeze(1).expand_as(traj)


            predicted_traj, mask = model(traj, interval, indices)
            loss = torch.mean((predicted_traj - traj) ** 2 * mask * atten_mask) / 0.5
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            train_losses.append(loss.item())
            
        avg_train_loss = np.mean(train_losses)
        logger.info(f"Epoch {epoch} Training Loss: {avg_train_loss:.5f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_val):
                traj, atten_mask = batch["trajectory"], batch["attention_mask"]
                interval = batch["intervals"]
                indices = batch["indices"]
                interval = interval.to(device)
                traj = traj.to(device)
                atten_mask = atten_mask.to(device)
                atten_mask = atten_mask.unsqueeze(1).expand_as(traj)
                
                predicted_traj, mask = model(traj, interval, indices)
                val_loss = torch.mean((predicted_traj - traj) ** 2 * mask * atten_mask) / 0.5
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        logger.info(f"Epoch {epoch} Validation Loss: {avg_val_loss:.5f}")
        
        scheduler.step(avg_val_loss)
    
        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # save best model
            m_path = model_save / f"best_model_epoch_{epoch}.pt"
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), m_path)
            else:
                torch.save(model.state_dict(), m_path)
            logger.info(f"Validation loss decreased,\nsaving model to {m_path}")
            
        else:
            trigger_times += 1
            logger.info(f"Validation loss did not decrease for {trigger_times} epochs")
            if trigger_times >= patience:
                m_path = model_save / f"Final_Model_{epoch}.pt"
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), m_path)
                else:
                    torch.save(model.state_dict(), m_path)
                logger.info("Early stopping triggered")
                break

    logger.info("<----Training Done---->")


def setup_experiment_directories(config, Exp_name="UniTraj"):
    root_dir = Path(__file__).resolve().parent
    result_name = f"{config.data.dataset}_bs={config.training.batch_size}"
    exp_dir = root_dir / Exp_name / result_name
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    exp_time_dir = exp_dir / timestamp
    files_save = exp_time_dir / "Files"
    result_save = exp_time_dir / "Results"
    model_save = exp_time_dir / "models"

    # Creating directories
    for directory in [files_save, result_save, model_save]:
        directory.mkdir(parents=True, exist_ok=True)

    # Copying files
    for filename in os.listdir(root_dir / "utils"):
        if filename.endswith(".py"):
            shutil.copy(root_dir / "utils" / filename, files_save)
    # Copying the current file itself
    this_file = Path(__file__)
    shutil.copy(this_file, files_save)

    print("All files saved path ---->>", exp_time_dir)
    logger = Logger(
        __name__, log_path=exp_dir / (timestamp + "/out.log"), colorize=True
    )
    return logger, files_save, result_save, model_save

if __name__ == "__main__":
    # Load configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    logger, files_save, result_save, model_save = setup_experiment_directories(
        config, Exp_name="UniTraj"
    )

    log_info(config, logger)
    main(config, logger)
