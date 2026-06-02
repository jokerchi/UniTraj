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


def masked_mae_rmse(predicted_traj, traj, mask, atten_mask):
    with torch.no_grad():
        metric_mask = mask * atten_mask
        denom = metric_mask.sum().clamp_min(1.0)
        error = predicted_traj - traj
        mae = (error.abs() * metric_mask).sum() / denom
        rmse = torch.sqrt((error ** 2 * metric_mask).sum() / denom)
        return mae, rmse


def restore_lonlat(trajectory, original, normalize_transform):
    mean = normalize_transform.mean.to(trajectory.device).view(1, 2, 1)
    std = normalize_transform.std.to(trajectory.device).view(1, 2, 1)
    original = original.to(trajectory.device).unsqueeze(-1)
    return trajectory * std + mean + original


def haversine_distance_meters(predicted_lonlat, target_lonlat):
    radius = 6371000.0
    pred_lon = torch.deg2rad(predicted_lonlat[:, 0])
    pred_lat = torch.deg2rad(predicted_lonlat[:, 1])
    target_lon = torch.deg2rad(target_lonlat[:, 0])
    target_lat = torch.deg2rad(target_lonlat[:, 1])

    dlon = pred_lon - target_lon
    dlat = pred_lat - target_lat
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(target_lat) * torch.cos(pred_lat) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt((1 - a).clamp_min(0.0)))
    return radius * c


def masked_meter_mae_rmse(predicted_traj, traj, original, normalize_transform, mask, atten_mask):
    with torch.no_grad():
        predicted_lonlat = restore_lonlat(predicted_traj, original, normalize_transform)
        target_lonlat = restore_lonlat(traj, original, normalize_transform)
        distance = haversine_distance_meters(predicted_lonlat, target_lonlat)
        point_mask = (mask * atten_mask)[:, 0]
        denom = point_mask.sum().clamp_min(1.0)
        mae = (distance * point_mask).sum() / denom
        rmse = torch.sqrt((distance ** 2 * point_mask).sum() / denom)
        return mae, rmse


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
    sampler_seed = getattr(config.data, "sampler_seed", 2024)
    normalize_transform = Normalize()
    train_set = TrajectoryDataset(
        data_path=train_file_path,
        max_len=config.data.traj_length,
        transform=normalize_transform,
        mode="train",
        seed=sampler_seed,
    )
    val_set = TrajectoryDataset(
        data_path=val_file_path,
        max_len=config.data.traj_length,
        transform=normalize_transform,
        mode="val",
        seed=sampler_seed,
    )
    logger.info(f"Train shards: {len(train_set.data_files)}, trajectories: {len(train_set)}")
    logger.info(f"Validation shards: {len(val_set.data_files)}, trajectories: {len(val_set)}")

    num_workers = max(0, int(config.data.num_workers))
    pin_memory = torch.cuda.is_available()
    train_sampler = ShardBatchSampler(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=False,
        seed=sampler_seed,
    )
    val_sampler = ShardBatchSampler(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False,
        seed=sampler_seed,
    )
    dataloader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataloader_val = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)  # Optimizer
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience = config.training.patience
    trigger_times = 0
    for epoch in range(0, config.training.n_epochs + 1):
        model.train()
        train_losses = []  # Store losses 
        train_maes = []
        train_rmses = []
        logger.info("<----- Epoch {} Training ---->".format(epoch))
        for batch_idx, batch in enumerate(dataloader):
            traj, atten_mask = batch["trajectory"], batch["attention_mask"]
            interval, indices = batch["intervals"], batch["indices"]
            if epoch == 0 and batch_idx == 0:
                logger.info(
                    "Train batch shapes: "
                    f"trajectory={tuple(traj.shape)}, "
                    f"attention_mask={tuple(atten_mask.shape)}, "
                    f"intervals={tuple(interval.shape)}, "
                    f"indices={tuple(indices.shape)}"
                )
            interval = interval.to(device)
            traj = traj.to(device)
            atten_mask = atten_mask.to(device)
            atten_mask = atten_mask.unsqueeze(1).expand_as(traj)


            predicted_traj, mask = model(traj, interval, indices)
            loss = torch.mean((predicted_traj - traj) ** 2 * mask * atten_mask) / 0.5
            mae, rmse = masked_mae_rmse(predicted_traj, traj, mask, atten_mask)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            train_losses.append(loss.item())
            train_maes.append(mae.item())
            train_rmses.append(rmse.item())
            
        avg_train_loss = np.mean(train_losses)
        avg_train_mae = np.mean(train_maes)
        avg_train_rmse = np.mean(train_rmses)
        logger.info(
            f"Epoch {epoch} Training Loss: {avg_train_loss:.5f}, "
            f"MAE: {avg_train_mae:.5f}, RMSE: {avg_train_rmse:.5f}"
        )

        model.eval()
        val_losses = []
        val_maes = []
        val_rmses = []
        val_meter_maes = []
        val_meter_rmses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_val):
                traj, atten_mask = batch["trajectory"], batch["attention_mask"]
                interval = batch["intervals"]
                indices = batch["indices"]
                original = batch["original"]
                if epoch == 0 and batch_idx == 0:
                    logger.info(
                        "Validation batch shapes: "
                        f"trajectory={tuple(traj.shape)}, "
                        f"attention_mask={tuple(atten_mask.shape)}, "
                        f"intervals={tuple(interval.shape)}, "
                        f"indices={tuple(indices.shape)}"
                    )
                interval = interval.to(device)
                traj = traj.to(device)
                atten_mask = atten_mask.to(device)
                original = original.to(device)
                atten_mask = atten_mask.unsqueeze(1).expand_as(traj)
                
                predicted_traj, mask = model(traj, interval, indices)
                val_loss = torch.mean((predicted_traj - traj) ** 2 * mask * atten_mask) / 0.5
                val_mae, val_rmse = masked_mae_rmse(predicted_traj, traj, mask, atten_mask)
                meter_mae, meter_rmse = masked_meter_mae_rmse(
                    predicted_traj, traj, original, normalize_transform, mask, atten_mask
                )
                val_losses.append(val_loss.item())
                val_maes.append(val_mae.item())
                val_rmses.append(val_rmse.item())
                val_meter_maes.append(meter_mae.item())
                val_meter_rmses.append(meter_rmse.item())
        
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        avg_val_rmse = np.mean(val_rmses)
        avg_val_meter_mae = np.mean(val_meter_maes)
        avg_val_meter_rmse = np.mean(val_meter_rmses)
        logger.info(
            f"Epoch {epoch} Validation Loss: {avg_val_loss:.5f}, "
            f"MAE: {avg_val_mae:.5f}, RMSE: {avg_val_rmse:.5f}, "
            f"Meter MAE: {avg_val_meter_mae:.2f}, Meter RMSE: {avg_val_meter_rmse:.2f}"
        )
        
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
