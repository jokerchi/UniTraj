#!/usr/bin/env python3
"""
UniTraj Training Script.

Supports:
- Single-GPU and multi-GPU (DDP via torchrun) training.
- Mixed-precision (AMP) with gradient accumulation.
- Two data backends: streaming Parquet (IterableDataset) and in-memory Pickle.
- Early stopping, learning rate scheduling, and model checkpointing.
- torch.compile support for accelerated training.

Usage:
    # Single-GPU
    python main.py

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 main.py
"""

import datetime
import logging
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.config import args
from utils.dataset import (
    Normalize,
    ShardBatchSampler,
    TrajectoryDataset,
    TrajectoryIterableDataset,
)
from utils.logger import Logger, log_info
from utils.unitraj import UniTraj


# ============================================================================
# Early Environment Setup (must precede torch import)
# ============================================================================

# Fix linker warnings on multiarch systems: ensure 64-bit libcuda.so is
# discoverable before the 32-bit version at /lib/i386-linux-gnu/libcuda.so.
_cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
if os.path.isdir(_cuda_home):
    _cuda_lib64 = os.path.join(_cuda_home, "lib64")
    _cuda_stubs = os.path.join(_cuda_lib64, "stubs")
    os.environ.setdefault("CUDA_HOME", _cuda_home)
    # LIBRARY_PATH: passed by gcc to ld as -L flags (compile-time search)
    _existing_lib = os.environ.get("LIBRARY_PATH", "")
    _add_paths = []
    if os.path.isdir(_cuda_stubs):
        _add_paths.append(_cuda_stubs)
    if os.path.isdir(_cuda_lib64):
        _add_paths.append(_cuda_lib64)
    if _add_paths:
        _prefix = ":".join(_add_paths)
        os.environ["LIBRARY_PATH"] = (
            f"{_prefix}:{_existing_lib}" if _existing_lib else _prefix
        )

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")


# ============================================================================
# DDP Utility Functions
# ============================================================================

def ddp_setup() -> Tuple[int, int]:
    """Initialize the DDP process group.

    Returns:
        (local_rank, world_size)
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def is_main_process() -> bool:
    """Check whether the current process is rank 0 (or running without DDP)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all DDP ranks (in-place)."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across all DDP ranks (in-place)."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def cleanup_ddp() -> None:
    """Destroy the DDP process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class DDPAwareLogger:
    """Logger wrapper that silences all ranks except rank 0.

    Non-rank-0 processes have their internal logger level raised above CRITICAL
    so no messages are emitted. Attribute access is transparently forwarded.
    """

    def __init__(self, logger: Logger):
        self._logger = logger
        if not is_main_process():
            logger.inner_logger.setLevel(logging.CRITICAL + 1)

    def __getattr__(self, name: str):
        if name == "_logger":
            raise AttributeError(name)
        return getattr(self._logger, name)


# ============================================================================
# Metric Functions
# ============================================================================

def masked_mae_rmse(
    predicted_traj: torch.Tensor,
    traj: torch.Tensor,
    mask: torch.Tensor,
    atten_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute MAE and RMSE over masked regions only.

    Args:
        predicted_traj: [B, 2, L] predicted trajectory.
        traj: [B, 2, L] ground-truth trajectory.
        mask: [B, 2, L] binary mask from the decoder.
        atten_mask: [B, 2, L] attention mask (1 = real, 0 = padding).

    Returns:
        (mae, rmse) scalar tensors.
    """
    with torch.no_grad():
        metric_mask = mask * atten_mask
        denom = metric_mask.sum().clamp_min(1.0)
        error = predicted_traj - traj
        mae = (error.abs() * metric_mask).sum() / denom
        rmse = torch.sqrt((error ** 2 * metric_mask).sum() / denom)
        return mae, rmse


def restore_lonlat(
    trajectory: torch.Tensor,
    original: torch.Tensor,
    normalize_transform: Normalize,
) -> torch.Tensor:
    """Invert normalization and re-add the origin offset.

    Args:
        trajectory: [B, 2, L] normalized trajectory.
        original: [B, 2] origin coordinates (first point).
        normalize_transform: Normalize instance with .mean and .std.

    Returns:
        [B, 2, L] trajectory in original lon/lat space.
    """
    mean = normalize_transform.mean.to(trajectory.device).view(1, 2, 1)
    std = normalize_transform.std.to(trajectory.device).view(1, 2, 1)
    original = original.to(trajectory.device).unsqueeze(-1)
    return trajectory * std + mean + original


def haversine_distance_meters(
    predicted_lonlat: torch.Tensor, target_lonlat: torch.Tensor
) -> torch.Tensor:
    """Haversine distance between two lon/lat points (in meters).

    Args:
        predicted_lonlat: [N, 2] (lon, lat).
        target_lonlat: [N, 2] (lon, lat).

    Returns:
        [N] distances in meters.
    """
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


def masked_meter_mae_rmse(
    predicted_traj: torch.Tensor,
    traj: torch.Tensor,
    original: torch.Tensor,
    normalize_transform: Normalize,
    mask: torch.Tensor,
    atten_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute MAE and RMSE in meters (Haversine) over masked regions.

    Restores normalized coordinates back to lon/lat, then computes the
    Haversine distance between predicted and ground-truth points.

    Returns:
        (meter_mae, meter_rmse) scalar tensors in meters.
    """
    with torch.no_grad():
        predicted_lonlat = restore_lonlat(predicted_traj, original, normalize_transform)
        target_lonlat = restore_lonlat(traj, original, normalize_transform)
        distance = haversine_distance_meters(predicted_lonlat, target_lonlat)
        point_mask = (mask * atten_mask)[:, 0]
        denom = point_mask.sum().clamp_min(1.0)
        mae = (distance * point_mask).sum() / denom
        rmse = torch.sqrt((distance ** 2 * point_mask).sum() / denom)
        return mae, rmse


# ============================================================================
# Helpers
# ============================================================================

def positive_int_or_none(value) -> Optional[int]:
    """Parse a positive integer or return None."""
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def model_state_dict(model: nn.Module) -> dict:
    """Get state_dict, unwrapping DataParallel / DDP if needed."""
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module.state_dict()
    return model.state_dict()


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize all CUDA streams if the device is CUDA."""
    if device.type == "cuda":
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_idx)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    minutes, seconds = divmod(float(seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:05.2f}s"
    return f"{minutes}m {seconds:05.2f}s"


# ============================================================================
# Model Factory
# ============================================================================

def create_model(config) -> UniTraj:
    """Build the UniTraj model from configuration.

    Args:
        config: Configuration namespace with a ``data`` attribute.

    Returns:
        UniTraj instance.
    """
    return UniTraj(
        trajectory_length=config.data.traj_length,
        patch_size=1,
        embedding_dim=config.data.emb_dim,
        encoder_layers=8,
        encoder_heads=4,
        decoder_layers=4,
        decoder_heads=4,
        mask_ratio=0.5,
    )


# ============================================================================
# Data Loading
# ============================================================================

def build_dataloaders(
    config,
    normalize_transform: Normalize,
    use_ddp: bool,
    world_size: int,
    pin_memory: bool,
    logger,
):
    """Construct training and validation DataLoaders.

    Supports two backends:
    - ``parquet``: streaming IterableDataset with shard-based DDP splitting.
    - ``pickle``: map-style Dataset with ShardBatchSampler.

    Returns:
        (dataloader, dataloader_val, steps_per_epoch, val_steps_per_epoch)
    """
    train_path = config.data.train_file_path
    val_path = config.data.val_file_path
    sampler_seed = getattr(config.data, "sampler_seed", 2024)
    data_format = getattr(config.data, "format", "pickle").lower()
    num_workers = max(0, int(config.data.num_workers))
    steps_per_epoch = None
    val_steps_per_epoch = None

    if data_format == "parquet":
        # --- Parquet streaming ---
        shuffle_buf = getattr(config.data, "shuffle_buffer_size", 4096)
        record_bs = getattr(config.data, "record_batch_size", 1024)

        train_set = TrajectoryIterableDataset(
            data_path=train_path,
            max_len=config.data.traj_length,
            transform=normalize_transform,
            mode="train",
            seed=sampler_seed,
            shuffle_buffer_size=shuffle_buf,
            record_batch_size=record_bs,
        )
        val_set = TrajectoryIterableDataset(
            data_path=val_path,
            max_len=config.data.traj_length,
            transform=normalize_transform,
            mode="val",
            seed=sampler_seed,
            shuffle_buffer_size=0,
            record_batch_size=record_bs,
        )

        steps_per_epoch = positive_int_or_none(
            getattr(config.data, "steps_per_epoch", None)
        )
        val_steps_per_epoch = positive_int_or_none(
            getattr(config.data, "val_steps_per_epoch", None)
        )
        if use_ddp and steps_per_epoch is not None:
            steps_per_epoch = max(1, steps_per_epoch // world_size)
        if use_ddp and val_steps_per_epoch is not None:
            val_steps_per_epoch = max(1, val_steps_per_epoch // world_size)

        logger.info(
            f"Train parquet shards: {len(train_set.data_files)}, "
            f"trajectories: {getattr(train_set, 'total_trajectories', 'unknown')}"
        )
        logger.info(
            f"Validation parquet shards: {len(val_set.data_files)}, "
            f"trajectories: {getattr(val_set, 'total_trajectories', 'unknown')}"
        )

        loader_kwargs = {
            "batch_size": config.training.batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(
                getattr(config.data, "prefetch_factor", 4)
            )

        dataloader = DataLoader(train_set, **loader_kwargs)
        dataloader_val = DataLoader(val_set, **loader_kwargs)

    elif data_format == "pickle":
        # --- Pickle (map-style) ---
        train_set = TrajectoryDataset(
            data_path=train_path,
            max_len=config.data.traj_length,
            transform=normalize_transform,
            mode="train",
            seed=sampler_seed,
        )
        val_set = TrajectoryDataset(
            data_path=val_path,
            max_len=config.data.traj_length,
            transform=normalize_transform,
            mode="val",
            seed=sampler_seed,
        )

        logger.info(
            f"Train shards: {len(train_set.data_files)}, "
            f"trajectories: {len(train_set)}"
        )
        logger.info(
            f"Validation shards: {len(val_set.data_files)}, "
            f"trajectories: {len(val_set)}"
        )

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
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    return dataloader, dataloader_val, steps_per_epoch, val_steps_per_epoch


# ============================================================================
# Training Loop Helpers
# ============================================================================

def _move_batch_to_device(
    batch: dict, device: torch.device, pin_memory: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transfer batch tensors to the target device and prepare attention mask.

    Returns:
        (traj, atten_mask, interval, indices, original):
        original is None for training batches.
    """
    interval = batch["intervals"].to(device, non_blocking=pin_memory)
    traj = batch["trajectory"].to(device, non_blocking=pin_memory)
    atten_mask = batch["attention_mask"].to(device, non_blocking=pin_memory)
    indices = batch["indices"]
    original = batch.get("original")
    if original is not None:
        original = original.to(device, non_blocking=pin_memory)
    atten_mask = atten_mask.unsqueeze(1).expand_as(traj)
    return traj, atten_mask, interval, indices, original


def _reduce_metrics(
    loss_sum: torch.Tensor,
    mae_sum: torch.Tensor,
    rmse_sum: torch.Tensor,
    count: torch.Tensor,
    use_ddp: bool,
    meter_mae_sum: Optional[torch.Tensor] = None,
    meter_rmse_sum: Optional[torch.Tensor] = None,
):
    """Aggregate metrics across DDP ranks (in-place sum reduction)."""
    if use_ddp:
        reduce_tensor(loss_sum)
        reduce_tensor(mae_sum)
        reduce_tensor(rmse_sum)
        reduce_tensor(count)
        if meter_mae_sum is not None:
            reduce_tensor(meter_mae_sum)
        if meter_rmse_sum is not None:
            reduce_tensor(meter_rmse_sum)


def _aggregate_scalar(loss_sum: torch.Tensor, count: torch.Tensor) -> float:
    """Compute average from summed loss and count tensors."""
    return (loss_sum / count).item()


# ============================================================================
# Main Training Function
# ============================================================================

def main(config, logger, local_rank=None, world_size=1):
    """Run the full training pipeline.

    Args:
        config: Configuration namespace.
        logger: DDPAwareLogger instance.
        local_rank: Local GPU rank for DDP (None if single-GPU).
        world_size: Total number of DDP processes.
    """
    # ---- Build model ----
    model = create_model(config)
    use_ddp = dist.is_available() and dist.is_initialized()

    # ---- Configure device and parallelism ----
    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)

        # Apply torch.compile before DDP wrapper (per-submodule compilation)
        if hasattr(torch, "compile"):
            logger.info("Applying torch.compile to model (pre-DDP)...")
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("torch.compile applied successfully.")
            except Exception as e:
                logger.warning(f"torch.compile failed, skipping: {e}")

        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        logger.info(f"DDP initialized: rank={local_rank}, world_size={world_size}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
            model = nn.DataParallel(model, device_ids=[0, 1])

        if hasattr(torch, "compile"):
            logger.info("Applying torch.compile to model...")
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("torch.compile applied successfully.")
            except Exception as e:
                logger.warning(f"torch.compile failed, skipping: {e}")

    # ---- Performance optimizations ----
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if is_main_process():
        print(next(model.parameters()).device)

    # ---- Build data loaders ----
    normalize_transform = Normalize()
    pin_memory = torch.cuda.is_available()

    dataloader, dataloader_val, steps_per_epoch, val_steps_per_epoch = build_dataloaders(
        config, normalize_transform, use_ddp, world_size, pin_memory, logger
    )

    # ---- Optimizer, scheduler, AMP ----
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=2)
    use_amp = bool(getattr(config.training, "use_amp", False)) and device.type == "cuda"
    grad_accum_steps = max(1, int(getattr(config.training, "grad_accum_steps", 1)))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger.info(f"AMP enabled: {use_amp}, grad_accum_steps: {grad_accum_steps}")

    # ---- Training loop ----
    best_val_loss = float("inf")
    patience = config.training.patience
    trigger_times = 0

    for epoch in range(config.training.n_epochs + 1):
        epoch_start = time.perf_counter()

        # ------ Training ------
        train_start = time.perf_counter()
        model.train()
        train_losses = []
        train_maes = []
        train_rmses = []
        logger.info(f"<----- Epoch {epoch} Training ---->")
        optim.zero_grad(set_to_none=True)
        train_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break

            traj, atten_mask, interval, indices, _ = _move_batch_to_device(
                batch, device, pin_memory
            )

            if epoch == 0 and batch_idx == 0:
                logger.info(
                    f"Train batch shapes: "
                    f"trajectory={tuple(traj.shape)}, "
                    f"attention_mask={tuple(atten_mask.shape)}, "
                    f"intervals={tuple(interval.shape)}, "
                    f"indices={tuple(indices.shape)}"
                )

            with torch.amp.autocast("cuda", enabled=use_amp):
                predicted_traj, mask = model(traj, interval, indices)
                loss = (
                    torch.mean((predicted_traj - traj) ** 2 * mask * atten_mask) / 0.5
                )

            mae, rmse = masked_mae_rmse(predicted_traj, traj, mask, atten_mask)

            train_batches += 1
            scaler.scale(loss / grad_accum_steps).backward()

            if train_batches % grad_accum_steps == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            train_losses.append(loss.item())
            train_maes.append(mae.item())
            train_rmses.append(rmse.item())

        # Handle remaining gradients
        if train_batches == 0:
            raise RuntimeError(
                "No training batches were produced. "
                "Check dataset paths and steps_per_epoch."
            )
        if train_batches % grad_accum_steps != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        train_seconds = time.perf_counter() - train_start

        # Aggregate training metrics across DDP ranks
        train_loss_sum = torch.tensor([sum(train_losses)], device=device)
        train_mae_sum = torch.tensor([sum(train_maes)], device=device)
        train_rmse_sum = torch.tensor([sum(train_rmses)], device=device)
        train_count = torch.tensor([len(train_losses)], device=device)
        _reduce_metrics(train_loss_sum, train_mae_sum, train_rmse_sum, train_count, use_ddp)

        avg_train_loss = _aggregate_scalar(train_loss_sum, train_count)
        avg_train_mae = _aggregate_scalar(train_mae_sum, train_count)
        avg_train_rmse = _aggregate_scalar(train_rmse_sum, train_count)
        logger.info(
            f"Epoch {epoch} Training Loss: {avg_train_loss:.5f}, "
            f"MAE: {avg_train_mae:.5f}, RMSE: {avg_train_rmse:.5f}, "
            f"Time: {format_duration(train_seconds)}"
        )

        # ------ Validation ------
        val_start = time.perf_counter()
        model.eval()
        val_losses = []
        val_maes = []
        val_rmses = []
        val_meter_maes = []
        val_meter_rmses = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_val):
                if val_steps_per_epoch is not None and batch_idx >= val_steps_per_epoch:
                    break

                traj, atten_mask, interval, indices, original = _move_batch_to_device(
                    batch, device, pin_memory
                )

                if epoch == 0 and batch_idx == 0:
                    logger.info(
                        f"Validation batch shapes: "
                        f"trajectory={tuple(traj.shape)}, "
                        f"attention_mask={tuple(atten_mask.shape)}, "
                        f"intervals={tuple(interval.shape)}, "
                        f"indices={tuple(indices.shape)}"
                    )

                with torch.amp.autocast("cuda", enabled=use_amp):
                    predicted_traj, mask = model(traj, interval, indices)
                    val_loss = (
                        torch.mean((predicted_traj - traj) ** 2 * mask * atten_mask)
                        / 0.5
                    )

                val_mae, val_rmse = masked_mae_rmse(predicted_traj, traj, mask, atten_mask)
                meter_mae, meter_rmse = masked_meter_mae_rmse(
                    predicted_traj, traj, original, normalize_transform, mask, atten_mask
                )

                val_losses.append(val_loss.item())
                val_maes.append(val_mae.item())
                val_rmses.append(val_rmse.item())
                val_meter_maes.append(meter_mae.item())
                val_meter_rmses.append(meter_rmse.item())

        if not val_losses:
            raise RuntimeError(
                "No validation batches were produced. "
                "Check dataset paths and val_steps_per_epoch."
            )

        val_seconds = time.perf_counter() - val_start

        # Aggregate validation metrics across DDP ranks
        val_loss_sum = torch.tensor([sum(val_losses)], device=device)
        val_mae_sum = torch.tensor([sum(val_maes)], device=device)
        val_rmse_sum = torch.tensor([sum(val_rmses)], device=device)
        val_meter_mae_sum = torch.tensor([sum(val_meter_maes)], device=device)
        val_meter_rmse_sum = torch.tensor([sum(val_meter_rmses)], device=device)
        val_count = torch.tensor([len(val_losses)], device=device)
        _reduce_metrics(
            val_loss_sum, val_mae_sum, val_rmse_sum, val_count, use_ddp,
            meter_mae_sum=val_meter_mae_sum, meter_rmse_sum=val_meter_rmse_sum,
        )

        avg_val_loss = _aggregate_scalar(val_loss_sum, val_count)
        avg_val_mae = _aggregate_scalar(val_mae_sum, val_count)
        avg_val_rmse = _aggregate_scalar(val_rmse_sum, val_count)
        avg_val_meter_mae = _aggregate_scalar(val_meter_mae_sum, val_count)
        avg_val_meter_rmse = _aggregate_scalar(val_meter_rmse_sum, val_count)
        logger.info(
            f"Epoch {epoch} Validation Loss: {avg_val_loss:.5f}, "
            f"MAE: {avg_val_mae:.5f}, RMSE: {avg_val_rmse:.5f}, "
            f"Meter MAE: {avg_val_meter_mae:.2f}, Meter RMSE: {avg_val_meter_rmse:.2f}, "
            f"Time: {format_duration(val_seconds)}"
        )

        scheduler.step(avg_val_loss)

        # ------ Checkpointing & Early Stopping (rank 0 only) ------
        if is_main_process():
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trigger_times = 0
                m_path = model_save / f"best_model_epoch_{epoch}.pt"
                torch.save(model_state_dict(model), m_path)
                logger.info(
                    f"Validation loss decreased,\nsaving model to {m_path}"
                )
            else:
                trigger_times += 1
                logger.info(
                    f"Validation loss did not decrease for {trigger_times} epochs"
                )
                if trigger_times >= patience:
                    m_path = model_save / f"Final_Model_{epoch}.pt"
                    torch.save(model_state_dict(model), m_path)
                    logger.info("Early stopping triggered")

            epoch_seconds = time.perf_counter() - epoch_start
            logger.info(
                f"Epoch {epoch} Time Summary: "
                f"train={format_duration(train_seconds)}, "
                f"val={format_duration(val_seconds)}, "
                f"total={format_duration(epoch_seconds)}"
            )

        # Broadcast early-stopping signal to all ranks
        stop_tensor = torch.tensor(
            [1 if (is_main_process() and trigger_times >= patience) else 0],
            device=device,
        )
        if use_ddp:
            dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() == 1:
            break

    logger.info("<----Training Done---->")


# ============================================================================
# Experiment Directory Setup
# ============================================================================

def setup_experiment_directories(
    config,
    exp_name: str = "UniTraj",
    timestamp: Optional[str] = None,
):
    """Create experiment directories and set up logging.

    Directory structure:
        {root}/{exp_name}/{dataset}_bs={batch_size}/{timestamp}/
        ├── Files/    -- copies of source code
        ├── Results/  -- placeholder for result files
        ├── models/   -- model checkpoints
        └── out.log   -- training log

    Args:
        config: Configuration namespace.
        exp_name: Top-level experiment name (default: "UniTraj").
        timestamp: Optional timestamp string (format: "MM-DD-HH-MM-SS").

    Returns:
        (logger, files_save, result_save, model_save)
    """
    root_dir = Path(__file__).resolve().parent
    result_name = f"{config.data.dataset}_bs={config.training.batch_size}"
    exp_dir = root_dir / exp_name / result_name

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    exp_time_dir = exp_dir / timestamp

    files_save = exp_time_dir / "Files"
    result_save = exp_time_dir / "Results"
    model_save = exp_time_dir / "models"

    # All ranks create directories (exist_ok=True for safety)
    for directory in [files_save, result_save, model_save]:
        directory.mkdir(parents=True, exist_ok=True)

    # Only rank 0 copies source files
    if is_main_process():
        utils_dir = root_dir / "utils"
        for filename in os.listdir(utils_dir):
            if filename.endswith(".py"):
                shutil.copy(utils_dir / filename, files_save)
        shutil.copy(Path(__file__), files_save)

    if is_main_process():
        print("All files saved path ---->>", exp_time_dir)

    # Create logger; non-rank-0 processes are silenced via DDPAwareLogger
    raw_logger = Logger(
        __name__, log_path=str(exp_dir / (timestamp + "/out.log")), colorize=True
    )
    logger = DDPAwareLogger(raw_logger)
    return logger, files_save, result_save, model_save


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # ---- Parse configuration ----
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    # ---- Detect DDP environment ----
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = local_rank >= 0

    if use_ddp:
        local_rank, world_size = ddp_setup()

    # ---- Generate unified timestamp across DDP ranks ----
    if use_ddp:
        if local_rank == 0:
            timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
            ts_val = int(timestamp.replace("-", ""))
        else:
            ts_val = 0
        ts_tensor = torch.tensor(
            [ts_val], dtype=torch.long, device=f"cuda:{local_rank}"
        )
        dist.broadcast(ts_tensor, src=0)
        if local_rank != 0:
            ts_str = f"{ts_tensor.item():010d}"
            timestamp = (
                f"{ts_str[:2]}-{ts_str[2:4]}-{ts_str[4:6]}"
                f"-{ts_str[6:8]}-{ts_str[8:10]}"
            )
    else:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

    # ---- Setup experiment ----
    logger, files_save, result_save, model_save = setup_experiment_directories(
        config, exp_name="UniTraj", timestamp=timestamp
    )

    if use_ddp:
        dist.barrier()  # Wait for rank 0 to finish file copying

    log_info(config, logger)
    main(config, logger, local_rank=local_rank, world_size=world_size)

    if use_ddp:
        cleanup_ddp()
