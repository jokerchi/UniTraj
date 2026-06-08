"""
Stage 1: MAE Pretraining Configuration.

Usage:
    python main.py --config configs/stage1_pretrain.py
"""

args = {
    "data": {
        "dataset": "worldtrace",
        # "pickle" (map-style) or "parquet" (streaming iterable)
        "format": "parquet",
        # Number of trajectory points (padding/truncation target)
        "traj_length": 200,
        # Model embedding dimension
        "emb_dim": 128,
        # DataLoader workers
        "num_workers": 12,
        # Random seed for dataset sampling
        "sampler_seed": 2024,
        # Shuffle buffer size for parquet streaming (0 = no shuffle)
        "shuffle_buffer_size": 4096,
        # Number of records per parquet read batch
        "record_batch_size": 32768,
        # DataLoader prefetch factor (only when num_workers > 0)
        "prefetch_factor": 3,
        # Number of batches per training epoch (None = iterate entire dataset)
        "steps_per_epoch": 900,
        # Number of batches per validation epoch (None = iterate entire dataset)
        "val_steps_per_epoch": 50,
        # Path to training data
        "train_file_path": "./data/parquet/train",
        # Path to validation data
        "val_file_path": "./data/parquet/val",
    },
    "training": {
        # Batch size per GPU
        "batch_size": 1024,
        # Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)
        "grad_accum_steps": 1,
        # Enable Automatic Mixed Precision (FP16)
        "use_amp": True,
        # Maximum number of training epochs
        "n_epochs": 1000,
        # Early stopping patience (epochs without improvement)
        "patience": 15,
        # ===== Stage 1: MAE Pretraining =====
        "stage": 1,
    },
}
