# UniTraj: Learning a Universal Trajectory Foundation Model from Billion-Scale Worldwide Traces

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of the NeurIPS 2025 paper: **UniTraj: Learning a Universal Trajectory Foundation Model from Billion-Scale Worldwide Traces**.

UniTraj is a universal trajectory foundation model pre-trained via masked autoencoding (MAE) on billion-scale worldwide GPS traces. It addresses the limitations of existing trajectory learning methods — task specificity, regional dependency, and data sensitivity — by learning general-purpose trajectory representations that transfer across downstream tasks.

## Overview

UniTraj uses a transformer-based encoder-decoder architecture:

1. **Trajectory Tokenization**: Raw (lon, lat) sequences are tokenized via Conv1D into patch embeddings.
2. **Masked Pretraining**: A portion of tokens are randomly masked; the encoder processes only unmasked tokens.
3. **Trajectory Reconstruction**: The decoder reconstructs the full trajectory, learning rich spatiotemporal representations.
4. **Rotary Position Embedding (RoPE)**: Encodes relative position information for better sequence modeling.
5. **Multi-Strategy Masking**: Random, block, last-N, and RDP-based key-point masking for robust pretraining.

<img src="./Logo.png" alt="UniTraj Logo" style="zoom:10%;" />

## Table of Contents

- [Project Structure](#project-structure)
- [Environment & Dependencies](#environment--dependencies)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Key Features](#key-features)
- [Citation](#citation)
- [License](#license)

## Project Structure

```
.
├── main.py                              # Main training script (single-GPU & DDP)
├── convert_matched_csv_to_pickle.py     # Data conversion utility (CSV/PKL → PKL/Parquet)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── LICENSE                              # License file
├── Logo.png                             # Project logo
│
├── utils/                               # Core modules
│   ├── __init__.py                      # Package init (empty)
│   ├── unitraj.py                       # UniTraj model: Encoder, Decoder, Transformer, RoPE
│   ├── dataset.py                       # Dataset classes, normalization, sampling, masking
│   ├── config.py                        # Training and data configuration
│   ├── logger.py                        # Colored logging system with file output
│   └── EMA.py                           # Exponential Moving Average helper
│
├── data/                                # Data storage (example)
│   ├── parquet/train/                   # Training parquet shards
│   │   ├── trajectories_000001.parquet
│   │   ├── trajectories_000002.parquet
│   │   └── metadata.json
│   ├── parquet/val/                     # Validation parquet shards
│   │   ├── trajectories_000001.parquet
│   │   └── metadata.json
│   └── worldtrace_sample.pkl            # Sample dataset (1,000 trajectories)
│
└── UniTraj/                             # Experiment output (auto-generated)
    └── worldtrace_bs=1024/
        └── MM-DD-HH-MM-SS/
            ├── Files/                   # Snapshots of source code
            ├── Results/                 # Training results
            ├── models/                  # Model checkpoints
            └── out.log                  # Training log
```

## Environment & Dependencies

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Runtime |
| PyTorch | 2.6.0 | Deep learning framework |
| NumPy | ≥ 1.19.0 | Numerical computation |
| Pandas | ≥ 1.1.0 | Data processing |
| PyArrow | * | Parquet file I/O (required for streaming dataset) |
| Timm | * | Vision Transformer utilities |
| Einops | * | Tensor operations |
| rdp | * | Ramer-Douglas-Peucker trajectory simplification |
| colored | * | ANSI-colored console output |
| Folium | * | Interactive map visualization (optional) |

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: ≥ 16 GB VRAM for batch_size=1024)
- **RAM**: ≥ 32 GB for large parquet datasets
- **Disk**: Depends on dataset size; parquet format is more storage-efficient

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd UniTraj

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: If you encounter CUDA linker warnings about `libcuda.so`, ensure your system has the 64-bit CUDA driver library properly linked. The training script automatically sets `LIBRARY_PATH` to include the CUDA stubs directory, but you may also create a symlink manually:
> ```bash
> sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
> ```

## Dataset

The full **WorldTrace** dataset (billion-scale worldwide GPS traces) is available on:

- 🤗 [Hugging Face](https://huggingface.co/datasets/OpenTrace/WorldTrace)
- [ModelScope](https://www.modelscope.cn/datasets/opentrace/WorldTrace)

A sample subset (1,000 trajectories) is included in `data/worldtrace_sample.pkl` for quick testing.

### Data Format

Trajectories are stored in one of two formats:

**Pickle (.pkl)** — Each file contains a DataFrame with columns:
- `time`: list of timestamps
- `trajectory`: list of `[latitude, longitude]` pairs

**Parquet (.parquet)** — Each file contains columns:
- `time`: list of Unix timestamps (int64)
- `latitude`: list of float32 values
- `longitude`: list of float32 values
- Plus a `metadata.json` with `total_trajectories` and shard info

## Data Preparation

### Converting CSV to Pickle

If your raw data is in CSV format (one file per trajectory) with columns `time, matched_latitude, matched_longitude`:

```bash
python convert_matched_csv_to_pickle.py \
    --input /path/to/csv_directory \
    --output data/custom.pkl \
    --input-format csv \
    --output-format pickle
```

### Converting Pickle to Parquet (Streaming)

For large-scale training, convert pickle shards to streaming-friendly parquet:

```bash
# Training data
python convert_matched_csv_to_pickle.py \
    --input data/shards/train \
    --input-format pickle \
    --output data/parquet/train \
    --output-format parquet \
    --shard-size 5000

# Validation data
python convert_matched_csv_to_pickle.py \
    --input data/shards/val \
    --input-format pickle \
    --output data/parquet/val \
    --output-format parquet \
    --shard-size 5000
```

This produces `trajectories_000001.parquet`, `trajectories_000002.parquet`, ..., plus a `metadata.json` file per directory.

### Quick Data Conversion Examples

```bash
# CSV files to a single pickle file
python convert_matched_csv_to_pickle.py \
    --input /path/to/csv_dir \
    --output data/custom.pkl

# CSV files to parquet shards
python convert_matched_csv_to_pickle.py \
    --input /path/to/csv_dir \
    --output /path/to/parquet_dir \
    --output-format parquet \
    --shard-size 50000
```

## Configuration

Edit `utils/config.py` to customize training. Key parameters:

```python
# Data settings
"data": {
    "format": "parquet",              # "parquet" or "pickle"
    "traj_length": 200,               # Fixed trajectory length
    "emb_dim": 128,                   # Model embedding dimension
    "steps_per_epoch": 900,           # Batches per training epoch
    "val_steps_per_epoch": 50,        # Batches per validation epoch
    "train_file_path": "./data/parquet/train",
    "val_file_path": "./data/parquet/val",
}

# Training settings
"training": {
    "batch_size": 1024,               # Per-GPU batch size
    "grad_accum_steps": 1,            # Gradient accumulation steps
    "use_amp": True,                  # Automatic Mixed Precision
    "n_epochs": 1000,                 # Max training epochs
    "patience": 15,                   # Early stopping patience
}
```

Both pickle and parquet paths support:
- A single file: `./data/worldtrace_sample.pkl`
- A directory of shards: `./data/parquet/train`
- A glob pattern: `./data/shards/*.pkl`
- A list: `["./dir1", "./dir2"]`

## Training

### Single-GPU Training

```bash
python main.py
```

### Multi-GPU Training (DDP)

```bash
# 2 GPUs
torchrun --nproc_per_node=2 main.py

# 4 GPUs
torchrun --nproc_per_node=4 main.py
```

### Training Output

During training, the following files are generated under `UniTraj/{dataset}_bs={batch_size}/{timestamp}/`:

| Path | Content |
|------|---------|
| `Files/` | Source code snapshot for reproducibility |
| `models/best_model_epoch_*.pt` | Best model checkpoints (by validation loss) |
| `models/Final_Model_*.pt` | Final model when early stopping triggers |
| `out.log` | Full training log with loss, metrics, and timing |

### Monitoring

The training log includes per-epoch metrics:
- **Loss**: MSE-based reconstruction loss on masked regions
- **MAE / RMSE**: Normalized coordinate error
- **Meter MAE / RMSE**: Haversine distance error in meters
- **Time**: Training, validation, and total epoch duration

## Key Features

### Model Architecture
- **Encoder-Decoder** transformer with masked autoencoding (MAE-style pretraining)
- **Rotary Position Embedding (RoPE)** for capturing relative positions
- **Learnable CLS token** for global trajectory representation
- **Multi-scale patching** via Conv1D tokenizer

### Data Pipeline
- **Dual backend**: Pickle (map-style) for smaller datasets; Parquet (streaming) for large-scale data
- **Adaptive resampling**: Logarithmic-ratio and fixed-interval down-sampling
- **Multi-strategy masking**: Random, block, last-N, and RDP key-point masking
- **Shuffle buffer**: Approximate shuffling for streaming datasets
- **DDP-aware shard splitting**: Automatic file distribution across ranks and workers

### Training
- **Mixed precision (AMP)** with gradient scaling
- **Gradient accumulation** for large effective batch sizes
- **Distributed Data Parallel (DDP)** with NCCL backend
- **torch.compile** support for accelerated execution
- **Early stopping** with patience-based termination
- **LR scheduling** via ReduceLROnPlateau
- **Haversine distance metrics** for real-world meter-level evaluation

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{unitraj2025,
  title={UniTraj: Learning a Universal Trajectory Foundation Model from Billion-Scale Worldwide Traces},
  author={Zhu, Yuanshao and Yu, James Jianqiao and Zhao, Xiangyu and Zhou, Xun and Han, Liang and Wei, Xuetao and Liang, Yuxuan},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
```

## License

This project is released under the [MIT License](LICENSE).
