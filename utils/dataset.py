"""
Trajectory Dataset Classes and Data Loading Utilities.

Provides two dataset implementations:
- ``TrajectoryDataset``: map-style (indexable) dataset for pickle files with
  in-memory shard caching, suitable for smaller datasets.
- ``TrajectoryIterableDataset``: streaming iterable dataset for parquet shards
  with optional shuffle buffer, suitable for large-scale data.

Also includes:
- ``Normalize``: standardisation transform for trajectory coordinates.
- ``ShardBatchSampler``: batch sampler that respects shard/file boundaries.
- ``logarithmic_sampling_ratio``: helper for adaptive trajectory down-sampling.
"""

import bisect
import glob
import json
import math
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdp import rdp
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, get_worker_info

# ============================================================================
# Constants
# ============================================================================

MIN_POINTS: int = 36
MAX_POINTS: int = 600
MIN_SAMPLING_RATIO: float = 0.35


# ============================================================================
# Coordinate Normalization
# ============================================================================

class Normalize:
    """Standardize trajectory coordinates using pre-computed mean and std.

    Default statistics are from the WorldTrace dataset.

    Args:
        mean: [lon_mean, lat_mean] or None to use WorldTrace defaults.
        std: [lon_std, lat_std] or None to use WorldTrace defaults.
    """

    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        self.mean = torch.tensor(
            mean if mean is not None else [5.3311563533497974e-05, -7.49477039789781e-05],
            dtype=torch.float32,
        )
        self.std = torch.tensor(
            std if std is not None else [0.049923088401556015, 0.040688566863536835],
            dtype=torch.float32,
        )

    def __call__(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization: (x - mean) / std."""
        return (trajectory - self.mean) / self.std


# ============================================================================
# Sampling Utilities
# ============================================================================

def logarithmic_sampling_ratio(
    length: int,
    min_points: int = MIN_POINTS,
    max_points: int = MAX_POINTS,
    min_ratio: float = MIN_SAMPLING_RATIO,
) -> float:
    """Compute a logarithmic down-sampling ratio based on trajectory length.

    Short trajectories (<= min_points) keep all points (ratio=1.0).
    Long trajectories (>= max_points) keep ``min_ratio`` of points.
    In between, the ratio decreases logarithmically.

    Args:
        length: Number of points in the trajectory.
        min_points: Length threshold for full retention.
        max_points: Length threshold for minimum retention.
        min_ratio: Minimum sampling ratio for very long trajectories.

    Returns:
        Sampling ratio in [min_ratio, 1.0].
    """
    if length <= min_points:
        return 1.0
    if length >= max_points:
        return min_ratio
    ratio = 1.0 - math.log(length - min_points + 1) / math.log(
        max_points - min_points + 1
    ) * (1.0 - min_ratio)
    return max(ratio, min_ratio)


# ============================================================================
# Map-style Trajectory Dataset (pickle)
# ============================================================================

class TrajectoryDataset(Dataset):
    """Map-style dataset for pickle-format trajectory data.

    Supports single pickle files, directories of pickle shards, or glob patterns.
    Uses single-file caching for efficiency when all data is in one file.

    Args:
        data_path: Path to a .pkl file, a directory of .pkl files, or a glob pattern.
        max_len: Fixed sequence length for padding/truncation.
        transform: Optional coordinate transform (e.g., ``Normalize``).
        mask_ratio: Fraction of points to mask during pre-training.
        mode: One of ``"train"``, ``"val"``, or ``"test"``.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        data_path: Union[str, List[str]],
        max_len: int = 200,
        transform: Optional[Normalize] = None,
        mask_ratio: float = 0.5,
        mode: str = "train",
        seed: int = 2024,
    ):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"Unsupported dataset mode: {mode}")

        self.data_path = data_path
        self.transform = transform
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.seed = seed
        self.num_masked_points = int(self.max_len * self.mask_ratio)

        # Precompute sampling ratios for adaptive trajectory down-sampling
        self.sampling_ratios = [
            logarithmic_sampling_ratio(length)
            for length in np.arange(MIN_POINTS, MAX_POINTS + 1, 1)
        ]

        # Collect pickle files
        self.data_files = self._collect_pickle_files(self.data_path)
        if not self.data_files:
            raise FileNotFoundError(f"No pickle files found: {self.data_path}")

        self._single_file = len(self.data_files) == 1
        self._cache_file_idx: Optional[int] = None
        self._cache_df: Optional[pd.DataFrame] = None

        # Build cumulative index for multi-shard access
        if self._single_file:
            self.data = pd.read_pickle(self.data_files[0])
            self.file_lengths = [len(self.data)]
            self.cum_lengths = [len(self.data)]
        else:
            self.data = None
            self.file_lengths = []
            for path in self.data_files:
                df = pd.read_pickle(path)
                self.file_lengths.append(len(df))
                del df
            self.cum_lengths = list(np.cumsum(self.file_lengths))

    # --- File collection ---

    @staticmethod
    def _collect_pickle_files(data_path: Union[str, List[str]]) -> List[Path]:
        """Resolve data_path into a flat list of .pkl file paths."""

        def collect_one(path_like: str) -> List[Path]:
            path = Path(path_like).expanduser()
            if path.is_file():
                return [path]
            if path.is_dir():
                return sorted(path.rglob("*.pkl"))
            return sorted(Path(p).expanduser() for p in glob.glob(str(path), recursive=True))

        if isinstance(data_path, (list, tuple)):
            files: List[Path] = []
            for item in data_path:
                files.extend(collect_one(item))
            return files
        return collect_one(data_path)

    # --- DataFrame access ---

    def _load_dataframe(self, file_idx: int) -> pd.DataFrame:
        """Load a shard DataFrame with simple single-entry cache."""
        if self._cache_file_idx == file_idx and self._cache_df is not None:
            return self._cache_df
        df = pd.read_pickle(self.data_files[file_idx])
        self._cache_file_idx = file_idx
        self._cache_df = df
        return df

    def _get_sample(self, idx: int) -> pd.Series:
        """Retrieve a single trajectory row by global index."""
        if self._single_file:
            return self.data.iloc[idx]
        file_idx = bisect.bisect_right(self.cum_lengths, idx)
        prev_cum = 0 if file_idx == 0 else self.cum_lengths[file_idx - 1]
        local_idx = idx - prev_cum
        df = self._load_dataframe(file_idx)
        return df.iloc[local_idx]

    # --- Public API ---

    def set_mask_ratio(self, mask_ratio: float) -> None:
        """Update the mask ratio and recompute the number of masked points."""
        self.mask_ratio = mask_ratio
        self.num_masked_points = int(self.max_len * self.mask_ratio)

    def __len__(self) -> int:
        return self.cum_lengths[-1]

    @contextmanager
    def _rng_context(self, idx: int):
        """Deterministic RNG context for validation/test (train uses global RNG)."""
        if self.mode == "train":
            yield
            return

        py_state = random.getstate()
        np_state = np.random.get_state()
        seed = (self.seed + int(idx)) % (2 ** 32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preprocessed trajectory sample.

        Returns:
            Dict with keys: trajectory, attention_mask, original, intervals, indices.
        """
        with self._rng_context(idx):
            return self._build_item(idx)

    # --- Item construction ---

    def _build_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Core preprocessing pipeline for a single trajectory."""
        # 1. Resample trajectory
        traj_df = self.resample_trajectory(idx)

        # 2. Extract coordinates and time intervals
        trajectory = torch.tensor(
            traj_df[["longitude", "latitude"]].values, dtype=torch.float32
        )
        intervals = torch.tensor(traj_df["interval"].values, dtype=torch.float32)

        # 3. Apply masking strategy
        trajectory_length = len(traj_df)
        mask_strategy = random.random()
        if mask_strategy < 0.7:
            mask = self.apply_random_mask(trajectory_length)
        elif mask_strategy < 0.85:
            mask = self.apply_rdp_mask(trajectory)
        elif mask_strategy < 0.9:
            mask = self.apply_block_mask(trajectory_length)
        else:
            mask = self.apply_last_n_mask(trajectory_length)

        # 4. Normalize (subtract origin, apply transform)
        original = trajectory[0]
        trajectory = trajectory - original
        if self.transform:
            trajectory = self.transform(trajectory)

        # 5. Pad or truncate to fixed length
        trajectory, attention_mask = self.pad_or_truncate(trajectory)
        intervals, _ = self.pad_or_truncate(intervals)

        # 6. Ensure mask covers exactly num_masked_points
        mask = np.pad(mask, (0, max(0, self.max_len - trajectory_length)), constant_values=0)
        current_masked_points = mask.sum()
        if current_masked_points < self.num_masked_points:
            additional = self.num_masked_points - current_masked_points
            padding_indices = np.where(mask == 0)[0]
            additional_indices = np.random.choice(
                padding_indices, size=additional, replace=False
            )
            mask[additional_indices] = 1

        # 7. Build output dict
        trajectory = trajectory.transpose(0, 1)  # [2, max_len]
        mask_indices = torch.tensor(np.where(mask == 1)[0]).long()

        return {
            "trajectory": trajectory,
            "attention_mask": attention_mask,
            "original": original,
            "intervals": intervals,
            "indices": mask_indices,
        }

    # --- Masking strategies ---

    def apply_random_mask(self, trajectory_length: int) -> np.ndarray:
        """Randomly select points to mask."""
        trajectory_length = min(trajectory_length, self.max_len)
        num_points = int(trajectory_length * self.mask_ratio)
        mask = np.full(trajectory_length, False, dtype=bool)
        mask_indices = np.random.choice(trajectory_length, size=num_points, replace=False)
        mask[mask_indices] = True
        return mask

    def apply_last_n_mask(self, trajectory_length: int, n: int = 8) -> np.ndarray:
        """Mask the last n points plus additional random points."""
        n = np.random.randint(3, 8)
        trajectory_length = min(trajectory_length, self.max_len)
        num_points = int(trajectory_length * self.mask_ratio)
        additional = num_points - n

        mask = np.full(trajectory_length, False, dtype=bool)
        mask[-n:] = True
        if additional > 0:
            candidates = np.arange(trajectory_length - n)
            additional_indices = np.random.choice(candidates, size=additional, replace=False)
            mask[additional_indices] = True
        return mask

    def apply_block_mask(self, trajectory_length: int, block_size: int = 8) -> np.ndarray:
        """Mask a contiguous block of points plus additional random points."""
        block_size = np.random.randint(5, 15)
        trajectory_length = min(trajectory_length, self.max_len)
        num_points = int(trajectory_length * self.mask_ratio)
        additional = num_points - block_size

        mask = np.full(trajectory_length, False, dtype=bool)
        start_idx = np.random.randint(0, trajectory_length - block_size + 1)
        mask[start_idx : start_idx + block_size] = True

        if additional > 0:
            non_block_indices = np.where(~mask)[0]
            additional_indices = np.random.choice(
                non_block_indices, size=additional, replace=False
            )
            mask[additional_indices] = True
        return mask

    def apply_rdp_mask(self, trajectory: torch.Tensor, epsilon: float = 1e-4) -> np.ndarray:
        """Mask key points detected by the Ramer-Douglas-Peucker algorithm."""
        trajectory = trajectory[: self.max_len]
        trajectory_length = len(trajectory)
        num_points = int(trajectory_length * self.mask_ratio)

        # Detect key points via RDP
        rdp_mask = rdp(trajectory, epsilon=epsilon, return_mask=True)
        rdp_mask = np.array(rdp_mask)
        rdp_mask[0], rdp_mask[-1] = False, False  # never mask endpoints

        num_rdp_mask = rdp_mask.sum()

        if num_rdp_mask > num_points:
            # Subsample RDP points
            indices = np.where(rdp_mask)[0]
            masked = np.random.choice(indices, size=num_points, replace=False)
            rdp_mask[:] = False
            rdp_mask[masked] = True
        elif num_rdp_mask < num_points:
            # Add random points to reach target count
            non_rdp_indices = np.where(~rdp_mask)[0]
            additional = num_points - num_rdp_mask
            additional_indices = np.random.choice(
                non_rdp_indices, size=additional, replace=False
            )
            rdp_mask[additional_indices] = True

        return rdp_mask

    # --- Trajectory resampling ---

    def resample_trajectory(self, idx: int) -> pd.DataFrame:
        """Load and resample a trajectory, producing (time, lon, lat, interval) DataFrame."""
        sample = self._get_sample(idx)
        full_df = pd.DataFrame({
            "time": sample["time"],
            "longitude": [point[1] for point in sample["trajectory"]],
            "latitude": [point[0] for point in sample["trajectory"]],
        })

        trajectory_length = len(full_df)

        if random.random() < 0.3 and trajectory_length >= 360:
            # Fixed-interval resampling for long trajectories
            if trajectory_length > 540:
                sampling_interval = random.randint(8, 15)
            elif trajectory_length > 360:
                sampling_interval = random.randint(6, 10)
            else:
                sampling_interval = random.randint(3, 6)

            full_df["time"] = pd.to_datetime(full_df["time"])
            full_df.set_index("time", inplace=True)
            resampled_df = full_df.resample(f"{sampling_interval}s").mean().reset_index()
            resampled_df["interval"] = (
                resampled_df["time"].diff().dt.total_seconds().fillna(0).astype("float32")
            )
        else:
            # Logarithmic-ratio random down-sampling
            if trajectory_length <= MIN_POINTS:
                sampling_ratio = 1.0
            elif trajectory_length >= MAX_POINTS:
                sampling_ratio = MIN_SAMPLING_RATIO
            else:
                sampling_ratio = self.sampling_ratios[trajectory_length - MIN_POINTS]

            num_sampled = int(trajectory_length * sampling_ratio)
            indices = np.random.choice(full_df.index, size=num_sampled, replace=False)
            resampled_df = full_df.loc[indices].sort_index().reset_index()
            resampled_df["interval"] = (
                resampled_df["time"].diff().dt.total_seconds().fillna(0).astype("float32")
            )

        return resampled_df

    # --- Padding / Truncation ---

    def pad_or_truncate(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pad or truncate a 1-D or 2-D tensor to ``self.max_len``.

        Returns:
            (padded_tensor, attention_mask): attention_mask is 1 for real points
            and 0 for padding; None for 1-D input.
        """
        seq_len = len(tensor)
        if seq_len > self.max_len:
            tensor = tensor[: self.max_len]
            attention_mask = torch.ones(self.max_len)
            return tensor, attention_mask

        if tensor.dim() == 2:
            padded_tensor = torch.zeros((self.max_len, 2), dtype=tensor.dtype)
            attention_mask = torch.zeros(self.max_len, dtype=torch.float32)
            attention_mask[:seq_len] = 1
        else:
            padded_tensor = torch.zeros(self.max_len, dtype=tensor.dtype)
            attention_mask = None

        padded_tensor[:seq_len] = tensor
        return padded_tensor, attention_mask


# ============================================================================
# Iterable Trajectory Dataset (parquet streaming)
# ============================================================================

class TrajectoryIterableDataset(IterableDataset):
    """Streaming iterable dataset for large-scale parquet trajectory data.

    Reads from parquet shards, supports multi-worker DDP shard splitting,
    and includes an optional shuffle buffer for training.

    Args:
        data_path: Path to a .parquet file, directory, or glob pattern.
        max_len: Fixed sequence length for padding/truncation.
        transform: Optional coordinate transform.
        mask_ratio: Fraction of points to mask during pre-training.
        mode: ``"train"``, ``"val"``, or ``"test"``.
        seed: Random seed for reproducibility.
        shuffle_buffer_size: Buffer size for approximate shuffling (0 = no shuffle).
        record_batch_size: Number of records per parquet read batch.
    """

    def __init__(
        self,
        data_path: Union[str, List[str]],
        max_len: int = 200,
        transform: Optional[Normalize] = None,
        mask_ratio: float = 0.5,
        mode: str = "train",
        seed: int = 2024,
        shuffle_buffer_size: int = 4096,
        record_batch_size: int = 1024,
    ):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"Unsupported dataset mode: {mode}")

        self.data_path = data_path
        self.data_files = self._collect_parquet_files(data_path)
        if not self.data_files:
            raise FileNotFoundError(f"No parquet files found: {data_path}")

        self.max_len = max_len
        self.transform = transform
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.seed = seed
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.record_batch_size = int(record_batch_size)
        self.num_masked_points = int(self.max_len * self.mask_ratio)

        self.sampling_ratios = [
            logarithmic_sampling_ratio(length)
            for length in np.arange(MIN_POINTS, MAX_POINTS + 1, 1)
        ]
        self.total_trajectories = self._load_total_trajectories()
        self._epoch = 0

    # --- File collection ---

    @staticmethod
    def _collect_parquet_files(data_path: Union[str, List[str]]) -> List[Path]:
        """Resolve data_path into a flat list of .parquet file paths."""

        def collect_one(path_like: str) -> List[Path]:
            path = Path(path_like).expanduser()
            if path.is_file():
                return [path]
            if path.is_dir():
                return sorted(path.rglob("*.parquet"))
            return sorted(Path(p).expanduser() for p in glob.glob(str(path), recursive=True))

        if isinstance(data_path, (list, tuple)):
            files: List[Path] = []
            for item in data_path:
                files.extend(collect_one(item))
            return files
        return collect_one(data_path)

    def _load_total_trajectories(self) -> Optional[int]:
        """Estimate total trajectory count from metadata.json or parquet metadata."""
        metadata_paths = sorted({
            path.parent / "metadata.json"
            for path in self.data_files
            if (path.parent / "metadata.json").exists()
        })
        if metadata_paths:
            total = 0
            for metadata_path in metadata_paths:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                total += int(metadata.get("total_trajectories", 0))
            return total

        try:
            import pyarrow.parquet as pq
        except ImportError:
            return None

        total = 0
        for path in self.data_files:
            total += pq.ParquetFile(str(path)).metadata.num_rows
        return total

    # --- Public API ---

    def __len__(self) -> int:
        if self.total_trajectories is None:
            raise TypeError("TrajectoryIterableDataset length is unknown.")
        return self.total_trajectories

    def __iter__(self):
        """Iterate over trajectories for one epoch.

        In training mode, shards are shuffled per epoch.
        Shards are split across DDP ranks and DataLoader workers.
        """
        epoch = self._epoch
        self._epoch += 1
        rng = np.random.default_rng(self.seed + epoch)

        files = list(self.data_files)
        if self.mode == "train":
            rng.shuffle(files)

        files = self._split_files(files)
        records = self._iter_parquet_records(files)
        if self.mode == "train" and self.shuffle_buffer_size > 1:
            records = self._shuffle_records(records, rng)

        for sample_idx, record in enumerate(records):
            sample_rng = rng
            if self.mode != "train":
                sample_rng = np.random.default_rng(self.seed + sample_idx)
            item = self._build_item_from_record(record, sample_rng)
            if item is not None:
                yield item

    # --- Shard splitting for DDP and workers ---

    def _split_files(self, files: List[Path]) -> List[Path]:
        """Split file list across DDP ranks and DataLoader workers."""
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        if world_size > 1:
            files = files[rank::world_size]

        worker_info = get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]
        return files

    # --- Parquet record iteration ---

    def _iter_parquet_records(self, files: List[Path]):
        """Yield individual records from parquet shards via pyarrow.

        Uses Arrow columnar access with offset-based slicing instead of
        ``to_pylist()``.  This avoids constructing Python dicts/lists for
        every value and is ~10–15× faster for list-typed columns.
        """
        if not files:
            return

        try:
            import pyarrow.dataset as ds
        except ImportError as exc:
            raise ImportError(
                "TrajectoryIterableDataset requires pyarrow. "
                "Install it with `pip install pyarrow`."
            ) from exc

        dataset = ds.dataset([str(path) for path in files], format="parquet")
        for batch in dataset.to_batches(
            columns=["time", "latitude", "longitude"],
            batch_size=self.record_batch_size,
        ):
            # Access Arrow columns directly — much faster than to_pylist().
            time_col = batch.column("time")       # ListArray<int64>
            lat_col = batch.column("latitude")    # ListArray<float32>
            lon_col = batch.column("longitude")   # ListArray<float32>

            # Extract flat arrays + offsets ONCE per batch.
            flat_times = time_col.values.to_numpy(zero_copy_only=False)
            flat_lats = lat_col.values.to_numpy(zero_copy_only=False)
            flat_lons = lon_col.values.to_numpy(zero_copy_only=False)
            offsets = time_col.offsets.to_numpy()  # [num_rows + 1]

            for i in range(batch.num_rows):
                s, e = offsets[i], offsets[i + 1]
                yield {
                    "time": flat_times[s:e],
                    "latitude": flat_lats[s:e],
                    "longitude": flat_lons[s:e],
                }

    # --- Shuffle buffer ---

    def _shuffle_records(self, records, rng: np.random.Generator):
        """Approximate shuffling via a fixed-size reservoir buffer."""
        buffer = []
        for record in records:
            buffer.append(record)
            if len(buffer) >= self.shuffle_buffer_size:
                idx = int(rng.integers(0, len(buffer)))
                yield buffer.pop(idx)

        rng.shuffle(buffer)
        yield from buffer

    # --- Item construction ---

    def _build_item_from_record(
        self, record: dict, rng: np.random.Generator
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Preprocess a single parquet record into a model-ready sample."""
        time = np.asarray(record["time"], dtype=np.int64)
        latitude = np.asarray(record["latitude"], dtype=np.float32)
        longitude = np.asarray(record["longitude"], dtype=np.float32)

        # Validate
        if len(time) == 0 or len(time) != len(latitude) or len(latitude) != len(longitude):
            return None

        # Filter invalid coordinates
        valid = np.isfinite(latitude) & np.isfinite(longitude)
        if not valid.all():
            time = time[valid]
            latitude = latitude[valid]
            longitude = longitude[valid]
        if len(time) == 0:
            return None

        # Resample
        time, longitude, latitude = self._resample_arrays(time, longitude, latitude, rng)
        if len(time) == 0:
            return None

        # Build arrays
        trajectory_np = np.stack([longitude, latitude], axis=1).astype(np.float32)
        intervals_np = np.diff(time, prepend=time[0]).astype(np.float32)
        seq_len = min(len(trajectory_np), self.max_len)
        if seq_len == 0:
            return None

        # Generate mask
        short_mask = self._make_mask(trajectory_np[:seq_len], seq_len, rng)
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:seq_len] = short_mask[:seq_len]
        if mask.sum() < self.num_masked_points:
            additional = self.num_masked_points - int(mask.sum())
            available = np.where(~mask)[0]
            mask[self._choice(available, additional, rng)] = True

        # Normalize
        trajectory = torch.as_tensor(trajectory_np[:seq_len], dtype=torch.float32)
        original = trajectory[0].clone()
        trajectory = trajectory - original
        if self.transform:
            trajectory = self.transform(trajectory)

        # Pad to fixed length
        padded_trajectory = torch.zeros((self.max_len, 2), dtype=torch.float32)
        padded_trajectory[:seq_len] = trajectory
        attention_mask = torch.zeros(self.max_len, dtype=torch.float32)
        attention_mask[:seq_len] = 1

        intervals = torch.zeros(self.max_len, dtype=torch.float32)
        intervals[:seq_len] = torch.as_tensor(intervals_np[:seq_len], dtype=torch.float32)

        return {
            "trajectory": padded_trajectory.transpose(0, 1),
            "attention_mask": attention_mask,
            "original": original,
            "intervals": intervals,
            "indices": torch.as_tensor(np.where(mask)[0], dtype=torch.long),
        }

    # --- Resampling ---

    def _resample_arrays(
        self,
        time: np.ndarray,
        longitude: np.ndarray,
        latitude: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Down-sample trajectory arrays for efficient training."""
        trajectory_length = len(time)

        # Fixed-interval resampling (30% chance for long trajectories)
        if rng.random() < 0.3 and trajectory_length >= 360:
            if trajectory_length > 540:
                sampling_interval = int(rng.integers(8, 16))
            elif trajectory_length > 360:
                sampling_interval = int(rng.integers(6, 11))
            else:
                sampling_interval = int(rng.integers(3, 7))

            resampled = self._interval_resample(time, longitude, latitude, sampling_interval)
            if len(resampled[0]) > 0:
                return resampled

        # Logarithmic-ratio random down-sampling
        if trajectory_length <= MIN_POINTS:
            sampling_ratio = 1.0
        elif trajectory_length >= MAX_POINTS:
            sampling_ratio = MIN_SAMPLING_RATIO
        else:
            sampling_ratio = self.sampling_ratios[trajectory_length - MIN_POINTS]

        num_sampled = max(1, int(trajectory_length * sampling_ratio))
        indices = rng.choice(trajectory_length, size=num_sampled, replace=False)
        indices.sort()
        return time[indices], longitude[indices], latitude[indices]

    @staticmethod
    def _interval_resample(
        time: np.ndarray,
        longitude: np.ndarray,
        latitude: np.ndarray,
        sampling_interval: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample by binning points into fixed time intervals (mean aggregation)."""
        bins = time // sampling_interval
        unique_bins, inverse = np.unique(bins, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float32)
        longitude_sum = np.bincount(inverse, weights=longitude)
        latitude_sum = np.bincount(inverse, weights=latitude)
        resampled_time = (unique_bins * sampling_interval).astype(np.int64)
        resampled_longitude = (longitude_sum / counts).astype(np.float32)
        resampled_latitude = (latitude_sum / counts).astype(np.float32)
        return resampled_time, resampled_longitude, resampled_latitude

    # --- Masking ---

    def _make_mask(
        self, trajectory: np.ndarray, trajectory_length: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Select a masking strategy and generate the mask."""
        mask_strategy = rng.random()
        if mask_strategy < 0.7:
            return self._random_mask(trajectory_length, rng)
        if mask_strategy < 0.85:
            return self._rdp_mask(trajectory, trajectory_length, rng)
        if mask_strategy < 0.9:
            return self._block_mask(trajectory_length, rng)
        return self._last_n_mask(trajectory_length, rng)

    def _random_mask(
        self, trajectory_length: int, rng: np.random.Generator
    ) -> np.ndarray:
        num_points = int(trajectory_length * self.mask_ratio)
        mask = np.full(trajectory_length, False, dtype=bool)
        mask[self._choice(np.arange(trajectory_length), num_points, rng)] = True
        return mask

    def _last_n_mask(
        self, trajectory_length: int, rng: np.random.Generator
    ) -> np.ndarray:
        num_points = int(trajectory_length * self.mask_ratio)
        n = (
            trajectory_length
            if trajectory_length < 3
            else int(rng.integers(3, min(8, trajectory_length) + 1))
        )
        mask = np.full(trajectory_length, False, dtype=bool)
        mask[-n:] = True
        additional = max(0, num_points - n)
        candidates = np.arange(max(0, trajectory_length - n))
        mask[self._choice(candidates, additional, rng)] = True
        return mask

    def _block_mask(
        self, trajectory_length: int, rng: np.random.Generator
    ) -> np.ndarray:
        num_points = int(trajectory_length * self.mask_ratio)
        block_size = (
            trajectory_length
            if trajectory_length < 5
            else int(rng.integers(5, min(15, trajectory_length) + 1))
        )
        mask = np.full(trajectory_length, False, dtype=bool)
        start_idx = int(rng.integers(0, trajectory_length - block_size + 1))
        mask[start_idx : start_idx + block_size] = True
        additional = max(0, num_points - block_size)
        candidates = np.where(~mask)[0]
        mask[self._choice(candidates, additional, rng)] = True
        return mask

    def _rdp_mask(
        self,
        trajectory: np.ndarray,
        trajectory_length: int,
        rng: np.random.Generator,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        num_points = int(trajectory_length * self.mask_ratio)
        mask = np.full(trajectory_length, False, dtype=bool)
        if trajectory_length <= 2 or num_points <= 0:
            return mask

        rdp_mask = np.asarray(
            rdp(trajectory, epsilon=epsilon, return_mask=True), dtype=bool
        )
        rdp_mask = rdp_mask[:trajectory_length]
        rdp_mask[0], rdp_mask[-1] = False, False
        num_rdp_mask = int(rdp_mask.sum())

        if num_rdp_mask > num_points:
            indices = np.where(rdp_mask)[0]
            masked = self._choice(indices, num_points, rng)
            rdp_mask[:] = False
            rdp_mask[masked] = True
        elif num_rdp_mask < num_points:
            candidates = np.where(~rdp_mask)[0]
            additional = self._choice(candidates, num_points - num_rdp_mask, rng)
            rdp_mask[additional] = True

        return rdp_mask

    @staticmethod
    def _choice(
        candidates: np.ndarray, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Safe random choice with size clamping."""
        candidates = np.asarray(candidates)
        size = min(max(int(size), 0), len(candidates))
        if size == 0:
            return np.asarray([], dtype=np.int64)
        return rng.choice(candidates, size=size, replace=False)


# ============================================================================
# Shard-aware Batch Sampler
# ============================================================================

class ShardBatchSampler(Sampler):
    """Batch sampler that generates batches within individual shard boundaries.

    This prevents cross-shard batching which is important for pickle-based
    ``TrajectoryDataset`` where each shard is loaded independently.

    Args:
        dataset: A ``TrajectoryDataset`` instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle within each shard.
        drop_last: Whether to drop the last incomplete batch per shard.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        dataset: TrajectoryDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 2024,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not hasattr(dataset, "file_lengths") or not hasattr(dataset, "cum_lengths"):
            raise ValueError(
                "ShardBatchSampler requires a TrajectoryDataset instance"
            )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        file_order = list(range(len(self.dataset.file_lengths)))
        if self.shuffle:
            rng.shuffle(file_order)

        for file_idx in file_order:
            file_len = self.dataset.file_lengths[file_idx]
            start_idx = 0 if file_idx == 0 else self.dataset.cum_lengths[file_idx - 1]
            indices = np.arange(start_idx, start_idx + file_len)
            if self.shuffle:
                rng.shuffle(indices)

            for batch_start in range(0, file_len, self.batch_size):
                batch = indices[batch_start : batch_start + self.batch_size].tolist()
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                length // self.batch_size for length in self.dataset.file_lengths
            )
        return sum(
            (length + self.batch_size - 1) // self.batch_size
            for length in self.dataset.file_lengths
        )
