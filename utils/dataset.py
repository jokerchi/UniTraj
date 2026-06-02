import torch
import pickle, random, math
import bisect
import glob
import json
import os
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from rdp import rdp
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset, get_worker_info

MIN_POINTS = 36
MAX_POINTS = 600
MIN_SAMPLING_RATIO = 0.35

class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = torch.tensor(mean if mean is not None else[5.3311563533497974e-05, -7.49477039789781e-05], dtype=torch.float32)
        self.std = torch.tensor(std if std is not None else [0.049923088401556015, 0.040688566863536835] , dtype=torch.float32)
        
    def __call__(self, trajectory):
        return (trajectory - self.mean) / self.std

# 根据轨迹长度length计算采样比例
def logarithmic_sampling_ratio(length, min_points=36, max_points=600, min_ratio=0.35):
    """
    Logarithmic sampling ratio: decreases logarithmically from 1.0 to min_ratio.
    """
    if length <= min_points:
        return 1.0
    elif length >= max_points:
        return min_ratio
    else:
        ratio = 1.0 - math.log(length - min_points + 1) / math.log(
            max_points - min_points + 1
        ) * (1.0 - min_ratio)
        return max(ratio, min_ratio)


class TrajectoryDataset(Dataset):
    def __init__(self, data_path, max_len=200, transform=None, mask_ratio=0.5, mode="train", seed=2024):
        self.data_path = data_path
        self.transform = transform
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        if mode not in ("train", "val", "test"):
            raise ValueError(f"Unsupported dataset mode: {mode}")
        self.mode = mode
        self.seed = seed
        self.num_masked_points = int(self.max_len * self.mask_ratio)
        self.sampling_ratios = [
            logarithmic_sampling_ratio(length)
            for length in np.arange(MIN_POINTS, MAX_POINTS + 1, 1)
        ]
        self.mask_strategy = "random"
        # # load data from pickle file: a pandas DataFrame
        # try:
        #     with open(self.data_path, "rb") as f:
        #         self.data = pd.read_pickle(f)
        # except:
        #     raise FileNotFoundError(f"File not found: {self.data_path}")
        self.data_files = self._collect_pickle_files(self.data_path)
        if not self.data_files:
            raise FileNotFoundError(f"No pickle files found: {self.data_path}")

        self._single_file = len(self.data_files) == 1
        self._cache_file_idx = None
        self._cache_df = None
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
    

    def _collect_pickle_files(self, data_path):
        def collect_one(path_like):
            path = Path(path_like).expanduser()
            if path.is_file():
                return [path]
            if path.is_dir():
                return sorted(path.rglob("*.pkl"))
            return sorted(Path(p).expanduser() for p in glob.glob(str(path), recursive=True))

        if isinstance(data_path, (list, tuple)):
            files = []
            for item in data_path:
                files.extend(collect_one(item))
            return files
        return collect_one(data_path)

    def _load_dataframe(self, file_idx):
        if self._cache_file_idx == file_idx and self._cache_df is not None:
            return self._cache_df
        df = pd.read_pickle(self.data_files[file_idx])
        self._cache_file_idx = file_idx
        self._cache_df = df
        return df

    def _get_sample(self, idx):
        if self._single_file:
            return self.data.iloc[idx]
        file_idx = bisect.bisect_right(self.cum_lengths, idx)
        prev_cum = 0 if file_idx == 0 else self.cum_lengths[file_idx - 1]
        local_idx = idx - prev_cum
        df = self._load_dataframe(file_idx)
        return df.iloc[local_idx]

    def set_mask_ratio(self, mask_ratio):
        self.mask_ratio = mask_ratio
        # update num_masked_points
        self.num_masked_points = int(self.max_len * self.mask_ratio)
    
    def __len__(self):
        return self.cum_lengths[-1]

    @contextmanager
    def _rng_context(self, idx):
        if self.mode == "train":
            yield
            return

        py_state = random.getstate()
        np_state = np.random.get_state()
        seed = (self.seed + int(idx)) % (2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def __getitem__(self, idx):
        with self._rng_context(idx):
            return self._build_item(idx)

    def _build_item(self, idx):
        # Step 1: get a trajectory data
        traj_df = self.resample_trajectory(idx)
        # Step 2: get coordinates and time interval
        trajectory = torch.tensor(traj_df[[ "longitude","latitude"]].values, dtype=torch.float32)
        intervals = torch.tensor(traj_df["interval"].values, dtype=torch.float32)
         # Step 3: masking
        trajectory_length = len(traj_df)
        mask_strategy = random.random()
        if mask_strategy < 0.7:
            # self.mask_strategy == "random"
            mask = self.apply_random_mask(trajectory_length)
        elif mask_strategy < 0.85:
            # self.mask_strategy == "rdp"
            mask = self.apply_rdp_mask(trajectory)
        elif mask_strategy < 0.9:
            # self.mask_strategy == "block"
            mask = self.apply_block_mask(trajectory_length)
        else:
            # self.mask_strategy == "lastn"
            mask = self.apply_last_n_mask(trajectory_length)
        
        original = trajectory[0]
        trajectory = trajectory - original
        # apply transform
        if self.transform:
            trajectory = self.transform(trajectory)
            
        # Step 4: padding or truncate
        trajectory, attention_mask = self.pad_or_truncate(trajectory)
        intervals, _ = self.pad_or_truncate(intervals)

        # Step 5: make sure mask is consistent as trajectory
        mask = np.pad(mask, (0, max(0, self.max_len - trajectory_length)), constant_values=0)
        current_masked_points = mask.sum()
        if current_masked_points < self.num_masked_points:
            additional_mask_needed = self.num_masked_points - current_masked_points
            padding_indices = np.where(mask == 0)[0]
            additional_indices = np.random.choice(padding_indices, size=additional_mask_needed, replace=False)
            mask[additional_indices] = 1

    
        # Step 6: return information
        trajectory = trajectory.transpose(0, 1)
        mask_indices = torch.tensor(np.where(mask==1)[0]).long()
        
        inputs = {
            'trajectory': trajectory,  # [2,200]
            'attention_mask': attention_mask,  #[200]
            'original': original, #[2]
            'intervals': intervals, #[200]
            'indices': mask_indices  #[100]
        }
      
        return inputs

    # 进行随机遮蔽
    def apply_random_mask(self, trajectory_length):
        # calculate the number of points need to be masked
        trajectory_length = min(trajectory_length, self.max_len)
        num_points = int(trajectory_length * self.mask_ratio)

        mask = np.full(trajectory_length, False, dtype=bool)

        mask_indices = np.random.choice(
            trajectory_length, size=num_points, replace=False
        )

        mask[mask_indices] = True
        # mask一个trajectory_length长度的bool向量，遮蔽位置是True，未遮蔽位置是false。
        return mask

    # 进行last n遮蔽
    def apply_last_n_mask(self, trajectory_length, n=8):
        """
        Mask the last n points of the trajectory.

        :param trajectory_length: The length of the trajectory.
        :param n: The number of points to mask from the end.
        :return: A mask array of shape (trajectory_length,), where the last n points are masked.
        """
        # set a random points for mask
        n = np.random.randint(3, 8)
        trajectory_length = min(trajectory_length, self.max_len)
        num_points = int(trajectory_length * self.mask_ratio)
        additional_mask_points = num_points - n
        mask = np.full(trajectory_length, False, dtype=bool)
        mask[-n:] = True
        indices = np.arange(trajectory_length - n)
        additional_indices = np.random.choice(
            indices, size=additional_mask_points, replace=False
        )
        mask[additional_indices] = True
        return mask
    
    # 进行block遮蔽
    def apply_block_mask(self, trajectory_length, block_size=8):
        """
        Apply a block mask strategy by masking a continuous region of the trajectory.

        :param trajectory_length: The length of the trajectory.
        :param block_size: The number of points to mask in the block (if None, it will be randomly chosen).
        :return: A mask array of shape (trajectory_length,), where a continuous block is masked.
        """
        block_size = np.random.randint(5, 15)  

        trajectory_length = min(trajectory_length, self.max_len)
        num_points = int(trajectory_length * self.mask_ratio)
        additional_mask_points = num_points - block_size
         
        mask = np.full(trajectory_length, False, dtype=bool)
        
        start_idx = np.random.randint(0, trajectory_length - block_size + 1)
         
        mask[start_idx : start_idx + block_size] = True
        non_block_indices = np.where(~mask)[0]
        additional_indices = np.random.choice(
                non_block_indices, size=additional_mask_points, replace=False
            )
        mask[additional_indices] = True

        return mask
    
    # 进行rdp关键点遮蔽
    def apply_rdp_mask(self, trajectory, epsilon=1e-4):
        trajectory = trajectory[: self.max_len]
        trajectory_length = len(trajectory)
        num_points = int(trajectory_length * self.mask_ratio)

        # using RDP algorithm to dectect the key points
        rdp_mask = rdp(trajectory, epsilon=epsilon, return_mask=True)
        rdp_mask = np.array(rdp_mask)
        rdp_mask[0], rdp_mask[-1] = False, False

        num_rdp_mask = rdp_mask.sum()  

     
        if num_rdp_mask > num_points:
            
            indices = np.where(rdp_mask)[0]
            masked_indices = np.random.choice(indices, size=num_points, replace=False)
            rdp_mask[:] = False  
            rdp_mask[masked_indices] = True  

        
        elif num_rdp_mask < num_points:
            non_rdp_indices = np.where(~rdp_mask)[0]
            additional_mask_points = num_points - num_rdp_mask
            additional_indices = np.random.choice(
                non_rdp_indices, size=additional_mask_points, replace=False
            )
            rdp_mask[additional_indices] = True

        return rdp_mask

    def resample_trajectory(self, idx):
        sample = self._get_sample(idx)
        full_df = pd.DataFrame(
            {
                "time": sample["time"],
                "longitude": [point[1] for point in sample["trajectory"]],
                "latitude": [point[0] for point in sample["trajectory"]],
            }
        )
        trajectory_length = len(full_df)
        
        if random.random() < 0.3 and trajectory_length >=360:
            # interval consistent resamping
            # 实行时间间隔一致的轨迹重采样，
            if trajectory_length > 540: 
                sampling_interval = random.randint(8, 15)
            elif trajectory_length > 360: 
                sampling_interval = random.randint(6, 10)
            elif trajectory_length >= 240: 
                sampling_interval = random.randint(3, 6)

            full_df['time'] = pd.to_datetime(full_df['time'])
            full_df.set_index('time', inplace=True)

            resampled_df = full_df.resample(f'{sampling_interval}s').mean().reset_index()

            # 计算时间间隔
            resampled_df["interval"] = (
                resampled_df["time"].diff().dt.total_seconds().fillna(0).astype('float32')
            )
            # 最终的resampled_df -> time, longitude, latitude, interval
        else:
            # dynamic resamping with logarithmic ratio
            sampling_ratio = (
                1.0
                if trajectory_length <= MIN_POINTS
                else (
                    MIN_SAMPLING_RATIO
                    if trajectory_length >= MAX_POINTS
                    else self.sampling_ratios[trajectory_length - MIN_POINTS]
                )
            )

            num_sampled_points = int(trajectory_length * sampling_ratio)
            resampled_indices = np.random.choice(
                full_df.index, size=num_sampled_points, replace=False
            )
            resampled_df = full_df.loc[resampled_indices].sort_index().reset_index()
            resampled_df["interval"] = (
                resampled_df["time"].diff().dt.total_seconds().fillna(0).astype('float32')
            )

        # 最终的resampled_df -> time, longitude, latitude, interval，是重采样过的轨迹。
        return resampled_df

    def pad_or_truncate(self, tensor):

        seq_len = len(tensor)
        if seq_len > self.max_len:
            tensor = tensor[: self.max_len]
            attention_mask = torch.ones(self.max_len)
            return tensor, attention_mask
        else:
            if tensor.dim() == 2:
                padded_tensor = torch.zeros((self.max_len, 2), dtype=tensor.dtype)
                attention_mask = torch.zeros(self.max_len, dtype=torch.float32)
                attention_mask[:seq_len] = 1
            else:
                padded_tensor = torch.zeros(self.max_len, dtype=tensor.dtype)
                attention_mask = None
            padded_tensor[:seq_len] = tensor
        
        # 把序列统一到max_len的长度，其中attention_mask中，1表示真实点，0表示补齐
        return padded_tensor, attention_mask


class TrajectoryIterableDataset(IterableDataset):
    def __init__(
        self,
        data_path,
        max_len=200,
        transform=None,
        mask_ratio=0.5,
        mode="train",
        seed=2024,
        shuffle_buffer_size=4096,
        record_batch_size=1024,
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

    def _collect_parquet_files(self, data_path):
        def collect_one(path_like):
            path = Path(path_like).expanduser()
            if path.is_file():
                return [path]
            if path.is_dir():
                return sorted(path.rglob("*.parquet"))
            return sorted(Path(p).expanduser() for p in glob.glob(str(path), recursive=True))

        if isinstance(data_path, (list, tuple)):
            files = []
            for item in data_path:
                files.extend(collect_one(item))
            return files
        return collect_one(data_path)

    def _load_total_trajectories(self):
        metadata_paths = sorted(
            {
                path.parent / "metadata.json"
                for path in self.data_files
                if (path.parent / "metadata.json").exists()
            }
        )
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

    def __len__(self):
        if self.total_trajectories is None:
            raise TypeError("TrajectoryIterableDataset length is unknown.")
        return self.total_trajectories

    def __iter__(self):
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

    def _split_files(self, files):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        if world_size > 1:
            files = files[rank::world_size]

        worker_info = get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]
        return files

    def _iter_parquet_records(self, files):
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
            for record in batch.to_pylist():
                yield record

    def _shuffle_records(self, records, rng):
        buffer = []
        for record in records:
            buffer.append(record)
            if len(buffer) >= self.shuffle_buffer_size:
                idx = int(rng.integers(0, len(buffer)))
                yield buffer.pop(idx)

        rng.shuffle(buffer)
        for record in buffer:
            yield record

    def _build_item_from_record(self, record, rng):
        time = np.asarray(record["time"], dtype=np.int64)
        latitude = np.asarray(record["latitude"], dtype=np.float32)
        longitude = np.asarray(record["longitude"], dtype=np.float32)
        if len(time) == 0 or len(time) != len(latitude) or len(latitude) != len(longitude):
            return None

        valid = np.isfinite(latitude) & np.isfinite(longitude)
        if not valid.all():
            time = time[valid]
            latitude = latitude[valid]
            longitude = longitude[valid]
        if len(time) == 0:
            return None

        time, longitude, latitude = self._resample_arrays(time, longitude, latitude, rng)
        if len(time) == 0:
            return None

        trajectory_np = np.stack([longitude, latitude], axis=1).astype(np.float32)
        intervals_np = np.diff(time, prepend=time[0]).astype(np.float32)
        seq_len = min(len(trajectory_np), self.max_len)
        if seq_len == 0:
            return None

        short_mask = self._make_mask(trajectory_np[:seq_len], seq_len, rng)
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:seq_len] = short_mask[:seq_len]
        current_masked_points = int(mask.sum())
        if current_masked_points < self.num_masked_points:
            additional_mask_needed = self.num_masked_points - current_masked_points
            available_indices = np.where(~mask)[0]
            additional_indices = self._choice(available_indices, additional_mask_needed, rng)
            mask[additional_indices] = True

        trajectory = torch.as_tensor(trajectory_np[:seq_len], dtype=torch.float32)
        original = trajectory[0].clone()
        trajectory = trajectory - original
        if self.transform:
            trajectory = self.transform(trajectory)

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

    def _resample_arrays(self, time, longitude, latitude, rng):
        trajectory_length = len(time)
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

        sampling_ratio = (
            1.0
            if trajectory_length <= MIN_POINTS
            else (
                MIN_SAMPLING_RATIO
                if trajectory_length >= MAX_POINTS
                else self.sampling_ratios[trajectory_length - MIN_POINTS]
            )
        )
        num_sampled_points = max(1, int(trajectory_length * sampling_ratio))
        indices = rng.choice(trajectory_length, size=num_sampled_points, replace=False)
        indices.sort()
        return time[indices], longitude[indices], latitude[indices]

    def _interval_resample(self, time, longitude, latitude, sampling_interval):
        bins = time // sampling_interval
        unique_bins, inverse = np.unique(bins, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float32)
        longitude_sum = np.bincount(inverse, weights=longitude)
        latitude_sum = np.bincount(inverse, weights=latitude)
        resampled_time = (unique_bins * sampling_interval).astype(np.int64)
        resampled_longitude = (longitude_sum / counts).astype(np.float32)
        resampled_latitude = (latitude_sum / counts).astype(np.float32)
        return resampled_time, resampled_longitude, resampled_latitude

    def _make_mask(self, trajectory, trajectory_length, rng):
        mask_strategy = rng.random()
        if mask_strategy < 0.7:
            return self._random_mask(trajectory_length, rng)
        if mask_strategy < 0.85:
            return self._rdp_mask(trajectory, trajectory_length, rng)
        if mask_strategy < 0.9:
            return self._block_mask(trajectory_length, rng)
        return self._last_n_mask(trajectory_length, rng)

    def _random_mask(self, trajectory_length, rng):
        num_points = int(trajectory_length * self.mask_ratio)
        mask = np.full(trajectory_length, False, dtype=bool)
        mask[self._choice(np.arange(trajectory_length), num_points, rng)] = True
        return mask

    def _last_n_mask(self, trajectory_length, rng):
        num_points = int(trajectory_length * self.mask_ratio)
        n = trajectory_length if trajectory_length < 3 else int(rng.integers(3, min(8, trajectory_length) + 1))
        mask = np.full(trajectory_length, False, dtype=bool)
        mask[-n:] = True
        additional_mask_points = max(0, num_points - n)
        candidates = np.arange(max(0, trajectory_length - n))
        mask[self._choice(candidates, additional_mask_points, rng)] = True
        return mask

    def _block_mask(self, trajectory_length, rng):
        num_points = int(trajectory_length * self.mask_ratio)
        block_size = trajectory_length if trajectory_length < 5 else int(rng.integers(5, min(15, trajectory_length) + 1))
        mask = np.full(trajectory_length, False, dtype=bool)
        start_idx = int(rng.integers(0, trajectory_length - block_size + 1))
        mask[start_idx : start_idx + block_size] = True
        additional_mask_points = max(0, num_points - block_size)
        candidates = np.where(~mask)[0]
        mask[self._choice(candidates, additional_mask_points, rng)] = True
        return mask

    def _rdp_mask(self, trajectory, trajectory_length, rng, epsilon=1e-4):
        num_points = int(trajectory_length * self.mask_ratio)
        mask = np.full(trajectory_length, False, dtype=bool)
        if trajectory_length <= 2 or num_points <= 0:
            return mask

        rdp_mask = np.asarray(rdp(trajectory, epsilon=epsilon, return_mask=True), dtype=bool)
        rdp_mask = rdp_mask[:trajectory_length]
        rdp_mask[0], rdp_mask[-1] = False, False
        num_rdp_mask = int(rdp_mask.sum())

        if num_rdp_mask > num_points:
            indices = np.where(rdp_mask)[0]
            masked_indices = self._choice(indices, num_points, rng)
            rdp_mask[:] = False
            rdp_mask[masked_indices] = True
        elif num_rdp_mask < num_points:
            candidates = np.where(~rdp_mask)[0]
            additional_indices = self._choice(candidates, num_points - num_rdp_mask, rng)
            rdp_mask[additional_indices] = True

        return rdp_mask

    def _choice(self, candidates, size, rng):
        candidates = np.asarray(candidates)
        size = min(max(int(size), 0), len(candidates))
        if size == 0:
            return np.asarray([], dtype=np.int64)
        return rng.choice(candidates, size=size, replace=False)


class ShardBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, seed=2024):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not hasattr(dataset, "file_lengths") or not hasattr(dataset, "cum_lengths"):
            raise ValueError("ShardBatchSampler requires a TrajectoryDataset instance")

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

    def __len__(self):
        if self.drop_last:
            return sum(length // self.batch_size for length in self.dataset.file_lengths)
        return sum((length + self.batch_size - 1) // self.batch_size for length in self.dataset.file_lengths)


if __name__ == '__main__':
    file_path = '../data/worldtrace_sample.pkl'
    normalize_transform = Normalize()
    dataset = TrajectoryDataset(data_path=file_path, max_len = 200, transform=normalize_transform, mask_ratio=0.5)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True,num_workers=16)
    for i, batch in enumerate(dataloader):
        print("trajectory:", batch['trajectory'].shape)
        print("attention_mask:", batch['attention_mask'].shape)
        print("original_location:", batch['original'].shape)
        print("intervals:", batch['intervals'].shape)
        print("indices:", batch["indices"].shape)
        break
        
    print("Done!")
