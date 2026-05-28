import torch
import pickle, random, math
import bisect
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from rdp import rdp
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, data_path, max_len=200, transform=None, mask_ratio=0.5):
        self.data_path = data_path
        self.transform = transform
        self.max_len = max_len
        self.mask_ratio = mask_ratio
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
        if isinstance(data_path, (list, tuple)):
            return [Path(p) for p in data_path]
        path = Path(data_path)
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(path.rglob("*.pkl"))
        return sorted(Path(p) for p in glob.glob(str(data_path), recursive=True))

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

    def __getitem__(self, idx):
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
 