# UniTraj: Learning a Universal Trajectory Foundation Model from Billion-Scale Worldwide Traces

## Overview
This repository contains the implementation of the NeurIPS 2025 accepted paper UniTraj. It is a universal trajectory foundation model designed to overcome the limitations of existing methods, such as task specificity, regional dependency, and data sensitivity. 


### Project Structure
```bash
.
├── data/                           # Data storage directory
│   └── worldtrace_sample.pkl      # WorldTrace dataset sample 
├── utils/                          # Utility modules and model components
│   ├── __init__.py                # Package initialization file
│   ├── unitraj.py                 # UniTraj model core implementation
│   ├── dataset.py                 # Dataset processing and loading, containing masking and resampling strategies
│   ├── config.py                  # Model and training configuration parameters
│   ├── logger.py                  # Colored logging system
│   └── EMA.py                     # Exponential Moving Average (EMA) helper
├── main.py                        # Main training script
├── load_see_data.ipynb           # Data loading and visualization notebook
├── requirements.txt              # Python dependency list
├── LICENSE                       # Open source license
└── README.md                     # Project documentation
```

### Requirements

This code is implemented in Python and based on the PyTorch framework. To ensure compatibility, please install the following dependencies:

#### Basic Environment
- **Python**: 3.8+
- **PyTorch**: 1.8.0+

#### Core Dependencies
- **numpy** (>=1.19.0): Numerical computation
- **pandas** (>=1.1.0): Data processing
- **matplotlib** (>=3.3.0): Data visualization
- **einops**: Simplified tensor operations
- **timm**: Vision Transformer model library
- **rdp**: Ramer-Douglas-Peucker algorithm (trajectory simplification)
- **colored**: Colored log output
- **folium**: Map visualization (for data display)


#### Running
you can run the code by running the following command:

```python

python main.py

```

## 📁 Dataset

<img src="./Logo.png" alt="Logo" style="zoom:10%;" />

The full WorldTrace dataset is released in 🤗 [Huggingface](https://huggingface.co/datasets/OpenTrace/WorldTrace) and  [Modelscope](https://www.modelscope.cn/datasets/opentrace/WorldTrace).

We also provide a sample of the WorldTrace dataset in the *data/directory* to help you get started quickly.

- data/worldtrace_sample.pkl: A subset of the dataset containing 1,000 trajectories.

- load_see_data.ipynb: A Jupyter Notebook that demonstrates how to load the sample data and visualize the trajectories.

### Converting trajectory data
If your data is stored as one CSV per trajectory with columns `time, matched_latitude, matched_longitude`, convert it to the original UniTraj pickle format:

```bash
python convert_matched_csv_to_pickle.py --input /path/to/csv_dir --output data/custom.pkl
```

For large pickle datasets, convert existing `.pkl` shards to streaming-friendly Parquet shards:

```bash
python convert_matched_csv_to_pickle.py \
  --input data/shards/train \
  --input-format pickle \
  --output data/parquet/train \
  --output-format parquet \
  --shard-size 5000
```

Repeat for validation data:

```bash
python convert_matched_csv_to_pickle.py \
  --input data/shards/val \
  --input-format pickle \
  --output data/parquet/val \
  --output-format parquet \
  --shard-size 5000
```

Parquet shards are written as `trajectories_000001.parquet`, `trajectories_000002.parquet`, and so on, plus a `metadata.json` file. The default training configuration reads these directories with `TrajectoryIterableDataset`.

To keep using pickle data, set the data format and paths in `utils/config.py`:

```python
'format': 'pickle',
'train_file_path': './data/shards/train',
'val_file_path': './data/shards/val',
```

Both pickle and Parquet paths support a single file, a shard directory, or a glob pattern.

## 📝 Citation
If you find our work useful in your research, please consider citing our paper:
```ini
@article{unitraj2025,
  title={UniTraj: Learning a Universal Trajectory Foundation Model from Billion-Scale Worldwide Traces},
  author={Zhu, Yuanshao and Yu, James Jianqiao and Zhao, Xiangyu and Zhou, Xun and Han, Liang and Wei, Xuetao and Liang, Yuxuan},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
```
