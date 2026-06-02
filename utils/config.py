args = {
    'data': {
        'dataset': 'worldtrace',
        'format': 'parquet',
        'traj_length': 200,
        'emb_dim': 128,
        'num_workers': 8,
        'sampler_seed': 2024,
        'shuffle_buffer_size': 4096,
        'record_batch_size': 1024,
        'prefetch_factor': 4,
        'steps_per_epoch': 1000,
        'val_steps_per_epoch': 100,
        # 'train_file_path': './data/worldtrace_sample.pkl',
        # 'val_file_path': './data/worldtrace_sample.pkl',
        # train/val paths can be parquet shard directories, a glob pattern, or a list.
        'train_file_path': './data/parquet/train',
        'val_file_path': './data/parquet/val',
    },
    'training': {
        'batch_size': 256,
        'grad_accum_steps': 2,
        'use_amp': True,
        'n_epochs': 1000,
        'patience': 5,
    },
}
