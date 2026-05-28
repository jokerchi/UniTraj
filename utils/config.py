args = {
    'data': {
        'dataset': 'worldtrace',
        'traj_length': 200,
        'emb_dim': 128,
        'num_workers': 16,
        'sampler_seed': 2024,
        # 'train_file_path': './data/worldtrace_sample.pkl',
        # 'val_file_path': './data/worldtrace_sample.pkl',
        # train/val paths can be a .pkl file, a shard directory, a glob pattern, or a list.
        'train_file_path': './data/shards/train',
        'val_file_path': './data/shards/val',
    },
    'training': {
        'batch_size': 512,
        'n_epochs': 1000,
    },
}
