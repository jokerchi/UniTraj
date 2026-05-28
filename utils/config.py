args = {
    'data': {
        'dataset': 'worldtrace',
        'traj_length': 200,
        'emb_dim': 128,
        'num_workers': 4,
        # 'train_file_path': './data/worldtrace_sample.pkl',
        # 'val_file_path': './data/worldtrace_sample.pkl',
        # You can also pass a list of .pkl files, e.g.:
        'train_file_path': ['~/UniTraj/data/shards/trajectories_000001.pkl'],
        'val_file_path': ['~/UniTraj/data/shards/trajectories_000011.pkl'],
    },
    'training': {
        'batch_size': 1024,
        'n_epochs': 1000,
    },
}
