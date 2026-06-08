"""
Stage 2: Domain Prompt Fine-tuning Configuration.

Loads a pretrained Stage-1 checkpoint, freezes the base model, and trains
only the DomainPromptNetwork.

Usage:
    python main.py --config configs/stage2_prompt.py

IMPORTANT: Update ``pretrained_checkpoint`` to point to your actual Stage-1
checkpoint file before running.
"""

args = {
    "data": {
        "dataset": "worldtrace",
        "format": "parquet",
        "traj_length": 200,
        "emb_dim": 128,
        "num_workers": 12,
        "sampler_seed": 2024,
        "shuffle_buffer_size": 4096,
        "record_batch_size": 32768,
        "prefetch_factor": 3,
        "steps_per_epoch": 900,
        "val_steps_per_epoch": 50,
        "train_file_path": "./data/parquet/train",
        "val_file_path": "./data/parquet/val",
    },
    "training": {
        "batch_size": 1024,
        "grad_accum_steps": 1,
        "use_amp": True,
        "n_epochs": 200,
        "patience": 10,
        # ===== Stage 2: Domain Prompt Fine-tuning =====
        "stage": 2,
        # Path to your Stage-1 checkpoint (REQUIRED — update this path)
        "pretrained_checkpoint": (
            "/home/stu252261/UniTraj/UniTraj/worldtrace_bs=1024/06-05-19-01-50/models/best_model_epoch_137.pt"
        ),
        # Freeze all base model parameters (only domain_prompt_net is trainable)
        "freeze_base": True,
        # Number of domain prompt tokens injected into the encoder
        "num_domain_prompts": 8,
        # Learning rate for the domain prompt network
        "prompt_lr": 1e-3,
    },
}
