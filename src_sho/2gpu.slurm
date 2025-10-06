#!/bin/bash
#SBATCH --job-name=debug_2gpu
#SBATCH --account=pccr
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=14
#SBATCH --time=3:00:00
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out
#SBATCH --mail-user=lu1202@purdue.edu
#SBATCH --mail-type=ALL

export HF_HOME=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/cache

accelerate launch --num_processes=$((2 * 1)) \
    --mixed_precision=fp16 train_1gpu.py \
    --data_path=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/datasets \
    --cache_dir=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/cache \
    --model_name=debug_unified_shared_smallModel_subSessions_2gpu \
    --multi_subject \
    --subj=1 \
    --batch_size=14 \
    --num_epochs=10 \
    --num_sessions=20 \
    --hidden_dim=64 \
    --n_blocks=2 \
    --use_prior \
    --blurry_recon \
    --ckpt_saving \
    --ckpt_interval=2 \
    --seed=42 \
    --wandb_log \
    --wandb_project='mindbridge'

