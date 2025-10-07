#!/bin/bash
#SBATCH --job-name=run_1gpu
#SBATCH --account=pccr
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=12:00:00
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out
#SBATCH --mail-user=lu1202@purdue.edu
#SBATCH --mail-type=ALL

export HF_HOME=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/cache

accelerate launch --num_processes=$((1 * 1)) \
    --mixed_precision=fp16 train_1gpu.py \
    --data_path=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/datasets \
    --cache_dir=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/cache \
    --model_name=full_model \
    --multi_subject \
    --subj=1 \
    --batch_size=7 \
    --num_epochs=150 \
    --num_sessions=40 \
    --use_prior \
    --blurry_recon \
    --ckpt_saving \
    --ckpt_interval=999 \
    --seed=42 \
    --wandb_log \
    --wandb_project='mindbridge'

