#!/bin/bash
#SBATCH --job-name=debug_multigpu
#SBATCH --account=pccr
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-gpu=14
#SBATCH --time=6:00:00
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out
#SBATCH --mail-user=lu1202@purdue.edu
#SBATCH --mail-type=ALL

# ============================================
# Environment Setup
# ============================================
export HF_HOME=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/cache

# ============================================
# Distributed Training Configuration
# ============================================
export NUM_GPUS=3  # Must match --gpus-per-node
export BATCH_SIZE=28
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Generate random port to avoid conflicts with other jobs
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))

# Get SLURM node information
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

# ============================================
# Debug Output (helpful for troubleshooting)
# ============================================
echo "============================================"
echo "DISTRIBUTED TRAINING INFO"
echo "============================================"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "COUNT_NODE=${COUNT_NODE}"
echo "WORLD_SIZE=$((NUM_GPUS * COUNT_NODE))"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "============================================"

# ============================================
# Launch Training
# ============================================
model_name="debug_fullModel_3gpu_multisubj_batch28"
echo model_name=${model_name}
accelerate launch \
    --num_processes=$(($NUM_GPUS * $COUNT_NODE)) \
    --num_machines=$COUNT_NODE \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --mixed_precision=fp16 \
    train_multigpu.py \
    --data_path=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/datasets \
    --cache_dir=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/cache \
    --model_name=${model_name} \
    --multi_subject \
    --subj=1 \
    --batch_size=${BATCH_SIZE} \
    --num_epochs=150 \
    --use_prior \
    --blurry_recon \
    --ckpt_saving \
    --ckpt_interval=999 \
    --seed=42 \
    --wandb_log \
    --wandb_project='mindbridge'