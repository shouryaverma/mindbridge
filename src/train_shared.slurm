#!/bin/bash
#SBATCH --account=pccr
#SBATCH --qos=preemptible
#SBATCH --partition=ai
#SBATCH --job-name=mb_pretrain
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue
#SBATCH --exclusive

# Make sure you activate your fmri environment
module load conda
conda activate /depot/natallah/data/roy/conda_envs/fmri

cd /depot/natallah/data/shourya/mindbridge/MindEyeV2/src

# multisubject pretraining
model_name="shared_backbone_pretrained"
echo model_name=${model_name}
accelerate launch --num_processes=$((1 * 1)) \
--num_machines=1 \
--mixed_precision=fp16 syn_train_shared.py \
--data_path=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/datasets \
--cache_dir=/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/cache \
--model_name=${model_name} \
--batch_size=28 \
--num_epochs=150 \
--num_sessions=40 \
--val_sessions=1 \
--hidden_dim=1024 \
--n_blocks=4 \
--use_prior \
--prior_scale=1 \
--blurry_recon \
--blur_scale=2 \
--clip_scale=1 \
--max_lr=3e-4 \
--mixup_pct=0.33 \
--ckpt_saving \
--ckpt_interval=10 \
--wandb_log \
--wandb_project='mindbridge'