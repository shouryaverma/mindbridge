#!/usr/bin/env python3
"""
MindEye V2 Shared Backbone Pretraining Script

This script pretrains the shared backbone on multiple subjects' data.
The pretrained model can then be fine-tuned on a new subject using the fine-tuning script.

Usage:
    $ accelerate launch train_shared_backbone.py \
        --data_path=/path/to/dataset \
        --train_subjects 2 5 7 \
        --num_sessions=32 \
        --model_name=shared_backbone_pretrained
"""

import os
import sys
import argparse
import random
import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import webdataset as wds
from accelerate import Accelerator
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
import kornia
from kornia.augmentation.container import AugmentationSequential

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/
try:
    sys.path.append('generative_models/')
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
except ImportError:
    print("Please ensure the 'generative-models' repository from Stability-AI is cloned and in the correct path.")
    sys.exit(1)

# Local module imports
import utils
from models_rect import BrainNetwork, PriorNetwork, BrainRectifiedFlow

# Enable TF32 for faster matmul operations on supported hardware
torch.backends.cuda.matmul.allow_tf32 = True

class MindEyeModule(nn.Module):
    """Main container module for all MindEye components."""
    def __init__(self):
        super(MindEyeModule, self).__init__()

    def forward(self, x):
        return x

class RidgeRegression(nn.Module):
    """
    A Ridge Regression module implemented as a PyTorch layer.
    It supports multiple linear layers for different subjects.
    """
    def __init__(self, input_sizes, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = nn.ModuleList([
            nn.Linear(input_size, out_features) for input_size in input_sizes
        ])

    def forward(self, x, subj_idx):
        # Select the linear layer corresponding to the subject index
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)

def parse_args():
    """Defines and parses command-line arguments for the shared backbone pretraining script."""
    parser = argparse.ArgumentParser(description="MindEye V2 Shared Backbone Pretraining Configuration")
    
    # --- Essential Args ---
    parser.add_argument("--model_name", type=str, default="shared_backbone_pretrained", help="Name for saving checkpoints and logging.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the MindEye V2 dataset.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for storing cached models (e.g., VAE).")
    parser.add_argument("--train_subjects", type=int, nargs='+', default=[2, 5, 7], help="Subject IDs to use for pretraining.")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size.")

    # --- Model & Training Strategy ---
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension of the brain network.")
    parser.add_argument("--n_blocks", type=int, default=4, help="Number of blocks in the brain network.")
    parser.add_argument("--use_prior", action=argparse.BooleanOptionalAction, default=True, help="Train the diffusion prior.")
    parser.add_argument("--blurry_recon", action=argparse.BooleanOptionalAction, default=True, help="Enable blurry image reconstruction loss.")
    
    # --- Loss Scaling ---
    parser.add_argument("--clip_scale", type=float, default=1.0, help="Scaling factor for the CLIP contrastive loss.")
    parser.add_argument("--prior_scale", type=float, default=30.0, help="Scaling factor for the diffusion prior loss.")
    parser.add_argument("--blur_scale", type=float, default=0.5, help="Scaling factor for the blurry reconstruction loss.")
    
    # --- Learning Rate & Optimizer ---
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Maximum learning rate for the scheduler.")
    parser.add_argument("--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'], help="Learning rate scheduler type.")
    parser.add_argument("--mixup_pct", type=float, default=0.33, help="Percentage of training to use MixCo before switching to SoftCLIP.")
    
    # --- Data & Augmentation ---
    parser.add_argument("--num_sessions", type=int, default=32, help="Number of fMRI sessions to use for training per subject.")
    parser.add_argument("--val_sessions", type=int, default=8, help="Number of sessions to reserve for validation per subject.")
    parser.add_argument("--use_image_aug", action=argparse.BooleanOptionalAction, default=False, help="Enable image augmentations.")

    # --- Logging & Checkpointing ---
    parser.add_argument("--wandb_log", action=argparse.BooleanOptionalAction, default=False, help="Enable logging to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="mindeye-v2-pretraining", help="W&B project name.")
    parser.add_argument("--ckpt_saving", action=argparse.BooleanOptionalAction, default=True, help="Enable saving model checkpoints.")
    parser.add_argument("--ckpt_interval", type=int, default=5, help="Epoch interval for saving checkpoints.")
    
    # --- System & Reproducibility ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser.parse_args()

def main():
    args = parse_args()
    utils.seed_everything(args.seed)

    # --- Accelerator and Distributed Setup ---
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = accelerator.device
    num_devices = accelerator.num_processes
    
    # Batch size per device
    batch_size = args.batch_size
    global_batch_size = batch_size * num_devices

    accelerator.print("--- Shared Backbone Pretraining Configuration ---")
    accelerator.print(f"Distributed: {accelerator.state.distributed_type != 'NO'}, Num Devices: {num_devices}")
    accelerator.print(f"Per-device Batch Size: {batch_size}, Global Batch Size: {global_batch_size}")
    accelerator.print(f"Mixed Precision: {accelerator.state.mixed_precision}")
    accelerator.print(f"Model: {args.model_name}")
    accelerator.print(f"Training Subjects: {args.train_subjects}")
    
    # --- Output Directory ---
    outdir = os.path.abspath(f'../train_logs/{args.model_name}')
    if accelerator.is_main_process and args.ckpt_saving:
        os.makedirs(outdir, exist_ok=True)

    # --- Data Loading ---
    accelerator.print("\n--- Loading Training Data ---")
    
    def my_split_by_node(urls): return urls
    
    # Calculate iterations per epoch
    train_batch_size = batch_size // len(args.train_subjects)
    num_samples_per_epoch = (750 * args.num_sessions) // num_devices
    num_iterations_per_epoch = num_samples_per_epoch // (train_batch_size * len(args.train_subjects))

    accelerator.print(f"Training sessions per subject: {args.num_sessions}")
    accelerator.print(f"Validation sessions per subject: {args.val_sessions}")
    accelerator.print(f"Batch size per subject: {train_batch_size}")
    accelerator.print(f"Iterations per epoch: {num_iterations_per_epoch}")

    train_data, train_dl, val_data, val_dl = {}, {}, {}, {}
    voxels, num_voxels, num_voxels_list = {}, {}, []
    nsessions_allsubj = np.array([0, 40, 40, 32, 30, 40, 32, 40, 30])  # Subj 0 is a placeholder

    for s in args.train_subjects:
        # Training data (first N sessions)
        train_url = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{args.num_sessions-1}" + "}.tar"
        
        # Validation data (last M sessions)
        val_start = nsessions_allsubj[s] - args.val_sessions
        val_url = f"{args.data_path}/wds/subj0{s}/train/" + "{" + f"{val_start}.." + f"{nsessions_allsubj[s]-1}" + "}.tar"

        # Training dataloader
        train_data[f'subj0{s}'] = wds.WebDataset(train_url, resampled=True, nodesplitter=my_split_by_node) \
            .shuffle(750, initial=1500, rng=random.Random(args.seed)) \
            .decode("torch") \
            .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
            .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
        
        train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=train_batch_size, shuffle=False, drop_last=True, pin_memory=True)

        # Validation dataloader
        val_data[f'subj0{s}'] = wds.WebDataset(val_url, resampled=False, nodesplitter=my_split_by_node) \
            .decode("torch") \
            .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
            .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
        
        val_dl[f'subj0{s}'] = torch.utils.data.DataLoader(val_data[f'subj0{s}'], batch_size=train_batch_size, shuffle=False, drop_last=True, pin_memory=True)

        # Load voxel data
        with h5py.File(f'{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r') as f:
            betas = torch.from_numpy(f['betas'][:]).to("cpu").to(torch.float16)
        num_voxels_list.append(betas.shape[1])
        num_voxels[f'subj0{s}'] = betas.shape[1]
        voxels[f'subj0{s}'] = betas
        accelerator.print(f"Loaded training data and {num_voxels[f'subj0{s}']} voxels for subj0{s}")
    
    # Load all 73k COCO images
    with h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r') as f:
        images = f['images'][:]
    accelerator.print("Loaded all 73k COCO images into memory.")

    # --- Model Initialization ---
    accelerator.print("\n--- Initializing Models ---")

    local_clip_path = '/depot/natallah/data/shourya/mindbridge/MindEyeV2/src/cache/CLIP_L/open_clip_pytorch_model.bin'

    # CLIP Image Encoder
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-L-14", 
        version=local_clip_path, 
        output_tokens=True, 
        only_tokens=True
    ).to(device)
    clip_seq_dim = 256
    clip_emb_dim = 1024

    # Main MindEye model
    model = MindEyeModule()
    model.ridge = RidgeRegression(num_voxels_list, out_features=args.hidden_dim)
    model.backbone = BrainNetwork(
        h=args.hidden_dim, in_dim=args.hidden_dim, seq_len=1, n_blocks=args.n_blocks,
        clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=args.blurry_recon, clip_scale=args.clip_scale
    )

    # Rectified Flow Prior model
    if args.use_prior:
        prior_network = PriorNetwork(
            dim=clip_emb_dim, depth=6, dim_head=64, heads=clip_emb_dim // 64,
            causal=False, num_tokens=clip_seq_dim, learned_query_mode="pos_emb"
        )
        model.rectified_flow = BrainRectifiedFlow(
            net=prior_network, image_embed_dim=clip_emb_dim,
            condition_on_text_encodings=False, text_cond_drop_prob=0.2
        )

    # Models for Blurry Reconstruction
    autoenc, cnx = None, None
    if args.blurry_recon:
        from autoencoder.convnext import ConvnextXL
        
        autoenc = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
        autoenc.eval().requires_grad_(False)
        
        cnx = ConvnextXL(f'{args.cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth').to(device)
        cnx.eval().requires_grad_(False)

    accelerator.print(f"Total model parameters: {utils.count_params(model)/1e6:.2f}M")
    
    # --- Optimizer and Scheduler ---
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_params = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.use_prior:
        opt_params.extend([
            {'params': [p for n, p in model.rectified_flow.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in model.rectified_flow.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ])
    
    optimizer = torch.optim.AdamW(opt_params, lr=args.max_lr)
    
    total_steps = num_iterations_per_epoch * args.num_epochs
    if args.lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    else:  # 'cycle'
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.max_lr, total_steps=total_steps,
            final_div_factor=1000, pct_start=2/args.num_epochs
        )

    # --- Prepare everything with Accelerator ---
    train_dls_prepared = [train_dl[f'subj0{s}'] for s in args.train_subjects]
    val_dls_prepared = [val_dl[f'subj0{s}'] for s in args.train_subjects]
    
    model, optimizer, *train_dls_prepared, lr_scheduler = accelerator.prepare(
        model, optimizer, *train_dls_prepared, lr_scheduler
    )
    val_dls_prepared = [accelerator.prepare(val_dl) for val_dl in val_dls_prepared]
    
    # --- Weights & Biases Logging ---
    if accelerator.is_main_process and args.wandb_log:
        import wandb
        wandb_config = {**vars(args), "global_batch_size": global_batch_size}
        wandb.init(
            project=args.wandb_project,
            name=args.model_name,
            config=wandb_config,
            resume="allow",
        )

    # --- Training Loop ---
    accelerator.print(f"\n--- Starting shared backbone pretraining ---")
    progress_bar = tqdm(range(args.num_epochs), disable=not accelerator.is_main_process)
    
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
    
    for epoch in progress_bar:
        model.train()
        
        # --- Pre-load all batches for the epoch for speed ---
        voxel_iters, perm_iters, betas_iters, select_iters = {}, {}, {}, {}
        skipped_iterations = set()
        image_iters = torch.zeros(num_iterations_per_epoch, global_batch_size, 3, 224, 224, dtype=torch.float16)

        for s_idx, s in enumerate(args.train_subjects):
            train_dl_iter = iter(train_dls_prepared[s_idx])
            for i in range(num_iterations_per_epoch):
                behav, _, _, _ = next(train_dl_iter)
                
                # Image data
                img_indices = behav[:,0,0].cpu().long().numpy()
                unique_indices, sorted_idx = np.unique(img_indices, return_index=True)
                if len(unique_indices) != len(img_indices): 
                    skipped_iterations.add(i)
                    continue
                
                image_batch = torch.from_numpy(images[unique_indices]).to(torch.float16)
                image_iters[i, s_idx*train_batch_size:(s_idx+1)*train_batch_size] = image_batch
                
                # Voxel data
                voxel_indices = behav[:,0,5].cpu().long().numpy()[sorted_idx]
                voxel_batch = voxels[f'subj0{s}'][voxel_indices].unsqueeze(1)
                
                if epoch < int(args.mixup_pct * args.num_epochs):
                    voxel_batch, perm, betas, select = utils.mixco(voxel_batch)
                    perm_iters[f"s{s}_i{i}"] = perm
                    betas_iters[f"s{s}_i{i}"] = betas
                    select_iters[f"s{s}_i{i}"] = select

                voxel_iters[f"s{s}_i{i}"] = voxel_batch

        # --- Train on pre-loaded batches ---
        epoch_loss = 0
        num_batches = 0
        
        for i in range(num_iterations_per_epoch):
            if i in skipped_iterations:
                continue
            optimizer.zero_grad()
            
            # Collate data from all subjects for the current iteration
            voxel_list = [voxel_iters[f"s{s}_i{i}"].to(device) for s in args.train_subjects]
            image_batch = image_iters[i].to(device)
            
            if args.use_image_aug:
                # Augmentations would be applied here if enabled
                pass
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                clip_target = clip_img_embedder(image_batch)
                
                # BiMixCo logic for early epochs
                if epoch < int(args.mixup_pct * args.num_epochs):
                    perm_list = [perm_iters[f"s{s}_i{i}"].to(device) for s in args.train_subjects]
                    betas_list = [betas_iters[f"s{s}_i{i}"].to(device) for s in args.train_subjects]
                    select_list = [select_iters[f"s{s}_i{i}"].to(device) for s in args.train_subjects]
                    perm, betas, select = torch.cat(perm_list), torch.cat(betas_list), torch.cat(select_list)
            
                voxel_ridge_list = [model.ridge(voxel_list[s_idx], s_idx) for s_idx, s in enumerate(args.train_subjects)]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)
                
                backbone_features, clip_voxels, blurry_image_enc = model.backbone(voxel_ridge)
                
                # --- Calculate Losses ---
                total_loss = 0.
                
                # Prior Loss
                if args.use_prior:
                    loss_prior, _ = model.rectified_flow(text_embed=backbone_features, image_embed=clip_target)
                    total_loss += args.prior_scale * loss_prior
                
                # CLIP Loss
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                
                if epoch < int(args.mixup_pct * args.num_epochs):
                    loss_clip = utils.mixco_nce(clip_voxels_norm, clip_target_norm, temp=0.006, perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch - int(args.mixup_pct * args.num_epochs)]
                    loss_clip = utils.soft_clip_loss(clip_voxels_norm, clip_target_norm, temp=epoch_temp)
                total_loss += args.clip_scale * loss_clip
                
                # Blurry Reconstruction Loss
                if args.blurry_recon:
                    image_enc_pred, _ = blurry_image_enc
                    with torch.no_grad():
                        image_enc = autoenc.encode(2 * image_batch - 1).latent_dist.mode() * 0.18215
                    loss_blurry = l1(image_enc_pred, image_enc)
                    total_loss += args.blur_scale * loss_blurry
                
                # Backward pass
                utils.check_loss(total_loss)
                accelerator.backward(total_loss)
                optimizer.step()
                lr_scheduler.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
        
        # --- End of Epoch Evaluation ---
        if accelerator.is_main_process:
            model.eval()
            val_loss = 0
            val_batches = 0
            val_fwd_acc = 0
            val_bwd_acc = 0
            
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                # Evaluate on validation sets from all training subjects
                for s_idx, s in enumerate(args.train_subjects):
                    val_dl_iter = iter(val_dls_prepared[s_idx])
                    
                    # Sample a fixed number of batches for validation
                    max_val_batches = 10  # Limit validation batches per subject
                    val_batch_count = 0
                    
                    while val_batch_count < max_val_batches:
                        try:
                            val_behav, _, _, _ = next(val_dl_iter)
                            val_batch_count += 1
                        except StopIteration:
                            # No more batches available for this subject
                            break
                            
                        # Process validation batch
                        val_img_indices = val_behav[:,0,0].cpu().long().numpy()
                        val_unique_indices, val_sorted_idx = np.unique(val_img_indices, return_index=True)
                        if len(val_unique_indices) != len(val_img_indices):
                            continue
                            
                        val_image_batch = torch.from_numpy(images[val_unique_indices]).to(torch.float16).to(device)
                        val_voxel_indices = val_behav[:,0,5].cpu().long().numpy()[val_sorted_idx]
                        val_voxel_batch = voxels[f'subj0{s}'][val_voxel_indices].unsqueeze(1).to(device)
                        
                        # Forward pass
                        val_clip_target = clip_img_embedder(val_image_batch)
                        val_voxel_ridge = accelerator.unwrap_model(model).ridge(val_voxel_batch, s_idx)
                        _, val_clip_voxels, _ = accelerator.unwrap_model(model).backbone(val_voxel_ridge)
                        
                        # Calculate validation metrics
                        val_clip_voxels_norm = nn.functional.normalize(val_clip_voxels.flatten(1), dim=-1)
                        val_clip_target_norm = nn.functional.normalize(val_clip_target.flatten(1), dim=-1)
                        
                        val_loss_clip = utils.soft_clip_loss(val_clip_voxels_norm, val_clip_target_norm, temp=0.006)
                        val_loss += val_loss_clip.item()
                        val_batches += 1
                        
                        # Retrieval metrics
                        labels = torch.arange(len(val_clip_voxels_norm)).to(device)
                        sims = utils.batchwise_cosine_similarity(val_clip_voxels_norm, val_clip_target_norm)
                        val_fwd_acc += utils.topk(sims, labels, k=1).item()
                        val_bwd_acc += utils.topk(sims.T, labels, k=1).item()
            
            # Average validation metrics
            if val_batches > 0:
                val_loss /= val_batches
                val_fwd_acc /= val_batches
                val_bwd_acc /= val_batches
            
            logs = {
                "epoch": epoch,
                "train/loss": epoch_loss / num_batches if num_batches > 0 else 0,
                "lr": lr_scheduler.get_last_lr()[0]
            }

            # add individual loss componants
            if args.use_prior:
                logs["train/prior_loss"] = loss_prior.item()
                logs["train/prior_loss_scaled"] = (args.prior_scale * loss_prior).item()

            logs["train/clip_loss"] = loss_clip.item()
            logs["train/clip_loss_scaled"] = (args.clip_scale * loss_clip).item()

            if args.blurry_recon:
                logs["train/blur_loss"] = loss_blurry.item()
                logs["train/blur_loss_scaled"] = (args.blur_scale * loss_blurry).item()

            # Validation metrics
            logs.update({
                "val/clip_loss": val_loss,
                "val/fwd_acc": val_fwd_acc,
                "val/bwd_acc": val_bwd_acc,
            })

            # Log loss ratios for monitoring balance
            if args.use_prior:
                logs["train/prior_ratio"] = (args.prior_scale * loss_prior / total_loss).item()
            logs["train/clip_ratio"] = (args.clip_scale * loss_clip / total_loss).item()
            if args.blurry_recon:
                logs["train/blur_ratio"] = (args.blur_scale * loss_blurry / total_loss).item()
            
            progress_bar.set_postfix(**logs)
            if args.wandb_log:
                wandb.log(logs)

        # --- Save Checkpoint ---
        if args.ckpt_saving and (epoch % args.ckpt_interval == 0 or epoch == args.num_epochs - 1):
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(outdir, 'shared_backbone_last.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'train_subjects': args.train_subjects,
                    'num_voxels_list': num_voxels_list,
                    'args': vars(args)
                }, save_path)
                accelerator.print(f"\n--- Saved checkpoint at {save_path} ---")

        accelerator.wait_for_everyone()
    
    accelerator.print("\n=== Shared Backbone Pretraining Finished! ===")
    if args.wandb_log and accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()