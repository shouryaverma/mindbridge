#!/usr/bin/env python3
"""
MindEye V2 Single Subject Fine-tuning Script

This script fine-tunes a pretrained shared backbone on a single target subject's data.
The pretrained backbone should be trained using train_shared_backbone.py first.

Usage:
    $ accelerate launch finetune_single_subject.py \
        --data_path=/path/to/dataset \
        --pretrained_model=/path/to/shared_backbone_last.pth \
        --target_subject=1 \
        --num_sessions=1 \
        --model_name=subj01_finetuned
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
    """Defines and parses command-line arguments for the fine-tuning script."""
    parser = argparse.ArgumentParser(description="MindEye V2 Single Subject Fine-tuning Configuration")
    
    # --- Essential Args ---
    parser.add_argument("--model_name", type=str, default="subj_finetuned", help="Name for saving checkpoints and logging.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the MindEye V2 dataset.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory for storing cached models (e.g., VAE).")
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to pretrained shared backbone checkpoint.")
    parser.add_argument("--target_subject", type=int, required=True, choices=[1, 2, 5, 7], help="Target subject ID to fine-tune on.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of fine-tuning epochs.")
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
    parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate for fine-tuning (lower than pretraining).")
    parser.add_argument("--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'], help="Learning rate scheduler type.")
    parser.add_argument("--mixup_pct", type=float, default=0.33, help="Percentage of training to use MixCo before switching to SoftCLIP.")
    
    # --- Data & Augmentation ---
    parser.add_argument("--num_sessions", type=int, default=1, help="Number of fMRI sessions to use for fine-tuning.")
    parser.add_argument("--use_image_aug", action=argparse.BooleanOptionalAction, default=False, help="Enable image augmentations.")
    parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True, help="Use the new, larger test set.")

    # --- Logging & Checkpointing ---
    parser.add_argument("--wandb_log", action=argparse.BooleanOptionalAction, default=False, help="Enable logging to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="mindeye-v2-finetuning", help="W&B project name.")
    parser.add_argument("--ckpt_saving", action=argparse.BooleanOptionalAction, default=True, help="Enable saving model checkpoints.")
    parser.add_argument("--ckpt_interval", type=int, default=10, help="Epoch interval for saving checkpoints.")
    
    # --- System & Reproducibility ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser.parse_args()

def load_pretrained_model(model, pretrained_path, target_subject, accelerator):
    """Load pretrained shared backbone and add target subject's ridge layer if needed."""
    accelerator.print(f"Loading pretrained model from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    pretrained_state_dict = checkpoint['model_state_dict']
    train_subjects = checkpoint['train_subjects']
    num_voxels_list = checkpoint['num_voxels_list']
    
    accelerator.print(f"Pretrained on subjects: {train_subjects}")
    
    # Check if target subject was in pretraining
    if target_subject in train_subjects:
        # Target subject already has a ridge layer
        accelerator.print(f"Target subject {target_subject} was in pretraining, using existing ridge layer")
        model.load_state_dict(pretrained_state_dict, strict=True)
        target_ridge_idx = train_subjects.index(target_subject)
    else:
        # Need to add new ridge layer for target subject
        accelerator.print(f"Target subject {target_subject} not in pretraining, adding new ridge layer")
        
        # Load target subject's voxel data to get dimensions
        data_path = checkpoint['args']['data_path']
        with h5py.File(f'{data_path}/betas_all_subj0{target_subject}_fp32_renorm.hdf5', 'r') as f:
            target_voxel_count = f['betas'].shape[1]
        
        # Extend ridge regression with new layer
        extended_voxel_counts = num_voxels_list + [target_voxel_count]
        new_ridge = RidgeRegression(extended_voxel_counts, model.ridge.out_features)
        
        # Copy existing ridge weights
        for i, linear in enumerate(model.ridge.linears):
            new_ridge.linears[i].load_state_dict(linear.state_dict())
        
        # Replace ridge in model
        model.ridge = new_ridge
        
        # Load pretrained weights (ridge layer will be partially loaded)
        model.load_state_dict(pretrained_state_dict, strict=False)
        target_ridge_idx = len(train_subjects)  # New ridge layer index
        
        accelerator.print(f"Added new ridge layer for {target_voxel_count} voxels")
    
    return target_ridge_idx

def main():
    args = parse_args()
    utils.seed_everything(args.seed)

    # --- Accelerator and Distributed Setup ---
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = accelerator.device
    num_devices = accelerator.num_processes
    
    batch_size = args.batch_size
    global_batch_size = batch_size * num_devices

    accelerator.print("--- Single Subject Fine-tuning Configuration ---")
    accelerator.print(f"Distributed: {accelerator.state.distributed_type != 'NO'}, Num Devices: {num_devices}")
    accelerator.print(f"Per-device Batch Size: {batch_size}, Global Batch Size: {global_batch_size}")
    accelerator.print(f"Mixed Precision: {accelerator.state.mixed_precision}")
    accelerator.print(f"Model: {args.model_name}, Target Subject: {args.target_subject}")
    
    # --- Output Directory ---
    outdir = os.path.abspath(f'../train_logs/{args.model_name}')
    if accelerator.is_main_process and args.ckpt_saving:
        os.makedirs(outdir, exist_ok=True)

    # --- Data Loading ---
    accelerator.print("\n--- Loading Target Subject Data ---")
    
    def my_split_by_node(urls): return urls
    
    num_samples_per_epoch = (750 * args.num_sessions) // num_devices
    num_iterations_per_epoch = num_samples_per_epoch // batch_size

    accelerator.print(f"Fine-tuning sessions: {args.num_sessions}")
    accelerator.print(f"Iterations per epoch: {num_iterations_per_epoch}")

    # Training dataloader for target subject
    train_url = f"{args.data_path}/wds/subj0{args.target_subject}/train/" + "{0.." + f"{args.num_sessions-1}" + "}.tar"
    train_data = wds.WebDataset(train_url, resampled=True, nodesplitter=my_split_by_node) \
        .shuffle(750, initial=1500, rng=random.Random(args.seed)) \
        .decode("torch") \
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)

    # Load target subject's voxel data
    with h5py.File(f'{args.data_path}/betas_all_subj0{args.target_subject}_fp32_renorm.hdf5', 'r') as f:
        voxels = torch.from_numpy(f['betas'][:]).to("cpu").to(torch.float16)
    accelerator.print(f"Loaded {voxels.shape[1]} voxels for target subject {args.target_subject}")

    # Validation dataloader (test set)
    test_url = f"{args.data_path}/wds/subj0{args.target_subject}/"
    if args.new_test:
        test_url += "new_test/0.tar"
        num_test_samples = {1: 3000, 2: 3000, 3: 2371, 4: 2188, 5: 3000, 6: 2371, 7: 3000, 8: 2188}
        num_test = num_test_samples[args.target_subject]
    else:
        test_url += "test/0.tar"
        num_test_samples = {1: 2770, 2: 2770, 3: 2113, 4: 1985, 5: 2770, 6: 2113, 7: 2770, 8: 1985}
        num_test = num_test_samples[args.target_subject]
        
    test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node) \
        .decode("torch") \
        .rename(behav="behav.npy") \
        .to_tuple("behav")
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
    accelerator.print(f"Loaded validation data with {num_test} samples.")
    
    # Load all 73k COCO images
    with h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r') as f:
        images = f['images'][:]
    accelerator.print("Loaded all 73k COCO images into memory.")

    # --- Model Initialization ---
    accelerator.print("\n--- Initializing Models ---")

    local_vae_path = f'{args.cache_dir}/sd-vae-ft-mse'
    local_clip_path = f'{args.cache_dir}/CLIP_L/open_clip_pytorch_model.bin'
    
    # CLIP Image Encoder
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-L-14", 
        version=local_clip_path, 
        output_tokens=True, 
        only_tokens=True
    ).to(device)
    clip_seq_dim = 256
    clip_emb_dim = 1024

    # Main MindEye model (will be loaded from pretrained)
    model = MindEyeModule()
    
    # Initialize with dummy ridge (will be replaced when loading pretrained)
    model.ridge = RidgeRegression([1000], out_features=args.hidden_dim)  # Dummy
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

    # Load pretrained model and get target ridge index
    target_ridge_idx = load_pretrained_model(model, args.pretrained_model, args.target_subject, accelerator)

    # Models for Blurry Reconstruction
    autoenc, cnx = None, None
    if args.blurry_recon:
        from autoencoder.convnext import ConvnextXL
        
        autoenc = AutoencoderKL.from_pretrained(local_vae_path).to(device)
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
    model, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, lr_scheduler
    )
    
    # --- Weights & Biases Logging ---
    if accelerator.is_main_process and args.wandb_log:
        import wandb
        wandb_config = {**vars(args), "global_batch_size": global_batch_size, "target_ridge_idx": target_ridge_idx}
        wandb.init(
            project=args.wandb_project,
            name=args.model_name,
            config=wandb_config,
            resume="allow",
        )

    # --- Training Loop ---
    accelerator.print(f"\n--- Starting fine-tuning for subject {args.target_subject} ---")
    progress_bar = tqdm(range(args.num_epochs), disable=not accelerator.is_main_process)
    
    test_image, test_voxel = None, None
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
    
    for epoch in progress_bar:
        model.train()
        
        # --- Pre-load all batches for the epoch for speed ---
        voxel_iters, perm_iters, betas_iters, select_iters = {}, {}, {}, {}
        skipped_iterations = set()
        image_iters = torch.zeros(num_iterations_per_epoch, batch_size, 3, 224, 224, dtype=torch.float16)

        train_dl_iter = iter(train_dl)
        for i in range(num_iterations_per_epoch):
            behav, _, _, _ = next(train_dl_iter)
            
            # Image data
            img_indices = behav[:,0,0].cpu().long().numpy()
            unique_indices, sorted_idx = np.unique(img_indices, return_index=True)
            if len(unique_indices) != len(img_indices): 
                skipped_iterations.add(i)
                continue
            
            image_batch = torch.from_numpy(images[unique_indices]).to(torch.float16)
            image_iters[i] = image_batch
            
            # Voxel data
            voxel_indices = behav[:,0,5].cpu().long().numpy()[sorted_idx]
            voxel_batch = voxels[voxel_indices].unsqueeze(1)
            
            if epoch < int(args.mixup_pct * args.num_epochs):
                voxel_batch, perm, betas, select = utils.mixco(voxel_batch)
                perm_iters[f"i{i}"] = perm
                betas_iters[f"i{i}"] = betas
                select_iters[f"i{i}"] = select

            voxel_iters[f"i{i}"] = voxel_batch

        # --- Train on pre-loaded batches ---
        epoch_loss = 0
        num_batches = 0
        
        for i in range(num_iterations_per_epoch):
            if i in skipped_iterations:
                continue
            optimizer.zero_grad()
            
            # Get data for this iteration
            voxel_batch = voxel_iters[f"i{i}"].to(device)
            image_batch = image_iters[i].to(device)
            
            if args.use_image_aug:
                # Augmentations would be applied here if enabled
                pass
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                clip_target = clip_img_embedder(image_batch)
                
                # BiMixCo logic for early epochs
                if epoch < int(args.mixup_pct * args.num_epochs):
                    perm = perm_iters[f"i{i}"].to(device)
                    betas = betas_iters[f"i{i}"].to(device)
                    select = select_iters[f"i{i}"].to(device)
            
                # Forward pass through target subject's ridge layer
                voxel_ridge = model.ridge(voxel_batch, target_ridge_idx)
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
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                # Prepare test data by averaging 3 repeats of the same image
                if test_image is None:
                    test_behav = next(iter(test_dl))[0]
                    voxel = voxels[test_behav[:,0,5].cpu().long()].unsqueeze(1)
                    image_indices = test_behav[:,0,0].cpu().long()
                    
                    unique_images = torch.unique(image_indices)
                    test_voxel_list, test_image_list = [], []
                    for img_idx in unique_images:
                        locs = torch.where(img_idx == image_indices)[0]
                        if len(locs) < 3: continue  # Skip images with fewer than 3 repeats
                        test_voxel_list.append(voxel[locs[:3]].unsqueeze(0))
                        test_image_list.append(torch.from_numpy(images[img_idx]).unsqueeze(0))
                    
                    test_voxel = torch.cat(test_voxel_list, dim=0).to(device)
                    test_image = torch.cat(test_image_list, dim=0).to(device)
                
                # Evaluate on a subset for speed
                eval_indices = torch.arange(len(test_voxel))[:300]
                voxel_eval = test_voxel[eval_indices].to(next(model.parameters()).dtype)
                image_eval = test_image[eval_indices].to(torch.float16)
                
                clip_target_eval = clip_img_embedder(image_eval)
                
                # Average voxel embeddings over 3 repeats
                clip_voxels_avg = 0
                for rep in range(3):
                    voxel_ridge_eval = accelerator.unwrap_model(model).ridge(voxel_eval[:, rep], target_ridge_idx)
                    _, clip_voxels_eval, _ = accelerator.unwrap_model(model).backbone(voxel_ridge_eval)
                    clip_voxels_avg += clip_voxels_eval
                clip_voxels_avg /= 3
                
                clip_voxels_norm_eval = nn.functional.normalize(clip_voxels_avg.flatten(1), dim=-1)
                clip_target_norm_eval = nn.functional.normalize(clip_target_eval.flatten(1), dim=-1)
                
                # --- Calculate Evaluation Metrics ---
                labels = torch.arange(len(clip_voxels_norm_eval)).to(device)
                sims = utils.batchwise_cosine_similarity(clip_voxels_norm_eval, clip_target_norm_eval)
                test_fwd_pct_correct = utils.topk(sims, labels, k=1).item()
                test_bwd_pct_correct = utils.topk(sims.T, labels, k=1).item()
                
                eval_loss_clip = utils.soft_clip_loss(clip_voxels_norm_eval, clip_target_norm_eval, temp=0.006)
            
            logs = {
                "epoch": epoch,
                "train/loss": epoch_loss / num_batches if num_batches > 0 else 0,
                "lr": lr_scheduler.get_last_lr()[0]
            }

            # Add individual loss components (same as shared backbone)
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
                "val/clip_loss": eval_loss_clip.item(),
                "val/fwd_acc": test_fwd_pct_correct,
                "val/bwd_acc": test_bwd_pct_correct,
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
                save_path = os.path.join(outdir, f'finetuned_subj0{args.target_subject}_last.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'target_subject': args.target_subject,
                    'target_ridge_idx': target_ridge_idx,
                    'args': vars(args)
                }, save_path)
                accelerator.print(f"\n--- Saved checkpoint at {save_path} ---")

        accelerator.wait_for_everyone()
    
    accelerator.print(f"\n=== Fine-tuning for subject {args.target_subject} finished! ===")
    if args.wandb_log and accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()