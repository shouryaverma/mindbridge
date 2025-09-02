#!/usr/bin/env python3
"""
MindEye V2 Inference Script with Rectified Flow

This script performs inference using a trained MindEye V2 model with rectified flow
to generate both unrefined and enhanced reconstructions from fMRI data.

Usage:
    python inference.py --model_name=subj01_finetuned --target_subject=1 --cache_dir=/path/to/cache
"""

import os
import sys
import argparse
import numpy as np
import h5py
from tqdm import tqdm
import webdataset as wds

import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/
try:
    sys.path.append('generative_models/')
    import sgm
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
    from generative_models.sgm.models.diffusion import DiffusionEngine
    from generative_models.sgm.util import append_dims
    from omegaconf import OmegaConf
except ImportError:
    print("Please ensure the 'generative-models' repository from Stability-AI is cloned and in the correct path.")
    sys.exit(1)

# Local module imports
import utils
from syn_models_rect import BrainNetwork, PriorNetwork, BrainRectifiedFlow

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
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)

class CLIPConverter(nn.Module):
    """Convert OpenCLIP bigG embeddings to CLIP L embeddings for text generation."""
    def __init__(self, clip_seq_dim=256, clip_text_seq_dim=257, 
                 clip_emb_dim=1024, clip_text_emb_dim=1024):
        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        return x

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MindEye V2 Inference Configuration")
    
    # Essential Args
    parser.add_argument("--model_name", type=str, required=True, 
                       help="Model name (will load from ../train_logs/model_name)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the MindEye V2 dataset")
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Directory for storing cached models")
    parser.add_argument("--target_subject", type=int, required=True, choices=[1,2,3,4,5,6,7,8],
                       help="Target subject for inference")
    
    # Model Architecture
    parser.add_argument("--hidden_dim", type=int, default=1024, 
                       help="Hidden dimension of the brain network")
    parser.add_argument("--n_blocks", type=int, default=4,
                       help="Number of blocks in the brain network")
    parser.add_argument("--blurry_recon", action=argparse.BooleanOptionalAction, default=True,
                       help="Enable blurry image reconstruction")
    
    # Inference Settings
    parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True,
                       help="Use the new, larger test set")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of steps for rectified flow inference")
    parser.add_argument("--cond_scale", type=float, default=1.0,
                       help="Conditioning scale for classifier-free guidance")
    parser.add_argument("--enhance_reconstructions", action=argparse.BooleanOptionalAction, default=True,
                       help="Generate enhanced reconstructions using SDXL")
    
    # System
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    
    return parser.parse_args()

def load_test_data(data_path, target_subject, new_test=True):
    """Load test data for the target subject."""
    # Load voxel data
    with h5py.File(f'{data_path}/betas_all_subj0{target_subject}_fp32_renorm.hdf5', 'r') as f:
        voxels = torch.from_numpy(f['betas'][:]).to("cpu").to(torch.float16)
    
    num_voxels = voxels.shape[1]
    print(f"Loaded {num_voxels} voxels for subj0{target_subject}")
    
    # Determine test set size and URL
    if new_test:
        test_sizes = {3: 2371, 4: 2188, 6: 2371, 8: 2188}
        num_test = test_sizes.get(target_subject, 3000)
        test_url = f"{data_path}/wds/subj0{target_subject}/new_test/0.tar"
    else:
        test_sizes = {3: 2113, 4: 1985, 6: 2113, 8: 1985}
        num_test = test_sizes.get(target_subject, 2770)
        test_url = f"{data_path}/wds/subj0{target_subject}/test/0.tar"
    
    print(f"Loading test data from: {test_url}")
    
    # Load test data
    def my_split_by_node(urls): return urls
    
    test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node) \
        .decode("torch") \
        .rename(behav="behav.npy", past_behav="past_behav.npy", 
               future_behav="future_behav.npy", olds_behav="olds_behav.npy") \
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, 
                                         shuffle=False, drop_last=True, pin_memory=True)
    
    # Process test indices
    for test_i, (behav, _, _, _) in enumerate(test_dl):
        test_voxel_indices = behav[:, 0, 5].cpu().long().numpy()
        test_image_indices = behav[:, 0, 0].cpu().long().numpy()
    
    print(f"Loaded {len(test_voxel_indices)} test samples")
    
    return voxels, test_voxel_indices, test_image_indices, num_voxels

def load_model(model_name, num_voxels, hidden_dim, n_blocks, device):
    """Load the trained MindEye model."""
    # Model architecture constants
    clip_seq_dim = 256
    clip_emb_dim = 1024
    
    # Initialize model
    model = MindEyeModule()
    model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)
    model.backbone = BrainNetwork(
        h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks,
        clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=True, clip_scale=1.0
    )
    
    # Rectified Flow Prior
    prior_network = PriorNetwork(
        dim=clip_emb_dim, depth=6, dim_head=64, heads=clip_emb_dim // 64,
        causal=False, num_tokens=clip_seq_dim, learned_query_mode="pos_emb"
    )
    model.rectified_flow = BrainRectifiedFlow(
        net=prior_network, image_embed_dim=clip_emb_dim,
        condition_on_text_encodings=False, text_cond_drop_prob=0.2
    )
    
    model.to(device)
    
    # Load checkpoint
    outdir = os.path.abspath(f'../train_logs/{model_name}')
    
    # Try different checkpoint naming conventions
    possible_names = ['last.pth', f'finetuned_subj0{args.target_subject}_last.pth']
    checkpoint_path = None
    
    for name in possible_names:
        path = os.path.join(outdir, name)
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {outdir}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded successfully!")
        
        # Get target ridge index if available
        target_ridge_idx = checkpoint.get('target_ridge_idx', 0)
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Try DeepSpeed format
        try:
            import deepspeed
            state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
                checkpoint_dir=outdir, tag='last')
            model.load_state_dict(state_dict, strict=False)
            target_ridge_idx = 0
            print("DeepSpeed checkpoint loaded!")
        except Exception as ds_e:
            raise Exception(f"Failed to load checkpoint: {e}, DeepSpeed attempt: {ds_e}")
    
    return model, target_ridge_idx

def setup_auxiliary_models(cache_dir, device):
    """Setup auxiliary models for text generation and VAE."""
    from diffusers import AutoencoderKL
    
    # VAE for blurry reconstruction
    autoenc = AutoencoderKL.from_pretrained(f'{cache_dir}/sd-vae-ft-mse').to(device)
    autoenc.eval().requires_grad_(False)
    
    # Text generation models
    from transformers import AutoProcessor, AutoModelForCausalLM
    try:
        # Try to import custom GIT model
        from modeling_git import GitForCausalLMClipEmb
        processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
        clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    except ImportError:
        print("Custom GIT model not found, using standard transformers")
        processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
        clip_text_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
    
    clip_text_model.to(device).eval().requires_grad_(False)
    
    # CLIP converter
    clip_convert = CLIPConverter()
    try:
        converter_path = f"{cache_dir}/bigG_to_L_epoch8.pth"
        state_dict = torch.load(converter_path, map_location='cpu')['model_state_dict']
        clip_convert.load_state_dict(state_dict, strict=True)
    except FileNotFoundError:
        print("Warning: CLIP converter checkpoint not found, using random weights")
    
    clip_convert.to(device)
    
    return autoenc, processor, clip_text_model, clip_convert

def setup_unclip_engine(cache_dir, device):
    """Setup unCLIP diffusion engine."""
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    
    # Modify sampling steps
    unclip_params["sampler_config"]["params"]["num_steps"] = 38
    
    diffusion_engine = DiffusionEngine(**unclip_params)
    diffusion_engine.eval().requires_grad_(False).to(device)
    
    # Load checkpoint
    ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        diffusion_engine.load_state_dict(ckpt['state_dict'])
        print(f"Loaded unCLIP checkpoint from {ckpt_path}")
    else:
        print("Warning: unCLIP checkpoint not found")
    
    # Setup conditioning
    batch = {
        "jpg": torch.randn(1, 3, 1, 1).to(device),
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device)
    }
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    
    return diffusion_engine, vector_suffix

def generate_reconstructions(model, target_ridge_idx, test_voxel_indices, test_image_indices, 
                           voxels, images, device, args):
    """Generate reconstructions from fMRI data."""
    model.eval()
    
    all_recons = []
    all_blurry_recons = []
    all_pred_captions = []
    all_clip_voxels = []
    
    unique_image_indices = np.unique(test_image_indices)
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for batch_start in tqdm(range(0, len(unique_image_indices), args.batch_size)):
            batch_imgs = unique_image_indices[batch_start:batch_start + args.batch_size]
            
            # Prepare voxel data (average across 3 repetitions)
            batch_voxels = []
            for img_idx in batch_imgs:
                locs = np.where(test_image_indices == img_idx)[0]
                
                # Ensure we have 3 repetitions
                if len(locs) == 1:
                    locs = np.repeat(locs, 3)
                elif len(locs) == 2:
                    locs = np.tile(locs, 2)[:3]
                else:
                    locs = locs[:3]
                
                # Average across repetitions
                voxel_data = voxels[test_voxel_indices[locs]].mean(dim=0, keepdim=True)
                batch_voxels.append(voxel_data)
            
            batch_voxels = torch.stack(batch_voxels).to(device)
            
            # Forward pass
            voxel_ridge = model.ridge(batch_voxels, target_ridge_idx)
            backbone_features, clip_voxels, blurry_image_enc = model.backbone(voxel_ridge)
            
            # Store retrieval outputs
            all_clip_voxels.append(clip_voxels.cpu())
            
            # Generate samples using Rectified Flow
            prior_samples = model.rectified_flow.sample(
                text_embed=backbone_features,
                num_steps=args.num_inference_steps,
                cond_scale=args.cond_scale
            )
            
            # Store results
            all_recons.append(prior_samples.cpu())
            
            if args.blurry_recon:
                blurry_recon, _ = blurry_image_enc
                all_blurry_recons.append(blurry_recon.cpu())
    
    # Concatenate results
    all_recons = torch.cat(all_recons, dim=0)
    all_clip_voxels = torch.cat(all_clip_voxels, dim=0)
    
    if args.blurry_recon:
        all_blurry_recons = torch.cat(all_blurry_recons, dim=0)
    else:
        all_blurry_recons = None
    
    return all_recons, all_blurry_recons, all_clip_voxels

def main():
    args = parse_args()
    
    # Setup
    utils.seed_everything(args.seed)
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = accelerator.device
    
    # Create output directory
    os.makedirs("evals", exist_ok=True)
    os.makedirs(f"evals/{args.model_name}", exist_ok=True)
    
    print(f"Running inference for {args.model_name} on subject {args.target_subject}")
    print(f"Device: {device}")
    
    # Load test data
    voxels, test_voxel_indices, test_image_indices, num_voxels = load_test_data(
        args.data_path, args.target_subject, args.new_test)
    
    # Load COCO images
    images_file = h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r')
    images = images_file['images']
    
    # Load model
    model, target_ridge_idx = load_model(
        args.model_name, num_voxels, args.hidden_dim, args.n_blocks, device)
    
    print(f"Model loaded with {utils.count_params(model)/1e6:.2f}M parameters")
    print(f"Target ridge index: {target_ridge_idx}")
    
    # Generate reconstructions
    all_recons, all_blurry_recons, all_clip_voxels = generate_reconstructions(
        model, target_ridge_idx, test_voxel_indices, test_image_indices, 
        voxels, images, device, args)
    
    # Resize and save results
    imsize = 256
    all_recons = transforms.Resize((imsize, imsize))(all_recons).float()
    
    # Save outputs
    output_dir = f"evals/{args.model_name}"
    torch.save(all_recons, f"{output_dir}/{args.model_name}_all_recons.pt")
    torch.save(all_clip_voxels, f"{output_dir}/{args.model_name}_all_clipvoxels.pt")
    
    if all_blurry_recons is not None:
        all_blurry_recons = transforms.Resize((imsize, imsize))(all_blurry_recons).float()
        torch.save(all_blurry_recons, f"{output_dir}/{args.model_name}_all_blurryrecons.pt")
    
    print(f"Saved reconstruction outputs to {output_dir}")
    print(f"Reconstructions shape: {all_recons.shape}")

if __name__ == "__main__":
    main()