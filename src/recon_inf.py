#!/usr/bin/env python3
"""
MindEye V2 Reconstruction Inference Script

This script loads a fine-tuned MindEye V2 model and generates initial reconstructions
from test fMRI data. It produces the unrefined reconstructions, predicted captions,
and intermediate outputs needed for the final evaluation pipeline.

Usage:
    python recon_inference.py \
        --model_path=/path/to/finetuned_subj01_last.pth \
        --data_path=/path/to/dataset \
        --subject=1 \
        --output_dir=./inference_outputs
"""

import os
import sys
import argparse
import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
import webdataset as wds
from diffusers import AutoencoderKL

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/
try:
    sys.path.append('generative_models/')
    import sgm
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
    from generative_models.sgm.models.diffusion import DiffusionEngine
    from omegaconf import OmegaConf
except ImportError:
    print("Please ensure the 'generative-models' repository from Stability-AI is cloned and in the correct path.")
    sys.exit(1)

# Local module imports
import utils
from models_rect import BrainNetwork, PriorNetwork, BrainRectifiedFlow

# Enable TF32 for faster matmul operations
torch.backends.cuda.matmul.allow_tf32 = True

class MindEyeModule(nn.Module):
    """Main container module for all MindEye components."""
    def __init__(self):
        super(MindEyeModule, self).__init__()
        
    def forward(self, x):
        return x

class RidgeRegression(nn.Module):
    """Ridge Regression module with multiple linear layers for different subjects."""
    def __init__(self, input_sizes, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = nn.ModuleList([
            nn.Linear(input_size, out_features) for input_size in input_sizes
        ])

    def forward(self, x, subj_idx):
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)

class CLIPConverter(nn.Module):
    """Converts OpenCLIP bigG embeddings to CLIP L embeddings for GIT captioning."""
    def __init__(self):
        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(256, 257)  # clip_seq_dim to clip_text_seq_dim
        self.linear2 = nn.Linear(1024, 1024)  # clip_emb_dim to clip_text_emb_dim
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        return x

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MindEye V2 Reconstruction Inference")
    
    # Essential arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to MindEye V2 dataset")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for cached models")
    parser.add_argument("--subject", type=int, required=True, choices=[1,2,3,4,5,6,7,8],
                        help="Target subject ID")
    parser.add_argument("--output_dir", type=str, default="./evals",
                        help="Output directory for results")
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per reconstruction")
    parser.add_argument("--prior_timesteps", type=int, default=20,
                        help="Number of timesteps for rectified flow sampling")
    parser.add_argument("--new_test", action='store_true', default=True,
                        help="Use new larger test set")
    
    # System
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()

def load_finetuned_model(model_path, device):
    """Load fine-tuned model and extract configuration."""
    print(f"Loading fine-tuned model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    config = checkpoint['args']
    target_ridge_idx = checkpoint['target_ridge_idx']
    
    # Initialize model
    model = MindEyeModule()
    
    # Get voxel dimensions from checkpoint
    if 'num_voxels_list' in checkpoint:
        num_voxels_list = checkpoint['num_voxels_list']
    else:
        # Fallback: infer from model state dict
        ridge_weights = [v for k, v in checkpoint['model_state_dict'].items() 
                        if k.startswith('ridge.linears') and k.endswith('.weight')]
        num_voxels_list = [w.shape[1] for w in ridge_weights]
    
    # Build model architecture
    model.ridge = RidgeRegression(num_voxels_list, out_features=config['hidden_dim'])
    model.backbone = BrainNetwork(
        h=config['hidden_dim'], 
        in_dim=config['hidden_dim'], 
        seq_len=1, 
        n_blocks=config['n_blocks'],
        clip_size=1024,  # clip_emb_dim
        out_dim=1024 * 256,  # clip_emb_dim * clip_seq_dim
        blurry_recon=config['blurry_recon'], 
        clip_scale=config['clip_scale']
    )
    
    if config['use_prior']:
        prior_network = PriorNetwork(
            dim=1024, depth=6, dim_head=64, heads=16,
            causal=False, num_tokens=256, learned_query_mode="pos_emb"
        )
        model.rectified_flow = BrainRectifiedFlow(
            net=prior_network, image_embed_dim=1024,
            condition_on_text_encodings=False, text_cond_drop_prob=0.2
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval().requires_grad_(False)
    
    return model, config, target_ridge_idx

def load_test_data(data_path, subject, new_test=True):
    """Load and prepare test data."""
    # Load voxel data
    with h5py.File(f'{data_path}/betas_all_subj0{subject}_fp32_renorm.hdf5', 'r') as f:
        voxels = torch.from_numpy(f['betas'][:]).to("cpu").to(torch.float16)
    
    # Load COCO images
    with h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r') as f:
        images = f['images'][:]
    
    # Test data counts per subject
    if new_test:
        num_test_samples = {1: 3000, 2: 3000, 3: 2371, 4: 2188, 
                           5: 3000, 6: 2371, 7: 3000, 8: 2188}
        test_url = f"{data_path}/wds/subj0{subject}/new_test/0.tar"
    else:
        num_test_samples = {1: 2770, 2: 2770, 3: 2113, 4: 1985, 
                           5: 2770, 6: 2113, 7: 2770, 8: 1985}
        test_url = f"{data_path}/wds/subj0{subject}/test/0.tar"
    
    num_test = num_test_samples[subject]
    
    # Load test indices
    def my_split_by_node(urls): return urls
    test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node) \
        .decode("torch") \
        .rename(behav="behav.npy") \
        .to_tuple("behav")
    
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, 
                                         shuffle=False, drop_last=True, pin_memory=True)
    
    # Extract test indices
    test_behav = next(iter(test_dl))[0]
    test_voxels_idx = test_behav[:, 0, 5].cpu().long().numpy()
    test_images_idx = test_behav[:, 0, 0].cpu().long().numpy()
    
    print(f"Loaded {len(voxels)} voxel samples and {len(images)} images")
    print(f"Test set: {len(test_images_idx)} samples, {len(np.unique(test_images_idx))} unique images")
    
    return voxels, images, test_voxels_idx, test_images_idx

def setup_auxiliary_models(cache_dir, device):
    """Setup unCLIP, captioning, and VAE models."""
    
    # CLIP Image Encoder for targets
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version=f"{cache_dir}/open_clip_pytorch_model.bin", 
        output_tokens=True,
        only_tokens=True
    ).to(device)
    
    # unCLIP Diffusion Engine
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    
    # Replace the problematic first_stage_config
    unclip_params["first_stage_config"] = {
        "target": "sgm.models.autoencoder.AutoencoderKL",  # Use base class
        "params": {
            "embed_dim": 4,
            "monitor": "val/rec_loss",
            "ddconfig": {
                "double_z": True,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 2, 4, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0
            }
        }
    }

    diffusion_engine = DiffusionEngine(
        network_config=unclip_params["network_config"],
        denoiser_config=unclip_params["denoiser_config"],
        first_stage_config=unclip_params["first_stage_config"],
        conditioner_config=unclip_params["conditioner_config"],
        sampler_config=unclip_params["sampler_config"],
        scale_factor=unclip_params["scale_factor"],
        disable_first_stage_autocast=unclip_params["disable_first_stage_autocast"]
    ).to(device).eval().requires_grad_(False)
    
    # Load unCLIP checkpoint
    ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])
    
    # Get vector suffix for unCLIP
    batch = {"jpg": torch.randn(1,3,1,1).to(device),
            "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
            "crop_coords_top_left": torch.zeros(1, 2).to(device)}
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    
    # VAE for blurry reconstruction
    vae_cache_path = f'{cache_dir}/sd-vae-ft-mse'
    autoenc = AutoencoderKL.from_pretrained(vae_cache_path).to(device).eval().requires_grad_(False)
    
    # Setup text captioning models
    from transformers import AutoProcessor
    from modeling_git import GitForCausalLMClipEmb
    
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    clip_text_model.to(device).eval().requires_grad_(False)
    
    # Load OpenCLIP to CLIP converter
    clip_converter = CLIPConverter()
    converter_state = torch.load(f"{cache_dir}/bigG_to_L_epoch8.pth", map_location='cpu')
    clip_converter.load_state_dict(converter_state['model_state_dict'])
    clip_converter.to(device)
    
    return (clip_img_embedder, diffusion_engine, vector_suffix, autoenc, 
            processor, clip_text_model, clip_converter)

def main():
    args = parse_args()
    utils.seed_everything(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create output directory
    model_name = os.path.basename(args.model_path).replace('.pth', '')
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    model, config, target_ridge_idx = load_finetuned_model(args.model_path, device)
    voxels, images, test_voxels_idx, test_images_idx = load_test_data(
        args.data_path, args.subject, args.new_test)
    
    # Setup auxiliary models
    (clip_img_embedder, diffusion_engine, vector_suffix, autoenc, 
     processor, clip_text_model, clip_converter) = setup_auxiliary_models(args.cache_dir, device)
    
    print(f"Starting inference for subject {args.subject}...")
    
    # Initialize output lists
    all_recons = []
    all_blurry_recons = []
    all_pred_captions = []
    all_clip_voxels = []
    all_backbone_features = []
    all_prior_outputs = []
    
    # Get unique images and their indices
    unique_images_idx = np.unique(test_images_idx)
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for batch_start in tqdm(range(0, len(unique_images_idx), args.batch_size)):
            batch_end = min(batch_start + args.batch_size, len(unique_images_idx))
            batch_unique_imgs = unique_images_idx[batch_start:batch_end]
            
            # Prepare voxel data for this batch
            voxel_batch_list = []
            for uniq_img in batch_unique_imgs:
                # Find all repetitions of this image
                locs = np.where(test_images_idx == uniq_img)[0]
                
                # Handle cases with fewer than 3 repetitions
                if len(locs) == 1:
                    locs = np.repeat(locs, 3)
                elif len(locs) == 2:
                    locs = np.tile(locs, 2)[:3]
                else:
                    locs = locs[:3]  # Use first 3 if more than 3
                
                assert len(locs) == 3
                voxel_batch_list.append(voxels[test_voxels_idx[locs]])
            
            voxel_batch = torch.stack(voxel_batch_list).to(device)  # [batch, 3_repeats, num_voxels]
            
            # Average over 3 repetitions through the model
            backbone_avg = 0
            clip_voxels_avg = 0
            blurry_enc_avg = 0
            
            for rep in range(3):
                voxel_ridge = model.ridge(voxel_batch[:, [rep]], target_ridge_idx)
                backbone_rep, clip_voxels_rep, blurry_enc_rep = model.backbone(voxel_ridge)
                
                backbone_avg += backbone_rep
                clip_voxels_avg += clip_voxels_rep
                if config['blurry_recon']:
                    blurry_enc_avg += blurry_enc_rep[0]  # Take first element of tuple
            
            # Average across repetitions
            backbone_features = backbone_avg / 3
            clip_voxels = clip_voxels_avg / 3
            if config['blurry_recon']:
                blurry_image_enc = blurry_enc_avg / 3
            
            # Store intermediate outputs
            all_backbone_features.append(backbone_features.cpu())
            all_clip_voxels.append(clip_voxels.cpu())
            
            # Generate CLIP embeddings via rectified flow
            if config['use_prior']:
                prior_out = model.rectified_flow.sample(
                    text_embed=backbone_features,
                    num_steps=args.prior_timesteps,
                    cond_scale=1.0
                )
            else:
                prior_out = backbone_features
            
            all_prior_outputs.append(prior_out.cpu())
            
            # Generate image captions
            pred_caption_emb = clip_converter(prior_out)
            generated_ids = clip_text_model.generate(
                pixel_values=pred_caption_emb, 
                max_length=20
            )
            generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_pred_captions.extend(generated_captions)
            
            # Generate unrefined reconstructions via unCLIP
            for i in range(len(prior_out)):
                samples = utils.unclip_recon(
                    prior_out[[i]],
                    diffusion_engine,
                    vector_suffix,
                    num_samples=args.num_samples
                )
                all_recons.append(samples.cpu())
            
            # Generate blurry reconstructions if enabled
            if config['blurry_recon']:
                blurry_images = autoenc.decode(blurry_image_enc / 0.18215).sample
                blurry_images = (blurry_images / 2 + 0.5).clamp(0, 1)
                all_blurry_recons.append(blurry_images.cpu())
    
    # Concatenate all outputs
    all_recons = torch.cat(all_recons, dim=0)
    all_clip_voxels = torch.cat(all_clip_voxels, dim=0) 
    all_backbone_features = torch.cat(all_backbone_features, dim=0)
    all_prior_outputs = torch.cat(all_prior_outputs, dim=0)
    
    if config['blurry_recon']:
        all_blurry_recons = torch.cat(all_blurry_recons, dim=0)
    
    # Resize outputs
    imsize = 256
    resize_transform = transforms.Resize((imsize, imsize))
    all_recons = resize_transform(all_recons).float()
    if config['blurry_recon']:
        all_blurry_recons = resize_transform(all_blurry_recons).float()
    
    # Save outputs
    print("Saving reconstruction outputs...")
    torch.save(all_recons, f"{output_dir}/{model_name}_all_recons.pt")
    torch.save(all_clip_voxels, f"{output_dir}/{model_name}_all_clipvoxels.pt") 
    torch.save(all_pred_captions, f"{output_dir}/{model_name}_all_predcaptions.pt")
    torch.save(all_backbone_features, f"{output_dir}/{model_name}_all_backbones.pt")
    torch.save(all_prior_outputs, f"{output_dir}/{model_name}_all_prior_out.pt")
    
    if config['blurry_recon']:
        torch.save(all_blurry_recons, f"{output_dir}/{model_name}_all_blurryrecons.pt")
    
    print(f"Saved outputs to {output_dir}/")
    print("Shapes:", all_recons.shape, all_clip_voxels.shape)
    if config['blurry_recon']:
        print("Blurry recons shape:", all_blurry_recons.shape)
    print(f"Generated {len(all_pred_captions)} captions")
    print("Inference complete!")

if __name__ == "__main__":
    main()