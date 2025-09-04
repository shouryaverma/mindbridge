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
from syn_model_rect import BrainNetwork, PriorNetwork, BrainRectifiedFlow

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

class EmbeddingExpansion(nn.Module):
    """Expand CLIP L embeddings (1024-dim) to OpenCLIP bigG format (1664-dim)."""
    def __init__(self, input_dim=1024, output_dim=1664):
        super(EmbeddingExpansion, self).__init__()
        self.expansion = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, 256, 1024) -> (batch, 256, 1664)
        return self.expansion(x)

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

def load_model(model_name, num_voxels, hidden_dim, n_blocks, device, target_subject):
    """Load the trained MindEye model."""
    # Model architecture constants
    clip_seq_dim = 256
    clip_emb_dim = 1024
    
    # Load checkpoint first to inspect structure
    outdir = os.path.abspath(f'../train_logs/{model_name}')
    
    # Try different checkpoint naming conventions
    possible_names = [f'finetuned_subj0{target_subject}_last.pth', 'last.pth']
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
        
        # Debug information
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        if 'target_ridge_idx' in checkpoint:
            print(f"Saved target_ridge_idx: {checkpoint['target_ridge_idx']}")
        print(f"Ridge layer 0 shape: {state_dict['ridge.linears.0.weight'].shape}")
        print(f"Ridge layer 7 shape: {state_dict['ridge.linears.7.weight'].shape}")
        
        # Create single-ridge model for inference
        model = MindEyeModule()
        model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)
        
        # Create other components
        model.backbone = BrainNetwork(
            h=hidden_dim, in_dim=hidden_dim, seq_len=1, n_blocks=n_blocks,
            clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim,
            blurry_recon=True, clip_scale=1.0
        )
        
        prior_network = PriorNetwork(
            dim=clip_emb_dim, depth=6, dim_head=64, heads=clip_emb_dim // 64,
            causal=False, num_tokens=clip_seq_dim, learned_query_mode="pos_emb"
        )
        model.rectified_flow = BrainRectifiedFlow(
            net=prior_network, image_embed_dim=clip_emb_dim,
            condition_on_text_encodings=False, text_cond_drop_prob=0.2
        )

        # expansion layer to map OpenCLIP ViT/L embeddings to OpenCLIP ViT/bigG embeddings
        model.expansion_layer = EmbeddingExpansion(input_dim=clip_emb_dim, output_dim=1664)
        
        model.to(device)
        
        # Extract ridge layer 7 (which has 15,724 voxels for your target subject)
        print(f"Extracting ridge layer 7 for target subject (15,724 voxels)")
        
        # Verify ridge layer 7 has correct dimensions
        ridge7_weight = state_dict['ridge.linears.7.weight']
        ridge7_bias = state_dict['ridge.linears.7.bias']
        
        if ridge7_weight.shape[1] != num_voxels:
            raise ValueError(f"Ridge layer 7 has {ridge7_weight.shape[1]} voxels, expected {num_voxels}")
        
        # Create new state dict with only target ridge (mapped to index 0)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('ridge.'):
                # Only keep ridge layer 7, map it to index 0
                if key == 'ridge.linears.7.weight':
                    new_state_dict['ridge.linears.0.weight'] = value
                elif key == 'ridge.linears.7.bias':
                    new_state_dict['ridge.linears.0.bias'] = value
                # Skip all other ridge layers (0-6)
            elif key.startswith('expansion_layer.'):
                # Skip expansion layer from checkpoint (it won't exist anyway)
                pass
            else:
                # Keep all non-ridge layers
                new_state_dict[key] = value
        
        state_dict = new_state_dict
        inference_ridge_idx = 0  # Always 0 since we only have one ridge layer now
        
        print(f"Filtered state dict - removed {8-1} unused ridge layers")
        
        # Load the filtered state dict with strict=False to allow missing expansion layer
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully!")
        if 'expansion_layer' in missing_keys:
            print("Note: Using randomly initialized expansion layer as it wasn't found in checkpoint")
        
        return model, inference_ridge_idx
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Try DeepSpeed format
        try:
            import deepspeed
            state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
                checkpoint_dir=outdir, tag='last')
            
            # Apply same ridge extraction for DeepSpeed checkpoint
            print("Extracting ridge layer 7 from DeepSpeed checkpoint")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('ridge.'):
                    if key == 'ridge.linears.7.weight':
                        new_state_dict['ridge.linears.0.weight'] = value
                    elif key == 'ridge.linears.7.bias':
                        new_state_dict['ridge.linears.0.bias'] = value
                elif key.startswith('expansion_layer.'):
                    # Skip expansion layer from checkpoint (it won't exist anyway)
                    pass
                else:
                    new_state_dict[key] = value
            
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            inference_ridge_idx = 0
            print("DeepSpeed checkpoint loaded!")
            if 'expansion_layer' in missing_keys:
                print("Note: Using randomly initialized expansion layer as it wasn't found in DeepSpeed checkpoint")
            return model, inference_ridge_idx
            
        except Exception as ds_e:
            raise Exception(f"Failed to load checkpoint: {e}, DeepSpeed attempt: {ds_e}")

def setup_auxiliary_models(cache_dir, device):
    """Setup auxiliary models for text generation and VAE."""
    from diffusers import AutoencoderKL
    
    # VAE for blurry reconstruction
    autoenc = AutoencoderKL.from_pretrained(f'{cache_dir}/sd-vae-ft-mse').to(device)
    autoenc.eval().requires_grad_(False)
    
    # SKIPPING TEXT GENERATION FOR NOW
    # # Text generation models
    # from transformers import AutoProcessor, AutoModelForCausalLM
    # try:
    #     # Try to import custom GIT model
    #     from modeling_git import GitForCausalLMClipEmb
    #     processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    #     clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    # except ImportError:
    #     print("Custom GIT model not found, using standard transformers")
    #     processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    #     clip_text_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")
    
    # clip_text_model.to(device).eval().requires_grad_(False)
    
    # # CLIP converter: removed because we are directly using OpenCLIP ViT/L embeddings
    # clip_convert = None # CLIPConverter() # no need for conversion since not using bigG
    # try:
    #     converter_path = f"{cache_dir}/bigG_to_L_epoch8.pth"
    #     state_dict = torch.load(converter_path, map_location='cpu')['model_state_dict']
    #     clip_convert.load_state_dict(state_dict, strict=True)
    # except FileNotFoundError:
    #     print("Warning: CLIP converter checkpoint not found, using random weights")
    # clip_convert.to(device)
    # clip_convert = 'direct'
    
    return autoenc, None, None, None #processor, clip_text_model, clip_convert

def setup_unclip_engine(cache_dir, device):
    """Setup unCLIP diffusion engine."""
    
    # Set environment to use your cache instead of downloading
    import os
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode
    
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    
    # Modify sampling steps
    unclip_params["sampler_config"]["params"]["num_steps"] = 38
    
    # Remove problematic keys
    problematic_keys = ['ckpt_config', 'ckpt_path']
    for key in problematic_keys:
        if key in unclip_params:
            print(f"Removing problematic key: {key}")
            del unclip_params[key]
    
    # Fix autoencoder class if needed
    if "first_stage_config" in unclip_params:
        first_stage = unclip_params["first_stage_config"]
        if "target" in first_stage and "AutoencoderKLInferenceWrapper" in first_stage["target"]:
            first_stage["target"] = "sgm.models.autoencoder.AutoencoderKL"
            print("Fixed autoencoder class name in config")
    
    try:
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
        
    except Exception as e:
        print(f"Error setting up unCLIP engine: {e}")
        raise e

def generate_reconstructions(model, target_ridge_idx, test_voxel_indices, test_image_indices, 
                           voxels, device, args, autoenc, processor, clip_text_model, clip_convert,
                           diffusion_engine, vector_suffix):
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
                # Keep all 3 repetitions separate - shape: (3, num_voxels)
                voxel_data = voxels[test_voxel_indices[locs]]
                batch_voxels.append(voxel_data)
            
            batch_voxels = torch.stack(batch_voxels).to(device)
            
            # Process each repetition separately, then average outputs
            for rep in range(3):
                voxel_ridge = model.ridge(batch_voxels[:, [rep]], target_ridge_idx)  # Process one rep at a time
                backbone_rep, clip_voxels_rep, blurry_rep = model.backbone(voxel_ridge)
                
                if rep == 0:
                    # Initialize accumulators
                    backbone_features = backbone_rep
                    clip_voxels = clip_voxels_rep
                    blurry_image_enc = blurry_rep[0] if args.blurry_recon else None
                else:
                    # Accumulate
                    backbone_features += backbone_rep
                    clip_voxels += clip_voxels_rep
                    if args.blurry_recon:
                        blurry_image_enc += blurry_rep[0]
        
            # Average the accumulated outputs
            backbone_features /= 3
            clip_voxels /= 3
            if args.blurry_recon:
                blurry_image_enc /= 3
            
            # Store retrieval outputs
            all_clip_voxels.append(clip_voxels.cpu())
            
            # Generate samples using Rectified Flow
            prior_samples_1024 = model.rectified_flow.sample(
                text_embed=backbone_features,
                num_steps=args.num_inference_steps,
                cond_scale=args.cond_scale
            )

            # Text Generation directly with OpenCLIP ViT/L embeddings
            if clip_convert is not None:
                try:
                    generated_ids = clip_text_model.generate(pixel_values=prior_samples_1024, max_length=20)
                    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    all_pred_captions.extend(generated_caption)
                    print(f"Generated captions (direct L): {generated_caption}")
                except Exception as e:
                    print(f"Direct CLIP L text generation failed: {e}")
                    all_pred_captions.extend(["" for _ in range(len(batch_imgs))])
            else:
                print("Using CLIP converter")
                all_pred_captions.extend(["" for _ in range(len(batch_imgs))])

            # Generate actual images via unCLIP (before refining)
            prior_samples_1664 = model.expansion_layer(prior_samples_1024)  # Expand for unCLIP
            for i in range(len(batch_voxels)):
                samples = utils.unclip_recon(prior_samples_1664[[i]], diffusion_engine, vector_suffix, num_samples=1)
                all_recons.append(samples.cpu())
            
            if args.blurry_recon:
                # blurry_recon, _ = blurry_image_enc
                blurred_image = (autoenc.decode(blurry_image_enc/0.18215).sample / 2 + 0.5).clamp(0,1)
                all_blurry_recons.append(blurred_image.cpu())
    
    # Concatenate results
    all_recons = torch.cat(all_recons, dim=0)
    all_clip_voxels = torch.cat(all_clip_voxels, dim=0)
    
    if args.blurry_recon:
        all_blurry_recons = torch.cat(all_blurry_recons, dim=0)
    else:
        all_blurry_recons = None
    
    return all_recons, all_blurry_recons, all_clip_voxels, all_pred_captions


def main():
    # Setup
    args = parse_args()
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
        args.model_name, num_voxels, args.hidden_dim, args.n_blocks, device, args.target_subject)
    
    print(f"Model loaded with {utils.count_params(model)/1e6:.2f}M parameters")
    print(f"Target ridge index: {target_ridge_idx}")

    # Setup auxiliary models - FIXED
    autoenc, processor, clip_text_model, clip_convert = setup_auxiliary_models(args.cache_dir, device)
    # Setup unCLIP - FIXED
    diffusion_engine, vector_suffix = setup_unclip_engine(args.cache_dir, device)
    
    # Generate reconstructions
    all_recons, all_blurry_recons, all_clip_voxels, all_pred_captions = generate_reconstructions(
        model, target_ridge_idx, test_voxel_indices, test_image_indices, 
        voxels, images, device, args, autoenc, processor, clip_text_model, clip_convert,
        diffusion_engine, vector_suffix)
    
    # Resize and save results
    imsize = 256
    all_recons = transforms.Resize((imsize, imsize))(all_recons).float()
    
    # Save outputs
    output_dir = f"evals/{args.model_name}"
    torch.save(all_recons, f"{output_dir}/{args.model_name}_all_recons.pt")
    torch.save(all_clip_voxels, f"{output_dir}/{args.model_name}_all_clipvoxels.pt")
    torch.save(all_pred_captions, f"{output_dir}/{args.model_name}_all_predcaptions.pt")
    
    if all_blurry_recons is not None:
        all_blurry_recons = transforms.Resize((imsize, imsize))(all_blurry_recons).float()
        torch.save(all_blurry_recons, f"{output_dir}/{args.model_name}_all_blurryrecons.pt")
    
    print(f"Saved reconstruction outputs to {output_dir}")
    print(f"Reconstructions shape: {all_recons.shape}")

if __name__ == "__main__":
    main()