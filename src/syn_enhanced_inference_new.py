#!/usr/bin/env python3
"""
Enhanced MindEye V2 Inference Script - Adapted from enhanced_recon_inference.py

This script takes outputs from syn_inference.py and applies SDXL-based enhancement
following the original MindEye2 approach but adapted for CLIP L architecture.

Usage:
    python syn_enhanced_inference.py --model_name=subj01_finetuned --cache_dir=/path/to/cache
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import json

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

# Enable TF32 for faster operations
torch.backends.cuda.matmul.allow_tf32 = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced MindEye V2 Inference")
    
    # Essential Args
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name (must match syn_inference.py output)")
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Directory containing cached models")
    # parser.add_argument("--target_subject", type=int, required=True, choices=[1,2,3,4,5,6,7,8],
    #                    help="Target subject for enhancement")
    
    # Enhancement Parameters (matching original)
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples per image (original uses 1)")
    parser.add_argument("--img2img_timepoint", type=int, default=13,
                       help="Denoising timepoint (higher = more prompt reliance)")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--num_steps", type=int, default=25,
                       help="Number of SDXL sampling steps")
    
    # System
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def load_inference_outputs(model_name):
    """Load outputs from syn_inference.py - following original structure."""
    output_dir = f"evals/{model_name}"
    
    print(f"Loading inference outputs from {output_dir}")
    
    # Load all required files (following original enhanced_recon_inference.py)
    all_images = torch.load(f"evals/all_images.pt")  # Ground truth images
    all_recons = torch.load(f"{output_dir}/{model_name}_all_recons.pt")  # Unrefined reconstructions
    all_clipvoxels = torch.load(f"{output_dir}/{model_name}_all_clipvoxels.pt")
    all_blurryrecons = torch.load(f"{output_dir}/{model_name}_all_blurryrecons.pt")
    all_predcaptions = torch.load(f"{output_dir}/{model_name}_all_predcaptions.pt")
    
    # Resize to 768x768 for SDXL processing (matching original)
    all_recons = transforms.Resize((768, 768))(all_recons).float()
    all_blurryrecons = transforms.Resize((768, 768))(all_blurryrecons).float()
    
    print(f"Loaded shapes: images={all_images.shape}, recons={all_recons.shape}")
    print(f"clipvoxels={all_clipvoxels.shape}, blurry={all_blurryrecons.shape}")
    
    return all_images, all_recons, all_clipvoxels, all_blurryrecons, all_predcaptions

def setup_sdxl_base_engine(cache_dir, device):
    """Setup SDXL base engine following original approach."""
    print("Setting up SDXL base engine...")
    
    # Set offline mode
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Block network access entirely during setup
    import socket
    def block_network(*args, **kwargs):
        raise ConnectionError("Network access blocked - use cached models only")

    original_getaddrinfo = socket.getaddrinfo
    socket.getaddrinfo = block_network
    
    # Load configs (following original)
    unclip_config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    unclip_config = OmegaConf.to_container(unclip_config, resolve=True)
    unclip_params = unclip_config["model"]["params"]
    sampler_config = unclip_params["sampler_config"]
    sampler_config['params']['num_steps'] = 38
    
    sdxl_config = OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml")
    sdxl_config = OmegaConf.to_container(sdxl_config, resolve=True)
    refiner_params = sdxl_config["model"]["params"]
    
    # Extract parameters
    network_config = refiner_params["network_config"]
    denoiser_config = refiner_params["denoiser_config"]
    first_stage_config = refiner_params["first_stage_config"]
    conditioner_config = refiner_params["conditioner_config"]
    scale_factor = refiner_params["scale_factor"]
    disable_first_stage_autocast = refiner_params["disable_first_stage_autocast"]
    
    # Try to find SDXL checkpoint
    possible_ckpt_paths = [
        f'{cache_dir}/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/sd_xl_base_1.0.safetensors',
        f'{cache_dir}/sd_xl_base_1.0.safetensors',
        f'{cache_dir}/stabilityai--stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors',
        # Add more possible paths as needed
    ]
    
    base_ckpt_path = None
    for path in possible_ckpt_paths:
        if os.path.exists(path):
            base_ckpt_path = path
            break
    
    if base_ckpt_path is None:
        print("Warning: No SDXL checkpoint found, using default initialization")
        base_ckpt_path = None
    
    # Create SDXL base engine
    base_engine = DiffusionEngine(
        network_config=network_config,
        denoiser_config=denoiser_config,
        first_stage_config=first_stage_config,
        conditioner_config=conditioner_config,
        sampler_config=sampler_config,
        scale_factor=scale_factor,
        disable_first_stage_autocast=disable_first_stage_autocast,
        ckpt_path=base_ckpt_path
    )
    base_engine.eval().requires_grad_(False).to(device)
    
    return base_engine, conditioner_config

# def setup_text_embedders(base_engine, device):
#     """Setup dual text embedders by extracting from the already-loaded conditioner."""
#     print("Setting up text embedders...")
    
#     # Extract the already-loaded embedders from the conditioner
#     embedders = base_engine.conditioner.embedders
    
#     # Find the CLIP and OpenCLIP embedders
#     base_text_embedder1 = None
#     base_text_embedder2 = None
    
#     for embedder in embedders:
#         if hasattr(embedder, 'transformer') and hasattr(embedder, 'tokenizer'):
#             if 'clip' in embedder.__class__.__name__.lower():
#                 if base_text_embedder1 is None:
#                     base_text_embedder1 = embedder
#                 else:
#                     base_text_embedder2 = embedder
    
#     # Fallback: if we can't find them, use the first two text embedders
#     if base_text_embedder1 is None or base_text_embedder2 is None:
#         text_embedders = [emb for emb in embedders if hasattr(emb, 'encode')]
#         base_text_embedder1 = text_embedders[0] if len(text_embedders) > 0 else None
#         base_text_embedder2 = text_embedders[1] if len(text_embedders) > 1 else None
    
#     if base_text_embedder1 is None or base_text_embedder2 is None:
#         raise RuntimeError("Could not find required text embedders in conditioner")
    
#     return base_text_embedder1, base_text_embedder2

def setup_text_embedders(conditioner_config, cache_dir, device):
    """Setup dual text embedders following original approach."""
    print("Setting up text embedders...")
    
    # OpenAI CLIP text embedder
    base_text_embedder1 = FrozenCLIPEmbedder(
        layer=conditioner_config['params']['emb_models'][0]['params']['layer'],
        layer_idx=conditioner_config['params']['emb_models'][0]['params']['layer_idx'],
        version=f"{cache_dir}/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41",
    )
    base_text_embedder1.to(device)
    
    # OpenCLIP text embedder
    base_text_embedder2 = FrozenOpenCLIPEmbedder2(
        arch=conditioner_config['params']['emb_models'][1]['params']['arch'],
        version=conditioner_config['params']['emb_models'][1]['params']['version'],
        freeze=conditioner_config['params']['emb_models'][1]['params']['freeze'],
        layer=conditioner_config['params']['emb_models'][1]['params']['layer'],
        always_return_pooled=conditioner_config['params']['emb_models'][1]['params']['always_return_pooled'],
        legacy=conditioner_config['params']['emb_models'][1]['params']['legacy'],
    )
    base_text_embedder2.to(device)
    
    return base_text_embedder1, base_text_embedder2

def setup_conditioning(base_engine, device):
    """Setup positive and negative conditioning following original."""
    print("Setting up conditioning...")
    
    # Positive conditioning (empty prompt)
    batch = {
        "txt": "",
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
        "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024
    }
    out = base_engine.conditioner(batch)
    crossattn = out["crossattn"].to(device)
    vector_suffix = out["vector"][:, -1536:].to(device)
    
    # Negative conditioning (following original negative prompt)
    negative_prompt = ("painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, "
                      "deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, "
                      "skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, "
                      "ugly face, distorted face, extra legs, anime")
    
    batch_uc = {
        "txt": negative_prompt,
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
        "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024
    }
    out_uc = base_engine.conditioner(batch_uc)
    crossattn_uc = out_uc["crossattn"].to(device)
    vector_uc = out_uc["vector"].to(device)
    
    print(f"Conditioning setup - crossattn: {crossattn.shape}, vector_suffix: {vector_suffix.shape}")
    
    return crossattn, vector_suffix, crossattn_uc, vector_uc

def setup_clip_evaluator(device):
    """Setup CLIP model for sample selection (bigG for consistency with original)."""
    print("Setting up CLIP evaluator...")
    try:
        clip_img_embedder = FrozenOpenCLIPImageEmbedder(
            arch="ViT-bigG-14",
            version="laion2b_s39b_b160k",
            output_tokens=True,
            only_tokens=True
        )
        clip_img_embedder.to(device)
        return clip_img_embedder
    except Exception as e:
        print(f"Warning: Could not setup CLIP evaluator: {e}")
        return None

def enhance_single_image(img_idx, image, prompt, all_images, base_engine, base_text_embedder1, 
                        base_text_embedder2, crossattn, vector_suffix, crossattn_uc, vector_uc,
                        clip_img_embedder, args, device):
    """Enhance a single image following the original approach."""
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), base_engine.ema_scope():
        # Set sampling parameters (following original)
        base_engine.sampler.num_steps = args.num_steps
        base_engine.sampler.guider.scale = args.cfg_scale
        
        # Ensure image is correct size
        assert image.shape[-1] == 768, f"Expected 768x768 image, got {image.shape}"
        
        # Encode image to latent space
        z = base_engine.encode_first_stage(image * 2 - 1).repeat(args.num_samples, 1, 1, 1)
        
        # Setup text conditioning
        openai_clip_text = base_text_embedder1(prompt)
        clip_text_tokenized, clip_text_emb = base_text_embedder2(prompt)
        clip_text_emb = torch.hstack((clip_text_emb, vector_suffix))
        clip_text_tokenized = torch.cat((openai_clip_text, clip_text_tokenized), dim=-1)
        
        c = {
            "crossattn": clip_text_tokenized.repeat(args.num_samples, 1, 1),
            "vector": clip_text_emb.repeat(args.num_samples, 1)
        }
        uc = {
            "crossattn": crossattn_uc.repeat(args.num_samples, 1, 1),
            "vector": vector_uc.repeat(args.num_samples, 1)
        }
        
        # Setup img2img denoising (following original)
        noise = torch.randn_like(z)
        sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)
        init_z = (z + noise * append_dims(sigmas[-args.img2img_timepoint], z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2.0)
        sigmas = sigmas[-args.img2img_timepoint:].repeat(args.num_samples, 1)
        
        # Perform sampling
        base_engine.sampler.num_steps = sigmas.shape[-1] - 1
        denoiser = lambda x, sigma, c: base_engine.denoiser(base_engine.model, x, sigma, c)
        
        noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(
            init_z, cond=c, uc=uc, num_steps=base_engine.sampler.num_steps
        )
        
        for timestep in range(base_engine.sampler.num_steps):
            noised_z = base_engine.sampler.sampler_step(
                sigmas[:, timestep], sigmas[:, timestep + 1], denoiser, noised_z, 
                cond=c, uc=uc, gamma=0
            )
        
        # Decode samples
        samples_z_base = noised_z
        samples_x = base_engine.decode_first_stage(samples_z_base)
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        
        # Select best sample based on CLIP similarity to ground truth (if available)
        if args.num_samples == 1:
            best_sample = samples[0]
        else:
            if clip_img_embedder is not None and img_idx < len(all_images):
                # Compute CLIP similarity to ground truth for sample selection
                sample_cossim = nn.functional.cosine_similarity(
                    clip_img_embedder(utils.resize(samples, 224).to(device)).flatten(1),
                    clip_img_embedder(utils.resize(all_images[[img_idx]].float(), 224).to(device)).flatten(1)
                )
                which_sample = torch.argmax(sample_cossim)
                best_sample = samples[which_sample]
            else:
                # No ground truth available, just take first sample
                best_sample = samples[0]
        
        return best_sample.cpu()

def enhance_all_reconstructions(all_images, all_recons, all_predcaptions, base_engine, 
                               base_text_embedder1, base_text_embedder2, crossattn, vector_suffix,
                               crossattn_uc, vector_uc, clip_img_embedder, args, device):
    """Enhance all reconstructions following original approach."""
    
    print(f"Enhancing {len(all_recons)} reconstructions...")
    
    # Progress checkpointing setup
    checkpoint_interval = 100  # Save every 100 images
    checkpoint_path = f"evals/{args.model_name}/{args.model_name}_enhanced_checkpoint.pt"
    
    # Check for existing checkpoint to resume from
    start_idx = 0
    all_enhanced_recons = []
    
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        all_enhanced_recons = checkpoint['enhanced_recons']
        start_idx = len(all_enhanced_recons)
        print(f"Resuming from image {start_idx}/{len(all_recons)}")
    
    # Main enhancement loop
    for img_idx in tqdm(range(start_idx, len(all_recons)), initial=start_idx, total=len(all_recons)):
        image = all_recons[[img_idx]].to(device)
        
        # Get caption for this image
        if img_idx < len(all_predcaptions):
            if isinstance(all_predcaptions[img_idx], list):
                prompt = all_predcaptions[img_idx][0] if all_predcaptions[img_idx] else ""
            else:
                prompt = str(all_predcaptions[img_idx])
        else:
            prompt = ""
        
        try:
            # Enhance the image
            enhanced_sample = enhance_single_image(
                img_idx, image, prompt, all_images, base_engine, base_text_embedder1,
                base_text_embedder2, crossattn, vector_suffix, crossattn_uc, vector_uc,
                clip_img_embedder, args, device
            )
            
            all_enhanced_recons.append(enhanced_sample[None])
            
            # Clear GPU cache periodically
            if (img_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error enhancing image {img_idx}: {e}")
            # Use original image as fallback
            all_enhanced_recons.append(image.cpu()[None])
        
        # Save checkpoint periodically
        if (img_idx + 1) % checkpoint_interval == 0 or img_idx == len(all_recons) - 1:
            checkpoint_data = {
                'enhanced_recons': all_enhanced_recons,
                'completed_images': img_idx + 1,
                'total_images': len(all_recons),
                'args': args
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved at image {img_idx + 1}/{len(all_recons)}")
    
    # Concatenate all enhanced reconstructions
    all_enhanced_recons = torch.cat(all_enhanced_recons, dim=0)
    
    # Resize to 256x256 for evaluation (following original)
    all_enhanced_recons = transforms.Resize((256, 256))(all_enhanced_recons).float()
    
    # Clean up checkpoint file when complete
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Removed checkpoint file - enhancement complete")
    
    return all_enhanced_recons

def main():
    # Setup
    args = parse_args()
    
    # Set random seeds
    utils.seed_everything(args.seed)
    
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = accelerator.device
    
    print(f"Running enhanced inference for {args.model_name}")
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs("evals", exist_ok=True)
    os.makedirs(f"evals/{args.model_name}", exist_ok=True)
    
    # Load outputs from base inference
    all_images, all_recons, all_clipvoxels, all_blurryrecons, all_predcaptions = load_inference_outputs(args.model_name)
    
    # Setup SDXL base engine
    base_engine, conditioner_config = setup_sdxl_base_engine(args.cache_dir, device)
    
    # Setup text embedders (reuse from base engine)
    base_text_embedder1, base_text_embedder2 = setup_text_embedders(conditioner_config, args.cache_dir, device)
    
    # Setup conditioning (reuse from base engine)
    crossattn, vector_suffix, crossattn_uc, vector_uc = setup_conditioning(base_engine, device)
    
    # Setup CLIP evaluator for sample selection
    clip_img_embedder = setup_clip_evaluator(device)
    
    # Enhance all reconstructions
    all_enhanced_recons = enhance_all_reconstructions(
        all_images, all_recons, all_predcaptions, base_engine,
        base_text_embedder1, base_text_embedder2, crossattn, vector_suffix,
        crossattn_uc, vector_uc, clip_img_embedder, args, device
    )
    
    # Save results (following original naming convention)
    output_path = f"evals/{args.model_name}/{args.model_name}_all_enhancedrecons.pt"
    torch.save(all_enhanced_recons, output_path)
    
    print(f"Enhanced reconstruction complete!")
    print(f"Saved {len(all_enhanced_recons)} enhanced reconstructions to {output_path}")
    print(f"Output shape: {all_enhanced_recons.shape}")

if __name__ == "__main__":
    main()