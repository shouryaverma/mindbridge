#!/usr/bin/env python3
"""
Enhanced MindEye V2 Inference Script

This script takes outputs from syn_inference.py and applies SDXL-based enhancement
to improve reconstruction quality using image-to-image refinement.

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
from PIL import Image
import clip
from accelerate import Accelerator

# Diffusers for SDXL
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)

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
    parser.add_argument("--target_subject", type=int, required=True, choices=[1,2,3,4,5,6,7,8],
                       help="Target subject for enhancement")
    
    # Enhancement Parameters
    parser.add_argument("--enhancement_strength", type=float, default=0.6,
                       help="Strength of SDXL enhancement (0.3-0.8 range)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale for SDXL generation")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                       help="Number of SDXL inference steps")
    parser.add_argument("--num_samples_per_image", type=int, default=1,
                       help="Number of enhanced samples per reconstruction")
    
    # Sample Selection
    parser.add_argument("--selection_method", type=str, default="random", 
                       choices=["random", "best", "all"],
                       help="Method for selecting samples to enhance")
    parser.add_argument("--num_samples_to_enhance", type=int, default=100,
                       help="Number of samples to enhance (if not 'all')")
    
    # System
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for enhancement")
    
    return parser.parse_args()

def load_inference_outputs(model_name):
    """Load outputs from syn_inference.py."""
    output_dir = f"evals/{model_name}"
    
    print(f"Loading inference outputs from {output_dir}")
    
    # Load reconstructions
    recons_path = f"{output_dir}/{model_name}_all_recons.pt"
    if not os.path.exists(recons_path):
        raise FileNotFoundError(f"Reconstructions not found: {recons_path}")
    all_recons = torch.load(recons_path, map_location='cpu')
    
    # Load CLIP embeddings for sample selection
    clipvoxels_path = f"{output_dir}/{model_name}_all_clipvoxels.pt"
    if os.path.exists(clipvoxels_path):
        all_clip_voxels = torch.load(clipvoxels_path, map_location='cpu')
    else:
        all_clip_voxels = None
        print("Warning: CLIP voxel embeddings not found")
    
    # Load captions (may be empty)
    captions_path = f"{output_dir}/{model_name}_all_predcaptions.pt"
    if os.path.exists(captions_path):
        all_captions = torch.load(captions_path, map_location='cpu')
    else:
        all_captions = ["" for _ in range(len(all_recons))]
        print("Warning: Captions not found, using empty strings")
    
    # Load blurry reconstructions if available
    blurry_path = f"{output_dir}/{model_name}_all_blurryrecons.pt"
    if os.path.exists(blurry_path):
        all_blurry_recons = torch.load(blurry_path, map_location='cpu')
    else:
        all_blurry_recons = None
        print("Blurry reconstructions not found")
    
    print(f"Loaded {len(all_recons)} reconstructions")
    return all_recons, all_clip_voxels, all_captions, all_blurry_recons

def setup_sdxl_pipeline(cache_dir, device):
    """Setup SDXL img2img pipeline for enhancement."""
    print("Setting up SDXL enhancement pipeline...")
    
    # Set offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        # Try to load from local cache
        sdxl_path = f"{cache_dir}/stabilityai--stable-diffusion-xl-base-1.0"
        if os.path.exists(sdxl_path):
            print(f"Loading SDXL from local cache: {sdxl_path}")
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                sdxl_path,
                torch_dtype=torch.float16,
                local_files_only=True
            )
        else:
            # Fallback to download (will need internet)
            print("Local SDXL not found, downloading...")
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                cache_dir=cache_dir
            )
    except Exception as e:
        print(f"Error loading SDXL: {e}")
        print("Falling back to basic pipeline...")
        return None
    
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    
    # Optimize for memory efficiency
    if hasattr(pipe, 'enable_model_cpu_offload'):
        pipe.enable_model_cpu_offload()
    
    print("SDXL pipeline ready")
    return pipe

def setup_clip_evaluator(cache_dir, device):
    """Setup CLIP model for sample evaluation."""
    try:
        # Try to load from local cache
        clip_path = f"{cache_dir}/openai--clip-vit-large-patch14"
        if os.path.exists(clip_path):
            print(f"Loading CLIP from local cache: {clip_path}")
            # Note: clip library doesn't directly support local loading
            # This is a simplified approach
            model, preprocess = clip.load("ViT-L/14", device=device)
        else:
            print("Loading CLIP model...")
            model, preprocess = clip.load("ViT-L/14", device=device)
        
        return model, preprocess
    except Exception as e:
        print(f"Error loading CLIP: {e}")
        return None, None

def select_samples_for_enhancement(all_recons, all_clip_voxels, args):
    """Select which samples to enhance based on criteria."""
    num_total = len(all_recons)
    
    if args.selection_method == "all":
        selected_indices = list(range(num_total))
    elif args.selection_method == "random":
        np.random.seed(args.seed)
        selected_indices = np.random.choice(
            num_total, 
            min(args.num_samples_to_enhance, num_total), 
            replace=False
        ).tolist()
    elif args.selection_method == "best":
        # Select based on some quality metric (placeholder)
        if all_clip_voxels is not None:
            # Use CLIP embedding norms as a simple quality proxy
            norms = torch.norm(all_clip_voxels, dim=-1).mean(dim=-1)
            _, top_indices = torch.topk(norms, min(args.num_samples_to_enhance, num_total))
            selected_indices = top_indices.tolist()
        else:
            # Fallback to random if no embeddings
            print("No CLIP embeddings for quality selection, using random")
            np.random.seed(args.seed)
            selected_indices = np.random.choice(
                num_total,
                min(args.num_samples_to_enhance, num_total),
                replace=False
            ).tolist()
    
    print(f"Selected {len(selected_indices)} samples for enhancement")
    return selected_indices

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    # tensor should be (C, H, W) in range [0, 1]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Ensure range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    np_image = tensor.permute(1, 2, 0).cpu().numpy()
    np_image = (np_image * 255).astype(np.uint8)
    
    return Image.fromarray(np_image)

def enhance_reconstructions(sdxl_pipe, selected_recons, selected_captions, args, device):
    """Apply SDXL enhancement to selected reconstructions."""
    enhanced_recons = []
    
    print(f"Enhancing {len(selected_recons)} reconstructions...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(selected_recons), args.batch_size)):
            batch_recons = selected_recons[i:i+args.batch_size]
            batch_captions = selected_captions[i:i+args.batch_size]
            
            batch_enhanced = []
            
            for recon, caption in zip(batch_recons, batch_captions):
                # Convert tensor to PIL Image
                pil_image = tensor_to_pil(recon)
                
                # Use empty prompt if caption is empty/meaningless
                prompt = caption if caption and len(caption.strip()) > 0 else ""
                
                # Enhance with SDXL img2img
                try:
                    enhanced = sdxl_pipe(
                        prompt=prompt,
                        image=pil_image,
                        strength=args.enhancement_strength,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                        num_images_per_prompt=args.num_samples_per_image
                    ).images
                    
                    # Convert back to tensor
                    if args.num_samples_per_image == 1:
                        enhanced_tensor = transforms.ToTensor()(enhanced[0])
                    else:
                        # Take the first sample if multiple
                        enhanced_tensor = transforms.ToTensor()(enhanced[0])
                    
                    batch_enhanced.append(enhanced_tensor)
                    
                except Exception as e:
                    print(f"Enhancement failed for sample {i}, using original: {e}")
                    batch_enhanced.append(recon)
            
            enhanced_recons.extend(batch_enhanced)
    
    return torch.stack(enhanced_recons)

def save_enhanced_outputs(enhanced_recons, selected_indices, model_name, args):
    """Save enhanced reconstructions and metadata."""
    output_dir = f"evals/{model_name}"
    
    # Save enhanced reconstructions
    enhanced_path = f"{output_dir}/{model_name}_all_enhancedrecons.pt"
    torch.save(enhanced_recons, enhanced_path)
    
    # Save metadata about which samples were enhanced
    metadata = {
        "selected_indices": selected_indices,
        "selection_method": args.selection_method,
        "num_enhanced": len(selected_indices),
        "enhancement_params": {
            "strength": args.enhancement_strength,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "num_samples_per_image": args.num_samples_per_image
        }
    }
    
    metadata_path = f"{output_dir}/{model_name}_enhancement_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved enhanced outputs to {output_dir}")
    print(f"Enhanced reconstructions: {enhanced_path}")
    print(f"Metadata: {metadata_path}")

def main():
    # Setup
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = accelerator.device
    
    print(f"Running enhanced inference for {args.model_name}")
    print(f"Device: {device}")
    
    # Load outputs from base inference
    all_recons, all_clip_voxels, all_captions, all_blurry_recons = load_inference_outputs(args.model_name)
    
    # Setup enhancement pipeline
    sdxl_pipe = setup_sdxl_pipeline(args.cache_dir, device)
    if sdxl_pipe is None:
        print("Could not setup SDXL pipeline, exiting")
        return
    
    # Select samples for enhancement
    selected_indices = select_samples_for_enhancement(all_recons, all_clip_voxels, args)
    selected_recons = [all_recons[i] for i in selected_indices]
    selected_captions = [all_captions[i] for i in selected_indices]
    
    # Apply enhancement
    enhanced_recons = enhance_reconstructions(
        sdxl_pipe, selected_recons, selected_captions, args, device
    )
    
    # Save results
    save_enhanced_outputs(enhanced_recons, selected_indices, args.model_name, args)
    
    print(f"Enhancement complete! Enhanced {len(enhanced_recons)} reconstructions")
    print(f"Original size: {all_recons[0].shape}")
    print(f"Enhanced size: {enhanced_recons[0].shape}")

if __name__ == "__main__":
    main()