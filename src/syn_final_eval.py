#!/usr/bin/env python3
"""
Final Evaluations for Rectified Flow MindEye V2 - Adapted from final_evaluations.py

This script evaluates enhanced reconstructions from syn_enhanced_inference.py
following the original MindEye2 evaluation protocol.

Usage:
    python syn_final_evaluations.py --model_name=subj01_finetuned --subj=1 --data_path=/path/to/data --cache_dir=/path/to/cache
"""

import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

from sentence_transformers import SentenceTransformer, util
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
import evaluate
import pandas as pd

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from syn_model_rect import GNet8_Encoder  # Updated import

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Final Evaluation Configuration")
    
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name of model, used for loading evaluation files")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to where NSD data is stored")
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Path to where misc. files are stored")
    parser.add_argument("--subj", type=int, required=True, choices=[1,2,3,4,5,6,7,8],
                       help="Evaluate on which subject?")
    parser.add_argument("--use_enhanced", action=argparse.BooleanOptionalAction, default=True,
                       help="Use enhanced reconstructions if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def load_evaluation_data(model_name, use_enhanced=True):
    """Load all required data for evaluation."""
    print(f"Loading evaluation data for {model_name}")
    
    # Determine which reconstructions to use
    if use_enhanced:
        enhanced_path = f"evals/{model_name}/{model_name}_all_enhancedrecons.pt"
        if os.path.exists(enhanced_path):
            print("Using enhanced reconstructions")
            all_recons = torch.load(enhanced_path)
        else:
            print("Enhanced reconstructions not found, using base reconstructions")
            all_recons = torch.load(f"evals/{model_name}/{model_name}_all_recons.pt")
    else:
        print("Using base reconstructions")
        all_recons = torch.load(f"evals/{model_name}/{model_name}_all_recons.pt")
    
    # Load other evaluation files
    all_images = torch.load("evals/all_images.pt")
    all_captions = torch.load("evals/all_captions.pt")
    
    # Load model outputs
    all_clipvoxels = torch.load(f"evals/{model_name}/{model_name}_all_clipvoxels.pt")
    all_blurryrecons = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons.pt")
    all_predcaptions = torch.load(f"evals/{model_name}/{model_name}_all_predcaptions.pt")
    
    # Load backbone and prior outputs (for UMAP analysis)
    try:
        all_backbones = torch.load(f"evals/{model_name}/{model_name}_all_backbones.pt")
        all_prior_out = torch.load(f"evals/{model_name}/{model_name}_all_prior_out.pt")
    except FileNotFoundError:
        print("Warning: Backbone or prior outputs not found. UMAP analysis will be skipped.")
        all_backbones = None
        all_prior_out = None
    
    print(f"Loaded shapes: images={all_images.shape}, recons={all_recons.shape}")
    
    return (all_images, all_recons, all_captions, all_clipvoxels, 
            all_blurryrecons, all_predcaptions, all_backbones, all_prior_out)

def prepare_images_for_evaluation(all_images, all_recons, all_blurryrecons, use_enhanced):
    """Prepare images for evaluation with proper sizing and blending."""
    imsize = 256
    
    # Resize all images to consistent size
    if all_images.shape[-1] != imsize:
        all_images = transforms.Resize((imsize, imsize))(all_images).float()
    if all_recons.shape[-1] != imsize:
        all_recons = transforms.Resize((imsize, imsize))(all_recons).float()
    if all_blurryrecons.shape[-1] != imsize:
        all_blurryrecons = transforms.Resize((imsize, imsize))(all_blurryrecons).float()
    
    # Apply weighted averaging for enhanced reconstructions to improve low-level metrics
    if use_enhanced and "enhanced" in str(all_recons.shape):
        print("Applying weighted averaging with blurry reconstructions for improved low-level metrics")
        all_recons = all_recons * 0.75 + all_blurryrecons * 0.25
    
    return all_images, all_recons, all_blurryrecons

def create_sample_visualization(all_images, all_recons, all_captions, all_predcaptions):
    """Create sample visualization comparing originals and reconstructions."""
    import textwrap
    
    def wrap_title(title, wrap_width):
        return "\n".join(textwrap.wrap(title, wrap_width))

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    jj, kk = -1, 0
    
    # Show sample reconstructions
    for j in np.array([2, 165, 119, 619, 231, 791]):
        jj += 1
        
        # Original image
        axes[kk][jj].imshow(utils.torch_to_Image(all_images[j]))
        axes[kk][jj].axis('off')
        axes[kk][jj].set_title(wrap_title(str(all_captions[j]), wrap_width=30), fontsize=8)
        
        jj += 1
        
        # Reconstructed image
        axes[kk][jj].imshow(utils.torch_to_Image(all_recons[j]))
        axes[kk][jj].axis('off')
        axes[kk][jj].set_title(wrap_title(str(all_predcaptions[j]), wrap_width=30), fontsize=8)
        
        if jj == 3: 
            kk += 1
            jj = -1

def evaluate_retrieval(all_images, all_clipvoxels, device, cache_dir):
    """Evaluate retrieval performance using OpenCLIP bigG."""
    print("Evaluating retrieval performance...")
    
    # Load embedding model (using bigG for consistency with training)
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device)
    
    from scipy import stats
    percent_correct_fwds, percent_correct_bwds = [], []
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for test_i in tqdm(range(30), desc="Retrieval evaluation"):
            random_samps = np.random.choice(np.arange(len(all_images)), size=300, replace=False)
            emb = clip_img_embedder(all_images[random_samps].to(device)).float()  # CLIP-Image
            emb_ = all_clipvoxels[random_samps].to(device).float()  # CLIP-Brain
            
            # Flatten and normalize
            emb = nn.functional.normalize(emb.reshape(len(emb), -1), dim=-1)
            emb_ = nn.functional.normalize(emb_.reshape(len(emb_), -1), dim=-1)
            
            labels = torch.arange(len(emb)).to(device)
            bwd_sim = utils.batchwise_cosine_similarity(emb, emb_)  # clip, brain
            fwd_sim = utils.batchwise_cosine_similarity(emb_, emb)  # brain, clip
            
            percent_correct_fwds = np.append(percent_correct_fwds, utils.topk(fwd_sim, labels, k=1).item())
            percent_correct_bwds = np.append(percent_correct_bwds, utils.topk(bwd_sim, labels, k=1).item())
    
    # Calculate statistics
    percent_correct_fwd = np.mean(percent_correct_fwds)
    fwd_sd = np.std(percent_correct_fwds) / np.sqrt(len(percent_correct_fwds))
    fwd_ci = stats.norm.interval(0.95, loc=percent_correct_fwd, scale=fwd_sd)
    
    percent_correct_bwd = np.mean(percent_correct_bwds)
    bwd_sd = np.std(percent_correct_bwds) / np.sqrt(len(percent_correct_bwds))
    bwd_ci = stats.norm.interval(0.95, loc=percent_correct_bwd, scale=bwd_sd)
    
    print(f"Forward retrieval: {percent_correct_fwd:.4f} 95% CI: [{fwd_ci[0]:.4f},{fwd_ci[1]:.4f}]")
    print(f"Backward retrieval: {percent_correct_bwd:.4f} 95% CI: [{bwd_ci[0]:.4f},{bwd_ci[1]:.4f}]")
    
    return percent_correct_fwd, percent_correct_bwd

@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, device='cuda'):
    """Compute two-way identification performance."""
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)
    perf = np.mean(success_cnt) / (len(all_images) - 1)
    
    return perf

def evaluate_image_metrics(all_images, all_recons, device):
    """Evaluate standard image quality metrics."""
    print("Evaluating image quality metrics...")
    
    results = {}
    
    # PixCorr
    print("Computing PixCorr...")
    preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_recons_flattened = preprocess(all_recons).view(len(all_recons), -1).cpu()
    
    corrsum = sum(np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1] 
                  for i in tqdm(range(len(all_images)), desc="PixCorr"))
    results['pixcorr'] = corrsum / len(all_images)
    
    # SSIM
    print("Computing SSIM...")
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim
    
    preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
    recon_gray = rgb2gray(preprocess(all_recons).permute((0,2,3,1)).cpu())
    
    ssim_scores = [ssim(rec, im, multichannel=True, gaussian_weights=True, 
                       sigma=1.5, use_sample_covariance=False, data_range=1.0)
                  for im, rec in tqdm(zip(img_gray, recon_gray), total=len(all_images), desc="SSIM")]
    results['ssim'] = np.mean(ssim_scores)
    
    # AlexNet
    print("Computing AlexNet metrics...")
    from torchvision.models import alexnet, AlexNet_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    
    alex_weights = AlexNet_Weights.IMAGENET1K_V1
    alex_model = create_feature_extractor(alexnet(weights=alex_weights), 
                                        return_nodes=['features.4','features.11']).to(device)
    alex_model.eval().requires_grad_(False)
    
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    results['alexnet2'] = two_way_identification(all_recons, all_images, alex_model, preprocess, 'features.4', device)
    results['alexnet5'] = two_way_identification(all_recons, all_images, alex_model, preprocess, 'features.11', device)
    
    # InceptionV3
    print("Computing InceptionV3 metrics...")
    from torchvision.models import inception_v3, Inception_V3_Weights
    
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                             return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False)
    
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    results['inception'] = two_way_identification(all_recons, all_images, inception_model, preprocess, 'avgpool', device)
    
    # CLIP
    print("Computing CLIP metrics...")
    import clip
    clip_model, _ = clip.load("ViT-L/14", device=device)
    
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    results['clip'] = two_way_identification(all_recons, all_images, clip_model.encode_image, preprocess, None, device)
    
    # EfficientNet
    print("Computing EfficientNet metrics...")
    import scipy as sp
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    
    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                       return_nodes=['avgpool']).to(device)
    eff_model.eval().requires_grad_(False)
    
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    gt = eff_model(preprocess(all_images))['avgpool'].reshape(len(all_images), -1).cpu().numpy()
    fake = eff_model(preprocess(all_recons))['avgpool'].reshape(len(all_recons), -1).cpu().numpy()
    results['effnet'] = np.array([sp.spatial.distance.correlation(gt[i], fake[i]) 
                                 for i in range(len(gt))]).mean()
    
    # SwAV
    print("Computing SwAV metrics...")
    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(swav_model, return_nodes=['avgpool']).to(device)
    swav_model.eval().requires_grad_(False)
    
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    gt = swav_model(preprocess(all_images))['avgpool'].reshape(len(all_images), -1).cpu().numpy()
    fake = swav_model(preprocess(all_recons))['avgpool'].reshape(len(all_recons), -1).cpu().numpy()
    results['swav'] = np.array([sp.spatial.distance.correlation(gt[i], fake[i]) 
                               for i in range(len(gt))]).mean()
    
    return results

def evaluate_brain_correlation(all_recons, subj, data_path, cache_dir, device):
    """Evaluate brain correlation using GNet."""
    print("Evaluating brain correlation...")
    
    # Load brain data and test information
    def my_split_by_node(urls): return urls

    # Load voxel data
    with h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r') as f:
        betas = torch.Tensor(f['betas'][:]).to("cpu")
    num_voxels = betas[0].shape[-1]
    
    # Determine test set size based on subject
    test_sizes = {3: 2371, 4: 2188, 6: 2371, 8: 2188}
    num_test = test_sizes.get(subj, 3000)
    
    test_url = f"{data_path}/wds/subj0{subj}/new_test/0.tar"
    test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", 
                           future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, 
                                        drop_last=True, pin_memory=True)
    
    # Process test data
    for test_i, (behav, _, _, _) in enumerate(test_dl):
        test_voxels = betas[behav[:,0,5].cpu().long()]
        test_images_idx = behav[:,0,0].cpu().numpy().astype(int)
    
    # Load brain region masks
    brain_region_masks = {}
    with h5py.File("brain_region_masks.hdf5", "r") as file:
        for subject in file.keys():
            subject_group = file[subject]
            subject_masks = {
                "nsd_general": subject_group["nsd_general"][:],
                "V1": subject_group["V1"][:], 
                "V2": subject_group["V2"][:], 
                "V3": subject_group["V3"][:], 
                "V4": subject_group["V4"][:],
                "higher_vis": subject_group["higher_vis"][:]
            }
            brain_region_masks[subject] = subject_masks
    
    subject_masks = brain_region_masks[f"subj0{subj}"]
    
    # Average voxels across repetitions
    test_voxels_averaged = torch.zeros((len(np.unique(test_images_idx)), num_voxels))
    uniq_imgs = np.unique(test_images_idx)
    
    for i, uniq_img in enumerate(uniq_imgs):
        locs = np.where(test_images_idx == uniq_img)[0]
        if len(locs) == 1:
            locs = locs.repeat(3)
        elif len(locs) == 2:
            locs = locs.repeat(2)[:3]
        test_voxels_averaged[i] = torch.mean(test_voxels[None, locs], dim=1)
    
    # Prepare reconstruction list for GNet
    recon_list = []
    for i in range(all_recons.shape[0]):
        img = all_recons[i].detach()
        img = transforms.ToPILImage()(img)
        recon_list.append(img)
    
    # Calculate brain correlations using GNet
    from torchmetrics import PearsonCorrCoef
    GNet = GNet8_Encoder(device=device, subject=subj, 
                        model_path=f"{cache_dir}/gnet_multisubject.pt")
    PeC = PearsonCorrCoef(num_outputs=len(recon_list))
    beta_primes = GNet.predict(recon_list)
    
    region_brain_correlations = {}
    for region, mask in subject_masks.items():
        score = PeC(test_voxels_averaged[:,mask].moveaxis(0,1), 
                   beta_primes[:,mask].moveaxis(0,1))
        region_brain_correlations[region] = float(torch.mean(score))
    
    return region_brain_correlations

def main():
    args = parse_args()
    utils.seed_everything(args.seed)
    
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = accelerator.device
    
    print(f"Running final evaluation for {args.model_name} on subject {args.subj}")
    print(f"Device: {device}")
    
    # Load evaluation data
    (all_images, all_recons, all_captions, all_clipvoxels, 
     all_blurryrecons, all_predcaptions, all_backbones, all_prior_out) = load_evaluation_data(
        args.model_name, args.use_enhanced)
    
    # Prepare images for evaluation
    all_images, all_recons, all_blurryrecons = prepare_images_for_evaluation(
        all_images, all_recons, all_blurryrecons, args.use_enhanced)
    
    # Create sample visualization
    create_sample_visualization(all_images, all_recons, all_captions, all_predcaptions)
    
    # Evaluate retrieval performance
    percent_correct_fwd, percent_correct_bwd = evaluate_retrieval(
        all_images, all_clipvoxels, device, args.cache_dir)
    
    # Evaluate image quality metrics
    image_metrics = evaluate_image_metrics(all_images, all_recons, device)
    
    # Evaluate brain correlation
    brain_correlations = evaluate_brain_correlation(
        all_recons, args.subj, args.data_path, args.cache_dir, device)
    
    # Compile results
    results_data = {
        "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", 
                  "EffNet-B", "SwAV", "FwdRetrieval", "BwdRetrieval",
                  "Brain Corr. nsd_general", "Brain Corr. V1", "Brain Corr. V2", 
                  "Brain Corr. V3", "Brain Corr. V4", "Brain Corr. higher_vis"],
        "Value": [image_metrics['pixcorr'], image_metrics['ssim'], 
                 image_metrics['alexnet2'], image_metrics['alexnet5'],
                 image_metrics['inception'], image_metrics['clip'], 
                 image_metrics['effnet'], image_metrics['swav'],
                 percent_correct_fwd, percent_correct_bwd,
                 brain_correlations["nsd_general"], brain_correlations["V1"], 
                 brain_correlations["V2"], brain_correlations["V3"], 
                 brain_correlations["V4"], brain_correlations["higher_vis"]]
    }
    
    df = pd.DataFrame(results_data)
    model_suffix = f"{args.model_name}_all_enhancedrecons" if args.use_enhanced else f"{args.model_name}_all_recons"
    
    print(f"\n=== Final Evaluation Results for {model_suffix} ===")
    print(df.to_string(index=False))
    
    # Save results
    os.makedirs('tables/', exist_ok=True)
    df.to_csv(f'tables/{model_suffix}.csv', sep='\t', index=False)
    print(f"\nResults saved to tables/{model_suffix}.csv")
    
    # Generate UMAP visualizations if data is available
    if all_backbones is not None and all_prior_out is not None:
        generate_umap_analysis(all_images, all_clipvoxels, all_backbones, 
                             all_prior_out, args.model_name, args.cache_dir, device)
    
    print("Evaluation complete!")

def generate_umap_analysis(all_images, all_clipvoxels, all_backbones, all_prior_out, 
                          model_name, cache_dir, device):
    """Generate UMAP visualizations for embedding space analysis."""
    print("Generating UMAP analysis...")
    
    try:
        import umap
        from scipy.spatial import distance
        
        # Load CLIP image embedder for ground truth embeddings
        clip_img_embedder = FrozenOpenCLIPImageEmbedder(
            arch="ViT-bigG-14",
            version="laion2b_s39b_b160k",
            output_tokens=True,
            only_tokens=True,
        )
        clip_img_embedder.to(device)
        
        # Get CLIP image embeddings
        with torch.cuda.amp.autocast(dtype=torch.float16):
            all_clipimgs = clip_img_embedder(all_images.to(device)).float()
        
        # Convert to numpy for UMAP
        clipimgs_np = all_clipimgs.flatten(1).detach().cpu().numpy()
        prior_out_np = all_prior_out.flatten(1).detach().cpu().numpy()
        backbones_np = all_backbones.flatten(1).detach().cpu().numpy()
        clipvoxels_np = all_clipvoxels.flatten(1).detach().cpu().numpy()
        
        reducer = umap.UMAP(random_state=42)
        
        # UMAP analysis for retrieval submodule
        plt.figure(figsize=(5, 5))
        data1, data2 = clipimgs_np, clipvoxels_np
        embedding = reducer.fit_transform(np.concatenate((data1, data2), axis=0))
        euclidean_dist = np.mean(np.linalg.norm(embedding[:len(data1), :] - embedding[len(data1):, :], axis=1))
        
        plt.scatter(embedding[:len(data1), 0], embedding[:len(data1), 1], c='blue', label='Images', alpha=0.5)
        plt.scatter(embedding[len(data1):, 0], embedding[len(data1):, 1], c='orange', label='Brain Voxels', alpha=0.5)
        plt.title(f'CLIP Image x Retrieval Submodule\nAvg. euclidean distance = {euclidean_dist:.2f}')
        plt.legend()
        plt.savefig(f'{model_name}_umap_retrieval.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # UMAP analysis for rectified flow (prior)
        plt.figure(figsize=(5, 5))
        data1, data2 = clipimgs_np, prior_out_np
        embedding = reducer.fit_transform(np.concatenate((data1, data2), axis=0))
        euclidean_dist = np.mean(np.linalg.norm(embedding[:len(data1), :] - embedding[len(data1):, :], axis=1))
        
        plt.scatter(embedding[:len(data1), 0], embedding[:len(data1), 1], c='blue', label='Images', alpha=0.5)
        plt.scatter(embedding[len(data1):, 0], embedding[len(data1):, 1], c='red', label='Rectified Flow', alpha=0.5)
        plt.title(f'CLIP Image x Rectified Flow\nAvg. euclidean distance = {euclidean_dist:.2f}')
        plt.legend()
        plt.savefig(f'{model_name}_umap_rectified_flow.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # UMAP analysis for backbone
        plt.figure(figsize=(5, 5))
        data1, data2 = clipimgs_np, backbones_np
        embedding = reducer.fit_transform(np.concatenate((data1, data2), axis=0))
        euclidean_dist = np.mean(np.linalg.norm(embedding[:len(data1), :] - embedding[len(data1):, :], axis=1))
        
        plt.scatter(embedding[:len(data1), 0], embedding[:len(data1), 1], c='blue', label='Images', alpha=0.5)
        plt.scatter(embedding[len(data1):, 0], embedding[len(data1):, 1], c='green', label='MLP Backbone', alpha=0.5)
        plt.title(f'CLIP Image x MLP Backbone\nAvg. euclidean distance = {euclidean_dist:.2f}')
        plt.legend()
        plt.savefig(f'{model_name}_umap_backbone.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("UMAP analysis completed successfully!")
        
    except ImportError:
        print("UMAP not available, skipping embedding space analysis")
    except Exception as e:
        print(f"Error in UMAP analysis: {e}")

if __name__ == "__main__":
    main()