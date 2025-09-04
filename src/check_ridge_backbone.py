import torch

# Load checkpoint
checkpoint_path = '../train_logs/debug_finetune_smallModel_subsetSessions/finetuned_subj01_last.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['model_state_dict']

# Check all ridge layer dimensions
print("Ridge layer dimensions:")
for i in range(8):  # 0-7 based on your error
    weight_key = f'ridge.linears.{i}.weight'
    bias_key = f'ridge.linears.{i}.bias'
    
    if weight_key in state_dict:
        weight_shape = state_dict[weight_key].shape
        bias_shape = state_dict[bias_key].shape
        print(f"Ridge {i}: weight {weight_shape}, bias {bias_shape}")
        print(f"  -> {weight_shape[1]} voxels -> {weight_shape[0]} hidden_dim")
    else:
        print(f"Ridge {i}: NOT FOUND")

# Check other key dimensions
backbone_linear = state_dict.get('backbone.backbone_linear.weight')
if backbone_linear is not None:
    print(f"\nBackbone linear: {backbone_linear.shape}")
    print(f"  -> {backbone_linear.shape[1]} input -> {backbone_linear.shape[0]} output")

# Check training info if available
if 'train_subjects' in checkpoint:
    print(f"\nTrained subjects: {checkpoint['train_subjects']}")
if 'num_voxels_list' in checkpoint:
    print(f"Voxel counts: {checkpoint['num_voxels_list']}")
if 'args' in checkpoint:
    args = checkpoint['args']
    print(f"Training args - hidden_dim: {args.get('hidden_dim')}, n_blocks: {args.get('n_blocks')}")