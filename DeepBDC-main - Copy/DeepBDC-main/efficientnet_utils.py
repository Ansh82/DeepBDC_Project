import numpy as np
import os
import glob
import argparse
import network.efficient_resnet as efficient_resnet
import torch
import random

# Dictionary of available efficient models
model_dict = dict(
    EfficientResNet10=efficient_resnet.EfficientResNet10,
    EfficientResNet12=efficient_resnet.EfficientResNet12,
    EfficientResNet18=efficient_resnet.EfficientResNet18,
    EfficientResNet34=efficient_resnet.EfficientResNet34,
    EfficientResNet34s=efficient_resnet.EfficientResNet34s,
    EfficientResNet50=efficient_resnet.EfficientResNet50,
    EfficientResNet101=efficient_resnet.EfficientResNet101)


def get_assigned_file(checkpoint_dir, num):
    """Get the specific checkpoint file based on epoch number."""
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    """Get the latest checkpoint file for resuming training."""
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    """Get the best model checkpoint file based on validation performance."""
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    print(best_file)
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def set_gpu(args):
    """Configure GPU settings based on command line arguments."""
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def load_model(model, dir):
    """Load model weights from a checkpoint file with compatibility handling."""
    model_dict = model.state_dict()
    file_dict = torch.load(dir)['state']
    
    # Handle parameter name differences between standard ResNet and EfficientResNet
    # Map standard ResNet parameter names to EfficientResNet parameter names
    renamed_file_dict = {}
    for k, v in file_dict.items():
        # Handle depthwise separable convolution parameter mapping
        if 'C1.' in k and not 'shortcut' in k:
            renamed_file_dict[k.replace('C1', 'C1_depthwise')] = v
            continue
        if 'C2.' in k and not 'shortcut' in k:
            renamed_file_dict[k.replace('C2', 'C2_depthwise')] = v
            continue
            
        # Add more mappings as needed for different parameter names
        renamed_file_dict[k] = v
    
    # Filter to only include parameters that exist in the model
    compatible_file_dict = {k: v for k, v in renamed_file_dict.items() if k in model_dict}
    
    # For parameters that couldn't be directly mapped, try to infer the mapping
    missing_keys = set(model_dict.keys()) - set(compatible_file_dict.keys())
    
    # Print warning about parameter mapping
    if len(missing_keys) > 0:
        print(f"Warning: {len(missing_keys)} parameters couldn't be loaded from checkpoint.")
        print("This is expected when loading standard ResNet weights into EfficientResNet.")
    
    model_dict.update(compatible_file_dict)
    model.load_state_dict(model_dict, strict=False)
    return model


def set_seed(seed):
    """Set random seeds for reproducibility."""
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def convert_resnet_to_efficient(original_model_path, efficient_model_type, output_path=None):
    """
    Convert weights from a standard ResNet model to an EfficientResNet model.
    
    Args:
        original_model_path: Path to the original ResNet model checkpoint
        efficient_model_type: Type of EfficientResNet model to create
        output_path: Where to save the converted model (optional)
    
    Returns:
        Loaded EfficientResNet model with weights converted from standard ResNet
    """
    # Load original model weights
    original_weights = torch.load(original_model_path)['state']
    
    # Create new efficient model
    if efficient_model_type not in model_dict:
        raise ValueError(f"Unknown model type: {efficient_model_type}")
    
    efficient_model = model_dict[efficient_model_type]()
    
    # Get weight mapping strategy
    weight_map = create_weight_mapping(original_weights, efficient_model.state_dict())
    
    # Apply weight mapping
    new_state_dict = efficient_model.state_dict()
    for efficient_key, original_key in weight_map.items():
        if original_key in original_weights:
            if new_state_dict[efficient_key].shape == original_weights[original_key].shape:
                new_state_dict[efficient_key] = original_weights[original_key]
            else:
                # Handle shape differences (e.g., for depthwise separable convs)
                print(f"Shape mismatch for {efficient_key}: "
                      f"{new_state_dict[efficient_key].shape} vs {original_weights[original_key].shape}")
    
    # Load mapped weights into efficient model
    efficient_model.load_state_dict(new_state_dict, strict=False)
    
    # Save converted model if output path is provided
    if output_path:
        torch.save({'state': new_state_dict}, output_path)
        print(f"Converted model saved to {output_path}")
    
    return efficient_model


def create_weight_mapping(original_dict, efficient_dict):
    """
    Create a mapping between original ResNet weights and EfficientResNet weights.
    
    Args:
        original_dict: State dict from original ResNet model
        efficient_dict: State dict from EfficientResNet model
    
    Returns:
        Dictionary mapping efficient_key -> original_key
    """
    mapping = {}
    
    # Create mapping rules
    for efficient_key in efficient_dict.keys():
        # Handle depthwise separable convolutions
        if '_depthwise' in efficient_key:
            # Map depthwise conv to original conv
            base_key = efficient_key.replace('_depthwise', '')
            if base_key in original_dict:
                mapping[efficient_key] = base_key
        elif '_pointwise' in efficient_key:
            # Pointwise convs have no direct mapping, initialize from scratch
            continue
        else:
            # For other parameters, try direct mapping
            if efficient_key in original_dict:
                mapping[efficient_key] = efficient_key
    
    return mapping


def get_parameter_count(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_speed(model, input_size=(1, 3, 224, 224), num_runs=100):
    """Measure the inference speed of a model."""
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure time
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time.record()
            _ = model(dummy_input)
            end_time.record()
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time))
    
    avg_time = sum(times) / len(times)
    return avg_time