"""
Model Pruning Utilities for YOLOv11
Supports structured and unstructured pruning for model compression.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple, Optional, Dict
import copy


def get_prunable_layers(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """
    Get all prunable layers (Conv2d and Linear) from model.
    
    Returns:
        List of (module, param_name) tuples
    """
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prunable.append((module, 'weight'))
        elif isinstance(module, nn.Linear):
            prunable.append((module, 'weight'))
    return prunable


def unstructured_pruning(
    model: nn.Module,
    amount: float = 0.3,
    method: str = 'l1'
) -> nn.Module:
    """
    Apply unstructured (weight-level) pruning.
    
    Prunes individual weights based on their magnitude.
    
    Args:
        model: Model to prune
        amount: Fraction of weights to prune (0.0-1.0)
        method: Pruning method ('l1', 'random')
    
    Returns:
        Pruned model
    """
    model = copy.deepcopy(model)
    layers = get_prunable_layers(model)
    
    # Select pruning method
    if method == 'l1':
        prune_fn = prune.L1Unstructured
    elif method == 'random':
        prune_fn = prune.RandomUnstructured
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply pruning to all layers
    for module, name in layers:
        prune_fn.apply(module, name, amount=amount)
    
    return model


def structured_pruning(
    model: nn.Module,
    amount: float = 0.3,
    dim: int = 0
) -> nn.Module:
    """
    Apply structured (channel-level) pruning.
    
    Prunes entire channels/filters based on their L2 norm.
    More hardware-friendly than unstructured pruning.
    
    Args:
        model: Model to prune
        amount: Fraction of channels to prune (0.0-1.0)
        dim: Dimension to prune (0=output channels, 1=input channels)
    
    Returns:
        Pruned model
    """
    model = copy.deepcopy(model)
    layers = get_prunable_layers(model)
    
    for module, name in layers:
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name, amount=amount, n=2, dim=dim)
    
    return model


def global_pruning(
    model: nn.Module,
    amount: float = 0.3
) -> nn.Module:
    """
    Apply global unstructured pruning.
    
    Prunes weights globally across all layers based on magnitude,
    which typically leads to better accuracy than layer-wise pruning.
    
    Args:
        model: Model to prune
        amount: Fraction of total weights to prune
    
    Returns:
        Pruned model
    """
    model = copy.deepcopy(model)
    layers = get_prunable_layers(model)
    
    prune.global_unstructured(
        layers,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    return model


def remove_pruning_reparametrization(model: nn.Module) -> nn.Module:
    """
    Remove pruning reparametrization to make pruning permanent.
    
    After this, the model can be saved without pruning masks.
    """
    model = copy.deepcopy(model)
    
    for module, name in get_prunable_layers(model):
        try:
            prune.remove(module, name)
        except ValueError:
            pass  # Not pruned
    
    return model


def get_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Calculate sparsity statistics for a model.
    
    Returns:
        Dictionary with sparsity metrics
    """
    total_zeros = 0
    total_params = 0
    layer_sparsities = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight
            zeros = (weight == 0).sum().item()
            total = weight.numel()
            
            total_zeros += zeros
            total_params += total
            
            layer_sparsities[name] = zeros / total
    
    return {
        'global_sparsity': total_zeros / total_params if total_params > 0 else 0,
        'total_zeros': total_zeros,
        'total_params': total_params,
        'layer_sparsities': layer_sparsities
    }


class GradualPruning:
    """
    Gradual pruning during training.
    
    Increases sparsity gradually over training epochs,
    which typically leads to better accuracy retention.
    """
    
    def __init__(
        self,
        model: nn.Module,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        begin_epoch: int = 0,
        end_epoch: int = 100,
        frequency: int = 5
    ):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch
        self.frequency = frequency
        self.current_sparsity = initial_sparsity
    
    def step(self, epoch: int):
        """Apply pruning for current epoch."""
        if epoch < self.begin_epoch or epoch > self.end_epoch:
            return
        
        if epoch % self.frequency != 0:
            return
        
        # Calculate target sparsity using cubic schedule
        progress = (epoch - self.begin_epoch) / (self.end_epoch - self.begin_epoch)
        target_sparsity = self.final_sparsity + \
            (self.initial_sparsity - self.final_sparsity) * (1 - progress) ** 3
        
        # Calculate amount to prune this step
        if target_sparsity > self.current_sparsity:
            # How much more to prune from remaining weights
            additional = (target_sparsity - self.current_sparsity) / (1 - self.current_sparsity)
            
            # Apply pruning
            layers = get_prunable_layers(self.model)
            for module, name in layers:
                prune.l1_unstructured(module, name, amount=additional)
            
            self.current_sparsity = target_sparsity


if __name__ == "__main__":
    from yolov11.model import YOLOv11
    
    print("Testing pruning utilities...")
    
    # Create model
    model = YOLOv11(num_classes=80, model_size='n')
    original_params = sum(p.numel() for p in model.parameters())
    
    # Test unstructured pruning
    pruned = unstructured_pruning(model, amount=0.3, method='l1')
    sparsity = get_sparsity(pruned)
    print(f"Unstructured pruning (30%): {sparsity['global_sparsity']:.2%} sparse")
    
    # Test global pruning
    pruned = global_pruning(model, amount=0.5)
    sparsity = get_sparsity(pruned)
    print(f"Global pruning (50%): {sparsity['global_sparsity']:.2%} sparse")
    
    # Remove reparametrization
    final = remove_pruning_reparametrization(pruned)
    sparsity = get_sparsity(final)
    print(f"After removing reparametrization: {sparsity['global_sparsity']:.2%} sparse")
    
    print("Pruning utilities OK!")
