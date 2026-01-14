"""
YOLOv11 Main Model
Combines backbone, neck, and task-specific heads
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any

from .backbone import CSPDarknet, MODEL_SCALES
from .neck import PANet
from .head import DetectionHead, SegmentationHead, PoseHead


class YOLOv11(nn.Module):
    """
    YOLOv11 Model - Unified architecture for Detection, Segmentation, and Pose Estimation
    
    Key improvements over YOLOv8:
        - C3k2 blocks replacing C2f for better efficiency
        - C2PSA spatial attention in backbone
        - Fewer parameters with higher accuracy
    
    Architecture:
        Input -> Backbone (CSPDarknet+C2PSA) -> Neck (PANet+C3k2) -> Head (task-specific)
    
    Args:
        num_classes: Number of detection classes
        task: Task type - 'detect', 'segment', or 'pose'
        model_size: Model size - 'n', 's', 'm', 'l', 'x'
        num_keypoints: Number of keypoints for pose task (default: 17 COCO)
        reg_max: Maximum regression value for DFL
    
    Example:
        >>> model = YOLOv11(num_classes=80, task='detect', model_size='s')
        >>> x = torch.randn(1, 3, 640, 640)
        >>> outputs = model(x)
    """
    
    SUPPORTED_TASKS = ['detect', 'segment', 'pose']
    
    def __init__(
        self,
        num_classes: int = 80,
        task: str = 'detect',
        model_size: str = 's',
        num_keypoints: int = 17,
        reg_max: int = 16,
        input_size: int = 640,
        use_cbam: bool = False
    ):
        super().__init__()
        
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Task must be one of {self.SUPPORTED_TASKS}, got {task}")
        
        if model_size not in MODEL_SCALES:
            raise ValueError(f"Model size must be one of {list(MODEL_SCALES.keys())}, got {model_size}")
        
        self.task = task
        self.model_size = model_size
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.reg_max = reg_max
        self.input_size = input_size
        self.use_cbam = use_cbam
        
        # Get depth multiplier for neck
        depth_mult, _ = MODEL_SCALES[model_size]
        
        # Build backbone with optional CBAM
        self.backbone = CSPDarknet(model_size=model_size, use_cbam=use_cbam)
        backbone_channels = self.backbone.get_out_channels()
        
        # Build neck
        self.neck = PANet(backbone_channels, depth_mult=depth_mult)
        neck_channels = self.neck.get_out_channels()
        
        # Build task-specific head
        if task == 'detect':
            self.head = DetectionHead(
                num_classes=num_classes,
                in_channels=neck_channels,
                reg_max=reg_max
            )
        elif task == 'segment':
            self.head = SegmentationHead(
                num_classes=num_classes,
                in_channels=neck_channels,
                reg_max=reg_max
            )
        elif task == 'pose':
            self.head = PoseHead(
                num_classes=1,  # Pose typically only detects 'person'
                in_channels=neck_channels,
                num_keypoints=num_keypoints,
                reg_max=reg_max
            )
        
        # Pre-compute strides and anchor grids
        self.strides = [8, 16, 32]  # P3, P4, P5 strides
        self._init_anchors()
    
    def _init_anchors(self):
        """Initialize anchor point grids for each scale."""
        self.anchor_grids = []
        
        for stride in self.strides:
            grid_size = self.input_size // stride
            
            # Create grid of anchor points
            ys, xs = torch.meshgrid(
                torch.arange(grid_size),
                torch.arange(grid_size),
                indexing='ij'
            )
            
            # Anchor points at center of each grid cell
            grid = torch.stack([xs, ys], dim=-1).float() + 0.5
            grid = grid * stride  # Scale to input coordinates
            
            self.register_buffer(f'anchor_grid_{stride}', grid)
            self.anchor_grids.append(grid)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Union[Dict[str, Any], List[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Dictionary with task-specific outputs:
                - detect: {'cls': [...], 'reg': [...]}
                - segment: {'cls': [...], 'reg': [...], 'mask': [...], 'proto': Tensor}
                - pose: {'cls': [...], 'reg': [...], 'kpt': [...]}
        """
        # Backbone
        features = self.backbone(x)
        
        # Neck
        features = self.neck(features)
        
        # Head
        outputs = self.head(features)
        
        # Format outputs based on task
        is_export = torch.onnx.is_in_onnx_export() or getattr(torch.jit, 'is_tracing', lambda: False)()
        
        if self.task == 'detect':
            cls_outputs, reg_outputs = outputs
            if is_export:
                # Return list for ONNX compatibility (no integers)
                return cls_outputs + reg_outputs
            return {
                'cls': cls_outputs,
                'reg': reg_outputs,
                'strides': self.strides
            }
        
        elif self.task == 'segment':
            cls_outputs, reg_outputs, mask_outputs, protos = outputs
            if is_export:
                return cls_outputs + reg_outputs + mask_outputs + [protos]
            return {
                'cls': cls_outputs,
                'reg': reg_outputs,
                'mask': mask_outputs,
                'proto': protos,
                'strides': self.strides
            }
        
        elif self.task == 'pose':
            cls_outputs, reg_outputs, kpt_outputs = outputs
            if is_export:
                return cls_outputs + reg_outputs + kpt_outputs
            return {
                'cls': cls_outputs,
                'reg': reg_outputs,
                'kpt': kpt_outputs,
                'strides': self.strides
            }
        
        return {}  # Default empty dict to satisfy type checker
    
    def get_anchors(self, device: torch.device) -> List[torch.Tensor]:
        """Get anchor grids on specified device."""
        return [getattr(self, f'anchor_grid_{s}').to(device) for s in self.strides]
    
    def fuse(self):
        """Fuse Conv2d + BatchNorm2d layers for inference optimization."""
        if not hasattr(self, 'fused'):
            self.fused = False
            
        if self.fused:
            print("Model already fused.")
            return self

        for m in self.modules():
            if hasattr(m, 'forward_fuse'):
                # Fuse conv and bn
                if hasattr(m, 'conv') and hasattr(m, 'bn') and isinstance(m.bn, nn.BatchNorm2d):
                    m.conv = self._fuse_conv_bn(m.conv, m.bn)
                    m.bn = nn.Identity()
                    m.forward = m.forward_fuse
        
        self.fused = True
        return self
    
    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        """Fuse convolution and batch normalization layers."""
        # Get parameters
        w_conv = conv.weight
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.out_channels, device=w_conv.device)
        
        w_bn = bn.weight
        b_bn = bn.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps
        
        # Calculate fused weights and bias
        std = torch.sqrt(running_var + eps)
        w_fused = w_conv * (w_bn / std).view(-1, 1, 1, 1)
        b_fused = (b_conv - running_mean) * w_bn / std + b_bn
        
        # Create fused conv
        fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            True  # Include bias
        )
        
        fused_conv.weight.data = w_fused
        fused_conv.bias.data = b_fused
        
        return fused_conv
    
    def info(self, verbose: bool = True) -> Dict[str, any]:
        """Print model information."""
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count layers
        num_layers = len(list(self.modules()))
        
        info = {
            'task': self.task,
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_layers': num_layers
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"YOLOv11{self.model_size}-{self.task}")
            print(f"{'='*60}")
            print(f"Task:             {self.task}")
            print(f"Model Size:       {self.model_size}")
            print(f"Num Classes:      {self.num_classes}")
            print(f"Input Size:       {self.input_size}x{self.input_size}")
            print(f"Total Params:     {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"Trainable Params: {trainable_params:,}")
            print(f"Num Layers:       {num_layers}")
            print(f"{'='*60}\n")
        
        return info
    
    def load_partial(self, state_dict: Dict[str, torch.Tensor], verbose: bool = True) -> Tuple[List[str], List[str]]:
        """
        Load weights from checkpoint with different architecture (partial loading).
        
        Useful for loading old weights into new architecture with additional modules (e.g., CBAM).
        
        Args:
            state_dict: State dict from checkpoint
            verbose: Whether to print loading info
        
        Returns:
            Tuple of (loaded_keys, skipped_keys)
        """
        current_state = self.state_dict()
        loaded_keys = []
        skipped_keys = []
        
        for name, param in state_dict.items():
            if name in current_state:
                if current_state[name].shape == param.shape:
                    current_state[name] = param
                    loaded_keys.append(name)
                else:
                    skipped_keys.append(f"{name} (shape mismatch: {param.shape} vs {current_state[name].shape})")
            else:
                skipped_keys.append(f"{name} (not in current model)")
        
        # Find new keys not in checkpoint
        new_keys = [k for k in current_state.keys() if k not in state_dict]
        
        self.load_state_dict(current_state)
        
        if verbose:
            print(f"Loaded {len(loaded_keys)}/{len(state_dict)} weights from checkpoint")
            if new_keys:
                print(f"New layers (randomly initialized): {len(new_keys)}")
                for k in new_keys[:5]:
                    print(f"  - {k}")
                if len(new_keys) > 5:
                    print(f"  ... and {len(new_keys) - 5} more")
            if skipped_keys:
                print(f"Skipped keys: {len(skipped_keys)}")
        
        return loaded_keys, skipped_keys


def create_model(
    num_classes: int = 80,
    task: str = 'detect',
    model_size: str = 's',
    pretrained: bool = False
) -> YOLOv11:
    """
    Factory function to create YOLOv11 model.
    
    Args:
        num_classes: Number of classes to detect
        task: One of 'detect', 'segment', 'pose'
        model_size: One of 'n', 's', 'm', 'l', 'x'
        pretrained: Whether to load pretrained weights (not implemented)
    
    Returns:
        YOLOv11 model instance
    """
    model = YOLOv11(
        num_classes=num_classes,
        task=task,
        model_size=model_size
    )
    
    if pretrained:
        # TODO: Load pretrained weights
        print("Warning: Pretrained weights not yet available")
    
    return model


if __name__ == "__main__":
    # Test all model variants
    print("Testing YOLOv11 models...\n")
    
    for task in ['detect', 'segment', 'pose']:
        for size in ['n', 's', 'm']:
            model = YOLOv11(num_classes=80, task=task, model_size=size)
            model.info()
            
            # Test forward pass
            x = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                out = model(x)
            
            print(f"  Outputs: {list(out.keys())}")
            for k, v in out.items():
                if isinstance(v, list):
                    print(f"    {k}: {[t.shape for t in v]}")
                elif isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
            print()
