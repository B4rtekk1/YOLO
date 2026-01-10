""" 
YOLOv11 Backbone - CSPDarknet with C3k2 and C2PSA
Extracts multi-scale features from input images
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List

from .blocks import Conv, C3k2, C2PSA, SPPF


# Model scaling configurations
# Format: (depth_multiplier, width_multiplier)
MODEL_SCALES: Dict[str, Tuple[float, float]] = {
    'n': (0.33, 0.25),  # nano
    's': (0.33, 0.50),  # small
    'm': (0.67, 0.75),  # medium
    'l': (1.00, 1.00),  # large
    'x': (1.33, 1.25),  # xlarge
}

# Base channel sizes at each stage
BASE_CHANNELS = [64, 128, 256, 512, 1024]

# Number of C3k2 blocks at each stage
BASE_DEPTHS = [3, 6, 6, 3]


def make_divisible(x: float, divisor: int = 8) -> int:
    """Make number divisible by divisor."""
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


class CSPDarknet(nn.Module):
    """
    YOLOv11 Backbone (CSPDarknet with C3k2 + C2PSA)
    
    Architecture:
        P1: Conv 3x3 s=2 -> 320x320
        P2: Conv 3x3 s=2 + C3k2 -> 160x160
        P3: Conv 3x3 s=2 + C3k2 -> 80x80   [output]
        P4: Conv 3x3 s=2 + C3k2 -> 40x40   [output]
        P5: Conv 3x3 s=2 + C3k2 + SPPF + C2PSA -> 20x20 [output]
    
    Key YOLOv11 improvements:
        - C3k2 blocks replace C2f for better efficiency
        - C2PSA adds spatial attention at P5
    
    Args:
        model_size: One of 'n', 's', 'm', 'l', 'x'
        in_channels: Input image channels (default: 3 for RGB)
    """
    
    def __init__(self, model_size: str = 's', in_channels: int = 3):
        super().__init__()
        
        if model_size not in MODEL_SCALES:
            raise ValueError(f"Model size must be one of {list(MODEL_SCALES.keys())}")
        
        depth_mult, width_mult = MODEL_SCALES[model_size]
        
        # Calculate actual channel sizes
        channels = [make_divisible(c * width_mult) for c in BASE_CHANNELS]
        # Calculate actual depths
        depths = [max(round(d * depth_mult), 1) for d in BASE_DEPTHS]
        
        # Store output channel sizes for neck
        self.out_channels = channels[2:]  # P3, P4, P5 channels
        
        # Stem: Initial convolution (P1)
        self.stem = Conv(in_channels, channels[0], 3, 2)  # 640->320
        
        # Stage 1 (P2): 320->160
        self.stage1 = nn.Sequential(
            Conv(channels[0], channels[1], 3, 2),
            C3k2(channels[1], channels[1], n=depths[0], shortcut=True)
        )
        
        # Stage 2 (P3): 160->80 - First output
        self.stage2 = nn.Sequential(
            Conv(channels[1], channels[2], 3, 2),
            C3k2(channels[2], channels[2], n=depths[1], shortcut=True)
        )
        
        # Stage 3 (P4): 80->40 - Second output
        self.stage3 = nn.Sequential(
            Conv(channels[2], channels[3], 3, 2),
            C3k2(channels[3], channels[3], n=depths[2], shortcut=True)
        )
        
        # Stage 4 (P5): 40->20 - Third output with SPPF + C2PSA (YOLOv11)
        self.stage4 = nn.Sequential(
            Conv(channels[3], channels[4], 3, 2),
            C3k2(channels[4], channels[4], n=depths[3], shortcut=True),
            SPPF(channels[4], channels[4], kernel_size=5),
            C2PSA(channels[4], channels[4], n=1)  # YOLOv11 spatial attention
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through backbone.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Tuple of feature maps (P3, P4, P5):
                P3: (B, C3, H/8, W/8)   - small objects
                P4: (B, C4, H/16, W/16) - medium objects  
                P5: (B, C5, H/32, W/32) - large objects
        """
        # Stem
        x = self.stem(x)
        
        # Stages
        x = self.stage1(x)
        p3 = self.stage2(x)   # 80x80 for 640 input
        p4 = self.stage3(p3)  # 40x40 for 640 input
        p5 = self.stage4(p4)  # 20x20 for 640 input
        
        return p3, p4, p5
    
    def get_out_channels(self) -> List[int]:
        """Get output channel sizes for P3, P4, P5."""
        return self.out_channels


if __name__ == "__main__":
    # Test backbone
    for size in ['n', 's', 'm', 'l', 'x']:
        model = CSPDarknet(model_size=size)
        x = torch.randn(1, 3, 640, 640)
        p3, p4, p5 = model(x)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"YOLOv11{size} backbone:")
        print(f"  Parameters: {params:.2f}M")
        print(f"  P3: {p3.shape}, P4: {p4.shape}, P5: {p5.shape}")
        print(f"  Channels: {model.get_out_channels()}")
        print()
