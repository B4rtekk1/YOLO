"""
yolov11.backbone – CSPDarknet Feature Extractor
================================================

Implements the YOLOv11 backbone based on CSPDarknet with two key additions:

* **C3k2** – a more efficient CSP bottleneck that replaces C2f from YOLOv8.
* **C2PSA** – Cross-Stage Partial with Spatial Attention, applied at the
  deepest feature level (P5) to capture long-range spatial dependencies.

The backbone produces three multi-scale feature maps (P3, P4, P5) that are
passed to the PANet neck for further fusion.

Model scaling
-------------
All channel widths and block depths are scaled by two multipliers:

=======  ================  ================
Size     depth_multiplier  width_multiplier
=======  ================  ================
n (nano)       0.33              0.25
s (small)      0.33              0.50
m (medium)     0.67              0.75
l (large)      1.00              1.00
x (xlarge)     1.33              1.25
=======  ================  ================
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List

from .blocks import Conv, C3k2, C2PSA, SPPF, CBAM


# Model scaling configurations: (depth_multiplier, width_multiplier)
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
    """Round *x* up to the nearest multiple of *divisor*.

    Used to ensure all channel counts are hardware-friendly (multiples of 8).

    Args:
        x: Raw (unrounded) channel count.
        divisor: Rounding granularity (default 8 for CUDA alignment).

    Returns:
        Smallest integer >= *x* that is divisible by *divisor*.
    """
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


class CSPDarknet(nn.Module):
    """
    YOLOv11 Backbone (CSPDarknet with C3k2 + C2PSA + optional CBAM)
    
    Architecture:
        P1: Conv 3x3 s=2 -> 320x320
        P2: Conv 3x3 s=2 + C3k2 -> 160x160
        P3: Conv 3x3 s=2 + C3k2 [+ CBAM] -> 80x80   [output]
        P4: Conv 3x3 s=2 + C3k2 [+ CBAM] -> 40x40   [output]
        P5: Conv 3x3 s=2 + C3k2 + SPPF + C2PSA [+ CBAM] -> 20x20 [output]
    
    Args:
        model_size: One of 'n', 's', 'm', 'l', 'x'
        in_channels: Input image channels (default: 3 for RGB)
        use_cbam: Whether to use CBAM attention at output stages
    """
    
    def __init__(self, model_size: str = 's', in_channels: int = 3, use_cbam: bool = False):
        super().__init__()
        
        if model_size not in MODEL_SCALES:
            raise ValueError(f"Model size must be one of {list(MODEL_SCALES.keys())}")
        
        depth_mult, width_mult = MODEL_SCALES[model_size]
        
        channels = [make_divisible(c * width_mult) for c in BASE_CHANNELS]
        depths = [max(round(d * depth_mult), 1) for d in BASE_DEPTHS]
        
        self.out_channels = channels[2:]  # P3, P4, P5 channels
        self.use_cbam = use_cbam
        
        # Stem
        self.stem = Conv(in_channels, channels[0], 3, 2)
        
        # Stage 1 (P2)
        self.stage1 = nn.Sequential(
            Conv(channels[0], channels[1], 3, 2),
            C3k2(channels[1], channels[1], n=depths[0], shortcut=True)
        )
        
        # Stage 2 (P3) - First output
        self.stage2 = nn.Sequential(
            Conv(channels[1], channels[2], 3, 2),
            C3k2(channels[2], channels[2], n=depths[1], shortcut=True)
        )
        
        # Stage 3 (P4) - Second output
        self.stage3 = nn.Sequential(
            Conv(channels[2], channels[3], 3, 2),
            C3k2(channels[3], channels[3], n=depths[2], shortcut=True)
        )
        
        # Stage 4 (P5) - Third output with SPPF + C2PSA
        self.stage4 = nn.Sequential(
            Conv(channels[3], channels[4], 3, 2),
            C3k2(channels[4], channels[4], n=depths[3], shortcut=True),
            SPPF(channels[4], channels[4], kernel_size=5),
            C2PSA(channels[4], channels[4], n=1)
        )
        
        # Optional CBAM attention at output stages
        if use_cbam:
            self.cbam_p3 = CBAM(channels[2])
            self.cbam_p4 = CBAM(channels[3])
            self.cbam_p5 = CBAM(channels[4])
        else:
            self.cbam_p3 = None
            self.cbam_p4 = None
            self.cbam_p5 = None
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise Conv2d and BatchNorm2d weights.

        * Conv2d weights: Kaiming Normal (fan_out, relu) – good default for
          networks with ReLU/SiLU activations.
        * Conv2d bias: zero-initialised (bias=False in most convs anyway).
        * BatchNorm2d weight: 1, bias: 0 (identity at start of training).
        """
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
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Tuple of feature maps (P3, P4, P5):
                P3: (B, C3, H/8, W/8)   - small objects
                P4: (B, C4, H/16, W/16) - medium objects  
                P5: (B, C5, H/32, W/32) - large objects
        """
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        
        # Apply CBAM if enabled
        if self.cbam_p3 is not None:
            p3 = self.cbam_p3(p3)
        if self.cbam_p4 is not None:
            p4 = self.cbam_p4(p4)
        if self.cbam_p5 is not None:
            p5 = self.cbam_p5(p5)
        
        return p3, p4, p5
    
    def get_out_channels(self) -> List[int]:
        """Return the number of output channels for each scale [P3, P4, P5].

        Used by the PANet neck to configure its input projections.

        Returns:
            List of three integers: ``[C_P3, C_P4, C_P5]``.
        """
        return self.out_channels


if __name__ == "__main__":
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
