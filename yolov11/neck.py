"""
yolov11.neck – PANet Feature Pyramid Neck
==========================================

Implements the Path Aggregation Network (PANet) that fuses multi-scale
features from the backbone into three output feature maps used by the
detection/segmentation/pose heads.

Two complementary pathways are used:

* **FPN (top-down)** – high-level semantic information flows *down* from P5
  to P3 via lateral connections and upsampling, enriching small-object
  feature maps with context.
* **PAN (bottom-up)** – fine-grained spatial information flows *up* from N3
  to N5 via strided convolutions, enriching large-object feature maps with
  precise localisation cues.

Each fusion step uses a :class:`~yolov11.blocks.C3k2` block (the YOLOv11
efficient CSP bottleneck) instead of the plain C2f used in YOLOv8.
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from .blocks import Conv, C3k2, Upsample


class PANet(nn.Module):
    """
    YOLOv11 Neck combining FPN and PAN with C3k2 blocks
    
    Architecture:
        FPN (Top-down):
            P5 -> Upsample + Concat(P4) -> C3k2 -> N4
            N4 -> Upsample + Concat(P3) -> C3k2 -> N3
        
        PAN (Bottom-up):
            N3 -> Conv s=2 + Concat(N4) -> C3k2 -> N4'
            N4' -> Conv s=2 + Concat(P5) -> C3k2 -> N5
    
    Args:
        in_channels: List of input channel sizes [P3, P4, P5]
        depth_mult: Depth multiplier for C3k2 blocks
    """
    
    def __init__(
        self,
        in_channels: List[int],
        depth_mult: float = 0.33
    ):
        super().__init__()
        
        c3, c4, c5 = in_channels
        n = max(round(3 * depth_mult), 1)
        
        self.out_channels = [c3, c4, c5]
        
        # FPN (Top-down)
        self.upsample1 = Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv1 = Conv(c5, c4, 1, 1)
        self.fpn_c2f1 = C3k2(c4 + c4, c4, n=n, shortcut=False)
        
        self.upsample2 = Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv2 = Conv(c4, c3, 1, 1)
        self.fpn_c2f2 = C3k2(c3 + c3, c3, n=n, shortcut=False)
        
        # PAN (Bottom-up)
        self.downsample1 = Conv(c3, c3, 3, 2)
        self.pan_c2f1 = C3k2(c3 + c4, c4, n=n, shortcut=False)
        
        self.downsample2 = Conv(c4, c4, 3, 2)
        self.pan_c2f2 = C3k2(c4 + c5, c5, n=n, shortcut=False)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise Conv2d (Kaiming Normal) and BatchNorm2d (identity) weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Tuple of (P3, P4, P5) from backbone
        
        Returns:
            Tuple of fused features (N3, N4, N5):
                N3: For small objects
                N4: For medium objects
                N5: For large objects
        """
        p3, p4, p5 = features
        
        # FPN (Top-down)
        fpn_p5 = self.lateral_conv1(p5)
        fpn_p5_up = self.upsample1(fpn_p5)
        fpn_n4 = self.fpn_c2f1(torch.cat([fpn_p5_up, p4], dim=1))
        
        fpn_n4_reduced = self.lateral_conv2(fpn_n4)
        fpn_n4_up = self.upsample2(fpn_n4_reduced)
        fpn_n3 = self.fpn_c2f2(torch.cat([fpn_n4_up, p3], dim=1))
        
        # PAN (Bottom-up)
        pan_n3_down = self.downsample1(fpn_n3)
        pan_n4 = self.pan_c2f1(torch.cat([pan_n3_down, fpn_n4], dim=1))
        
        pan_n4_down = self.downsample2(pan_n4)
        pan_n5 = self.pan_c2f2(torch.cat([pan_n4_down, p5], dim=1))
        
        return fpn_n3, pan_n4, pan_n5
    
    def get_out_channels(self) -> List[int]:
        """Return output channel counts ``[C_N3, C_N4, C_N5]``.

        The values equal the backbone's P3/P4/P5 channel counts because the
        PANet preserves channel width at each scale.
        """
        return self.out_channels


if __name__ == "__main__":
    in_channels = [128, 256, 512]
    
    neck = PANet(in_channels, depth_mult=0.33)
    
    p3 = torch.randn(1, 128, 80, 80)
    p4 = torch.randn(1, 256, 40, 40)
    p5 = torch.randn(1, 512, 20, 20)
    
    n3, n4, n5 = neck((p3, p4, p5))
    
    params = sum(p.numel() for p in neck.parameters()) / 1e6
    print(f"PANet:")
    print(f"  Parameters: {params:.2f}M")
    print(f"  N3: {n3.shape}, N4: {n4.shape}, N5: {n5.shape}")
