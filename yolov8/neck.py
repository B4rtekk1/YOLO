"""
YOLOv11 Neck - PANet (Path Aggregation Network)
Combines FPN (top-down) and PAN (bottom-up) pathways with C3k2 blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from .blocks import Conv, C3k2, Concat, Upsample


class PANet(nn.Module):
    """
    YOLOv11 Neck combining FPN and PAN with C3k2 blocks
    
    Architecture:
        FPN (Top-down pathway):
            P5 -> Upsample + Concat(P4) -> C3k2 -> N4
            N4 -> Upsample + Concat(P3) -> C3k2 -> N3
        
        PAN (Bottom-up pathway):
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
        
        c3, c4, c5 = in_channels  # e.g., [128, 256, 512] for YOLOv8s
        
        # Calculate depth for C3k2 blocks
        n = max(round(3 * depth_mult), 1)
        
        # Store output channels
        self.out_channels = [c3, c4, c5]
        
        # =================
        # FPN (Top-down)
        # =================
        
        # P5 -> N4: Upsample P5, concat with P4
        self.upsample1 = Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv1 = Conv(c5, c4, 1, 1)  # Reduce channels before concat
        self.fpn_c2f1 = C3k2(c4 + c4, c4, n=n, shortcut=False)
        
        # N4 -> N3: Upsample N4, concat with P3
        self.upsample2 = Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv2 = Conv(c4, c3, 1, 1)  # Reduce channels before concat
        self.fpn_c2f2 = C3k2(c3 + c3, c3, n=n, shortcut=False)
        
        # =================
        # PAN (Bottom-up)
        # =================
        
        # N3 -> N4': Downsample N3, concat with N4
        self.downsample1 = Conv(c3, c3, 3, 2)  # Stride 2 conv
        self.pan_c2f1 = C3k2(c3 + c4, c4, n=n, shortcut=False)
        
        # N4' -> N5: Downsample N4', concat with P5
        self.downsample2 = Conv(c4, c4, 3, 2)  # Stride 2 conv
        self.pan_c2f2 = C3k2(c4 + c5, c5, n=n, shortcut=False)
        
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
    
    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through neck.
        
        Args:
            features: Tuple of (P3, P4, P5) from backbone
        
        Returns:
            Tuple of fused features (N3, N4, N5):
                N3: Same resolution as P3, for small objects
                N4: Same resolution as P4, for medium objects
                N5: Same resolution as P5, for large objects
        """
        p3, p4, p5 = features
        
        # =================
        # FPN (Top-down)
        # =================
        
        # P5 -> N4
        fpn_p5 = self.lateral_conv1(p5)  # Reduce channels
        fpn_p5_up = self.upsample1(fpn_p5)  # Upsample
        fpn_n4 = self.fpn_c2f1(torch.cat([fpn_p5_up, p4], dim=1))
        
        # N4 -> N3
        fpn_n4_reduced = self.lateral_conv2(fpn_n4)  # Reduce channels
        fpn_n4_up = self.upsample2(fpn_n4_reduced)  # Upsample
        fpn_n3 = self.fpn_c2f2(torch.cat([fpn_n4_up, p3], dim=1))
        
        # =================
        # PAN (Bottom-up)
        # =================
        
        # N3 -> N4'
        pan_n3_down = self.downsample1(fpn_n3)  # Downsample
        pan_n4 = self.pan_c2f1(torch.cat([pan_n3_down, fpn_n4], dim=1))
        
        # N4' -> N5
        pan_n4_down = self.downsample2(pan_n4)  # Downsample
        pan_n5 = self.pan_c2f2(torch.cat([pan_n4_down, p5], dim=1))
        
        return fpn_n3, pan_n4, pan_n5
    
    def get_out_channels(self) -> List[int]:
        """Get output channel sizes."""
        return self.out_channels


if __name__ == "__main__":
    # Test neck
    in_channels = [128, 256, 512]  # YOLOv8s backbone outputs
    
    neck = PANet(in_channels, depth_mult=0.33)
    
    # Simulate backbone outputs
    p3 = torch.randn(1, 128, 80, 80)
    p4 = torch.randn(1, 256, 40, 40)
    p5 = torch.randn(1, 512, 20, 20)
    
    n3, n4, n5 = neck((p3, p4, p5))
    
    params = sum(p.numel() for p in neck.parameters()) / 1e6
    print(f"PANet:")
    print(f"  Parameters: {params:.2f}M")
    print(f"  N3: {n3.shape}, N4: {n4.shape}, N5: {n5.shape}")
