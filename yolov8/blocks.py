"""
YOLOv11 Building Blocks
Basic components used throughout the architecture
Includes: C3k2 (faster CSP), C2PSA (spatial attention), and legacy C2f
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    """Calculate padding to maintain spatial dimensions."""
    if padding is None:
        padding = (kernel_size - 1) // 2 * dilation
    return padding


class Conv(nn.Module):
    """
    Standard Convolution block: Conv2d + BatchNorm2d + SiLU activation
    
    This is the most basic building block used throughout YOLOv8
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        activation: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without batch norm (for inference after fusion)."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """
    Depthwise Convolution
    Uses groups=in_channels for depthwise separable convolution
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        activation: bool = True
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=in_channels,
            dilation=dilation,
            activation=activation
        )


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depthwise Transposed Convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups=in_channels,
            bias=False
        )


class Bottleneck(nn.Module):
    """
    Standard Bottleneck block with residual connection
    
    Structure: 1x1 conv -> 3x3 conv with optional residual
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions (faster version of C3)
    
    This is a key component in YOLOv8 that improves gradient flow
    while maintaining computational efficiency.
    
    Structure:
        Input -> [Conv 1x1] -> Split -> [n x Bottleneck] -> Concat -> [Conv 1x1] -> Output
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        expansion: float = 0.5
    ):
        super().__init__()
        self.c = int(out_channels * expansion)  # hidden channels
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, groups, expansion=1.0)
            for _ in range(n)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input into two parts
        y = list(self.cv1(x).chunk(2, 1))
        # Apply bottlenecks sequentially, keeping all intermediate outputs
        y.extend(m(y[-1]) for m in self.m)
        # Concatenate all features and apply final conv
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Alternative forward with split instead of chunk."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF)
    
    Uses sequential max pooling operations instead of parallel ones
    for better computational efficiency while maintaining receptive field.
    
    Structure:
        Input -> Conv 1x1 -> [MaxPool 5x5]×3 (sequential) -> Concat -> Conv 1x1 -> Output
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class Concat(nn.Module):
    """Concatenation layer along specified dimension."""
    
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


class Upsample(nn.Module):
    """Upsampling layer using bilinear interpolation."""
    
    def __init__(self, scale_factor: int = 2, mode: str = 'nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Proto(nn.Module):
    """
    YOLOv8 mask Proto module for segmentation
    
    Generates prototype masks that are combined with mask coefficients
    to produce instance-specific masks.
    """
    
    def __init__(
        self,
        in_channels: int,
        proto_channels: int = 256,
        num_protos: int = 32
    ):
        super().__init__()
        self.cv1 = Conv(in_channels, proto_channels, 3)
        self.upsample = nn.ConvTranspose2d(
            proto_channels, proto_channels, 2, 2, 0, bias=True
        )
        self.cv2 = Conv(proto_channels, proto_channels, 3)
        self.cv3 = Conv(proto_channels, num_protos, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class DFL(nn.Module):
    """
    Distribution Focal Loss layer
    
    Integral module for distribution-based box regression in YOLOv8.
    Converts discrete probability distribution to continuous values.
    
    reg_max: Maximum discrete regression distance
    """
    
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        # Create fixed weight for integration
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        # Initialize weights as [0, 1, 2, ..., reg_max-1]
        x = torch.arange(reg_max, dtype=torch.float)
        self.conv.weight.data[:] = x.view(1, reg_max, 1, 1)
        self.conv.weight.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply softmax and integrate to get continuous value.
        
        Args:
            x: Input tensor of shape (B, 4*reg_max, H, W) or (B, N, 4*reg_max)
        
        Returns:
            Integrated values of shape (B, 4, H, W) or (B, N, 4)
        """
        b, c, *hw = x.shape
        # Reshape to (B, 4, reg_max, H*W) for softmax
        x = x.view(b, 4, self.reg_max, -1).transpose(2, 3)
        # Apply softmax over reg_max dimension
        x = F.softmax(x, dim=-1)
        # Integrate: multiply by [0, 1, 2, ...] and sum
        x = x.transpose(2, 3).reshape(b, 4 * self.reg_max, *hw)
        # Use conv to integrate
        x_split = x.chunk(4, dim=1)
        return torch.cat([self.conv(xi) for xi in x_split], dim=1)


# ============================================================
# YOLOv11 NEW BLOCKS
# ============================================================

class C3k(nn.Module):
    """
    CSP Bottleneck with 3 convolutions and customizable kernel size.
    
    Args:
        c1: Input channels
        c2: Output channels
        n: Number of bottleneck blocks
        shortcut: Whether to use residual connection
        g: Groups for convolution
        e: Expansion ratio
        k: Kernel size for bottleneck convs
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, expansion=1.0) for _ in range(n)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    """
    YOLOv11 CSP Bottleneck with 2 convolutions (faster version).
    
    This is a key improvement in YOLOv11, replacing C2f with a more
    computationally efficient implementation using smaller kernels.
    
    Main differences from C2f:
    - Uses C3k sub-blocks for better feature extraction
    - More efficient gradient flow
    - Reduced computational cost
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True
    ):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        
        # Use C3k blocks if c3k=True, otherwise standard Bottleneck
        if c3k:
            self.m = nn.ModuleList(
                C3k(self.c, self.c, 2, shortcut, g) for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                Bottleneck(self.c, self.c, shortcut, g, expansion=1.0)
                for _ in range(n)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    """
    Simplified Spatial Attention Module for YOLOv11.
    
    Uses channel attention and spatial attention mechanisms.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = max(1, min(num_heads, dim // 8))  # Ensure valid num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection - output same dimension
        self.qkv = Conv(dim, dim * 3, 1)
        
        # Output projection
        self.proj = Conv(dim, dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        
        # Compute QKV
        qkv = self.qkv(x)  # (B, 3C, H, W)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (B, num_heads, head_dim, N)
        
        # Attention: (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values: (B, num_heads, head_dim, N) @ (B, num_heads, N, N) -> (B, num_heads, head_dim, N)
        out = v @ attn.transpose(-2, -1)
        out = out.view(B, C, H, W)
        
        return self.proj(out)



class C2PSA(nn.Module):
    """
    YOLOv11 Cross Stage Partial with Spatial Attention (C2PSA).
    
    This is a key new block in YOLOv11 that enhances spatial attention
    within feature maps, enabling the model to focus more effectively
    on critical regions in an image.
    
    Structure:
        Input -> Conv 1x1 -> Split -> [Attention blocks] -> Concat -> Conv 1x1 -> Output
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1, 1)
        
        # Stack of attention blocks
        self.m = nn.Sequential(
            *[
                nn.Sequential(
                    Attention(self.c, num_heads=self.c // 64 if self.c >= 64 else 1),
                    nn.Sequential(
                        Conv(self.c, self.c * 2, 1),
                        Conv(self.c * 2, self.c, 1)
                    )  # FFN
                )
                for _ in range(n)
            ]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into two branches
        a, b = self.cv1(x).chunk(2, 1)
        
        # Apply attention to first branch
        b = self.m(b)
        
        # Concatenate and project
        return self.cv2(torch.cat([a, b], 1))
