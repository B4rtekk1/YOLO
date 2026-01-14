"""
YOLOv11 Building Blocks
Core components: C3k2 (efficient CSP), C2PSA (spatial attention), and standard blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    """Calculate padding to maintain spatial dimensions."""
    if padding is None:
        padding = (kernel_size - 1) // 2 * dilation
    return padding


class Conv(nn.Module):
    """
    Standard convolution block: Conv2d + BatchNorm2d + SiLU
    
    The basic building block used throughout the architecture.
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
        """Forward pass without batch norm (for fused inference)."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depthwise convolution with groups=in_channels."""
    
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
    """Depthwise transposed convolution."""
    
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
    Standard bottleneck block with optional residual connection.
    Structure: 1x1 conv -> 3x3 conv [+ residual]
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


class SE(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Adaptively recalibrates channel-wise feature responses.
    Reference: https://arxiv.org/abs/1709.01507
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.avg_pool(x).view(b, c)
        max_ = self.max_pool(x).view(b, c)
        w = torch.sigmoid(self.fc(avg) + self.fc(max_)).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * w


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies sequential channel and spatial attention.
    Reference: https://arxiv.org/abs/1807.06521
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions.
    Improves gradient flow while maintaining computational efficiency.
    
    Structure: Input -> Conv1x1 -> Split -> [n x Bottleneck] -> Concat -> Conv1x1 -> Output
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
        self.c = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, groups, expansion=1.0)
            for _ in range(n)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Alternative forward using split instead of chunk."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast
    
    Uses sequential max pooling for efficiency while maintaining receptive field.
    Structure: Input -> Conv1x1 -> [MaxPool5x5]x3 (sequential) -> Concat -> Conv1x1 -> Output
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
    """Upsampling layer using interpolation."""
    
    def __init__(self, scale_factor: int = 2, mode: str = 'nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Proto(nn.Module):
    """
    Mask prototype module for instance segmentation.
    Generates prototype masks combined with mask coefficients for instance-specific masks.
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
    Distribution Focal Loss layer for box regression (vectorized).
    
    Converts discrete probability distribution over bins [0, 1, ..., reg_max-1]
    to continuous values, capturing localization uncertainty.
    """
    
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        # Register weights as buffer for efficient computation
        self.register_buffer('weights', torch.arange(reg_max, dtype=torch.float).view(1, 1, reg_max, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized DFL forward pass.
        
        Args:
            x: Input tensor (B, 4*reg_max, H, W) or (B, N, 4*reg_max)
        
        Returns:
            Integrated values (B, 4, H, W) or (B, N, 4)
        """
        b, c, *hw = x.shape
        
        # Reshape to (B, 4, reg_max, H*W)
        x = x.view(b, 4, self.reg_max, -1)
        
        # Softmax over reg_max dimension
        x = F.softmax(x, dim=2)
        
        # Vectorized weighted sum: (B, 4, reg_max, N) * (1, 1, reg_max, 1) -> sum -> (B, 4, N)
        x = (x * self.weights).sum(dim=2)
        
        # Reshape back to original spatial dimensions
        if hw:
            x = x.view(b, 4, *hw)
        
        return x


# YOLOv11 Specific Blocks

class C3k(nn.Module):
    """
    CSP Bottleneck with 3 convolutions and customizable kernel size.
    
    Args:
        c1: Input channels
        c2: Output channels
        n: Number of bottleneck blocks
        shortcut: Use residual connection
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
    YOLOv11 CSP Bottleneck with 2 convolutions (efficient version).
    
    Replaces C2f with more efficient implementation using smaller kernels.
    Key differences from C2f:
    - Uses C3k sub-blocks for improved feature extraction
    - Better gradient flow
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
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        
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
    Spatial attention module for YOLOv11.
    Uses PyTorch 2.0+ scaled_dot_product_attention for efficiency (Flash Attention).
    """
    
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = max(1, min(num_heads, dim // 8))
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = Conv(dim, dim * 3, 1)
        self.proj = Conv(dim, dim, 1)
        
        # Check if scaled_dot_product_attention is available (PyTorch 2.0+)
        self._use_sdpa = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        
        qkv = self.qkv(x)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (B, heads, head_dim, N)
        
        if self._use_sdpa:
            # Use PyTorch 2.0+ Flash Attention (2-3x faster)
            # Transpose to (B, heads, N, head_dim) for SDPA
            q = q.transpose(-2, -1)  # (B, heads, N, head_dim)
            k = k.transpose(-2, -1)
            v = v.transpose(-2, -1)
            
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            out = out.transpose(-2, -1)  # (B, heads, head_dim, N)
        else:
            # Fallback to manual attention for older PyTorch
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = v @ attn.transpose(-2, -1)
        
        out = out.reshape(B, C, H, W)
        return self.proj(out)


class C2PSA(nn.Module):
    """
    YOLOv11 Cross Stage Partial with Spatial Attention (C2PSA).
    
    Enhances spatial attention within feature maps, enabling the model
    to focus on critical image regions.
    
    Structure: Input -> Conv1x1 -> Split -> [Attention blocks] -> Concat -> Conv1x1 -> Output
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
        
        self.m = nn.Sequential(
            *[
                nn.Sequential(
                    Attention(self.c, num_heads=self.c // 64 if self.c >= 64 else 1),
                    nn.Sequential(
                        Conv(self.c, self.c * 2, 1),
                        Conv(self.c * 2, self.c, 1)
                    )
                )
                for _ in range(n)
            ]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, 1)
        b = self.m(b)
        return self.cv2(torch.cat([a, b], 1))
