"""
YOLOv11 Implementation
Supports Detection, Segmentation, and Pose Estimation

Key improvements over YOLOv8:
- C3k2 blocks for improved efficiency
- C2PSA spatial attention mechanism
- Reduced parameters with higher accuracy
"""

from .model import YOLOv11, create_model
from .backbone import CSPDarknet
from .neck import PANet
from .head import DetectionHead, SegmentationHead, PoseHead

__version__ = "11.0.0"
__all__ = [
    "YOLOv11",
    "create_model",
    "CSPDarknet", 
    "PANet",
    "DetectionHead",
    "SegmentationHead", 
    "PoseHead"
]
