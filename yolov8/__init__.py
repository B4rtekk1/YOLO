"""
YOLOv11 From Scratch Implementation
Supports: Detection, Segmentation, Pose Estimation

Key improvements over YOLOv8:
- C3k2 blocks for better efficiency
- C2PSA spatial attention
- Fewer parameters with higher accuracy
"""

from .model import YOLOv11, create_model
from .backbone import CSPDarknet
from .neck import PANet
from .head import DetectionHead, SegmentationHead, PoseHead

# Backward compatibility alias
YOLOv8 = YOLOv11

__version__ = "11.0.0"
__all__ = [
    "YOLOv11",
    "YOLOv8",  # alias for backward compatibility
    "create_model",
    "CSPDarknet", 
    "PANet",
    "DetectionHead",
    "SegmentationHead", 
    "PoseHead"
]
