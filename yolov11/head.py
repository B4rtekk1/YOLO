"""
YOLOv11 Detection Heads
Supports Detection, Segmentation, and Pose Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math

from .blocks import Conv, DFL, Proto


# COCO keypoint definitions (17 keypoints)
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# Skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]


class DetectionHead(nn.Module):
    """
    Anchor-free detection head with decoupled classification and regression.
    
    Uses Distribution Focal Loss (DFL) for box regression.
    
    Args:
        num_classes: Number of detection classes
        in_channels: List of input channels from neck [N3, N4, N5]
        reg_max: Maximum discrete regression value for DFL
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        in_channels: List[int] = [128, 256, 512],
        reg_max: int = 16
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs_per_anchor = num_classes + 4 * reg_max
        
        self.dfl = DFL(reg_max)
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        for ch in in_channels:
            self.cls_convs.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1)
            ))
            
            self.reg_convs.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1)
            ))
            
            self.cls_preds.append(nn.Conv2d(ch, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(ch, 4 * reg_max, 1))
        
        self._initialize_biases()
    
    def _initialize_biases(self):
        """Initialize prediction biases for training stability."""
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(-1, )
            b.data.fill_(-math.log((1 - 0.01) / 0.01))  # Prior probability 0.01
            cls_pred.bias = nn.Parameter(b, requires_grad=True)
    
    def forward(
        self,
        features: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Tuple of feature maps (N3, N4, N5) from neck
        
        Returns:
            Tuple of (cls_outputs, reg_outputs) per scale
        """
        cls_outputs = []
        reg_outputs = []
        
        for i, feat in enumerate(features):
            cls_feat = self.cls_convs[i](feat)
            cls_out = self.cls_preds[i](cls_feat)
            
            reg_feat = self.reg_convs[i](feat)
            reg_out = self.reg_preds[i](reg_feat)
            
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
        
        return cls_outputs, reg_outputs
    
    def decode_boxes(
        self,
        reg_outputs: List[torch.Tensor],
        anchors: List[torch.Tensor],
        strides: List[int]
    ) -> torch.Tensor:
        """
        Decode box predictions using DFL.
        
        Args:
            reg_outputs: List of regression outputs per scale
            anchors: List of anchor points per scale
            strides: List of strides for each scale
        
        Returns:
            Decoded boxes in xyxy format
        """
        boxes = []
        
        for reg_out, anchor, stride in zip(reg_outputs, anchors, strides):
            b, _, h, w = reg_out.shape
            
            reg_dist = self.dfl(reg_out)
            reg_dist = reg_dist.view(b, 4, -1).permute(0, 2, 1)
            anchor = anchor.view(-1, 2)
            
            lt = reg_dist[..., :2]
            rb = reg_dist[..., 2:]
            
            x1y1 = anchor - lt * stride
            x2y2 = anchor + rb * stride
            
            boxes.append(torch.cat([x1y1, x2y2], dim=-1))
        
        return torch.cat(boxes, dim=1)


class SegmentationHead(nn.Module):
    """
    Instance segmentation head extending detection with mask coefficients.
    
    Uses prototype masks linearly combined with per-instance coefficients.
    
    Args:
        num_classes: Number of detection classes
        in_channels: List of input channels from neck
        num_protos: Number of prototype masks
        proto_channels: Channels for prototype generation
        reg_max: Maximum discrete regression value for DFL
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        in_channels: List[int] = [128, 256, 512],
        num_protos: int = 32,
        proto_channels: int = 256,
        reg_max: int = 16
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_protos = num_protos
        self.reg_max = reg_max
        
        self.dfl = DFL(reg_max)
        self.proto = Proto(in_channels[0], proto_channels, num_protos)
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.mask_preds = nn.ModuleList()
        
        for ch in in_channels:
            self.cls_convs.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1)
            ))
            
            self.reg_convs.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1)
            ))
            
            self.cls_preds.append(nn.Conv2d(ch, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(ch, 4 * reg_max, 1))
            self.mask_preds.append(nn.Conv2d(ch, num_protos, 1))
        
        self._initialize_biases()
    
    def _initialize_biases(self):
        """Initialize prediction biases."""
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(-1, )
            b.data.fill_(-math.log((1 - 0.01) / 0.01))
            cls_pred.bias = nn.Parameter(b, requires_grad=True)
    
    def forward(
        self,
        features: Tuple[torch.Tensor, ...]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Tuple of feature maps (N3, N4, N5) from neck
        
        Returns:
            Tuple of (cls_outputs, reg_outputs, mask_coeffs, protos)
        """
        cls_outputs = []
        reg_outputs = []
        mask_outputs = []
        
        protos = self.proto(features[0])
        
        for i, feat in enumerate(features):
            cls_feat = self.cls_convs[i](feat)
            cls_out = self.cls_preds[i](cls_feat)
            
            reg_feat = self.reg_convs[i](feat)
            reg_out = self.reg_preds[i](reg_feat)
            mask_out = self.mask_preds[i](reg_feat)
            
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            mask_outputs.append(mask_out)
        
        return cls_outputs, reg_outputs, mask_outputs, protos
    
    def assemble_masks(
        self,
        mask_coeffs: torch.Tensor,
        protos: torch.Tensor,
        boxes: torch.Tensor,
        img_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Assemble instance masks from prototypes and coefficients.
        
        Args:
            mask_coeffs: Mask coefficients (N, num_protos)
            protos: Prototype masks (B, num_protos, H, W)
            boxes: Detection boxes in xyxy format (N, 4)
            img_size: Original image size (H, W)
        
        Returns:
            Instance masks (N, H, W)
        """
        ph, pw = protos.shape[2:]
        protos_flat = protos[0].view(self.num_protos, -1)
        
        masks = torch.mm(mask_coeffs, protos_flat)
        masks = masks.sigmoid().view(-1, ph, pw)
        
        masks = self._crop_masks(masks, boxes, img_size)
        
        return masks
    
    def _crop_masks(
        self,
        masks: torch.Tensor,
        boxes: torch.Tensor,
        img_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Crop and resize masks to image size."""
        h, w = img_size
        n = masks.shape[0]
        
        masks = F.interpolate(
            masks.unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        output_masks = torch.zeros((n, h, w), device=masks.device)
        
        for i in range(n):
            x1, y1, x2, y2 = boxes[i].int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            output_masks[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
        
        return output_masks


class PoseHead(nn.Module):
    """
    Pose estimation head extending detection with keypoint prediction.
    
    Predicts 17 COCO keypoints per detection.
    
    Args:
        num_classes: Number of detection classes (typically 1 for person)
        in_channels: List of input channels from neck
        num_keypoints: Number of keypoints to predict (17 for COCO)
        reg_max: Maximum discrete regression value for DFL
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        in_channels: List[int] = [128, 256, 512],
        num_keypoints: int = 17,
        reg_max: int = 16
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.reg_max = reg_max
        
        self.dfl = DFL(reg_max)
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kpt_convs = nn.ModuleList()
        
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.kpt_preds = nn.ModuleList()
        
        for ch in in_channels:
            self.cls_convs.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1)
            ))
            
            self.reg_convs.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1)
            ))
            
            self.kpt_convs.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1)
            ))
            
            self.cls_preds.append(nn.Conv2d(ch, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(ch, 4 * reg_max, 1))
            self.kpt_preds.append(nn.Conv2d(ch, num_keypoints * 3, 1))
        
        self._initialize_biases()
    
    def _initialize_biases(self):
        """Initialize prediction biases."""
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(-1, )
            b.data.fill_(-math.log((1 - 0.01) / 0.01))
            cls_pred.bias = nn.Parameter(b, requires_grad=True)
    
    def forward(
        self,
        features: Tuple[torch.Tensor, ...]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: Tuple of feature maps (N3, N4, N5) from neck
        
        Returns:
            Tuple of (cls_outputs, reg_outputs, kpt_outputs)
        """
        cls_outputs = []
        reg_outputs = []
        kpt_outputs = []
        
        for i, feat in enumerate(features):
            cls_feat = self.cls_convs[i](feat)
            cls_out = self.cls_preds[i](cls_feat)
            
            reg_feat = self.reg_convs[i](feat)
            reg_out = self.reg_preds[i](reg_feat)
            
            kpt_feat = self.kpt_convs[i](feat)
            kpt_out = self.kpt_preds[i](kpt_feat)
            
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            kpt_outputs.append(kpt_out)
        
        return cls_outputs, reg_outputs, kpt_outputs
    
    def decode_keypoints(
        self,
        kpt_outputs: List[torch.Tensor],
        anchors: List[torch.Tensor],
        strides: List[int]
    ) -> torch.Tensor:
        """
        Decode keypoint predictions.
        
        Args:
            kpt_outputs: Keypoint predictions per scale
            anchors: Anchor points per scale
            strides: Strides for each scale
        
        Returns:
            Decoded keypoints (B, N, num_keypoints, 3) with (x, y, visibility)
        """
        keypoints = []
        
        for kpt_out, anchor, stride in zip(kpt_outputs, anchors, strides):
            b, c, h, w = kpt_out.shape
            
            kpt_out = kpt_out.view(b, self.num_keypoints, 3, -1)
            kpt_out = kpt_out.permute(0, 3, 1, 2)
            
            anchor = anchor.view(-1, 2)
            
            xy = kpt_out[..., :2] * stride + anchor.unsqueeze(0).unsqueeze(2)
            visibility = kpt_out[..., 2:].sigmoid()
            
            keypoints.append(torch.cat([xy, visibility], dim=-1))
        
        return torch.cat(keypoints, dim=1)


if __name__ == "__main__":
    in_channels = [128, 256, 512]
    
    print("Testing DetectionHead...")
    det_head = DetectionHead(80, in_channels)
    feats = [torch.randn(1, c, 80//(2**i), 80//(2**i)) for i, c in enumerate(in_channels)]
    cls_out, reg_out = det_head(feats)
    print(f"  Cls shapes: {[c.shape for c in cls_out]}")
    print(f"  Reg shapes: {[r.shape for r in reg_out]}")
    
    print("\nTesting SegmentationHead...")
    seg_head = SegmentationHead(80, in_channels)
    cls_out, reg_out, mask_out, protos = seg_head(feats)
    print(f"  Cls shapes: {[c.shape for c in cls_out]}")
    print(f"  Mask shapes: {[m.shape for m in mask_out]}")
    print(f"  Proto shape: {protos.shape}")
    
    print("\nTesting PoseHead...")
    pose_head = PoseHead(1, in_channels, 17)
    cls_out, reg_out, kpt_out = pose_head(feats)
    print(f"  Cls shapes: {[c.shape for c in cls_out]}")
    print(f"  Kpt shapes: {[k.shape for k in kpt_out]}")
