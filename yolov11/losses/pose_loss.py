"""
Pose Estimation Loss for YOLOv11
Keypoint loss using OKS (Object Keypoint Similarity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# COCO keypoint sigmas for OKS calculation
# Smaller sigma = stricter matching required
COCO_KEYPOINT_SIGMAS = torch.tensor([
    0.026,  # nose
    0.025,  # left_eye
    0.025,  # right_eye
    0.035,  # left_ear
    0.035,  # right_ear
    0.079,  # left_shoulder
    0.079,  # right_shoulder
    0.072,  # left_elbow
    0.072,  # right_elbow
    0.062,  # left_wrist
    0.062,  # right_wrist
    0.107,  # left_hip
    0.107,  # right_hip
    0.087,  # left_knee
    0.087,  # right_knee
    0.089,  # left_ankle
    0.089,  # right_ankle
])


class OKSLoss(nn.Module):
    """
    Object Keypoint Similarity (OKS) Loss for pose estimation.
    
    OKS measures similarity between predicted and ground truth keypoints,
    accounting for object scale and keypoint-specific uncertainty.
    
    OKS = exp(-d^2 / (2 * s^2 * k^2))
    
    where:
    - d: Euclidean distance between predicted and GT keypoint
    - s: Object scale (sqrt of bounding box area)
    - k: Keypoint-specific sigma (from COCO)
    
    Args:
        sigmas: Per-keypoint sigma values
        use_visibility: Whether to use visibility flags
    """
    
    def __init__(
        self,
        sigmas: torch.Tensor = None,
        use_visibility: bool = True,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        if sigmas is None:
            sigmas = COCO_KEYPOINT_SIGMAS
        
        self.register_buffer('sigmas', sigmas)
        self.use_visibility = use_visibility
        self.reduction = reduction
    
    def forward(
        self,
        pred_kpts: torch.Tensor,
        target_kpts: torch.Tensor,
        target_vis: torch.Tensor,
        area: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate OKS loss.
        
        Args:
            pred_kpts: Predicted keypoints (N, 17, 2) - x, y coordinates
            target_kpts: Target keypoints (N, 17, 2)
            target_vis: Target visibility flags (N, 17)
            area: Object areas for scale (N,)
        
        Returns:
            OKS loss value
        """
        # Squared distance between predicted and target
        d2 = ((pred_kpts - target_kpts) ** 2).sum(dim=-1)  # (N, 17)
        
        # Scale factor
        s2 = area.unsqueeze(-1)  # (N, 1)
        
        # Keypoint-specific variance
        k2 = (self.sigmas ** 2).unsqueeze(0)  # (1, 17)
        
        # OKS for each keypoint
        oks = torch.exp(-d2 / (2 * s2 * k2 + 1e-9))  # (N, 17)
        
        # Apply visibility mask
        if self.use_visibility:
            valid_mask = target_vis > 0
            oks = oks * valid_mask.float()
            num_valid = valid_mask.sum(dim=-1).clamp(min=1)
            oks = oks.sum(dim=-1) / num_valid  # (N,)
        else:
            oks = oks.mean(dim=-1)  # (N,)
        
        # Loss = 1 - OKS
        loss = 1.0 - oks
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class KeypointL1Loss(nn.Module):
    """
    L1 Loss for keypoint regression.
    
    Simple but effective loss for keypoint coordinate regression.
    Typically combined with OKS for better results.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred_kpts: torch.Tensor,
        target_kpts: torch.Tensor,
        target_vis: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculate L1 loss for keypoints.
        
        Args:
            pred_kpts: Predicted keypoints (N, 17, 2)
            target_kpts: Target keypoints (N, 17, 2)
            target_vis: Optional visibility flags (N, 17)
        
        Returns:
            L1 loss value
        """
        loss = F.l1_loss(pred_kpts, target_kpts, reduction='none')
        loss = loss.sum(dim=-1)  # Sum x, y components (N, 17)
        
        if target_vis is not None:
            valid_mask = target_vis > 0
            loss = loss * valid_mask.float()
            loss = loss.sum() / valid_mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss


class VisibilityLoss(nn.Module):
    """
    Binary Cross Entropy Loss for keypoint visibility prediction.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(
        self,
        pred_vis: torch.Tensor,
        target_vis: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate visibility loss.
        
        Args:
            pred_vis: Predicted visibility logits (N, 17)
            target_vis: Target visibility flags (N, 17) - 0, 1, or 2
                       0: not labeled, 1: labeled but not visible, 2: visible
        
        Returns:
            Visibility loss value
        """
        # Convert to binary: visible (2) vs not visible (0, 1)
        target_binary = (target_vis == 2).float()
        
        return self.bce(pred_vis, target_binary)


class PoseLoss(nn.Module):
    """
    Combined Pose Estimation Loss for YOLOv11.
    
    Combines:
    - OKS Loss for keypoint similarity
    - L1 Loss for coordinate regression
    - BCE Loss for visibility prediction
    
    Args:
        oks_weight: Weight for OKS loss
        l1_weight: Weight for L1 loss
        vis_weight: Weight for visibility loss
        sigmas: Per-keypoint sigma values for OKS
    """
    
    def __init__(
        self,
        oks_weight: float = 1.0,
        l1_weight: float = 0.5,
        vis_weight: float = 1.0,
        sigmas: torch.Tensor = None
    ):
        super().__init__()
        
        self.oks_weight = oks_weight
        self.l1_weight = l1_weight
        self.vis_weight = vis_weight
        
        self.oks_loss = OKSLoss(sigmas=sigmas)
        self.l1_loss = KeypointL1Loss()
        self.vis_loss = VisibilityLoss()
    
    def forward(
        self,
        pred_kpts: torch.Tensor,
        pred_vis: torch.Tensor,
        target_kpts: torch.Tensor,
        target_vis: torch.Tensor,
        area: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate combined pose loss.
        
        Args:
            pred_kpts: Predicted keypoints (N, 17, 2)
            pred_vis: Predicted visibility logits (N, 17)
            target_kpts: Target keypoints (N, 17, 2)
            target_vis: Target visibility flags (N, 17)
            area: Object areas (N,)
        
        Returns:
            Tuple of (total_loss, dict with individual losses)
        """
        # OKS loss
        oks = self.oks_loss(pred_kpts, target_kpts, target_vis, area)
        
        # L1 loss
        l1 = self.l1_loss(pred_kpts, target_kpts, target_vis)
        
        # Visibility loss
        vis = self.vis_loss(pred_vis, target_vis)
        
        # Total loss
        total = (self.oks_weight * oks + 
                 self.l1_weight * l1 + 
                 self.vis_weight * vis)
        
        return total, {
            'oks': oks,
            'l1': l1,
            'vis': vis
        }


class WingLoss(nn.Module):
    """
    Wing Loss for keypoint regression.
    
    Better than L1/L2 for small errors, commonly used in facial landmark detection.
    
    w(x) = ln(1 + |x|/ε) * w  if |x| < w
         = |x| - C            otherwise
    
    where C = w - w * ln(1 + w/ε)
    
    Args:
        width: Width parameter (w)
        epsilon: Curvature parameter (ε)
    """
    
    def __init__(
        self,
        width: float = 10.0,
        epsilon: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.width = width
        self.epsilon = epsilon
        self.reduction = reduction
        
        # Precompute constant
        self.C = width - width * torch.log(torch.tensor(1.0 + width / epsilon))
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculate Wing loss.
        
        Args:
            pred: Predicted coordinates (N, K, 2)
            target: Target coordinates (N, K, 2)
            mask: Optional mask for valid keypoints (N, K)
        
        Returns:
            Wing loss value
        """
        diff = pred - target
        abs_diff = diff.abs()
        
        # Wing loss formula
        small_error = abs_diff < self.width
        loss = torch.where(
            small_error,
            self.width * torch.log(1.0 + abs_diff / self.epsilon),
            abs_diff - self.C
        )
        
        loss = loss.sum(dim=-1)  # Sum x, y components
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


if __name__ == "__main__":
    # Test pose losses
    print("Testing Pose Losses...")
    
    N = 8
    pred_kpts = torch.randn(N, 17, 2) * 100  # Random keypoints
    target_kpts = pred_kpts + torch.randn(N, 17, 2) * 5  # Add noise
    target_vis = torch.randint(0, 3, (N, 17))  # Random visibility
    area = torch.rand(N) * 10000 + 1000  # Random areas
    
    # OKS Loss
    oks_loss = OKSLoss()
    loss = oks_loss(pred_kpts, target_kpts, target_vis, area)
    print(f"OKS Loss: {loss.item():.4f}")
    
    # Keypoint L1 Loss
    l1_loss = KeypointL1Loss()
    loss = l1_loss(pred_kpts, target_kpts, target_vis)
    print(f"Keypoint L1 Loss: {loss.item():.4f}")
    
    # Combined Pose Loss
    pred_vis = torch.randn(N, 17)
    pose_loss = PoseLoss()
    total, loss_dict = pose_loss(pred_kpts, pred_vis, target_kpts, target_vis, area)
    print(f"Pose Loss: {total.item():.4f}")
    print(f"  OKS: {loss_dict['oks'].item():.4f}")
    print(f"  L1: {loss_dict['l1'].item():.4f}")
    print(f"  Vis: {loss_dict['vis'].item():.4f}")
    
    # Wing Loss
    wing_loss = WingLoss()
    loss = wing_loss(pred_kpts, target_kpts, (target_vis > 0).float())
    print(f"Wing Loss: {loss.item():.4f}")
