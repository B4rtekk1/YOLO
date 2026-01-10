"""
Enhanced Loss Functions for YOLOv11
Includes: Wise-IoU, Quality Focal Loss, Label Smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WiseIoULoss(nn.Module):
    """
    Wise-IoU Loss with dynamic non-monotonic focusing mechanism.
    
    Handles outliers better than standard IoU losses by using
    a dynamic gradient weighting strategy.
    
    Reference: https://arxiv.org/abs/2301.10051
    """
    
    def __init__(self, eps: float = 1e-7, ratio: float = 0.8):
        super().__init__()
        self.eps = eps
        self.ratio = ratio
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted boxes (N, 4) in xyxy format
            target: Target boxes (N, 4) in xyxy format
        """
        # Calculate intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Calculate union
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = pred_area + target_area - inter_area + self.eps
        
        # IoU
        iou = inter_area / union
        
        # Calculate center distance
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        
        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Enclosing box diagonal
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        
        c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + self.eps
        
        # Wise-IoU focusing coefficient
        # Uses ratio of current IoU to expected IoU
        wise_scale = torch.exp((center_dist / c2) * self.ratio)
        
        # Wise-IoU loss
        loss = (1 - iou) * wise_scale
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss for dense object detection.
    
    Combines classification and localization quality into a single loss,
    enabling joint optimization.
    
    Reference: https://arxiv.org/abs/2006.04388
    """
    
    def __init__(self, beta: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        quality: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted class logits (N, C)
            target: Target class indices (N,)
            quality: IoU quality scores (N,), if None uses 1.0
        """
        # Convert to one-hot
        num_classes = pred.shape[-1]
        target_one_hot = F.one_hot(target.long(), num_classes).float()
        
        # Apply quality score
        if quality is not None:
            quality = quality.unsqueeze(-1)
            target_one_hot = target_one_hot * quality
        
        # Sigmoid activation
        pred_sigmoid = torch.sigmoid(pred)
        
        # Calculate focal weight
        pt = pred_sigmoid * target_one_hot + (1 - pred_sigmoid) * (1 - target_one_hot)
        focal_weight = (target_one_hot - pred_sigmoid).abs().pow(self.beta)
        
        # BCE loss with focal weight
        bce = F.binary_cross_entropy_with_logits(pred, target_one_hot, reduction='none')
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss - improved version of Quality Focal Loss.
    
    Uses IoU-aware classification score (IACS) as target.
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_score: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted class logits (N, C)
            target: Target class indices (N,)
            target_score: IoU quality scores (N,), if None uses 1.0
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Create soft labels with IoU score
        num_classes = pred.shape[-1]
        target_one_hot = F.one_hot(target.long().clamp(0, num_classes-1), num_classes).float()
        
        if target_score is not None:
            target_one_hot = target_one_hot * target_score.unsqueeze(-1)
        
        # Varifocal weighting
        focal_weight = target_one_hot * (target_one_hot > 0).float() + \
                       self.alpha * pred_sigmoid.pow(self.gamma) * (target_one_hot == 0).float()
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target_one_hot, reduction='none')
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingCE(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.
    
    Prevents overconfident predictions and improves generalization.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (N, C)  
            target: Target class indices (N,)
        """
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1).long(), self.confidence)
        
        loss = -true_dist * log_probs
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == "__main__":
    # Test losses
    print("Testing enhanced losses...")
    
    # Wise-IoU
    wise_iou = WiseIoULoss()
    pred_boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32)
    target_boxes = torch.tensor([[12, 12, 48, 48], [22, 22, 58, 58]], dtype=torch.float32)
    loss = wise_iou(pred_boxes, target_boxes)
    print(f"Wise-IoU Loss: {loss.item():.4f}")
    
    # Quality Focal Loss
    qfl = QualityFocalLoss()
    pred_cls = torch.randn(4, 80)
    target_cls = torch.randint(0, 80, (4,))
    quality = torch.rand(4)
    loss = qfl(pred_cls, target_cls, quality)
    print(f"Quality Focal Loss: {loss.item():.4f}")
    
    # Label Smoothing
    lsce = LabelSmoothingCE(num_classes=80)
    loss = lsce(pred_cls, target_cls)
    print(f"Label Smoothing CE: {loss.item():.4f}")
    
    print("All losses OK!")
