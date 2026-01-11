"""
Box Regression Losses for YOLOv11
Implements CIoU Loss and Distribution Focal Loss (DFL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = False,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = True,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between boxes.
    
    Supports multiple IoU variants:
    - Standard IoU
    - GIoU (Generalized IoU)
    - DIoU (Distance IoU)
    - CIoU (Complete IoU)
    
    Args:
        box1: First set of boxes (N, 4)
        box2: Second set of boxes (N, 4) or (M, 4)
        xywh: If True, boxes are in (x, y, w, h) format, else (x1, y1, x2, y2)
        GIoU: Calculate Generalized IoU
        DIoU: Calculate Distance IoU
        CIoU: Calculate Complete IoU (default)
        eps: Small value to avoid division by zero
    
    Returns:
        IoU values (N,) or (N, M)
    """
    # Convert to xyxy format if needed
    if xywh:
        # (x_center, y_center, width, height) -> (x1, y1, x2, y2)
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
        
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    
    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + eps
    
    # IoU
    iou = inter_area / union_area
    
    if GIoU or DIoU or CIoU:
        # Enclosing box
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)
        
        if GIoU:
            # Generalized IoU
            enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + eps
            return iou - (enclose_area - union_area) / enclose_area
        
        if DIoU or CIoU:
            # Distance IoU / Complete IoU
            # Diagonal distance of enclosing box
            c_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
            
            # Center distance
            b1_cx = (b1_x1 + b1_x2) / 2
            b1_cy = (b1_y1 + b1_y2) / 2
            b2_cx = (b2_x1 + b2_x2) / 2
            b2_cy = (b2_y1 + b2_y2) / 2
            center_dist = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2
            
            if DIoU:
                return iou - center_dist / c_diag
            
            if CIoU:
                # Aspect ratio consistency
                b1_w = b1_x2 - b1_x1
                b1_h = b1_y2 - b1_y1
                b2_w = b2_x2 - b2_x1
                b2_h = b2_y2 - b2_y1
                
                v = (4 / (torch.pi ** 2)) * torch.pow(
                    torch.atan(b2_w / (b2_h + eps)) - torch.atan(b1_w / (b1_h + eps)), 
                    2
                )
                
                with torch.no_grad():
                    alpha = v / (1 - iou + v + eps)
                
                return iou - (center_dist / c_diag + v * alpha)
    
    return iou


class CIoULoss(nn.Module):
    """
    Complete IoU Loss for box regression.
    
    CIoU considers:
    1. Overlap area (IoU)
    2. Center point distance
    3. Aspect ratio consistency
    
    Loss = 1 - CIoU
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        pred_boxes: torch.Tensor, 
        target_boxes: torch.Tensor,
        weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calculate CIoU loss.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in xyxy format
            target_boxes: Target boxes (N, 4) in xyxy format
            weights: Optional sample weights (N,)
        
        Returns:
            CIoU loss value
        """
        ciou = bbox_iou(pred_boxes, target_boxes, CIoU=True)
        loss = 1.0 - ciou
        
        if weights is not None:
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DFLoss(nn.Module):
    """
    Distribution Focal Loss for box regression.
    
    DFL represents box coordinates as probability distributions
    over discrete bins [0, 1, 2, ..., reg_max-1] instead of
    directly regressing float values.
    
    This helps capture uncertainty in box localization.
    
    Args:
        reg_max: Maximum regression range (default: 16)
    """
    
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
    
    def forward(
        self,
        pred_dist: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Distribution Focal Loss.
        
        Args:
            pred_dist: Predicted distribution logits (N, reg_max)
            target: Target regression values (N,) in range [0, reg_max-1]
        
        Returns:
            DFL loss value
        """
        # Get left and right bins
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # left bin index
        tr = tl + 1  # right bin index
        
        # Weight for left bin (closer to target = higher weight)
        wl = tr.float() - target
        wr = 1 - wl
        
        # Cross entropy loss for left and right bins
        loss = (
            F.cross_entropy(pred_dist, tl, reduction='none') * wl +
            F.cross_entropy(pred_dist, tr, reduction='none') * wr
        )
        
        return loss.mean()


class BboxLoss(nn.Module):
    """
    Combined Box Loss for YOLOv11.
    
    Combines:
    - CIoU Loss for decoded boxes
    - DFL Loss for distribution regression
    
    Args:
        reg_max: Maximum regression range for DFL
    """
    
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.ciou_loss = CIoULoss(reduction='none')
        self.dfl_loss = DFLoss(reg_max)
    
    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_boxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate combined box loss.
        
        Args:
            pred_dist: Predicted distributions (B, N, 4*reg_max)
            pred_boxes: Decoded predicted boxes (B, N, 4)
            anchor_points: Anchor points (N, 2)
            target_boxes: Target boxes (B, N, 4)
            target_scores: Target scores/weights (B, N)
            target_scores_sum: Sum of target scores
            fg_mask: Foreground mask (B, N)
        
        Returns:
            Tuple of (ciou_loss, dfl_loss)
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # CIoU loss
        ciou = bbox_iou(pred_boxes[fg_mask], target_boxes[fg_mask], CIoU=True)
        loss_iou = ((1.0 - ciou) * weight).sum() / target_scores_sum
        
        # DFL loss
        target_ltrb = self._bbox2dist(anchor_points, target_boxes, self.reg_max)
        loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
        
        return loss_iou, loss_dfl
    
    def _bbox2dist(
        self,
        anchor_points: torch.Tensor,
        bbox: torch.Tensor,
        reg_max: int
    ) -> torch.Tensor:
        """Convert bbox to distance format (left, top, right, bottom)."""
        x1y1, x2y2 = bbox.chunk(2, -1)
        lt = anchor_points - x1y1
        rb = x2y2 - anchor_points
        return torch.cat([lt, rb], -1).clamp(0, reg_max - 0.01)
    
    def _df_loss(
        self,
        pred_dist: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate DFL loss."""
        target = target.view(-1)
        tl = target.long()
        tr = tl + 1
        wl = tr.float() - target
        wr = 1 - wl
        
        return (
            F.cross_entropy(pred_dist, tl, reduction='none') * wl +
            F.cross_entropy(pred_dist, tr.clamp(max=self.reg_max - 1), reduction='none') * wr
        ).view(-1, 4).mean(-1)


if __name__ == "__main__":
    # Test losses
    print("Testing Box Losses...")
    
    # CIoU Loss
    pred = torch.tensor([[10, 10, 50, 50], [20, 20, 80, 80]], dtype=torch.float32)
    target = torch.tensor([[15, 15, 55, 55], [25, 25, 75, 75]], dtype=torch.float32)
    
    ciou_loss = CIoULoss()
    loss = ciou_loss(pred, target)
    print(f"CIoU Loss: {loss.item():.4f}")
    
    # IoU variants
    print(f"IoU: {bbox_iou(pred, target, CIoU=False).mean().item():.4f}")
    print(f"GIoU: {bbox_iou(pred, target, GIoU=True).mean().item():.4f}")
    print(f"DIoU: {bbox_iou(pred, target, DIoU=True).mean().item():.4f}")
    print(f"CIoU: {bbox_iou(pred, target, CIoU=True).mean().item():.4f}")
    
    # DFL Loss
    dfl_loss = DFLoss(reg_max=16)
    pred_dist = torch.randn(10, 16)
    target_val = torch.rand(10) * 15
    loss = dfl_loss(pred_dist, target_val)
    print(f"DFL Loss: {loss.item():.4f}")
