"""
Segmentation Loss for YOLOv11
Instance mask loss using BCE and Dice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
    Good for handling class imbalance in segmentation.
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            pred: Predicted mask probabilities (N, H, W) or (N, 1, H, W)
            target: Target binary masks (N, H, W) or (N, 1, H, W)
        
        Returns:
            Dice loss value
        """
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        intersection = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SegmentationLoss(nn.Module):
    """
    Instance Segmentation Loss for YOLOv11.
    
    Combines:
    - Binary Cross Entropy for pixel-wise classification
    - Dice Loss for region overlap
    
    Args:
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        reduction: Loss reduction method
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.reduction = reduction
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_loss = DiceLoss(reduction='none')
    
    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        mask_weights: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate segmentation loss.
        
        Args:
            pred_masks: Predicted mask logits (N, H, W)
            target_masks: Target binary masks (N, H, W)
            mask_weights: Optional per-instance weights (N,)
        
        Returns:
            Tuple of (total_loss, dict with individual losses)
        """
        # BCE Loss
        bce = self.bce_loss(pred_masks, target_masks)
        
        if mask_weights is not None:
            bce = bce * mask_weights.view(-1, 1, 1)
        
        bce = bce.mean(dim=(1, 2))  # Mean over spatial dims
        
        # Dice Loss
        pred_probs = torch.sigmoid(pred_masks)
        dice = self.dice_loss(pred_probs, target_masks)
        
        if mask_weights is not None:
            dice = dice * mask_weights
        
        # Combine losses
        if self.reduction == 'mean':
            bce_loss = bce.mean()
            dice_loss = dice.mean()
        elif self.reduction == 'sum':
            bce_loss = bce.sum()
            dice_loss = dice.sum()
        else:
            bce_loss = bce
            dice_loss = dice
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss, {'bce': bce_loss, 'dice': dice_loss}
    
    def single_mask_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        area: torch.Tensor,
        proto: torch.Tensor,
        xyxy: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss for a single instance mask.
        
        Args:
            pred: Mask coefficients (num_protos,)
            target: Target mask (H, W)
            area: Instance area for weighting
            proto: Prototype masks (num_protos, H, W)
            xyxy: Bounding box for cropping
        
        Returns:
            Single instance mask loss
        """
        # Generate mask from coefficients
        pred_mask = (pred @ proto.view(proto.shape[0], -1)).view(proto.shape[1:])
        
        # Crop to bounding box
        x1, y1, x2, y2 = xyxy.int().tolist()
        pred_crop = pred_mask[y1:y2, x1:x2]
        target_crop = target[y1:y2, x1:x2]
        
        # BCE loss within bounding box
        loss = F.binary_cross_entropy_with_logits(pred_crop, target_crop, reduction='mean')
        
        return loss


class MaskIoU(nn.Module):
    """
    Mask IoU calculation for evaluation and quality-aware training.
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate IoU between predicted and target masks.
        
        Args:
            pred_masks: Predicted mask probabilities (N, H, W)
            target_masks: Target binary masks (N, H, W)
        
        Returns:
            IoU values (N,)
        """
        # Binarize predictions
        pred_binary = (pred_masks > self.threshold).float()
        
        # Flatten spatial dimensions
        pred_flat = pred_binary.flatten(1)
        target_flat = target_masks.flatten(1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1) - intersection
        
        # IoU
        iou = intersection / (union + 1e-7)
        
        return iou


if __name__ == "__main__":
    # Test segmentation losses
    print("Testing Segmentation Losses...")
    
    pred_masks = torch.randn(4, 160, 160)
    target_masks = (torch.rand(4, 160, 160) > 0.5).float()
    
    # Dice Loss
    dice_loss = DiceLoss()
    loss = dice_loss(torch.sigmoid(pred_masks), target_masks)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Combined Segmentation Loss
    seg_loss = SegmentationLoss()
    total_loss, loss_dict = seg_loss(pred_masks, target_masks)
    print(f"Segmentation Loss: {total_loss.item():.4f}")
    print(f"  BCE: {loss_dict['bce'].item():.4f}")
    print(f"  Dice: {loss_dict['dice'].item():.4f}")
    
    # Mask IoU
    mask_iou = MaskIoU()
    iou = mask_iou(torch.sigmoid(pred_masks), target_masks)
    print(f"Mask IoU: {iou.mean().item():.4f}")
