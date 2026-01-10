"""
Classification Loss for YOLOv8
Binary Cross Entropy with Task-Aligned Assigner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassificationLoss(nn.Module):
    """
    Classification loss using Binary Cross Entropy.
    
    Supports both soft labels (for task-aligned assignment) 
    and hard labels.
    
    Args:
        use_sigmoid: Use sigmoid activation (BCE) or softmax (CE)
        reduction: Loss reduction method
    """
    
    def __init__(
        self,
        use_sigmoid: bool = True,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if use_sigmoid:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                reduction='none',
                label_smoothing=label_smoothing
            )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate classification loss.
        
        Args:
            pred: Predicted logits (N, num_classes) or (B, N, num_classes)
            target: Target labels (N,) or soft labels (N, num_classes)
            weight: Optional sample weights
        
        Returns:
            Classification loss value
        """
        if self.use_sigmoid:
            # Binary Cross Entropy
            loss = self.loss_fn(pred, target)
            
            if weight is not None:
                loss = loss * weight.unsqueeze(-1)
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss
        else:
            # Cross Entropy
            loss = self.loss_fn(pred, target)
            
            if weight is not None:
                loss = loss * weight
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive samples
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: Loss reduction method
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Focal Loss.
        
        Args:
            pred: Predicted logits (N, num_classes)
            target: Target labels (N, num_classes) one-hot or soft
        
        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(pred)
        
        # Calculate cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Calculate p_t
        p_t = p * target + (1 - p) * (1 - target)
        
        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Final focal loss
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss - improved focal loss with IoU-aware targets.
    
    For positive samples: loss = -q * (q * log(p) + (1-q) * log(1-p))
    For negative samples: loss = -alpha * p^gamma * log(1-p)
    
    where q is the soft target (IoU score).
    
    Args:
        alpha: Weighting factor for negative samples
        gamma: Focusing parameter for negative samples
        reduction: Loss reduction method
    """
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Varifocal Loss.
        
        Args:
            pred: Predicted logits (N, num_classes)
            target: Soft target labels (N, num_classes) with IoU scores
        
        Returns:
            Varifocal loss value
        """
        p = torch.sigmoid(pred)
        
        # Binary cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Positive mask
        pos_mask = target > 0
        
        # Varifocal weight
        weight = torch.zeros_like(target)
        weight[pos_mask] = target[pos_mask]  # q for positives
        weight[~pos_mask] = self.alpha * p[~pos_mask].pow(self.gamma)  # alpha * p^gamma for negatives
        
        loss = weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == "__main__":
    # Test classification losses
    print("Testing Classification Losses...")
    
    pred = torch.randn(10, 80)
    target = torch.zeros(10, 80)
    target[range(10), torch.randint(0, 80, (10,))] = 1
    
    # BCE Loss
    bce_loss = ClassificationLoss(use_sigmoid=True)
    loss = bce_loss(pred, target)
    print(f"BCE Loss: {loss.item():.4f}")
    
    # Focal Loss
    focal_loss = FocalLoss()
    loss = focal_loss(pred, target)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Varifocal Loss with soft targets
    soft_target = target * torch.rand(10, 1)  # Simulate IoU scores
    vfl = VarifocalLoss()
    loss = vfl(pred, soft_target)
    print(f"Varifocal Loss: {loss.item():.4f}")
