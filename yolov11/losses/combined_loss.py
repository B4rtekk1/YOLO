"""
Combined YOLOv11 Loss Function
Unifies all task-specific losses with Task-Aligned Assigner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .box_loss import CIoULoss, DFLoss, bbox_iou
from .cls_loss import ClassificationLoss, VarifocalLoss
from .enhanced_loss import WiseIoULoss, QualityFocalLoss
from .seg_loss import SegmentationLoss
from .pose_loss import PoseLoss


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Label Assignment for anchor-free detection.

    Assigns ground-truth boxes to anchor points by ranking candidate anchors
    using a unified alignment score that balances classification confidence
    and localisation quality:

    .. math::

        s_i = p_{c,i}^{\\alpha} \\cdot \\text{IoU}(b_i, b_{gt})^{\\beta}

    Algorithm
    ---------
    1. For each ground-truth box, compute the alignment score for every anchor
       that falls *inside* the GT box.
    2. Select the top-*k* anchors per GT as positives.
    3. Resolve conflicts when an anchor is assigned to multiple GTs by keeping
       the assignment with the highest alignment score.
    4. Compute soft classification targets as the normalised alignment scores
       (encourages the model to predict high confidence only for well-localised
       detections).

    Args:
        topk: Maximum number of anchors assigned per ground-truth box.
        num_classes: Number of object categories.
        alpha: Exponent for the classification score term.
        beta: Exponent for the IoU term.
        eps: Small constant for numerical stability.

    References:
        - TOOD: Task-aligned One-stage Object Detection (Feng et al., 2021)
          https://arxiv.org/abs/2108.07755
    """
    
    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = 1e-9
    ):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assign targets to predictions.
        
        Args:
            pred_scores: Predicted class scores (B, N, num_classes)
            pred_bboxes: Predicted boxes (B, N, 4)
            anchor_points: Anchor points (N, 2)
            gt_labels: Ground truth labels (B, M)
            gt_bboxes: Ground truth boxes (B, M, 4)
            mask_gt: Valid GT mask (B, M)
        
        Returns:
            Tuple of (target_labels, target_bboxes, target_scores, fg_mask)
        """
        batch_size = pred_scores.shape[0]
        num_anchors = anchor_points.shape[0]
        num_max_boxes = gt_bboxes.shape[1]
        
        if num_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device),
                torch.zeros(batch_size, num_anchors, 4, device=device),
                torch.zeros(batch_size, num_anchors, self.num_classes, device=device),
                torch.zeros(batch_size, num_anchors, dtype=torch.bool, device=device)
            )
        
        # Get positive mask based on anchor points inside GT boxes
        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt
        )
        
        # Select top-k anchors for each GT
        target_gt_idx, fg_mask, mask_pos = self._select_topk(
            mask_pos, align_metric, overlaps, num_max_boxes
        )
        
        # Get targets
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )
        
        # Normalize target scores by max alignment metric
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2)
        target_scores = target_scores * norm_align_metric.unsqueeze(-1)
        
        return target_labels, target_bboxes, target_scores, fg_mask
    
    def _get_pos_mask(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        mask_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get positive mask and alignment metrics."""
        batch_size = pred_scores.shape[0]
        num_anchors = anchor_points.shape[0]
        num_gt = gt_bboxes.shape[1]
        
        # Check if anchor center is inside GT box
        mask_in_gts = self._is_anchor_in_gt(anchor_points, gt_bboxes)
        
        # Calculate alignment metric
        # Get predicted class scores for GT classes
        batch_idx = torch.arange(batch_size, device=pred_scores.device)[:, None, None]
        gt_idx = torch.arange(num_gt, device=pred_scores.device)[None, :, None]
        
        # pred_scores: (B, N, C), gt_labels: (B, M) -> (B, M, N)
        cls_scores = pred_scores.permute(0, 2, 1)  # (B, C, N)
        gt_labels_expanded = gt_labels.unsqueeze(-1).expand(-1, -1, num_anchors)
        pred_cls_scores = cls_scores.gather(1, gt_labels_expanded.clamp(0, self.num_classes - 1))  # (B, M, N)
        # pred_cls_scores = pred_cls_scores.sigmoid() # REMOVED: redundant as input already has sigmoid
        
        # Calculate IoU between predictions and GTs
        # pred_bboxes: (B, N, 4), gt_bboxes: (B, M, 4)
        overlaps = self._batch_bbox_iou(pred_bboxes, gt_bboxes)  # (B, M, N)
        
        # Alignment metric
        align_metric = pred_cls_scores.pow(self.alpha) * overlaps.pow(self.beta)
        
        # Combine masks
        mask_pos = mask_gt.unsqueeze(-1) * mask_in_gts  # (B, M, N)
        
        return mask_pos, align_metric, overlaps
    
    def _is_anchor_in_gt(
        self,
        anchor_points: torch.Tensor,
        gt_bboxes: torch.Tensor
    ) -> torch.Tensor:
        """Check if anchor centers are inside GT boxes."""
        # anchor_points: (N, 2), gt_bboxes: (B, M, 4)
        x, y = anchor_points.T  # (N,), (N,)
        x1, y1, x2, y2 = gt_bboxes.permute(2, 0, 1)  # (B, M) each
        
        # Compare: anchor (N,) vs GT (B, M)
        x1 = x1.unsqueeze(-1)  # (B, M, 1)
        y1 = y1.unsqueeze(-1)
        x2 = x2.unsqueeze(-1)
        y2 = y2.unsqueeze(-1)
        
        lt = (x > x1) & (y > y1)  # (B, M, N)
        rb = (x < x2) & (y < y2)
        
        return lt & rb
    
    def _batch_bbox_iou(
        self,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor
    ) -> torch.Tensor:
        """Calculate IoU between all pred and GT boxes."""
        # pred_bboxes: (B, N, 4), gt_bboxes: (B, M, 4)
        pred = pred_bboxes.unsqueeze(1)  # (B, 1, N, 4)
        gt = gt_bboxes.unsqueeze(2)  # (B, M, 1, 4)
        
        # Intersection
        inter_x1 = torch.max(pred[..., 0], gt[..., 0])
        inter_y1 = torch.max(pred[..., 1], gt[..., 1])
        inter_x2 = torch.min(pred[..., 2], gt[..., 2])
        inter_y2 = torch.min(pred[..., 3], gt[..., 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        # Union
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        gt_area = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
        union_area = pred_area + gt_area - inter_area
        
        return inter_area / (union_area + self.eps)  # (B, M, N)
    
    def _select_topk(
        self,
        mask_pos: torch.Tensor,
        align_metric: torch.Tensor,
        overlaps: torch.Tensor,
        num_gt: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k anchors for each GT (vectorized implementation)."""
        batch_size, _, num_anchors = mask_pos.shape
        
        # Top-k alignment metric per GT
        topk_metric, topk_idx = torch.topk(
            align_metric * mask_pos.float(),
            k=min(self.topk, num_anchors),
            dim=-1,
            largest=True
        )
        
        # Create top-k mask using vectorized scatter_ operation
        topk_mask = torch.zeros_like(mask_pos, dtype=torch.bool)
        # Expand valid GT mask to match topk_idx shape for scattering
        valid_gt_mask = mask_pos.any(dim=-1, keepdim=True).expand_as(topk_idx)
        # Use scatter to set True at topk indices where GT is valid
        topk_mask.scatter_(-1, topk_idx, valid_gt_mask)
        
        mask_pos = mask_pos & topk_mask
        
        # Resolve conflicts: assign each anchor to only one GT
        fg_mask = mask_pos.any(dim=1)  # (B, N)
        
        # Get target GT index for each anchor
        overlaps_masked = overlaps * mask_pos.float()
        target_gt_idx = overlaps_masked.argmax(dim=1)  # (B, N)
        
        return target_gt_idx, fg_mask, mask_pos
    
    def _get_targets(
        self,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get target labels, boxes, and scores."""
        batch_size, num_anchors = target_gt_idx.shape
        
        # Get target labels
        batch_idx = torch.arange(batch_size, device=gt_labels.device)[:, None]
        target_labels = gt_labels[batch_idx, target_gt_idx]  # (B, N)
        target_labels = torch.where(fg_mask, target_labels, torch.zeros_like(target_labels))
        
        # Get target boxes
        target_bboxes = gt_bboxes[batch_idx, target_gt_idx]  # (B, N, 4)
        
        # Create one-hot target scores
        target_scores = F.one_hot(
            target_labels.long(),
            num_classes=self.num_classes
        ).float()  # (B, N, num_classes)
        target_scores = target_scores * fg_mask.unsqueeze(-1).float()
        
        return target_labels, target_bboxes, target_scores


class YOLOv11Loss(nn.Module):
    """
    Combined YOLOv11 Loss for all tasks.
    
    Supports:
    - Detection: box + classification loss
    - Segmentation: box + classification + mask loss
    - Pose: box + classification + keypoint loss
    
    Args:
        task: Task type ('detect', 'segment', 'pose')
        num_classes: Number of detection classes
        reg_max: Maximum regression value for DFL
        box_weight: Weight for box loss
        cls_weight: Weight for classification loss
        dfl_weight: Weight for DFL loss
    """
    
    def __init__(
        self,
        task: str = 'detect',
        num_classes: int = 80,
        reg_max: int = 16,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        seg_weight: float = 3.0,
        pose_weight: float = 12.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.task = task
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Loss weights
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.seg_weight = seg_weight
        self.pose_weight = pose_weight
        self.label_smoothing = label_smoothing
        
        # Task-aligned assigner
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=num_classes
        )
        
        # Loss functions
        # self.bce_loss = nn.BCEWithLogitsLoss(reduction='none') 
        self.qfl_loss = QualityFocalLoss(reduction='none')
        # self.ciou_loss = CIoULoss(reduction='none')
        self.wise_iou = WiseIoULoss()
        self.dfl_loss = DFLoss(reg_max)
        
        if task == 'segment':
            self.seg_loss = SegmentationLoss()
        elif task == 'pose':
            self.pose_loss = PoseLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Extract predictions
        cls_preds = predictions['cls']  # List of (B, C, H, W)
        reg_preds = predictions['reg']  # List of (B, 4*reg_max, H, W)
        strides = predictions['strides']
        
        device = cls_preds[0].device
        batch_size = cls_preds[0].shape[0]
        
        # Flatten predictions across scales and get feature map sizes
        cls_flat, reg_flat, anchor_points, feat_sizes = self._flatten_predictions(
            cls_preds, reg_preds, strides, device
        )
        
        # Decode boxes for assignment using actual feature map sizes
        pred_bboxes = self._decode_boxes(reg_flat, anchor_points, strides, feat_sizes)
        
        # Get targets
        gt_labels = targets['labels']  # (B, M)
        gt_bboxes = targets['bboxes']  # (B, M, 4)
        mask_gt = targets.get('mask_gt', gt_labels >= 0)  # (B, M)
        
        # Assign targets
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            cls_flat.sigmoid(),
            pred_bboxes,
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        
        target_scores_sum = target_scores.sum().clamp(min=1)
        
        # Apply label smoothing to target scores
        if self.label_smoothing > 0:
            target_scores = target_scores * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        
        # Classification loss
        # Use QualityFocalLoss instead of standard BCE
        # Extract quality scores from target_scores (max value per anchor)
        # target_scores is (B, N, C), one-hot scaled by quality
        quality_scores, _ = target_scores.max(dim=-1)
        
        # Flatten for QFL
        loss_cls = self.qfl_loss(
            cls_flat.view(-1, self.num_classes), 
            target_labels.view(-1),
            quality=quality_scores.view(-1)
        ).sum() / target_scores_sum
        
        # Box losses (only for foreground)
        if fg_mask.any():
            loss_box = self.wise_iou(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask],
                reduction='none'
            )
            loss_box = (loss_box * target_scores[fg_mask].sum(-1)).sum() / target_scores_sum
            
            # DFL loss
            # Get stride for each anchor
            num_per_level = [h * w for h, w in feat_sizes]
            stride_tensor = torch.cat([
                torch.full((n,), s, device=device, dtype=torch.float)
                for n, s in zip(num_per_level, strides)
            ]).view(1, -1, 1)
            
            target_ltrb = self._box2dist(anchor_points, target_bboxes, stride_tensor)
            loss_dfl = self._compute_dfl_loss(reg_flat[fg_mask], target_ltrb[fg_mask])
        else:
            loss_box = torch.tensor(0.0, device=device)
            loss_dfl = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = (
            self.box_weight * loss_box +
            self.cls_weight * loss_cls +
            self.dfl_weight * loss_dfl
        )
        
        loss_dict = {
            'loss_box': loss_box.detach(),
            'loss_cls': loss_cls.detach(),
            'loss_dfl': loss_dfl.detach()
        }
        
        # Task-specific losses
        if self.task == 'segment' and 'masks' in targets:
            loss_seg = self._compute_seg_loss(predictions, targets, fg_mask)
            loss = loss + self.seg_weight * loss_seg
            loss_dict['loss_seg'] = loss_seg.detach()
        
        elif self.task == 'pose' and 'keypoints' in targets:
            loss_pose = self._compute_pose_loss(predictions, targets, fg_mask)
            loss = loss + self.pose_weight * loss_pose
            loss_dict['loss_pose'] = loss_pose.detach()
        
        loss_dict['loss'] = loss.detach()
        
        return loss, loss_dict
    
    def _flatten_predictions(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        strides: List[int],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """Flatten multi-scale predictions and return feature map sizes."""
        cls_flat = []
        reg_flat = []
        anchor_points = []
        feat_sizes = []  # Store actual feature map sizes
        
        for cls_pred, reg_pred, stride in zip(cls_preds, reg_preds, strides):
            b, c, h, w = cls_pred.shape
            feat_sizes.append((h, w))  # Store actual size
            
            # Flatten spatial dimensions
            cls_flat.append(cls_pred.view(b, c, -1).permute(0, 2, 1))  # (B, H*W, C)
            reg_flat.append(reg_pred.view(b, -1, h * w).permute(0, 2, 1))  # (B, H*W, 4*reg_max)
            
            # Generate anchor points
            ys, xs = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            grid = torch.stack([xs, ys], dim=-1).float() + 0.5
            anchor_points.append(grid.view(-1, 2) * stride)
        
        cls_flat = torch.cat(cls_flat, dim=1)  # (B, N, C)
        reg_flat = torch.cat(reg_flat, dim=1)  # (B, N, 4*reg_max)
        anchor_points = torch.cat(anchor_points, dim=0)  # (N, 2)
        
        return cls_flat, reg_flat, anchor_points, feat_sizes
    
    def _decode_boxes(
        self,
        reg_preds: torch.Tensor,
        anchor_points: torch.Tensor,
        strides: List[int],
        feat_sizes: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Decode box predictions using actual feature map sizes."""
        b = reg_preds.shape[0]
        
        # Apply softmax and integrate
        reg_preds = reg_preds.view(b, -1, 4, self.reg_max)
        reg_preds = F.softmax(reg_preds, dim=-1)
        
        # Weighted sum
        weights = torch.arange(self.reg_max, device=reg_preds.device, dtype=torch.float)
        reg_preds = (reg_preds * weights).sum(dim=-1)  # (B, N, 4)
        
        # ltrb to xyxy
        lt = reg_preds[..., :2]
        rb = reg_preds[..., 2:]
        
        # Get stride for each anchor using actual feature map sizes
        num_per_level = [h * w for h, w in feat_sizes]
        stride_tensor = torch.cat([
            torch.full((n,), s, device=reg_preds.device, dtype=torch.float)
            for n, s in zip(num_per_level, strides)
        ])
        stride_tensor = stride_tensor.view(1, -1, 1)
        
        x1y1 = anchor_points - lt * stride_tensor
        x2y2 = anchor_points + rb * stride_tensor
        
        return torch.cat([x1y1, x2y2], dim=-1)
    
    def _box2dist(
        self,
        anchor_points: torch.Tensor,
        bboxes: torch.Tensor,
        stride: torch.Tensor
    ) -> torch.Tensor:
        """Convert boxes to distance format normalized by stride."""
        x1y1, x2y2 = bboxes[..., :2], bboxes[..., 2:]
        lt = (anchor_points - x1y1) / stride
        rb = (x2y2 - anchor_points) / stride
        return torch.cat([lt, rb], dim=-1).clamp(0, self.reg_max - 0.01)
    
    def _compute_dfl_loss(
        self,
        pred_dist: torch.Tensor,
        target_dist: torch.Tensor
    ) -> torch.Tensor:
        """Compute DFL loss."""
        pred_dist = pred_dist.view(-1, self.reg_max)
        target_dist = target_dist.view(-1)
        
        target_dist = target_dist.clamp(0, self.reg_max - 0.01)
        tl = target_dist.long()
        tr = tl + 1
        wl = tr.float() - target_dist
        wr = 1 - wl
        
        loss = (
            F.cross_entropy(pred_dist, tl, reduction='none') * wl +
            F.cross_entropy(pred_dist, tr.clamp(max=self.reg_max - 1), reduction='none') * wr
        )
        
        return loss.mean()
    
    def _compute_seg_loss(
        self,
        predictions: Dict,
        targets: Dict,
        fg_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute instance segmentation loss.
        
        Args:
            predictions: Model outputs containing 'masks' (coefficients) and 'protos'
            targets: Ground truth containing 'masks' (N, H, W) binary masks
            fg_mask: Foreground mask (B, num_anchors)
            
        Returns:
            Segmentation loss tensor
        """
        if not fg_mask.any():
            return torch.tensor(0.0, device=fg_mask.device)
        
        # Get mask predictions
        mask_coeffs = predictions.get('masks')  # List of (B, num_protos, H, W) per scale
        protos = predictions.get('protos')  # (B, num_protos, proto_H, proto_W)
        
        if mask_coeffs is None or protos is None:
            return torch.tensor(0.0, device=fg_mask.device)
        
        # Flatten mask coefficients across scales
        mask_flat = []
        for mask in mask_coeffs:
            b, c, h, w = mask.shape
            mask_flat.append(mask.view(b, c, -1).permute(0, 2, 1))  # (B, H*W, num_protos)
        mask_flat = torch.cat(mask_flat, dim=1)  # (B, N, num_protos)
        
        # Get target masks
        target_masks = targets['masks']  # (B, M, mask_H, mask_W)
        target_bboxes = targets['bboxes']  # (B, M, 4)
        
        batch_size = fg_mask.shape[0]
        device = fg_mask.device
        total_loss = torch.tensor(0.0, device=device)
        num_fg = 0
        
        for b in range(batch_size):
            fg_idx = fg_mask[b].nonzero(as_tuple=True)[0]
            if len(fg_idx) == 0:
                continue
            
            # Get coefficients for foreground predictions
            coeffs = mask_flat[b, fg_idx]  # (num_fg, num_protos)
            
            # Get prototypes for this batch
            proto = protos[b]  # (num_protos, pH, pW)
            proto_h, proto_w = proto.shape[1:]
            
            # Assemble predicted masks: coeffs @ proto -> (num_fg, pH, pW)
            proto_flat = proto.view(proto.shape[0], -1)  # (num_protos, pH*pW)
            pred_masks = torch.mm(coeffs, proto_flat)  # (num_fg, pH*pW)
            pred_masks = pred_masks.view(-1, proto_h, proto_w)  # (num_fg, pH, pW)
            
            # Get target masks for this batch
            n_targets = min(len(fg_idx), target_masks.shape[1])
            if n_targets == 0:
                continue
                
            gt_masks = target_masks[b, :n_targets]  # (n_targets, mask_H, mask_W)
            
            # Resize target masks to proto size
            gt_masks = F.interpolate(
                gt_masks.unsqueeze(1).float(),
                size=(proto_h, proto_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # (n_targets, pH, pW)
            
            # Match predictions to targets (simplified - use first n_targets)
            pred_masks_matched = pred_masks[:n_targets]
            
            # Compute BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_masks_matched,
                gt_masks,
                reduction='none'
            )
            
            # Compute Dice loss
            pred_sigmoid = pred_masks_matched.sigmoid()
            pred_flat = pred_sigmoid.flatten(1)
            gt_flat = gt_masks.flatten(1)
            
            intersection = (pred_flat * gt_flat).sum(1)
            union = pred_flat.sum(1) + gt_flat.sum(1)
            dice_loss = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
            
            # Combine losses
            batch_loss = bce_loss.mean() + dice_loss.mean()
            total_loss = total_loss + batch_loss
            num_fg += n_targets
        
        if num_fg > 0:
            return total_loss / batch_size
        return torch.tensor(0.0, device=device)
    
    def _compute_pose_loss(
        self,
        predictions: Dict,
        targets: Dict,
        fg_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pose estimation loss using OKS.
        
        Args:
            predictions: Model outputs containing 'kpts' (keypoint predictions)
            targets: Ground truth containing 'keypoints' (N, 17, 3) and 'areas'
            fg_mask: Foreground mask (B, num_anchors)
            
        Returns:
            Pose loss tensor
        """
        if not fg_mask.any():
            return torch.tensor(0.0, device=fg_mask.device)
        
        # Get keypoint predictions for foreground anchors
        kpt_preds = predictions.get('kpts')
        if kpt_preds is None:
            return torch.tensor(0.0, device=fg_mask.device)
        
        # Flatten keypoint predictions across scales
        kpt_flat = []
        for kpt in kpt_preds:
            b, c, h, w = kpt.shape
            # c = num_keypoints * 3 (x, y, visibility) = 51
            kpt_flat.append(kpt.view(b, c, -1).permute(0, 2, 1))  # (B, H*W, 51)
        kpt_flat = torch.cat(kpt_flat, dim=1)  # (B, N, 51)
        
        # Reshape to (B, N, 17, 3)
        num_keypoints = 17
        kpt_flat = kpt_flat.view(kpt_flat.shape[0], kpt_flat.shape[1], num_keypoints, 3)
        
        # Get target keypoints
        target_kpts = targets['keypoints']  # (B, M, 17, 3) - x, y, visibility
        target_areas = targets.get('areas', None)
        
        # For each foreground prediction, get the corresponding target keypoints
        # This is a simplified version - full implementation would use target assignment
        batch_size = fg_mask.shape[0]
        
        pred_kpts_fg = kpt_flat[fg_mask]  # (num_fg, 17, 3)
        
        if pred_kpts_fg.shape[0] == 0:
            return torch.tensor(0.0, device=fg_mask.device)
        
        # Get coordinates and visibility
        pred_coords = pred_kpts_fg[..., :2]  # (num_fg, 17, 2)
        pred_vis = pred_kpts_fg[..., 2]      # (num_fg, 17)
        
        # Compute areas if not provided (use bbox area approximation)
        if target_areas is None:
            # Approximate area from keypoint spread
            target_bboxes = targets['bboxes']  # (B, M, 4)
            areas_flat = (target_bboxes[..., 2] - target_bboxes[..., 0]) * \
                        (target_bboxes[..., 3] - target_bboxes[..., 1])
            # Get areas for foreground (simplified - use mean)
            areas = areas_flat[fg_mask[:, :areas_flat.shape[1]] if fg_mask.shape[1] > areas_flat.shape[1] 
                              else fg_mask].clamp(min=1.0)
        else:
            areas = target_areas[fg_mask[:, :target_areas.shape[1]]].clamp(min=1.0)
        
        # Match predictions to targets (simplified - use first M targets repeated)
        # Full implementation would use Hungarian matching or assignment from detector
        num_fg = pred_kpts_fg.shape[0]
        target_kpts_flat = target_kpts.view(-1, 17, 3)  # (B*M, 17, 3)
        
        # Repeat targets to match number of foreground predictions
        if target_kpts_flat.shape[0] < num_fg:
            repeat_factor = (num_fg // target_kpts_flat.shape[0]) + 1
            target_kpts_flat = target_kpts_flat.repeat(repeat_factor, 1, 1)[:num_fg]
        else:
            target_kpts_flat = target_kpts_flat[:num_fg]
        
        target_coords = target_kpts_flat[..., :2]  # (num_fg, 17, 2)
        target_vis = target_kpts_flat[..., 2]      # (num_fg, 17)
        
        # Ensure areas match
        if areas.shape[0] != num_fg:
            areas = areas.mean().expand(num_fg)
        
        # Compute OKS Loss
        loss, _ = self.pose_loss(
            pred_coords, pred_vis,
            target_coords, target_vis,
            areas
        )
        
        return loss


if __name__ == "__main__":
    print("Testing YOLOv11 Loss...")
    
    # Create loss
    loss_fn = YOLOv11Loss(task='detect', num_classes=80)
    
    # Mock predictions
    predictions = {
        'cls': [torch.randn(2, 80, 80, 80), torch.randn(2, 80, 40, 40), torch.randn(2, 80, 20, 20)],
        'reg': [torch.randn(2, 64, 80, 80), torch.randn(2, 64, 40, 40), torch.randn(2, 64, 20, 20)],
        'strides': [8, 16, 32]
    }
    
    # Mock targets
    targets = {
        'labels': torch.randint(0, 80, (2, 10)),
        'bboxes': torch.rand(2, 10, 4) * 640,
    }
    # Convert to xyxy format
    targets['bboxes'][..., 2:] = targets['bboxes'][..., :2] + targets['bboxes'][..., 2:]
    
    loss, loss_dict = loss_fn(predictions, targets)
    print(f"Total Loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
