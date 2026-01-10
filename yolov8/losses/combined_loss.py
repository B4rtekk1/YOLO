"""
Combined YOLOv8 Loss Function
Unifies all task-specific losses with Task-Aligned Assigner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .box_loss import CIoULoss, DFLoss, bbox_iou
from .cls_loss import ClassificationLoss, VarifocalLoss
from .seg_loss import SegmentationLoss
from .pose_loss import PoseLoss


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner for YOLOv8.
    
    Assigns ground truth targets to anchor points based on both
    classification and localization quality.
    
    alignment_metric = cls_score^alpha * iou^beta
    
    Args:
        topk: Maximum number of anchors to assign per GT
        alpha: Classification score weight
        beta: IoU weight
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
        pred_cls_scores = pred_cls_scores.sigmoid()
        
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
        """Select top-k anchors for each GT."""
        batch_size, _, num_anchors = mask_pos.shape
        
        # Top-k alignment metric per GT
        topk_metric, topk_idx = torch.topk(
            align_metric * mask_pos.float(),
            k=min(self.topk, num_anchors),
            dim=-1,
            largest=True
        )
        
        # Create top-k mask
        topk_mask = torch.zeros_like(mask_pos, dtype=torch.bool)
        for b in range(batch_size):
            for gt in range(num_gt):
                if mask_pos[b, gt].any():
                    topk_mask[b, gt, topk_idx[b, gt]] = True
        
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


class YOLOv8Loss(nn.Module):
    """
    Combined YOLOv8 Loss for all tasks.
    
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
        pose_weight: float = 12.0
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
        
        # Task-aligned assigner
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=num_classes
        )
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ciou_loss = CIoULoss(reduction='none')
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
        
        # Flatten predictions across scales
        cls_flat, reg_flat, anchor_points = self._flatten_predictions(
            cls_preds, reg_preds, strides, device
        )
        
        # Decode boxes for assignment
        pred_bboxes = self._decode_boxes(reg_flat, anchor_points, strides)
        
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
        
        # Classification loss
        loss_cls = self.bce_loss(cls_flat, target_scores).sum() / target_scores_sum
        
        # Box losses (only for foreground)
        if fg_mask.any():
            loss_box = self.ciou_loss(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask]
            )
            loss_box = (loss_box * target_scores[fg_mask].sum(-1)).sum() / target_scores_sum
            
            # DFL loss
            target_ltrb = self._box2dist(anchor_points, target_bboxes, self.reg_max)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten multi-scale predictions."""
        cls_flat = []
        reg_flat = []
        anchor_points = []
        
        for cls_pred, reg_pred, stride in zip(cls_preds, reg_preds, strides):
            b, c, h, w = cls_pred.shape
            
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
        
        return cls_flat, reg_flat, anchor_points
    
    def _decode_boxes(
        self,
        reg_preds: torch.Tensor,
        anchor_points: torch.Tensor,
        strides: List[int]
    ) -> torch.Tensor:
        """Decode box predictions."""
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
        
        # Get stride for each anchor
        num_per_level = [(640 // s) ** 2 for s in strides]
        stride_tensor = torch.cat([
            torch.full((n,), s, device=reg_preds.device)
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
        reg_max: int
    ) -> torch.Tensor:
        """Convert boxes to distance format."""
        x1y1, x2y2 = bboxes[..., :2], bboxes[..., 2:]
        lt = anchor_points - x1y1
        rb = x2y2 - anchor_points
        return torch.cat([lt, rb], dim=-1).clamp(0, reg_max - 0.01)
    
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
        """Compute segmentation loss."""
        # Placeholder - full implementation requires mask assembly
        return torch.tensor(0.0, device=fg_mask.device)
    
    def _compute_pose_loss(
        self,
        predictions: Dict,
        targets: Dict,
        fg_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute pose loss."""
        # Placeholder - full implementation requires keypoint assembly
        return torch.tensor(0.0, device=fg_mask.device)


if __name__ == "__main__":
    print("Testing YOLOv8 Loss...")
    
    # Create loss
    loss_fn = YOLOv8Loss(task='detect', num_classes=80)
    
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
