"""
yolov11.utils.nms – Non-Maximum Suppression
============================================

Post-processing utilities that filter redundant bounding-box predictions.

Algorithm (standard NMS)
------------------------
1. Filter all predictions below *conf_thres*.
2. For each class, sort remaining predictions by confidence (descending).
3. Greedily keep the highest-confidence box and suppress all boxes with
   IoU > *iou_thres* relative to it.
4. Repeat until no boxes remain.

The implementation uses ``torchvision.ops.nms`` (CUDA-accelerated when
available) and applies a class-offset trick to perform batched NMS across
all classes in a single call.
"""

import torch
import torchvision
from typing import List, Tuple, Optional


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Calculate IoU between two sets of boxes (xyxy format)."""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2 - inter
    
    return inter / (union + 1e-7)


def non_max_suppression(
    predictions: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    max_det: int = 300,
    multi_label: bool = True
) -> List[torch.Tensor]:
    """
    Non-Maximum Suppression for object detection.
    
    Args:
        predictions: (batch, num_anchors, 4 + num_classes) or with masks/keypoints
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        classes: Filter by class indices
        max_det: Maximum detections per image
        multi_label: Allow multiple labels per box
    
    Returns:
        List of detections per image: (n, 6) [x1, y1, x2, y2, conf, cls]
    """
    batch_size = predictions.shape[0]
    num_classes = predictions.shape[2] - 4
    
    # Settings
    max_wh = 7680  # Max box width/height
    max_nms = 30000  # Max boxes for torchvision.ops.nms
    
    output = [torch.zeros((0, 6), device=predictions.device)] * batch_size
    
    for xi, x in enumerate(predictions):
        # Filter by confidence: keep only anchors where any class score > conf_thres
        xc = x[:, 4:].amax(1) > conf_thres
        x = x[xc]
        
        if not x.shape[0]:
            continue
        
        # Separate box coordinates from class scores
        box = x[:, :4]
        cls_scores = x[:, 4:]
        
        if multi_label:
            # Allow a single box to be detected as multiple classes
            i, j = (cls_scores > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat([box[i], cls_scores[i, j, None], j[:, None].float()], 1)
        else:
            # Single label: keep only the highest-scoring class per box
            conf, j = cls_scores.max(1, keepdim=True)
            x = torch.cat([box, conf, j.float()], 1)
            x = x[conf.view(-1) > conf_thres]
        
        # Filter by class whitelist
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        
        n = x.shape[0]
        if not n:
            continue
        
        # Sort by confidence and cap at max_nms to avoid OOM
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        
        # Batched NMS: offset boxes by class index so different classes
        # never suppress each other (they will have non-overlapping coordinates)
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        
        output[xi] = x[i]
    
    return output


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_thres: float = 0.5
) -> torch.Tensor:
    """Batched NMS - applies NMS per class."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    max_offset = boxes.max()
    offsets = class_ids.to(boxes) * (max_offset + 1)
    boxes_offset = boxes + offsets[:, None]
    
    return torchvision.ops.nms(boxes_offset, scores, iou_thres)


def nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    """NMS for rotated boxes (OBB)."""
    # Simplified: convert to axis-aligned for now
    return torchvision.ops.nms(boxes[:, :4], scores, iou_thres)
