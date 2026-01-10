"""
Evaluation Metrics for YOLOv8
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes (xyxy format)."""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2 - inter
    
    return inter / (union + 1e-7)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    # 101-point interpolation
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    
    return float(ap)


def compute_ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute AP for each class.
    
    Returns: (tp, fp, precision, recall, ap) per class
    """
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    unique_classes = np.unique(target_cls)
    nc = len(unique_classes)
    
    ap = np.zeros(nc)
    precision = np.zeros(nc)
    recall = np.zeros(nc)
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_pred = i.sum()
        
        if n_pred == 0 or n_gt == 0:
            continue
        
        # Cumulative sums
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        
        # Recall
        rec = tpc / (n_gt + eps)
        
        # Precision
        prec = tpc / (tpc + fpc + eps)
        
        # AP
        ap[ci] = compute_ap(rec, prec)
        precision[ci] = prec[-1]
        recall[ci] = rec[-1]
    
    return precision, recall, ap, unique_classes


def compute_oks(
    pred_kpts: np.ndarray,
    gt_kpts: np.ndarray,
    gt_area: float,
    sigmas: Optional[np.ndarray] = None
) -> float:
    """
    Compute Object Keypoint Similarity (OKS).
    
    Args:
        pred_kpts: Predicted keypoints (17, 3)
        gt_kpts: Ground truth keypoints (17, 3)
        gt_area: Object area
        sigmas: Per-keypoint sigmas
    """
    if sigmas is None:
        sigmas = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ])
    
    # Get visibility
    vis = gt_kpts[:, 2] > 0
    
    if not vis.any():
        return 0.0
    
    # Distance
    d = (pred_kpts[:, 0] - gt_kpts[:, 0]) ** 2 + (pred_kpts[:, 1] - gt_kpts[:, 1]) ** 2
    
    # OKS
    k2 = 2 * sigmas ** 2
    oks = np.exp(-d / (2 * gt_area * k2 + 1e-9))
    
    return float(oks[vis].mean())


class ConfusionMatrix:
    """Confusion matrix for object detection."""
    
    def __init__(self, nc: int, conf: float = 0.25, iou_thres: float = 0.5):
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        self.matrix = np.zeros((nc + 1, nc + 1))
    
    def process_batch(self, detections: torch.Tensor, labels: torch.Tensor):
        """Process a batch of detections."""
        if detections is None or len(detections) == 0:
            for label in labels:
                self.matrix[self.nc, int(label[0])] += 1
            return
        
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        
        iou = compute_iou(labels[:, 1:5], detections[:, :4])
        
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))
        
        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and j.sum() == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1
            else:
                self.matrix[self.nc, gc] += 1
        
        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1
    
    @property
    def tp(self) -> np.ndarray:
        return self.matrix.diagonal()[:self.nc]
    
    @property
    def fp(self) -> np.ndarray:
        return self.matrix[:self.nc, self.nc]
    
    @property
    def fn(self) -> np.ndarray:
        return self.matrix[self.nc, :self.nc]


class Metrics:
    """Detection metrics calculator."""
    
    def __init__(self):
        self.stats = []
    
    def process(
        self,
        tp: np.ndarray,
        conf: np.ndarray,
        pred_cls: np.ndarray,
        target_cls: np.ndarray
    ):
        """Process batch results."""
        self.stats.append((tp, conf, pred_cls, target_cls))
    
    def compute(self) -> dict:
        """Compute final metrics."""
        if not self.stats:
            return {'mAP50': 0, 'mAP50-95': 0, 'precision': 0, 'recall': 0}
        
        tp = np.concatenate([s[0] for s in self.stats])
        conf = np.concatenate([s[1] for s in self.stats])
        pred_cls = np.concatenate([s[2] for s in self.stats])
        target_cls = np.concatenate([s[3] for s in self.stats])
        
        precision, recall, ap, _ = compute_ap_per_class(tp, conf, pred_cls, target_cls)
        
        return {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'mAP50': ap.mean(),
            'mAP50-95': ap.mean()  # Simplified
        }
