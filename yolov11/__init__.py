"""
YOLOv11 - Custom PyTorch Implementation
========================================

A from-scratch implementation of YOLOv11 supporting three computer-vision tasks:

    * **Object Detection** – anchor-free, multi-scale detection with DFL box regression.
    * **Instance Segmentation** – extends detection with prototype-based mask prediction.
    * **Pose Estimation** – extends detection with 17-keypoint COCO skeleton prediction.

Architecture overview
---------------------
::

    Input image (B, 3, H, W)
        │
        ▼
    CSPDarknet backbone  (C3k2 + SPPF + C2PSA)
        │  P3 (stride 8)  – small objects
        │  P4 (stride 16) – medium objects
        │  P5 (stride 32) – large objects
        ▼
    PANet neck  (FPN top-down + PAN bottom-up, C3k2 blocks)
        │  N3, N4, N5
        ▼
    Task head  (DetectionHead / SegmentationHead / PoseHead)
        │
        ▼
    Loss  (TaskAlignedAssigner + WiseIoU + QFL + DFL)

Key improvements over YOLOv8
------------------------------
* **C3k2** replaces C2f – smaller kernels, better gradient flow.
* **C2PSA** adds cross-stage partial spatial attention at P5.
* **WiseIoU** loss handles bounding-box outliers more robustly.
* **QualityFocalLoss** jointly optimises classification and localisation quality.
* Optional **CBAM** channel+spatial attention at every output stage.

Quick start
-----------
>>> from yolov11 import YOLOv11
>>> model = YOLOv11(num_classes=80, task='detect', model_size='s')
>>> model.info()
>>> import torch
>>> out = model(torch.randn(1, 3, 640, 640))
>>> print(out.keys())  # dict_keys(['cls', 'reg', 'strides'])

Module map
----------
* ``yolov11.model``    – :class:`YOLOv11` main model and :func:`create_model` factory.
* ``yolov11.backbone`` – :class:`CSPDarknet` feature extractor.
* ``yolov11.neck``     – :class:`PANet` feature pyramid neck.
* ``yolov11.head``     – :class:`DetectionHead`, :class:`SegmentationHead`, :class:`PoseHead`.
* ``yolov11.losses``   – :class:`YOLOv11Loss`, :class:`TaskAlignedAssigner`, and individual losses.
* ``yolov11.data``     – :class:`COCODataset`, :class:`YOLODataset`, augmentations.
* ``yolov11.utils``    – NMS, metrics, export, training helpers (EMA, schedulers).
"""

from .model import YOLOv11, create_model
from .backbone import CSPDarknet
from .neck import PANet
from .head import DetectionHead, SegmentationHead, PoseHead

__version__ = "11.0.0"
__all__ = [
    "YOLOv11",
    "create_model",
    "CSPDarknet", 
    "PANet",
    "DetectionHead",
    "SegmentationHead", 
    "PoseHead"
]
