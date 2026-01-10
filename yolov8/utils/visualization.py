"""
Visualization Utilities for YOLOv8
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# COCO class colors (80 classes)
np.random.seed(42)
COCO_COLORS = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(80)]

# COCO skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Keypoint colors
KEYPOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170)
]


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray = None,
    scores: np.ndarray = None,
    class_names: List[str] = None,
    colors: List[Tuple] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: BGR image
        boxes: (N, 4) xyxy format
        labels: (N,) class indices
        scores: (N,) confidence scores
        class_names: List of class names
        colors: List of colors per class
        thickness: Line thickness
    """
    image = image.copy()
    
    if colors is None:
        colors = COCO_COLORS
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Get color
        color = colors[int(labels[i]) % len(colors)] if labels is not None else (0, 255, 0)
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if labels is not None:
            label = class_names[int(labels[i])] if class_names else str(int(labels[i]))
            if scores is not None:
                label = f"{label} {scores[i]:.2f}"
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def draw_masks(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray = None,
    colors: List[Tuple] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Draw instance segmentation masks.
    
    Args:
        image: BGR image
        masks: (N, H, W) binary masks
        labels: (N,) class indices
        colors: List of colors per class
        alpha: Transparency
    """
    image = image.copy()
    
    if colors is None:
        colors = COCO_COLORS
    
    overlay = image.copy()
    
    for i, mask in enumerate(masks):
        color = colors[int(labels[i]) % len(colors)] if labels is not None else (0, 255, 0)
        overlay[mask > 0.5] = color
    
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray = None,
    threshold: float = 0.5,
    radius: int = 5,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw human pose keypoints and skeleton.
    
    Args:
        image: BGR image
        keypoints: (N, 17, 3) keypoints [x, y, visibility]
        scores: (N,) detection scores
        threshold: Visibility threshold
        radius: Keypoint radius
        thickness: Skeleton line thickness
    """
    image = image.copy()
    
    for person_idx, kpts in enumerate(keypoints):
        # Draw skeleton
        for i, (p1, p2) in enumerate(SKELETON):
            if kpts[p1, 2] > threshold and kpts[p2, 2] > threshold:
                pt1 = (int(kpts[p1, 0]), int(kpts[p1, 1]))
                pt2 = (int(kpts[p2, 0]), int(kpts[p2, 1]))
                color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
                cv2.line(image, pt1, pt2, color, thickness)
        
        # Draw keypoints
        for j, (x, y, v) in enumerate(kpts):
            if v > threshold:
                color = KEYPOINT_COLORS[j % len(KEYPOINT_COLORS)]
                cv2.circle(image, (int(x), int(y)), radius, color, -1)
                cv2.circle(image, (int(x), int(y)), radius + 1, (0, 0, 0), 1)
    
    return image


def visualize_predictions(
    image: np.ndarray,
    predictions: dict,
    task: str = 'detect',
    class_names: List[str] = None,
    conf_thres: float = 0.25
) -> np.ndarray:
    """
    Visualize model predictions.
    
    Args:
        image: BGR image
        predictions: Model output dict
        task: 'detect', 'segment', or 'pose'
        class_names: List of class names
        conf_thres: Confidence threshold
    """
    # Filter by confidence
    if 'scores' in predictions:
        mask = predictions['scores'] > conf_thres
        for k in predictions:
            if isinstance(predictions[k], np.ndarray):
                predictions[k] = predictions[k][mask]
    
    # Draw based on task
    if task == 'detect':
        image = draw_boxes(
            image,
            predictions.get('boxes', np.zeros((0, 4))),
            predictions.get('labels'),
            predictions.get('scores'),
            class_names
        )
    
    elif task == 'segment':
        if 'masks' in predictions:
            image = draw_masks(
                image,
                predictions['masks'],
                predictions.get('labels')
            )
        image = draw_boxes(
            image,
            predictions.get('boxes', np.zeros((0, 4))),
            predictions.get('labels'),
            predictions.get('scores'),
            class_names
        )
    
    elif task == 'pose':
        image = draw_boxes(
            image,
            predictions.get('boxes', np.zeros((0, 4))),
            predictions.get('labels'),
            predictions.get('scores'),
            class_names
        )
        if 'keypoints' in predictions:
            image = draw_keypoints(
                image,
                predictions['keypoints'],
                predictions.get('scores')
            )
    
    return image
