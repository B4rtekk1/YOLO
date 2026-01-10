"""
Data Augmentations for YOLOv8
"""

import cv2
import numpy as np
import torch
import random
from typing import Tuple, List, Dict, Callable


class Compose:
    """Compose multiple augmentations."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray, labels: Dict) -> Tuple[np.ndarray, Dict]:
        for t in self.transforms:
            image, labels = t(image, labels)
        return image, labels


class LetterBox:
    """Resize and pad image maintaining aspect ratio."""
    
    def __init__(self, new_shape=(640, 640), color=(114, 114, 114)):
        self.new_shape = new_shape if isinstance(new_shape, tuple) else (new_shape, new_shape)
        self.color = color
    
    def __call__(self, image: np.ndarray, labels: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        shape = image.shape[:2]
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (self.new_shape[1] - new_unpad[0]) / 2, (self.new_shape[0] - new_unpad[1]) / 2
        
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        
        if labels and 'bboxes' in labels and len(labels['bboxes']) > 0:
            labels['bboxes'] = labels['bboxes'].copy()
            labels['bboxes'][:, [0, 2]] = labels['bboxes'][:, [0, 2]] * r + dw
            labels['bboxes'][:, [1, 3]] = labels['bboxes'][:, [1, 3]] * r + dh
        
        return image, labels #type: ignore


class RandomHSV:
    """Random HSV augmentation."""
    
    def __init__(self, h_gain=0.015, s_gain=0.7, v_gain=0.4):
        self.h_gain, self.s_gain, self.v_gain = h_gain, s_gain, v_gain
    
    def __call__(self, image: np.ndarray, labels: Dict) -> Tuple[np.ndarray, Dict]:
        if random.random() < 0.5:
            return image, labels
        r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256, dtype=r.dtype)
        hue = cv2.LUT(hue, ((x * r[0]) % 180).astype(image.dtype))
        sat = cv2.LUT(sat, np.clip(x * r[1], 0, 255).astype(image.dtype))
        val = cv2.LUT(val, np.clip(x * r[2], 0, 255).astype(image.dtype))
        return cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2BGR), labels


class RandomFlip:
    """Random horizontal flip."""
    
    def __init__(self, p=0.5):
        self.p = p
        self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # COCO keypoint swap
    
    def __call__(self, image: np.ndarray, labels: Dict) -> Tuple[np.ndarray, Dict]:
        if random.random() < self.p:
            h, w = image.shape[:2]
            image = np.fliplr(image)
            if 'bboxes' in labels and len(labels['bboxes']) > 0:
                bboxes = labels['bboxes'].copy()
                bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
                labels['bboxes'] = bboxes
            if 'keypoints' in labels and len(labels['keypoints']) > 0:
                kpts = labels['keypoints'].copy()
                kpts[:, :, 0] = w - kpts[:, :, 0]
                labels['keypoints'] = kpts[:, self.flip_idx, :]
        return np.ascontiguousarray(image), labels


class Mosaic:
    """Mosaic augmentation - combines 4 images."""
    
    def __init__(self, img_size=640, p=1.0):
        self.img_size = img_size
        self.p = p
    
    def __call__(self, images: List[np.ndarray], labels_list: List[Dict]) -> Tuple[np.ndarray, Dict]:
        if random.random() > self.p or len(images) < 4:
            return images[0], labels_list[0]
        
        s = self.img_size
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        bboxes_list, labels_cls = [], []
        
        for i, (img, labels) in enumerate(zip(images[:4], labels_list[:4])):
            h, w = img.shape[:2]
            if i == 0: x1a, y1a, x2a, y2a = max(xc-w, 0), max(yc-h, 0), xc, yc
            elif i == 1: x1a, y1a, x2a, y2a = xc, max(yc-h, 0), min(xc+w, s*2), yc
            elif i == 2: x1a, y1a, x2a, y2a = max(xc-w, 0), yc, xc, min(s*2, yc+h)
            else: x1a, y1a, x2a, y2a = xc, yc, min(xc+w, s*2), min(s*2, yc+h)
            
            # Calculate source coordinates
            if i == 0:  # top-left
                x1b = w - (x2a - x1a)
                y1b = h - (y2a - y1a)
                x2b, y2b = w, h
            elif i == 1:  # top-right
                x1b = 0
                y1b = h - (y2a - y1a)
                x2b = min(x2a - x1a, w)
                y2b = h
            elif i == 2:  # bottom-left
                x1b = w - (x2a - x1a)
                y1b = 0
                x2b, y2b = w, min(y2a - y1a, h)
            else:  # bottom-right
                x1b, y1b = 0, 0
                x2b = min(x2a - x1a, w)
                y2b = min(y2a - y1a, h)
            
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            
            if 'bboxes' in labels and len(labels['bboxes']) > 0:
                bboxes = labels['bboxes'].copy()
                bboxes[:, [0, 2]] += (x1a - x1b)
                bboxes[:, [1, 3]] += (y1a - y1b)
                bboxes_list.append(bboxes)
                labels_cls.append(labels['labels'])
        
        result = {}
        if bboxes_list:
            result['bboxes'] = np.clip(np.concatenate(bboxes_list), 0, 2*s)
            result['labels'] = np.concatenate(labels_cls)
        
        img4 = img4[yc-s//2:yc+s//2, xc-s//2:xc+s//2]
        if 'bboxes' in result:
            result['bboxes'][:, [0,2]] -= (xc - s//2)
            result['bboxes'][:, [1,3]] -= (yc - s//2)
        return img4, result


class MixUp:
    """MixUp augmentation."""
    
    def __init__(self, alpha=32.0, p=0.5):
        self.alpha, self.p = alpha, p
    
    def __call__(self, img1, labels1, img2, labels2):
        if random.random() > self.p:
            return img1, labels1
        r = np.random.beta(self.alpha, self.alpha)
        img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        labels = {}
        if 'bboxes' in labels1 and 'bboxes' in labels2:
            labels['bboxes'] = np.concatenate([labels1['bboxes'], labels2['bboxes']])
            labels['labels'] = np.concatenate([labels1['labels'], labels2['labels']])
        return img, labels


class ToTensor:
    """Convert to PyTorch tensor."""
    
    def __call__(self, image: np.ndarray, labels: Dict):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0 #type: ignore
        for k, v in labels.items():
            if isinstance(v, np.ndarray):
                labels[k] = torch.from_numpy(v)
        return image, labels


class CutMix:
    """
    CutMix augmentation for object detection.
    
    Cuts a random patch from one image and pastes it onto another,
    mixing labels proportionally.
    """
    
    def __init__(self, p: float = 0.5, beta: float = 1.0):
        self.p = p
        self.beta = beta
    
    def __call__(self, img1, labels1, img2, labels2):
        if random.random() > self.p:
            return img1, labels1
        
        h, w = img1.shape[:2]
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.beta, self.beta)
        
        # Calculate cut size
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Random center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Cut region
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        img = img1.copy()
        img[bby1:bby2, bbx1:bbx2] = img2[bby1:bby2, bbx1:bbx2]
        
        # Merge labels - keep boxes that are still mostly visible
        labels = {}
        if 'bboxes' in labels1 and 'bboxes' in labels2:
            # Filter boxes from img1 (keep if not in cut region)
            boxes1 = labels1['bboxes'].copy()
            keep1 = []
            for i, box in enumerate(boxes1):
                # Check if box is mostly outside cut region
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                inter_x1 = max(box[0], bbx1)
                inter_y1 = max(box[1], bby1)
                inter_x2 = min(box[2], bbx2)
                inter_y2 = min(box[3], bby2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                if inter_area < 0.5 * box_area:  # Keep if <50% covered
                    keep1.append(i)
            
            # Filter boxes from img2 (keep if in cut region)
            boxes2 = labels2['bboxes'].copy()
            keep2 = []
            for i, box in enumerate(boxes2):
                # Check if box is mostly inside cut region
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                inter_x1 = max(box[0], bbx1)
                inter_y1 = max(box[1], bby1)
                inter_x2 = min(box[2], bbx2)
                inter_y2 = min(box[3], bby2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                if inter_area > 0.5 * box_area:  # Keep if >50% inside
                    keep2.append(i)
            
            # Concatenate labels
            kept_boxes1 = boxes1[keep1] if len(keep1) > 0 else np.zeros((0, 4))
            kept_boxes2 = boxes2[keep2] if len(keep2) > 0 else np.zeros((0, 4))
            kept_labels1 = labels1['labels'][keep1] if len(keep1) > 0 else np.array([])
            kept_labels2 = labels2['labels'][keep2] if len(keep2) > 0 else np.array([])
            
            labels['bboxes'] = np.concatenate([kept_boxes1, kept_boxes2]) if len(kept_boxes1) + len(kept_boxes2) > 0 else np.zeros((0, 4))
            labels['labels'] = np.concatenate([kept_labels1, kept_labels2]) if len(kept_labels1) + len(kept_labels2) > 0 else np.array([])
        
        return img, labels


class CopyPaste:
    """
    Copy-Paste augmentation for instance segmentation.
    
    Copies instances from one image and pastes them onto another.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img1, labels1, img2, labels2):
        if random.random() > self.p or 'masks' not in labels2:
            return img1, labels1
        
        # Simple implementation: copy random instances
        h, w = img1.shape[:2]
        n_instances = len(labels2.get('masks', []))
        
        if n_instances == 0:
            return img1, labels1
        
        # Copy up to 3 random instances
        n_copy = min(3, n_instances)
        indices = random.sample(range(n_instances), n_copy)
        
        for idx in indices:
            mask = labels2['masks'][idx]
            # Paste instance
            img1[mask > 0] = img2[mask > 0]
        
        return img1, labels1

