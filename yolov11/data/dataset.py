"""
Dataset Loaders for YOLOv11
Supports COCO and YOLO format datasets
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .augmentations import Compose, LetterBox, RandomHSV, RandomFlip, Mosaic, ToTensor


class COCODataset(Dataset):
    """
    COCO format dataset for detection, segmentation, and pose estimation.
    
    Args:
        root: Path to images directory
        ann_file: Path to COCO annotation JSON file
        task: 'detect', 'segment', or 'pose'
        img_size: Target image size
        augment: Apply augmentations
        mosaic: Use mosaic augmentation
    """
    
    def __init__(
        self,
        root: str,
        ann_file: str,
        task: str = 'detect',
        img_size: int = 640,
        augment: bool = True,
        mosaic: bool = True
    ):
        super().__init__()
        self.root = Path(root)
        self.task = task
        self.img_size = img_size
        self.augment = augment
        self.use_mosaic = mosaic and augment
        
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        
        self.images = {img['id']: img for img in coco['images']}
        self.categories = {cat['id']: cat for cat in coco['categories']}
        
        self.cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(sorted(self.categories.keys()))}
        
        self.img_to_anns = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        self.img_ids = [img_id for img_id in self.images.keys() if img_id in self.img_to_anns]
        
        self.letterbox = LetterBox((img_size, img_size))
        self.mosaic = Mosaic(img_size) if self.use_mosaic else None
        
        if augment:
            self.transforms = Compose([RandomHSV(), RandomFlip()])
        else:
            self.transforms = None
        
        self.to_tensor = ToTensor()
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        if self.use_mosaic and np.random.random() < 0.5:
            return self._load_mosaic(idx)
        return self._load_single(idx)
    
    def _load_single(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        
        filename = img_info['file_name']
        img_path = self.root / filename
        
        # If image not found in root, check subdirectories (e.g., train2017, val2017)
        if not img_path.exists():
            for split in ['train2017', 'val2017', 'test2017']:
                alt_path = self.root / split / filename
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        labels = self._parse_annotations(img_id, img_info)
        
        image, labels = self.letterbox(image, labels)
        
        if self.transforms:
            image, labels = self.transforms(image, labels)
        
        image, labels = self.to_tensor(image, labels)
        
        return image, labels
    
    def _load_mosaic(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        indices = [idx] + [np.random.randint(len(self)) for _ in range(3)]
        images, labels_list = [], []
        
        for i in indices:
            img_id = self.img_ids[i]
            img_info = self.images[img_id]
            filename = img_info['file_name']
            img_path = self.root / filename
            
            if not img_path.exists():
                for split in ['train2017', 'val2017', 'test2017']:
                    alt_path = self.root / split / filename
                    if alt_path.exists():
                        img_path = alt_path
                        break
            
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(cv2.resize(img, (self.img_size, self.img_size)))
                labels_list.append(self._parse_annotations(img_id, img_info))
        
        if len(images) < 4:
            return self._load_single(idx)
        
        image, labels = self.mosaic(images, labels_list)
        
        if self.transforms:
            image, labels = self.transforms(image, labels)
        
        image, labels = self.to_tensor(image, labels)
        return image, labels
    
    def _parse_annotations(self, img_id: int, img_info: Dict) -> Dict:
        anns = self.img_to_anns.get(img_id, [])
        h, w = img_info['height'], img_info['width']
        
        bboxes, labels, keypoints, masks = [], [], [], []
        
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            bboxes.append([x, y, x + bw, y + bh])
            labels.append(self.cat_id_to_idx[ann['category_id']])
            
            if self.task == 'pose' and 'keypoints' in ann:
                kpts = np.array(ann['keypoints']).reshape(-1, 3)
                keypoints.append(kpts)
            
            if self.task == 'segment' and 'segmentation' in ann:
                mask = self._decode_segmentation(ann['segmentation'], h, w)
                masks.append(mask)
        
        result = {
            'bboxes': np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64),
            'image_id': img_id
        }
        
        if keypoints:
            result['keypoints'] = np.array(keypoints, dtype=np.float32)
        if masks:
            result['masks'] = np.array(masks, dtype=np.float32)
        
        return result
    
    def _decode_segmentation(self, segm, height: int, width: int) -> np.ndarray:
        """Decode COCO segmentation to binary mask."""
        from pycocotools import mask as maskUtils
        
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        else:
            rle = segm
        
        return maskUtils.decode(rle)


class YOLODataset(Dataset):
    """
    YOLO format dataset (txt files with labels).
    Label format: class x_center y_center width height (normalized)
    """
    
    def __init__(
        self,
        root: str,
        img_size: int = 640,
        augment: bool = True
    ):
        super().__init__()
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        
        self.img_files = list(self.root.glob('images/**/*.jpg')) + \
                         list(self.root.glob('images/**/*.png'))
        
        self.letterbox = LetterBox((img_size, img_size))
        self.transforms = Compose([RandomHSV(), RandomFlip()]) if augment else None
        self.to_tensor = ToTensor()
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_path = self.img_files[idx]
        
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        
        label_path = str(img_path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        labels = self._load_labels(label_path, w, h)
        
        image, labels = self.letterbox(image, labels)
        if self.transforms:
            image, labels = self.transforms(image, labels)
        image, labels = self.to_tensor(image, labels)
        
        return image, labels
    
    def _load_labels(self, path: str, img_w: int, img_h: int) -> Dict:
        bboxes, labels = [], []
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, xc, yc, w, h = map(float, parts[:5])
                        x1 = (xc - w/2) * img_w
                        y1 = (yc - h/2) * img_h
                        x2 = (xc + w/2) * img_w
                        y2 = (yc + h/2) * img_h
                        bboxes.append([x1, y1, x2, y2])
                        labels.append(int(cls))
        
        return {
            'bboxes': np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4)),
            'labels': np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64)
        }


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, Dict]:
    """Custom collate function for variable-size targets."""
    images, labels_list = zip(*batch)
    images = torch.stack(images, 0)
    
    max_objs = max(len(lab['labels']) for lab in labels_list)
    batch_size = len(labels_list)
    
    batch_labels = torch.zeros(batch_size, max_objs, dtype=torch.int64)
    batch_bboxes = torch.zeros(batch_size, max_objs, 4)
    mask_gt = torch.zeros(batch_size, max_objs, dtype=torch.bool)
    
    for i, lab in enumerate(labels_list):
        n = len(lab['labels'])
        if n > 0:
            batch_labels[i, :n] = lab['labels'] if isinstance(lab['labels'], torch.Tensor) else torch.tensor(lab['labels'])
            batch_bboxes[i, :n] = lab['bboxes'] if isinstance(lab['bboxes'], torch.Tensor) else torch.tensor(lab['bboxes'])
            mask_gt[i, :n] = True
    
    targets = {'labels': batch_labels, 'bboxes': batch_bboxes, 'mask_gt': mask_gt}
    
    if 'keypoints' in labels_list[0]:
        batch_kpts = torch.zeros(batch_size, max_objs, 17, 3)
        for i, lab in enumerate(labels_list):
            if 'keypoints' in lab and len(lab['keypoints']) > 0:
                n = len(lab['keypoints'])
                kpts = lab['keypoints'] if isinstance(lab['keypoints'], torch.Tensor) else torch.tensor(lab['keypoints'])
                batch_kpts[i, :n] = kpts
        targets['keypoints'] = batch_kpts
    
    return images, targets


def create_dataloader(
    root: str,
    ann_file: str = None,
    task: str = 'detect',
    img_size: int = 640,
    batch_size: int = 16,
    augment: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False
) -> DataLoader:
    """Create a DataLoader for training or validation."""
    
    if ann_file:
        dataset = COCODataset(root, ann_file, task, img_size, augment)
    else:
        dataset = YOLODataset(root, img_size, augment)
    
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
