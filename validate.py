"""
Validate YOLOv11 model on COCO validation dataset
Uses the same inference code as inference.py (which works correctly)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from inference import YOLOv11Predictor


def validate(
    predictor,
    images_dir: str,
    ann_file: str,
    save_json: bool = False,
    output_path: str = None
):
    """Run validation using the same predictor as inference.py."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    # Load annotations
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    
    all_predictions = []
    
    # COCO class mapping (model outputs 0-79, COCO uses 1-90 with gaps)
    COCO_CATEGORIES = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90
    ]
    
    print(f"\nValidating on {len(img_ids)} images...")
    
    processed = 0
    for img_id in tqdm(img_ids, desc="Validating"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            continue
        
        # Load image
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        
        orig_h, orig_w = image.shape[:2]
        processed += 1
        
        # Run inference using the same code as inference.py
        results = predictor.predict(image)
        
        boxes = results['boxes']
        scores = results['scores']
        labels = results['labels']
        
        # Convert to COCO format
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]
            w, h = x2 - x1, y2 - y1
            
            if w <= 0 or h <= 0:
                continue
            
            # Map model class (0-79) to COCO category ID (1-90 with gaps)
            cls_idx = int(labels[j])
            if cls_idx < len(COCO_CATEGORIES):
                coco_cat_id = COCO_CATEGORIES[cls_idx]
            else:
                coco_cat_id = cls_idx + 1
            
            all_predictions.append({
                'image_id': int(img_id),
                'category_id': int(coco_cat_id),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(scores[j])
            })
    
    print(f"\nProcessed images: {processed}")
    print(f"Total predictions: {len(all_predictions)}")
    
    if len(all_predictions) == 0:
        print("No predictions! Check model and data.")
        return {}
    
    # Save predictions if requested
    if save_json and output_path:
        with open(output_path, 'w') as f:
            json.dump(all_predictions, f)
        print(f"Predictions saved to: {output_path}")
    
    # Evaluate using pycocotools
    print("\nRunning COCO evaluation...")
    coco_dt = coco.loadRes(all_predictions)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    results = {
        'mAP50-95': coco_eval.stats[0],
        'mAP50': coco_eval.stats[1],
        'mAP75': coco_eval.stats[2],
        'mAP_small': coco_eval.stats[3],
        'mAP_medium': coco_eval.stats[4],
        'mAP_large': coco_eval.stats[5],
    }
    
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  mAP50-95: {results['mAP50-95']:.4f}")
    print(f"  mAP50:    {results['mAP50']:.4f}")
    print(f"  mAP75:    {results['mAP75']:.4f}")
    print(f"  mAP (S):  {results['mAP_small']:.4f}")
    print(f"  mAP (M):  {results['mAP_medium']:.4f}")
    print(f"  mAP (L):  {results['mAP_large']:.4f}")
    print(f"{'='*50}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate YOLOv11 on COCO')
    parser.add_argument('--weights', type=str, required=True, help='Model weights (.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--ann', type=str, default=None, help='Annotation file (auto-detected if not provided)')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='0', help='Device (0, 1, cpu)')
    parser.add_argument('--save-json', action='store_true', help='Save predictions to JSON')
    args = parser.parse_args()
    
    print(f"Using device: cuda:{args.device}" if args.device != 'cpu' else "Using device: cpu")
    
    # Find images and annotations
    data_path = Path(args.data)
    
    if args.ann:
        ann_file = args.ann
    else:
        # Auto-detect annotation file
        ann_candidates = [
            data_path / 'annotations' / 'instances_val2017.json',
            data_path / 'annotations' / 'instances_val.json',
            data_path / 'val.json',
        ]
        ann_file = None
        for candidate in ann_candidates:
            if candidate.exists():
                ann_file = str(candidate)
                break
        
        if ann_file is None:
            raise FileNotFoundError(f"Could not find annotation file in {data_path}")
    
    # Find images directory
    img_candidates = [
        data_path / 'val2017',
        data_path / 'val',
        data_path / 'images' / 'val2017',
        data_path / 'images' / 'val',
    ]
    images_dir = None
    for candidate in img_candidates:
        if candidate.exists():
            images_dir = str(candidate)
            break
    
    if images_dir is None:
        raise FileNotFoundError(f"Could not find images directory in {data_path}")
    
    print(f"Images: {images_dir}")
    print(f"Annotations: {ann_file}")
    
    # Create predictor (same as inference.py)
    predictor = YOLOv11Predictor(
        weights=args.weights,
        device=args.device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        img_size=args.img_size
    )
    
    # Output path for predictions
    output_path = str(Path(args.weights).parent / 'predictions_val.json') if args.save_json else None
    
    # Run validation
    results = validate(
        predictor=predictor,
        images_dir=images_dir,
        ann_file=ann_file,
        save_json=args.save_json,
        output_path=output_path
    )


if __name__ == '__main__':
    main()
