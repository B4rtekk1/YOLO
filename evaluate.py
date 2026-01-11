"""
YOLOv11 Evaluation Script
Runs full COCO benchmark on val2017 or generates predictions for test-dev
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from yolov11.model import YOLOv11
from yolov11.data import create_dataloader
from yolov11.data.augmentations import LetterBox
from yolov11.utils.nms import non_max_suppression


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    model_size = config.get('model_size', 's')
    num_classes = config.get('num_classes', 80)
    task = config.get('task', 'detect')
    
    # Create model
    model = YOLOv11(
        num_classes=num_classes,
        task=task,
        model_size=model_size
    ).to(device)
    
    # Load weights (try EMA first, then model_state_dict)
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print(f"Loaded EMA weights from: {weights_path}")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from: {weights_path}")
    else:
        # Assume checkpoint is just the state_dict
        model.load_state_dict(checkpoint)
        print(f"Loaded state_dict from: {weights_path}")
    
    model.eval()
    model.fuse()  # Fuse Conv+BN for inference
    
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    save_json: bool = False,
    save_path: str = None
) -> Dict:
    """Run evaluation on dataset with ground truth."""
    
    from yolov11.utils.metrics import Metrics
    
    metrics = Metrics()
    all_predictions = []  # For COCO JSON format
    
    print(f"\nEvaluating on {len(dataloader)} batches...")
    print(f"Confidence threshold: {conf_thres}, IoU threshold: {iou_thres}")
    
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = images.to(device)
        for k in targets:
            if isinstance(targets[k], torch.Tensor):
                targets[k] = targets[k].to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Decode predictions
        cls_outputs, reg_outputs = outputs['cls'], outputs['reg']
        strides = outputs['strides']
        anchors = model.get_anchors(device)
        
        # Decode boxes
        decoded_boxes = model.head.decode_boxes(reg_outputs, anchors, strides)
        
        # Format class predictions
        cls_preds = []
        for cls in cls_outputs:
            b, c, fh, fw = cls.shape
            cls_preds.append(cls.view(b, c, -1))
        cls_preds = torch.cat(cls_preds, dim=-1).permute(0, 2, 1).sigmoid()
        
        # Combine for NMS
        predictions = torch.cat([decoded_boxes, cls_preds], dim=-1)
        
        # NMS
        results = non_max_suppression(predictions, conf_thres=conf_thres, iou_thres=iou_thres)
        
        # Process each image in batch
        for i, det in enumerate(results):
            # Get ground truth
            mask = targets['mask_gt'][i]
            gt_boxes = targets['bboxes'][i][mask]
            gt_labels = targets['labels'][i][mask]
            
            # Update metrics
            metrics.process_batch(det, gt_boxes, gt_labels)
            
            # Save predictions for COCO JSON if needed
            if save_json and len(det) > 0:
                # Get image ID (if available in targets)
                img_id = targets.get('image_id', [batch_idx * len(images) + i])[i]
                if isinstance(img_id, torch.Tensor):
                    img_id = img_id.item()
                
                for *xyxy, conf, cls_id in det.cpu().numpy():
                    # Convert to COCO format (x, y, w, h)
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    
                    all_predictions.append({
                        'image_id': int(img_id),
                        'category_id': int(cls_id) + 1,  # COCO uses 1-indexed categories
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(conf)
                    })
    
    # Compute final metrics
    results = metrics.compute()
    
    # Save predictions JSON if requested
    if save_json and save_path:
        with open(save_path, 'w') as f:
            json.dump(all_predictions, f)
        print(f"\nPredictions saved to: {save_path}")
    
    return results


@torch.no_grad()
def generate_test_predictions(
    model: nn.Module,
    test_images_dir: str,
    image_info_file: str,
    device: torch.device,
    img_size: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    save_path: str = "predictions.json",
    half: bool = False
) -> None:
    """Generate predictions for test-dev (no ground truth, for CodaBench submission)."""
    
    # Load image info
    with open(image_info_file, 'r') as f:
        image_info = json.load(f)
    
    images_list = image_info['images']
    print(f"\nGenerating predictions for {len(images_list)} test images...")
    print(f"Confidence threshold: {conf_thres}, IoU threshold: {iou_thres}")
    
    all_predictions = []
    letterbox = LetterBox((img_size, img_size))
    
    for img_info in tqdm(images_list, desc="Processing test images"):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = os.path.join(test_images_dir, img_filename)
        
        # Load image
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess
        img_resized, _ = letterbox(image, {})
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        img_tensor /= 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        # Forward pass
        outputs = model(img_tensor)
        
        # Decode predictions
        cls_outputs, reg_outputs = outputs['cls'], outputs['reg']
        strides = outputs['strides']
        anchors = model.get_anchors(device)
        
        # Decode boxes
        decoded_boxes = model.head.decode_boxes(reg_outputs, anchors, strides)
        
        # Format class predictions
        cls_preds = []
        for cls in cls_outputs:
            b, c, fh, fw = cls.shape
            cls_preds.append(cls.view(b, c, -1))
        cls_preds = torch.cat(cls_preds, dim=-1).permute(0, 2, 1).sigmoid()
        
        # Combine for NMS
        predictions = torch.cat([decoded_boxes, cls_preds], dim=-1)
        
        # NMS
        results = non_max_suppression(predictions, conf_thres=conf_thres, iou_thres=iou_thres)
        det = results[0]
        
        if len(det) > 0:
            # Scale boxes back to original image size
            scale = min(img_size / orig_h, img_size / orig_w)
            pad = ((img_size - orig_w * scale) / 2, (img_size - orig_h * scale) / 2)
            
            boxes = det[:, :4].cpu().numpy()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
            
            # Clip to image bounds
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
            
            scores = det[:, 4].cpu().numpy()
            cls_ids = det[:, 5].cpu().numpy().astype(int)
            
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j]
                w, h = x2 - x1, y2 - y1
                
                all_predictions.append({
                    'image_id': int(img_id),
                    'category_id': int(cls_ids[j]) + 1,  # COCO uses 1-indexed
                    'bbox': [round(float(x1), 2), round(float(y1), 2), 
                             round(float(w), 2), round(float(h), 2)],
                    'score': round(float(scores[j]), 4)
                })
    
    # Save predictions
    with open(save_path, 'w') as f:
        json.dump(all_predictions, f)
    
    print(f"\n{'='*60}")
    print(f"  Predictions saved to: {save_path}")
    print(f"  Total detections: {len(all_predictions)}")
    print(f"  Images processed: {len(images_list)}")
    print(f"{'='*60}")
    print(f"\nNext steps for CodaBench submission:")
    print(f"  1. Zip the predictions file:")
    print(f"     Compress-Archive -Path '{save_path}' -DestinationPath 'submission.zip'")
    print(f"  2. Upload submission.zip to CodaBench")


def print_results(results: Dict):
    """Print evaluation results in a nice format."""
    print("\n" + "=" * 60)
    print("                    EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP@0.5:        {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:   {results['mAP50-95']:.4f}")
    print("=" * 60)
    
    # Per-class results if available
    if 'per_class_ap' in results and results['per_class_ap'] is not None:
        print("\nPer-Class AP@0.5:")
        print("-" * 40)
        for cls_id, ap in enumerate(results['per_class_ap']):
            if ap > 0:
                print(f"  Class {cls_id:3d}: {ap:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv11 on COCO')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path (.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to data directory (e.g., coco_mini)')
    parser.add_argument('--ann', type=str, default=None, help='Path to annotation file (auto-detected if not provided)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--save-json', action='store_true', help='Save predictions as COCO JSON')
    parser.add_argument('--half', action='store_true', help='Use FP16 half precision')
    parser.add_argument('--test-dev', action='store_true', help='Generate predictions for test-dev (no mAP, for CodaBench)')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.weights, device)
    
    if args.half and device.type == 'cuda':
        model.half()
        print("Using FP16 half precision")
    
    data_path = Path(args.data)
    
    # Test-dev mode: generate predictions without ground truth
    if args.test_dev:
        # Find test images and info
        test_images_dir = data_path / "test2017"
        image_info_file = data_path / "annotations/image_info_test-dev2017.json"
        
        # Try alternative paths
        if not image_info_file.exists():
            image_info_file = data_path / "annotations/image_info_test2017.json"
        
        if not test_images_dir.exists():
            print(f"Error: Test images directory not found: {test_images_dir}")
            return
        
        if not image_info_file.exists():
            print(f"Error: Image info file not found: {image_info_file}")
            print("Please download image_info_test2017.zip from COCO website")
            return
        
        save_path = str(data_path / "predictions_test-dev.json")
        
        generate_test_predictions(
            model=model,
            test_images_dir=str(test_images_dir),
            image_info_file=str(image_info_file),
            device=device,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            save_path=save_path,
            half=args.half
        )
        return
    
    # Standard evaluation with ground truth
    if args.ann:
        ann_file = args.ann
    else:
        # Try common locations
        possible_anns = [
            data_path / "annotations/instances_val2017.json",
            data_path / "annotations/detect_val2017.json",
        ]
        ann_file = None
        for p in possible_anns:
            if p.exists():
                ann_file = str(p)
                break
        
        if ann_file is None:
            print("Error: Could not find annotation file. Please specify with --ann")
            return
    
    print(f"Using annotations: {ann_file}")
    
    # Create dataloader
    dataloader = create_dataloader(
        root=args.data,
        ann_file=ann_file,
        task='detect',
        img_size=args.img_size,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        num_workers=args.workers,
        distributed=False
    )
    
    # Run evaluation
    save_path = str(Path(args.weights).parent / 'predictions.json') if args.save_json else None
    
    results = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        save_json=args.save_json,
        save_path=save_path
    )
    
    # Print results
    print_results(results)


if __name__ == '__main__':
    main()
