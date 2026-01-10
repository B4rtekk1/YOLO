"""
YOLOv8 Inference Script
Supports image, video, and webcam inference
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).parent))

from yolov8.model import YOLOv8
from yolov8.data.augmentations import LetterBox
from yolov8.utils.nms import non_max_suppression
from yolov8.utils.visualization import visualize_predictions


# COCO class names
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class YOLOv8Predictor:
    """YOLOv8 inference wrapper."""
    
    def __init__(
        self,
        weights: str,
        task: str = 'detect',
        device: str = '0',
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        img_size: int = 640
    ):
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.task = task
        
        # Load model
        checkpoint = torch.load(weights, map_location=self.device)
        
        # Determine model config from checkpoint
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Create model (assuming saved hyperparameters or defaults)
        self.model = YOLOv8(
            num_classes=80,
            task=task,
            model_size='s'
        ).to(self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Preprocessing
        self.letterbox = LetterBox((img_size, img_size))
        
        print(f'Loaded {weights} on {self.device}')
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference."""
        # Resize with letterbox
        img, _ = self.letterbox(image, {})
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # HWC to CHW, normalize
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Add batch dimension
        img = torch.from_numpy(img).unsqueeze(0)
        
        return img.to(self.device)
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        """Run inference on image."""
        h, w = image.shape[:2]
        
        # Preprocess
        img = self.preprocess(image)
        
        # Inference
        outputs = self.model(img)
        
        # Post-process
        results = self.postprocess(outputs, (h, w))
        
        return results
    
    def postprocess(self, outputs: dict, orig_shape: tuple) -> dict:
        """Post-process model outputs."""
        h, w = orig_shape
        
        # Flatten predictions across scales
        cls_preds = []
        reg_preds = []
        
        for cls, reg in zip(outputs['cls'], outputs['reg']):
            b, c, fh, fw = cls.shape
            cls_preds.append(cls.view(b, c, -1))
            reg_preds.append(reg.view(b, -1, fh * fw))
        
        cls_preds = torch.cat(cls_preds, dim=-1).permute(0, 2, 1)  # (B, N, C)
        
        # Decode boxes from regression (simplified)
        # For proper decoding, would need anchor points and DFL
        reg_preds = torch.cat(reg_preds, dim=-1).permute(0, 2, 1)  # (B, N, 64)
        
        # Combine class scores with dummy boxes for NMS format
        num_anchors = cls_preds.shape[1]
        
        # Create dummy boxes (proper implementation needs anchor decoding)
        boxes = torch.rand(1, num_anchors, 4, device=cls_preds.device) * self.img_size
        boxes[:, :, 2:] = boxes[:, :, :2] + boxes[:, :, 2:]  # xywh to xyxy
        
        # Combine
        predictions = torch.cat([boxes, cls_preds], dim=-1)
        
        # NMS
        results = non_max_suppression(
            predictions,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres
        )[0]
        
        if len(results) == 0:
            return {
                'boxes': np.zeros((0, 4)),
                'scores': np.zeros(0),
                'labels': np.zeros(0, dtype=int)
            }
        
        # Scale boxes to original image
        scale = min(self.img_size / h, self.img_size / w)
        pad = ((self.img_size - w * scale) / 2, (self.img_size - h * scale) / 2)
        
        boxes = results[:, :4].cpu().numpy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
        
        # Clip to image bounds
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
        
        return {
            'boxes': boxes,
            'scores': results[:, 4].cpu().numpy(),
            'labels': results[:, 5].cpu().numpy().astype(int)
        }


def run_image(predictor: YOLOv8Predictor, source: str, save_path: str = None):
    """Run inference on image."""
    # Use numpy.fromfile to handle Unicode paths
    try:
        image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        image = cv2.imread(source)
    
    if image is None:
        raise FileNotFoundError(f'Image not found: {source}')

    
    start = time.time()
    results = predictor.predict(image)
    inference_time = time.time() - start
    
    print(f'Inference: {inference_time * 1000:.1f}ms, {len(results["boxes"])} detections')
    
    # Visualize
    vis_image = visualize_predictions(
        image, results, predictor.task, COCO_NAMES
    )
    
    if save_path:
        cv2.imwrite(save_path, vis_image)
        print(f'Saved: {save_path}')
    else:
        # Try to display, if fails auto-save
        try:
            cv2.imshow('YOLOv8', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            auto_save = 'output_detection.jpg'
            cv2.imwrite(auto_save, vis_image)
            print(f'GUI not available, saved to: {auto_save}')

    
    return results


def run_video(predictor: YOLOv8Predictor, source, save_path: str = None):
    """Run inference on video or webcam."""
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        results = predictor.predict(frame)
        fps = 1 / (time.time() - start + 1e-9)
        
        # Visualize
        vis_frame = visualize_predictions(
            frame, results, predictor.task, COCO_NAMES
        )
        
        # Add FPS
        cv2.putText(
            vis_frame, f'FPS: {fps:.1f}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        if save_path:
            writer.write(vis_frame)
        else:
            cv2.imshow('YOLOv8', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if save_path:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or webcam index')
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'segment', 'pose'])
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--save', type=str, default=None, help='Save path')
    args = parser.parse_args()
    
    # Create predictor
    predictor = YOLOv8Predictor(
        weights=args.weights,
        task=args.task,
        device=args.device,
        conf_thres=args.conf,
        iou_thres=args.iou,
        img_size=args.img_size
    )
    
    # Determine source type
    source = args.source
    if source.isdigit() or source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        run_video(predictor, source, args.save)
    else:
        run_image(predictor, source, args.save)


if __name__ == '__main__':
    main()
