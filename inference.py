"""
YOLOv11 Inference Script
Supports image, video, and webcam inference
"""

import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time
from typing import Union
import sys
sys.path.insert(0, str(Path(__file__).parent))

from yolov11.model import YOLOv11
from yolov11.data.augmentations import LetterBox
from yolov11.utils.nms import non_max_suppression
from yolov11.utils.visualization import visualize_predictions


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


class YOLOv11Predictor:
    """YOLOv11 inference wrapper."""
    
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
        self.weights_path = Path(weights)
        self.format = self.weights_path.suffix.lower() # '.pt', '.onnx', '.engine'
        self.half = False
        self.strides = [8, 16, 32]
        self._anchors = None # Cache for anchors
        
        if self.format == '.pt':
            # Load PyTorch model
            checkpoint = torch.load(weights, map_location=self.device, weights_only=False)
            config = checkpoint.get('config', {})
            model_size = config.get('model_size', 's')
            num_classes = config.get('num_classes', 80)
            self.task = config.get('task', task)
            
            self.model = YOLOv11(
                num_classes=num_classes,
                task=self.task,
                model_size=model_size
            ).to(self.device)
            
            if 'ema_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['ema_state_dict'])
            else:
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.fuse() # Fuse Conv + BN for inference
            print(f'Loaded PyTorch model: {weights}')
            
        elif self.format == '.onnx':
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(weights, sess_options, providers=providers)
            print(f'Loaded ONNX model: {weights}')
            
        elif self.format == '.engine':
            import tensorrt as trt
            from yolov11.utils.export import TRTInference # We should add a helper for this
            self.trt_model = TRTInference(weights)
            print(f'Loaded TensorRT engine: {weights}')
            
        else:
            raise ValueError(f"Unsupported weight format: {self.format}. Supported: .pt, .onnx, .engine")
            
        # Preprocessing
        self.letterbox = LetterBox((img_size, img_size))

    
    def set_half(self, half: bool):
        """Toggle FP16 (half precision) inference."""
        if half and self.device.type == 'cuda':
            self.half = True
            if self.format == '.pt':
                self.model.half()
            print("FP16 (Half precision) enabled.")
        else:
            self.half = False

    def get_anchors(self):
        """Generate anchor points for decoding."""
        if self._anchors is not None:
            return self._anchors
        
        anchors = []
        for stride in self.strides:
            grid_size = self.img_size // stride
            ys, xs = torch.meshgrid(
                torch.arange(grid_size),
                torch.arange(grid_size),
                indexing='ij'
            )
            grid = torch.stack([xs, ys], dim=-1).float() + 0.5
            grid = grid * stride
            anchors.append(grid.to(self.device))
        
        self._anchors = anchors
        return anchors

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference."""
        # Resize with letterbox
        img, _ = self.letterbox(image, {})
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # HWC to CHW, normalize, move to device
        img = torch.from_numpy(img.transpose(2, 0, 1)).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        
        # Add batch dimension
        return img.unsqueeze(0)
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        """Run inference on image."""
        h, w = image.shape[:2]
        img = self.preprocess(image)
        
        # Inference based on format
        if self.format == '.pt':
            outputs = self.model(img)
        elif self.format == '.onnx':
            ort_inputs = {self.session.get_inputs()[0].name: img.cpu().numpy()}
            outputs = self.session.run(None, ort_inputs)
            # Move outputs to device
            outputs = [torch.from_numpy(x).to(self.device) for x in outputs]
        elif self.format == '.engine':
            # TensorRT outputs are on host (CPU), move to device
            outputs = [x.to(self.device) for x in self.trt_model(img)]
        else:
            raise ValueError(f"Cannot run prediction on unsupported format: {self.format}")
        
        return self.postprocess(outputs, (h, w))
    
    def postprocess(self, outputs: Union[dict, list], orig_shape: tuple) -> dict:
        """Post-process model outputs."""
        h, w = orig_shape
        
        if isinstance(outputs, dict):
            cls_outputs, reg_outputs = outputs['cls'], outputs['reg']
            strides = outputs['strides']
            anchors = self.model.get_anchors(self.device)
            
            # Decode actual boxes
            if hasattr(self.model, 'head') and hasattr(self.model.head, 'decode_boxes'):
                boxes = self.model.head.decode_boxes(reg_outputs, anchors, strides)
            else:
                # Fallback decoding if head is not accessible
                boxes = self._decode_boxes(reg_outputs, anchors, strides)
                
            cls_preds = []
            for cls in cls_outputs:
                b, c, fh, fw = cls.shape
                cls_preds.append(cls.view(b, c, -1))
            cls_preds = torch.cat(cls_preds, dim=-1).permute(0, 2, 1).sigmoid()
        else:
            # ONNX/TRT return a flat list of 6 tensors for Detection (3 cls, 3 reg)
            cls_outputs, reg_outputs = outputs[:3], outputs[3:6]
            strides = self.strides
            anchors = self.get_anchors()
            
            # Decode boxes for ONNX/TRT
            boxes = self._decode_boxes(reg_outputs, anchors, strides)
            
            cls_preds = []
            for cls in cls_outputs:
                b, c, fh, fw = cls.shape
                cls_preds.append(cls.view(b, c, -1))
            cls_preds = torch.cat(cls_preds, dim=-1).permute(0, 2, 1).sigmoid()
            
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
        
        output = {
            'boxes': boxes,
            'scores': results[:, 4].cpu().numpy(),
            'labels': results[:, 5].cpu().numpy().astype(int)
        }
        
        # Handle pose keypoints
        if self.task == 'pose' and isinstance(outputs, dict) and 'kpts' in outputs:
            kpt_outputs = outputs['kpts']
            anchors = self.model.get_anchors(self.device)
            strides = outputs['strides']
            
            # Decode keypoints
            if hasattr(self.model, 'head') and hasattr(self.model.head, 'decode_keypoints'):
                keypoints = self.model.head.decode_keypoints(kpt_outputs, anchors, strides)
            else:
                keypoints = self._decode_keypoints(kpt_outputs, anchors, strides)
            
            # Get keypoints for NMS results (need to track indices)
            # For simplicity, we get keypoints from flattened predictions
            # and match by spatial location
            keypoints = keypoints[0].cpu().numpy()  # (N, 17, 3)
            
            # Scale keypoints to original image
            keypoints[..., 0] = (keypoints[..., 0] - pad[0]) / scale
            keypoints[..., 1] = (keypoints[..., 1] - pad[1]) / scale
            
            # Clip to image bounds
            keypoints[..., 0] = np.clip(keypoints[..., 0], 0, w)
            keypoints[..., 1] = np.clip(keypoints[..., 1], 0, h)
            
            # Match keypoints to detection results (simplified - take first N)
            num_dets = len(boxes)
            if keypoints.shape[0] >= num_dets:
                output['keypoints'] = keypoints[:num_dets]
            else:
                # Pad with zeros if not enough keypoints
                output['keypoints'] = np.zeros((num_dets, 17, 3))
                output['keypoints'][:len(keypoints)] = keypoints
        
        # Handle segmentation masks
        if self.task == 'segment' and isinstance(outputs, dict) and 'masks' in outputs and 'protos' in outputs:
            mask_coeffs = outputs['masks']  # List of (B, num_protos, H, W) per scale
            protos = outputs['protos']  # (B, num_protos, proto_H, proto_W)
            
            # Flatten mask coefficients
            mask_flat = []
            for mask in mask_coeffs:
                b, c, mh, mw = mask.shape
                mask_flat.append(mask.view(b, c, -1).permute(0, 2, 1))
            mask_flat = torch.cat(mask_flat, dim=1)[0]  # (N, num_protos)
            
            # Get prototypes
            proto = protos[0]  # (num_protos, pH, pW)
            proto_h, proto_w = proto.shape[1:]
            proto_flat = proto.view(proto.shape[0], -1)  # (num_protos, pH*pW)
            
            # Assemble all masks
            all_masks = torch.mm(mask_flat, proto_flat)  # (N, pH*pW)
            all_masks = all_masks.sigmoid().view(-1, proto_h, proto_w)  # (N, pH, pW)
            
            # Select masks for detections (simplified - take first num_dets)
            num_dets = len(boxes)
            if all_masks.shape[0] >= num_dets:
                masks = all_masks[:num_dets]
            else:
                masks = all_masks
            
            # Resize masks to original image size
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # Apply letterbox inverse transform
            masks = masks.cpu().numpy()
            
            # Crop padding (masks are already in letterbox space)
            pad_h = int(pad[1] / scale)
            pad_w = int(pad[0] / scale)
            
            output['masks'] = (masks > 0.5).astype(np.uint8)
        
        return output

    def _decode_boxes(self, reg_outputs, anchors, strides) -> torch.Tensor:
        """Internal helper for box decoding (DFL excluded for simplicity in ONNX)."""
        decoded_boxes = []
        for reg_out, anchor, stride in zip(reg_outputs, anchors, strides):
            b, c, h, w = reg_out.shape
            
            # For simplicity, if reg_out has 64 channels (4*16), we take the mean or first 4
            # A full DFL decoding would be better but this is more robust for exported models
            if c == 64:
                # Simplified DFL integration: take the mean of distribution
                reg_dist = reg_out.view(b, 4, 16, -1).softmax(2)
                weights = torch.arange(16, device=reg_out.device).view(1, 1, 16, 1)
                reg_dist = (reg_dist * weights).sum(2) # (B, 4, H*W)
            else:
                reg_dist = reg_out.view(b, 4, -1)
            
            reg_dist = reg_dist.permute(0, 2, 1) # (B, H*W, 4)
            anchor = anchor.view(-1, 2)
            
            lt = reg_dist[..., :2]
            rb = reg_dist[..., 2:]
            
            x1y1 = anchor - lt * stride
            x2y2 = anchor + rb * stride
            decoded_boxes.append(torch.cat([x1y1, x2y2], dim=-1))
            
        return torch.cat(decoded_boxes, dim=1)
    
    def _decode_keypoints(self, kpt_outputs, anchors, strides) -> torch.Tensor:
        """Internal helper for keypoint decoding."""
        decoded_kpts = []
        for kpt_out, anchor, stride in zip(kpt_outputs, anchors, strides):
            b, c, h, w = kpt_out.shape
            num_keypoints = c // 3  # x, y, visibility per keypoint
            
            # Reshape: (B, 51, H, W) -> (B, 17, 3, H*W)
            kpt = kpt_out.view(b, num_keypoints, 3, -1)
            
            # Decode: keypoint offset from anchor
            anchor_flat = anchor.view(-1, 2)  # (H*W, 2)
            
            # x, y offsets (scaled by stride)
            xy = kpt[:, :, :2, :]  # (B, 17, 2, H*W)
            vis = kpt[:, :, 2:3, :]  # (B, 17, 1, H*W)
            
            # Decode coordinates
            xy = xy.permute(0, 3, 1, 2)  # (B, H*W, 17, 2)
            vis = vis.permute(0, 3, 1, 2)  # (B, H*W, 17, 1)
            
            # Add anchor offset and scale
            xy = xy * stride + anchor_flat.view(1, -1, 1, 2)
            
            # Combine
            kpts = torch.cat([xy, vis.sigmoid()], dim=-1)  # (B, H*W, 17, 3)
            decoded_kpts.append(kpts)
        
        return torch.cat(decoded_kpts, dim=1)  # (B, N, 17, 3)

def run_image(predictor: YOLOv11Predictor, source: str, save_path: str = None):
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
            cv2.imshow('YOLOv11', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            auto_save = 'output_detection.jpg'
            cv2.imwrite(auto_save, vis_image)
            print(f'GUI not available, saved to: {auto_save}')

    
    return results


def run_video(predictor: YOLOv11Predictor, source, save_path: str = None):
    """Run inference on video or webcam with FPS counter."""
    # Convert string digit to int (for webcam source)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return
    
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') #type: ignore
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps_video, (w, h))
    
    print("Press 'q' to quit")
    
    # Accurate FPS calculation across frames
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inference
        results = predictor.predict(frame)
        
        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        
        # Visualize
        start_vis = time.time()
        vis_frame = visualize_predictions(
            frame, results, predictor.task, COCO_NAMES
        )
        vis_time = time.time() - start_vis
        
        # Add FPS Overlay
        cv2.putText(
            vis_frame, f'FPS: {fps:.1f} (Vis: {vis_time*1000:.1f}ms)', (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        if save_path:
            writer.write(vis_frame)
        else:
            cv2.imshow('YOLOv11 Inference', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if save_path:
        writer.release()
    cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Inference')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or webcam index')
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'segment', 'pose'])
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--save', type=str, default=None, help='Save path')
    parser.add_argument('--half', action='store_true', help='Use FP16 half precision')
    args = parser.parse_args()
    
    # Create predictor
    predictor = YOLOv11Predictor(
        weights=args.weights,
        task=args.task,
        device=args.device,
        conf_thres=args.conf,
        iou_thres=args.iou,
        img_size=args.img_size
    )
    
    if args.half:
        predictor.set_half(True)
    
    # Determine source type
    source = args.source
    if source.isdigit() or source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        run_video(predictor, source, args.save)
    else:
        run_image(predictor, source, args.save)


if __name__ == '__main__':
    main()
