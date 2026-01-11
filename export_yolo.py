"""
Export YOLOv11 model to ONNX or TensorRT
"""

import argparse
import torch
from yolov11.model import YOLOv11
from yolov11.utils.export import export_onnx, export_tensorrt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='PyTorch weights (.pt)')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'engine'], help='Export format')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--half', action='store_true', help='FP16 quantization')
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    model_size = config.get('model_size', 's')
    num_classes = config.get('num_classes', 80)
    task = config.get('task', 'detect')

    # Create model
    model = YOLOv11(num_classes=num_classes, task=task, model_size=model_size)
    
    # Load weights
    state_dict = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    save_path = Path(args.weights).with_suffix(f'.{args.format}')

    if args.format == 'onnx':
        export_onnx(model, str(save_path), input_size=(args.imgsz, args.imgsz))
    elif args.format == 'engine':
        # First export to ONNX
        onnx_path = Path(args.weights).with_suffix('.onnx')
        export_onnx(model, str(onnx_path), input_size=(args.imgsz, args.imgsz))
        
        # Then convert ONNX to TensorRT
        export_tensorrt(str(onnx_path), str(save_path), fp16=args.half)

if __name__ == '__main__':
    main()
