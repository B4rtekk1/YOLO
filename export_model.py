import torch
from pathlib import Path
from yolov11.model import YOLOv11
from yolov11.utils.export import export_onnx
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='saves/last.pt', help='weights path')
    parser.add_argument('--img-size', type=int, default=640, help='input size')
    args = parser.parse_args()

    # 1. Load checkpoint
    print(f"Loading weights from {args.weights}...")
    ckpt = torch.load(args.weights, map_location='cpu')
    
    # 2. Extract config
    cfg = ckpt.get('config', {})
    model_size = cfg.get('model_size', 's')
    num_classes = cfg.get('num_classes', 80)
    task = cfg.get('task', 'detect')
    
    # 3. Initialize model
    model = YOLOv11(
        num_classes=num_classes,
        task=task,
        model_size=model_size
    )
    
    # 4. Load state dict
    sd = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    
    model.load_state_dict(sd)
    model.eval()
    model.fuse() # Fuse for better performance
    
    # ---- FP16 Conversion ----
    print("Converting model to FP16 (Half Precision)...")
    model.half()
    # Move to GPU for export to ensure CUDA ops are preserved if any
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 5. Export to ONNX
    output_path = Path(args.weights).parent / (Path(args.weights).stem + "_fp16.onnx")
    print(f"Exporting to {output_path}...")
    
    # Create dummy input on the correct device and dtype
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device).half()
    
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        opset_version=17,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=None
    )
    
    # Simplify
    try:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(str(output_path))
        onnx_model, check = simplify(onnx_model)
        if check:
            onnx.save(onnx_model, str(output_path))
            print("ONNX model simplified successfully")
    except Exception as e:
        print(f"Simplification skipped: {e}")
        
    print(f"Done! FP16 model saved to {output_path}")

if __name__ == "__main__":
    main()
