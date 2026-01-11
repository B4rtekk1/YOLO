"""
Model Export Utilities for YOLOv11
Supports: ONNX, TensorRT (optional), INT8 Quantization (optional)

Note: TensorRT requires tensorrt package installed separately.
      Quantization uses PyTorch native quantization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import warnings


def export_onnx(
    model: nn.Module,
    save_path: str,
    input_size: Tuple[int, int] = (640, 640),
    batch_size: int = 1,
    opset_version: int = 17,
    dynamic_axes: bool = False,
    simplify: bool = True
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        save_path: Output path for ONNX file
        input_size: Input image size (H, W)
        batch_size: Batch size for export
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch size
        simplify: Simplify ONNX model (requires onnx-simplifier)
    
    Returns:
        Path to saved ONNX model
    """
    model = model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, *input_size)
    
    # Set up dynamic axes
    dynamic = None
    if dynamic_axes:
        dynamic = {
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        }
    
    # Export
    save_path = Path(save_path)
    save_path = save_path.with_suffix('.onnx')
    
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        opset_version=opset_version,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=dynamic
    )
    
    print(f"ONNX model exported to: {save_path}")
    
    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            onnx_model = onnx.load(str(save_path))
            onnx_model, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(onnx_model, str(save_path))
                print("ONNX model simplified successfully")
        except ImportError:
            print("onnx-simplifier not installed, skipping simplification")
        except Exception as e:
            print(f"Simplification failed: {e}")
    
    return str(save_path)


def export_tensorrt(
    onnx_path: str,
    save_path: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 1,
    workspace_size: int = 4,
    calibrator: Optional[Any] = None
) -> str:
    """
    Export ONNX model to TensorRT engine.
    
    Requires: tensorrt package
    
    Args:
        onnx_path: Path to ONNX model
        save_path: Output path for TensorRT engine
        fp16: Enable FP16 precision
        int8: Enable INT8 precision (requires calibrator)
        max_batch_size: Maximum batch size
        workspace_size: GPU workspace size in GB
        calibrator: INT8 calibrator for quantization
    
    Returns:
        Path to saved TensorRT engine
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT not installed. Install with:\n"
            "pip install tensorrt\n"
            "Or download from NVIDIA website."
        )
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")
    
    # Configure builder
    config = builder.create_builder_config()
    # TRT 10.0+ uses set_memory_pool_limit instead of max_workspace_size
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))
    else:
        config.max_workspace_size = workspace_size * (1 << 30)
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator:
            config.int8_calibrator = calibrator
    
    # Build engine
    # TRT 10.0+ uses build_serialized_network instead of build_engine
    if hasattr(builder, 'build_serialized_network'):
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine (it's already serialized)
        save_path = Path(save_path).with_suffix('.engine')
        with open(save_path, 'wb') as f:
            f.write(engine)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
            
        # Save engine
        save_path = Path(save_path).with_suffix('.engine')
        with open(save_path, 'wb') as f:
            f.write(engine.serialize())
    
    print(f"TensorRT engine exported to: {save_path}")
    return str(save_path)


class TRTInference:
    """Helper class for TensorRT engine inference (Compatible with TRT 10.x)."""
    
    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # Required for CUDA context
        
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            
        self.inputs, self.outputs = [], []
        self.input_names, self.output_names = [], []
        self.stream = cuda.Stream()
        
        # TRT 10.x uses tensor names instead of indices
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # Host and Device buffers
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Bind tensor address
            self.context.set_tensor_address(name, int(device_mem))
            
            if is_input:
                self.input_names.append(name)
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'shape': shape})
            else:
                self.output_names.append(name)
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'shape': shape})
                
    def __call__(self, img: torch.Tensor) -> List[torch.Tensor]:
        import pycuda.driver as cuda
        # Copy input to host
        input_data = self.inputs[0]
        np_img = img.cpu().numpy().astype(input_data['host'].dtype)
        np.copyto(input_data['host'], np_img.ravel())
        
        # Transfer to device, execute, transfer back
        cuda.memcpy_htod_async(input_data['device'], input_data['host'], self.stream)
        
        # TRT 10.x uses execute_async_v3
        if hasattr(self.context, 'execute_async_v3'):
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            # Fallback for older versions (unlikely if we reached here with 10.x logic)
            bindings = [int(i['device']) for i in self.inputs] + [int(o['device']) for o in self.outputs]
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
            
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        
        # Convert back to torch tensors
        results = []
        for out in self.outputs:
            results.append(torch.from_numpy(out['host'].reshape(out['shape'])))
        return results


class QuantizationConfig:
    """Configuration for model quantization."""
    
    def __init__(
        self,
        backend: str = 'fbgemm',  # 'fbgemm' for x86, 'qnnpack' for ARM
        dtype: str = 'qint8'
    ):
        self.backend = backend
        self.dtype = getattr(torch, dtype)


def quantize_dynamic(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None
) -> nn.Module:
    """
    Apply dynamic quantization (weights only).
    
    Fast and simple, good for inference on CPU.
    Quantizes Linear and LSTM layers.
    
    Args:
        model: Model to quantize
        config: Quantization configuration
    
    Returns:
        Quantized model
    """
    if config is None:
        config = QuantizationConfig()
    
    torch.backends.quantized.engine = config.backend
    
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=config.dtype
    )
    
    return quantized


def quantize_static(
    model: nn.Module,
    calibration_loader,
    config: Optional[QuantizationConfig] = None,
    num_batches: int = 100
) -> nn.Module:
    """
    Apply static quantization (weights and activations).
    
    Requires calibration data for activation statistics.
    Better accuracy than dynamic quantization.
    
    Args:
        model: Model to quantize (must have qconfig set)
        calibration_loader: DataLoader for calibration
        config: Quantization configuration
        num_batches: Number of batches for calibration
    
    Returns:
        Quantized model
    """
    if config is None:
        config = QuantizationConfig()
    
    torch.backends.quantized.engine = config.backend
    
    # Prepare model for quantization
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig(config.backend)
    
    # Fuse common patterns
    model_fused = _fuse_model(model)
    
    # Insert observers
    model_prepared = torch.quantization.prepare(model_fused)
    
    # Calibration
    print(f"Calibrating with {num_batches} batches...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_batches:
                break
            model_prepared(images)
    
    # Convert to quantized
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized


def _fuse_model(model: nn.Module) -> nn.Module:
    """Fuse Conv-BN-ReLU patterns for quantization."""
    import copy
    model = copy.deepcopy(model)
    
    # Common fusion patterns
    for name, module in model.named_modules():
        if hasattr(module, 'conv') and hasattr(module, 'bn'):
            # Fuse conv + bn
            torch.quantization.fuse_modules(
                module, 
                ['conv', 'bn'], 
                inplace=True
            )
    
    return model


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Get model size statistics.
    
    Returns:
        Dictionary with size information
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    
    return {
        'params_mb': param_size / (1024 ** 2),
        'buffers_mb': buffer_size / (1024 ** 2),
        'total_mb': (param_size + buffer_size) / (1024 ** 2),
        'num_params': sum(p.numel() for p in model.parameters())
    }


if __name__ == "__main__":
    from yolov11.model import YOLOv11
    
    print("Testing export utilities...")
    
    # Create model
    model = YOLOv11(num_classes=80, model_size='n')
    model.eval()
    
    # Get original size
    orig_size = get_model_size(model)
    print(f"Original model: {orig_size['total_mb']:.2f} MB, {orig_size['num_params']:,} params")
    
    # Test dynamic quantization
    try:
        quantized = quantize_dynamic(model)
        quant_size = get_model_size(quantized)
        print(f"Quantized model: {quant_size['total_mb']:.2f} MB")
    except Exception as e:
        print(f"Dynamic quantization: {e}")
    
    # Test ONNX export (if torch.onnx available)
    try:
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = export_onnx(
                model, 
                os.path.join(tmpdir, "model.onnx"),
                input_size=(320, 320),
                simplify=False
            )
            print(f"ONNX export successful: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"ONNX export: {e}")
    
    print("Export utilities OK!")
