"""
train.py \u2013 YOLOv11 Training Script
=====================================

End-to-end training script supporting single-GPU, multi-GPU (DataParallel),
and distributed multi-GPU (DistributedDataParallel / torchrun) training for
all three YOLOv11 tasks: detection, instance segmentation, and pose estimation.

Training pipeline
-----------------
1. Parse CLI arguments and initialise the distributed process group (if any).
2. Build :class:`~yolov11.model.YOLOv11` and optionally wrap in DDP.
3. Optionally compile the model with ``torch.compile`` (PyTorch 2.0+).
4. Create :class:`~yolov11.losses.YOLOv11Loss` with the requested task.
5. Set up SGD + cosine-annealing LR scheduler.
6. Create train / val :class:`~torch.utils.data.DataLoader` instances.
7. Optionally resume from a checkpoint (full or partial weight loading).
8. Initialise :class:`~yolov11.utils.training.ModelEMA` and
   :class:`~yolov11.utils.training.WarmupScheduler`.
9. Run the training loop:
   - :func:`train_one_epoch` with AMP, gradient scaling, warmup, and EMA.
   - :func:`validate` every 5 epochs using the EMA model.
   - Save ``best.pt`` (EMA, highest score) and ``last.pt`` (raw model).

CLI reference
-------------
::

    python train.py \\
        --task detect          # detect | segment | pose
        --model s              # n | s | m | l | x
        --data /path/to/data   # root directory of images
        --ann  /path/to/train_annotations.json   # COCO format (optional)
        --val-ann /path/to/val_annotations.json  # separate val set (optional)
        --epochs 100
        --batch  16            # total batch size (divided across GPUs)
        --img-size 640
        --lr 0.01
        --workers 4
        --device 0             # GPU index, or 'cpu'
        --resume /path/best.pt # resume from checkpoint
        --save-dir runs/train
        --num-classes 80
        --compile              # torch.compile (PyTorch 2.0+, ~20-50% speedup)
        --ema-decay 0.9999
        --warmup-epochs 3
        --label-smoothing 0.0
        --use-cbam             # enable CBAM attention in backbone

Distributed training (torchrun)
--------------------------------
::

    torchrun --nproc_per_node=4 train.py --batch 64 --data /path/to/data ...
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from yolov11.model import YOLOv11, create_model
from yolov11.losses import YOLOv11Loss
from yolov11.data import create_dataloader
from yolov11.utils.training import ModelEMA, WarmupScheduler


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: 'Any',
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    warmup: 'WarmupScheduler | None' = None,
    ema: 'ModelEMA | None' = None
) -> dict:
    """Train for one epoch with warmup and EMA support."""
    model.train()
    
    total_loss = 0
    loss_items = {'loss_box': 0, 'loss_cls': 0, 'loss_dfl': 0}
    num_batches = len(dataloader)
    rank = int(os.environ.get('RANK', -1))
    
    pbar = None
    if rank in [-1, 0]:
        pbar = tqdm(dataloader, desc=f'Ep{epoch}', bar_format='{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}')
    
    for batch_idx, (images, targets) in enumerate(pbar or dataloader):
        # Apply warmup scheduler
        if warmup is not None:
            warmup.step(epoch, batch_idx, num_batches)
        
        images = images.to(device)
        for k in targets:
            if isinstance(targets[k], torch.Tensor):
                targets[k] = targets[k].to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=use_amp):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA after optimizer step
        if ema is not None:
            ema.update(model)
        
        total_loss += loss.item()
        for k in loss_items:
            if k in loss_dict:
                loss_items[k] += loss_dict[k].item()
        
        if pbar and (batch_idx + 1) % 5 == 0:  # Update every 5 batches for smoothness and less stutter
            avg_loss = total_loss / (batch_idx + 1)
            avg_box = loss_items['loss_box'] / (batch_idx + 1)
            avg_cls = loss_items['loss_cls'] / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'box': f'{avg_box:.3f}',
                'cls': f'{avg_cls:.3f}'
            })
    
    n = len(dataloader)
    if n == 0:
        return {
            'loss': 0,
            **{k: 0 for k in loss_items}
        }
    
    return {
        'loss': total_loss / n,
        **{k: v / n for k, v in loss_items.items()}
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    Validate model (loss only).
    For accurate mAP evaluation, use validate.py with pycocotools.
    """
    model.eval()
    
    total_loss = 0
    loss_items = {'loss_box': 0, 'loss_cls': 0, 'loss_dfl': 0}
    
    rank = int(os.environ.get('RANK', -1))
    
    loader_iter = tqdm(dataloader, desc='Validating') if rank in [-1, 0] else dataloader
    for images, targets in loader_iter:
        images = images.to(device)
        for k in targets:
            if isinstance(targets[k], torch.Tensor):
                targets[k] = targets[k].to(device)
        
        outputs = model(images)
        loss, loss_dict = criterion(outputs, targets)
        
        total_loss += loss.item()
        for k in loss_items:
            if k in loss_dict:
                loss_items[k] += loss_dict[k].item()
    
    n = len(dataloader)
    if n == 0:
        return {'val_loss': 0, 'mAP50': 0, 'mAP50-95': 0}
    
    # In DDP, aggregate loss
    if dist.is_initialized():
        loss_tensor = torch.tensor([total_loss, n], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = loss_tensor[0] / max(1, loss_tensor[1].item())
        return {
            'val_loss': avg_val_loss.item(),
            'mAP50': 0,  # Use validate.py for accurate mAP
            'mAP50-95': 0
        }
    
    return {
        'val_loss': total_loss / n,
        **{f'val_{k}': v / max(1, n) for k, v in loss_items.items()},
        'mAP50': 0,  # Use validate.py for accurate mAP
        'mAP50-95': 0
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f'Checkpoint saved: {path}')


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11')
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'segment', 'pose'])
    parser.add_argument('--model', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--ann', type=str, default=None, help='Path to COCO training annotation file')
    parser.add_argument('--val-ann', type=str, default=None, help='Path to COCO validation annotation file (separate from training)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save-dir', type=str, default='runs/train')
    parser.add_argument('--num-classes', type=int, default=80)
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for 20-50% speedup (PyTorch 2.0+)')
    # Training enhancements
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='EMA decay factor')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Number of warmup epochs')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing factor (0.0-0.1)')
    parser.add_argument('--copypaste', type=float, default=0.0, help='CopyPaste augmentation probability')
    parser.add_argument('--use-cbam', action='store_true', help='Use CBAM attention in backbone')
    parser.add_argument('--partial-load', action='store_true', help='Allow partial weight loading (for architecture changes)')
    parser.add_argument('--reset-optimizer', action='store_true', help='Discard optimizer state and start fresh (useful when changing batch size)')
    parser.add_argument('--reset-lr', action='store_true', help='Reset learning rate when resuming (ignore saved optimizer LR)')
    args = parser.parse_args()
    
    # Multi-GPU Setup at the very beginning
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    distributed = rank != -1
    
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        if rank == 0:
            print(f'Distributed training initialized: Rank {rank}, World Size {world_size}')
        
        # Adjust batch size for distributed training
        # args.batch is the TOTAL batch size across all GPUs
        per_gpu_batch = max(1, args.batch // world_size)
        if rank == 0:
            print(f'Adjusted batch size per GPU: {per_gpu_batch} (Total: {args.batch})')
    else:
        if args.device == 'cpu':
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        per_gpu_batch = args.batch

        # Use DataParallel if multiple GPUs available and not distributed
        if not distributed and torch.cuda.device_count() > 1 and args.device != 'cpu':
            if rank in [-1, 0]:
                print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
    
    if rank in [-1, 0]:
        print(f'Using device: {device}')
    
    # Create save directory (only on rank 0)
    save_dir = Path(args.save_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    if rank in [-1, 0]:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f'Saving to: {save_dir}')
    
    # Create model
    from yolov11.model import YOLOv11
    model = YOLOv11(
        num_classes=args.num_classes,
        task=args.task,
        model_size=args.model,
        use_cbam=args.use_cbam
    ).to(device)
    
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif not distributed and torch.cuda.device_count() > 1 and args.device != 'cpu':
        model = nn.DataParallel(model)
        
    if rank in [-1, 0]:
        # Unwrap for info
        (model.module if hasattr(model, 'module') else model).info()
    
    # Apply torch.compile for PyTorch 2.0+ speedup
    if args.compile:
        if hasattr(torch, 'compile'):
            print('Compiling model with torch.compile (reduce-overhead mode)...')
            model = torch.compile(model, mode='reduce-overhead')
            print('Model compiled successfully!')
        else:
            print('Warning: torch.compile requires PyTorch 2.0+, skipping compilation')
    
    # Create loss function with label smoothing
    criterion = YOLOv11Loss(
        task=args.task,
        num_classes=args.num_classes,
        label_smoothing=args.label_smoothing
    )
    
    # Create optimizer
    optimizer = optim.SGD(
        (model.module if hasattr(model, 'module') else model).parameters(),
        lr=args.lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Create dataloader
    train_loader = create_dataloader(
        root=args.data,
        ann_file=args.ann,
        task=args.task,
        img_size=args.img_size,
        batch_size=per_gpu_batch,
        augment=True,
        shuffle=True,
        num_workers=args.workers,
        distributed=distributed
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        base_model = model.module if hasattr(model, 'module') else model
        
        if args.partial_load:
            # Partial loading for architecture changes (e.g., adding CBAM)
            base_model.load_partial(checkpoint['model_state_dict'])
            if rank in [-1, 0]:
                print(f'Partially loaded weights from {args.resume}')
        else:
            # Standard loading
            base_model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            if args.reset_optimizer or args.reset_lr:
                # Don't load optimizer state - use fresh LR and empty momentum buffers
                if rank in [-1, 0]:
                    status = "optimizer RESET" if args.reset_optimizer else "LR RESET"
                    print(f'Resumed from epoch {start_epoch} with {status}: {args.lr}')
            else:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if rank in [-1, 0]:
                    print(f'Resumed from epoch {start_epoch}')
    
    # Mixed precision scaler
    # Use torch.amp.GradScaler for newer PyTorch versions
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = GradScaler()
    
    # Initialize EMA (all ranks to support distributed validation)
    ema_model = model.module if hasattr(model, 'module') else model
    ema = ModelEMA(ema_model, decay=args.ema_decay)
    if rank in [-1, 0]:
        print(f'EMA initialized with decay={args.ema_decay}')
    
    # Initialize warmup scheduler (pass start_epoch if resetting optimizer for correct progress calculation)
    warmup_start = start_epoch if (args.reset_optimizer or args.reset_lr) else 0
    warmup = WarmupScheduler(optimizer, warmup_epochs=args.warmup_epochs, start_epoch=warmup_start)
    if rank in [-1, 0]:
        print(f'Warmup scheduled for {args.warmup_epochs} epochs')
    
    # Training loop
    best_loss = float('inf')
    
    # Create val dataloader if annotations provided or auto-detected
    val_loader = None
    if args.val_ann:
        # Use explicitly provided validation annotations
        val_ann = args.val_ann
    else:
        # Try to auto-detect coco_mini val set
        val_ann_path = Path(args.data) / "annotations/instances_val2017.json"
        val_ann = str(val_ann_path) if val_ann_path.exists() else None
        
    if val_ann:
        print(f"Using validation annotations: {val_ann}")
        val_loader = create_dataloader(
            root=args.data,
            ann_file=val_ann,
            task=args.task,
            img_size=args.img_size,
            batch_size=per_gpu_batch,
            augment=False,
            shuffle=False,
            num_workers=args.workers,
            distributed=distributed
        )
    elif rank in [-1, 0]:
        print("Warning: No validation annotations found. mAP evaluation will be skipped.")
    
    print(f'\nStarting training for {args.epochs} epochs...\n')
    
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
            
        # Determine warmup limit (relative to start_epoch if optimizer was reset)
        warmup_limit = start_epoch + args.warmup_epochs if (args.reset_optimizer or args.reset_lr) else args.warmup_epochs
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            warmup=warmup if epoch < warmup_limit else None,
            ema=ema
        )
        
        # Update scheduler (only after warmup completes)
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        # Periodic Evaluation every 5 epochs using EMA model
        eval_metrics = {}
        if val_loader and (epoch + 1) % 5 == 0:
            eval_metrics = validate(ema.ema, val_loader, criterion, device)
            if rank in [-1, 0]:
                print(f'Evaluation Epoch {epoch}: mAP50={eval_metrics["mAP50"]:.4f}, '
                      f'mAP50-95={eval_metrics["mAP50-95"]:.4f}')
        
        # Synchronize ranks if needed, but validate is only on rank 0
        if distributed:
            dist.barrier()
        
        # Save checkpoints (only on rank 0)
        if rank in [-1, 0]:
            save_score = eval_metrics.get('mAP50-95', -train_metrics['loss'])
            if save_score > best_loss: # Higher mAP or lower negative loss
                best_loss = save_score
                save_checkpoint(ema.ema, optimizer, epoch, train_metrics['loss'], str(save_dir / 'best.pt'))
            
            save_checkpoint(model.module if hasattr(model, 'module') else model, optimizer, epoch, train_metrics['loss'], str(save_dir / 'last.pt'))
            torch.save(ema.ema.state_dict(), str(save_dir / 'last_ema.pt'))
    
    if rank in [-1, 0]:
        print(f'\nTraining complete! Best Score: {best_loss:.4f}')
        print(f'Weights saved to: {save_dir}')
    
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
