"""
YOLOv8 Training Script
Supports training for detection, segmentation, and pose estimation
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from yolov8.model import YOLOv8, create_model
from yolov8.losses import YOLOv8Loss
from yolov8.data import create_dataloader


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    loss_items = {'loss_box': 0, 'loss_cls': 0, 'loss_dfl': 0}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        for k in targets:
            if isinstance(targets[k], torch.Tensor):
                targets[k] = targets[k].to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k in loss_items:
            if k in loss_dict:
                loss_items[k] += loss_dict[k].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'box': f'{loss_dict.get("loss_box", 0):.4f}',
            'cls': f'{loss_dict.get("loss_cls", 0):.4f}'
        })
    
    n = len(dataloader)
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
    """Validate model."""
    model.eval()
    
    total_loss = 0
    loss_items = {'loss_box': 0, 'loss_cls': 0, 'loss_dfl': 0}
    
    for images, targets in tqdm(dataloader, desc='Validating'):
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
    return {
        'val_loss': total_loss / n,
        **{f'val_{k}': v / n for k, v in loss_items.items()}
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
    parser = argparse.ArgumentParser(description='Train YOLOv8')
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'segment', 'pose'])
    parser.add_argument('--model', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--ann', type=str, default=None, help='Path to COCO annotation file')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save-dir', type=str, default='runs/train')
    parser.add_argument('--num-classes', type=int, default=80)
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    save_dir = Path(args.save_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'Saving to: {save_dir}')
    
    # Create model
    model = create_model(
        num_classes=args.num_classes,
        task=args.task,
        model_size=args.model
    ).to(device)
    model.info()
    
    # Create loss function
    criterion = YOLOv8Loss(
        task=args.task,
        num_classes=args.num_classes
    )
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
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
        batch_size=args.batch,
        augment=True,
        shuffle=True,
        num_workers=args.workers
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_loss = float('inf')
    
    print(f'\nStarting training for {args.epochs} epochs...\n')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        print(f'\nEpoch {epoch}: loss={train_metrics["loss"]:.4f}, '
              f'box={train_metrics["loss_box"]:.4f}, '
              f'cls={train_metrics["loss_cls"]:.4f}, '
              f'lr={scheduler.get_last_lr()[0]:.6f}')
        
        # Save checkpoint
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            save_checkpoint(model, optimizer, epoch, best_loss, str(save_dir / 'best.pt'))
        
        save_checkpoint(model, optimizer, epoch, train_metrics['loss'], str(save_dir / 'last.pt'))
    
    print(f'\nTraining complete! Best loss: {best_loss:.4f}')
    print(f'Weights saved to: {save_dir}')


if __name__ == '__main__':
    main()
