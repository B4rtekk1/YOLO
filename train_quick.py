"""
YOLOv11 Quick Training Script
Downloads a subset of COCO and trains with enhanced features:
- EMA (Exponential Moving Average)
- Warmup scheduler
- CutMix augmentation
- Enhanced losses (Wise-IoU optional)
"""

import os
import sys
import json
import urllib.request
import zipfile
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from yolov8.model import YOLOv11
from yolov8.losses import YOLOv8Loss
from yolov8.data import COCODataset, create_dataloader
from yolov8.utils.training import ModelEMA, WarmupScheduler, EarlyStopping


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'num_images': 200,          # Number of images to download
    'split': 'train',             # 'train' or 'val'
    'task': 'detect',           # 'detect', 'segment', 'pose'
    'model_size': 'n',          # 'n', 's', 'm', 'l', 'x'
    'num_classes': 80,          # COCO has 80 classes
    'img_size': 640,
    'batch_size': 8,
    'epochs': 20,
    'lr': 0.01,
    'output_dir': 'coco_mini',
    'save_dir': 'runs/train_mini',
    'num_workers': 4,
    # New YOLOv11 features
    'use_ema': True,            # Use Exponential Moving Average
    'ema_decay': 0.9999,
    'warmup_epochs': 3,         # Warmup epochs
    'early_stopping': 10,       # Patience for early stopping (0 to disable)
    'label_smoothing': 0.0,     # Label smoothing (0.0-0.1)
}


# ============================================================
# DOWNLOAD COCO SUBSET
# ============================================================

def download_file(url: str, dest: str) -> bool:
    """Download a single file."""
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        return False


def download_image(args):
    """Download single image (for ThreadPoolExecutor)."""
    img_info, dest_dir, base_url = args
    filename = img_info['file_name']
    url = f"{base_url}/{filename}"
    dest_path = dest_dir / filename
    
    if dest_path.exists():
        return True
    
    return download_file(url, str(dest_path))


def download_coco_subset(config: dict):
    """Download COCO subset."""
    output_dir = Path(config['output_dir'])
    images_dir = output_dir / f"{config['split']}2017"
    annotations_dir = output_dir / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    subset_ann_path = annotations_dir / f"{config['task']}_{config['split']}2017_subset_{config['num_images']}.json"
    if subset_ann_path.exists() and len(list(images_dir.glob('*.jpg'))) >= config['num_images'] * 0.9:
        print(f"✅ Dataset already exists: {subset_ann_path}")
        return str(images_dir), str(subset_ann_path)
    
    print(f"\n📥 Downloading COCO {config['split']}2017 ({config['num_images']} images)...")
    
    # Download annotations
    ann_filename = 'instances_val2017.json' if config['split'] == 'val' else 'instances_train2017.json'
    if config['task'] == 'pose':
        ann_filename = f"person_keypoints_{config['split']}2017.json"
    
    ann_path = annotations_dir / ann_filename
    
    if not ann_path.exists():
        print("Downloading annotations (~250MB)...")
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = output_dir / "annotations.zip"
        
        with tqdm(unit='B', unit_scale=True, desc="annotations") as pbar:
            def reporthook(count, block_size, total_size):
                if pbar.total is None and total_size > 0:
                    pbar.total = total_size
                pbar.update(block_size)
            
            urllib.request.urlretrieve(ann_url, str(zip_path), reporthook)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)
        zip_path.unlink()
    
    # Load annotations
    print(f"Loading annotations from {ann_path}...")
    
    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("Installing pycocotools...")
        os.system("pip install pycocotools -q")
        from pycocotools.coco import COCO
    
    coco = COCO(str(ann_path))
    
    # Select images
    if config['task'] == 'pose':
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
    else:
        img_ids = list(coco.imgs.keys())
    
    img_ids = img_ids[:config['num_images']]
    images_info = coco.loadImgs(img_ids)
    
    print(f"📸 Downloading {len(images_info)} images...")
    
    base_url = f"http://images.cocodataset.org/{config['split']}2017"
    download_args = [(img, images_dir, base_url) for img in images_info]
    
    successful = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_image, arg): arg for arg in download_args}
        
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                pbar.update(1)
    
    print(f"✅ Downloaded {successful}/{len(images_info)} images")
    
    # Create subset annotation file
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    annotations = coco.loadAnns(ann_ids)
    
    subset_coco = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'images': images_info,
        'annotations': annotations,
        'categories': coco.dataset.get('categories', [])
    }
    
    with open(subset_ann_path, 'w') as f:
        json.dump(subset_coco, f)
    
    print(f"✅ Annotations: {subset_ann_path}")
    
    return str(images_dir), str(subset_ann_path)


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, 
                warmup_scheduler=None, use_amp=True):
    """Train for one epoch with warmup support."""
    model.train()
    total_loss = 0
    loss_items = {'box': 0, 'cls': 0, 'dfl': 0}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        # Warmup
        if warmup_scheduler and epoch <= warmup_scheduler.warmup_epochs:
            warmup_scheduler.step(epoch, batch_idx, len(dataloader))
        
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
        for k in ['loss_box', 'loss_cls', 'loss_dfl']:
            if k in loss_dict:
                loss_items[k.replace('loss_', '')] += loss_dict[k].item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'box': f'{loss_dict.get("loss_box", 0):.3f}',
            'cls': f'{loss_dict.get("loss_cls", 0):.3f}'
        })
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        **{k: v / n for k, v in loss_items.items()}
    }


def main():
    print("\n" + "="*60)
    print("  YOLOv11 TRAINING ON COCO SUBSET")
    print("  With EMA, Warmup, and Enhanced Features")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Print config
    print(f"\n📋 Configuration:")
    for k, v in CONFIG.items():
        print(f"   {k}: {v}")
    
    # Download COCO subset
    print("\n" + "-"*60)
    images_dir, ann_path = download_coco_subset(CONFIG)
    
    # Create dataloader
    print("\n" + "-"*60)
    print("📦 Creating DataLoader...")
    
    dataloader = create_dataloader(
        root=images_dir,
        ann_file=ann_path,
        task=CONFIG['task'],
        img_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        augment=True,
        shuffle=True,
        num_workers=CONFIG['num_workers']
    )
    
    print(f"   Batches per epoch: {len(dataloader)}")
    
    # Create model
    print("\n" + "-"*60)
    print("🧠 Creating YOLOv11 model...")
    
    nc = 1 if CONFIG['task'] == 'pose' else CONFIG['num_classes']
    model = YOLOv11(
        num_classes=nc,
        task=CONFIG['task'],
        model_size=CONFIG['model_size']
    ).to(device)
    
    model.info()
    
    # Create EMA model
    ema = None
    if CONFIG['use_ema']:
        ema = ModelEMA(model, decay=CONFIG['ema_decay'])
        print(f"✅ EMA enabled (decay={CONFIG['ema_decay']})")
    
    # Create loss function
    criterion = YOLOv8Loss(
        task=CONFIG['task'],
        num_classes=nc
    )
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=CONFIG['lr'],
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True
    )
    
    # Warmup scheduler
    warmup_scheduler = None
    if CONFIG['warmup_epochs'] > 0:
        warmup_scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=CONFIG['warmup_epochs']
        )
        print(f"✅ Warmup enabled ({CONFIG['warmup_epochs']} epochs)")
    
    # Main scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['epochs'],
        eta_min=CONFIG['lr'] * 0.01
    )
    
    # Early stopping
    early_stopping = None
    if CONFIG['early_stopping'] > 0:
        early_stopping = EarlyStopping(patience=CONFIG['early_stopping'])
        print(f"✅ Early stopping enabled (patience={CONFIG['early_stopping']})")
    
    # Mixed precision
    scaler = GradScaler()
    
    # Save directory
    save_dir = Path(CONFIG['save_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training
    print("\n" + "="*60)
    print(f"  TRAINING: {CONFIG['epochs']} epochs")
    print("="*60 + "\n")
    
    best_loss = float('inf')
    history = []
    
    start_time = time.time()
    
    for epoch in range(CONFIG['epochs']):
        # Train
        metrics = train_epoch(
            model, dataloader, criterion, optimizer, scaler, device, 
            epoch + 1, warmup_scheduler
        )
        
        # Update EMA
        if ema:
            ema.update(model)
        
        # Update scheduler (after warmup)
        if epoch >= CONFIG['warmup_epochs']:
            scheduler.step()
        
        # Log
        history.append(metrics)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
              f"loss={metrics['loss']:.4f}, "
              f"box={metrics['box']:.4f}, "
              f"cls={metrics['cls']:.4f}, "
              f"lr={lr:.6f}")
        
        # Save best
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': CONFIG
            }
            if ema:
                save_dict['ema_state_dict'] = ema.ema.state_dict()
            torch.save(save_dict, save_dir / 'best.pt')
        
        # Save last
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics['loss'],
            'config': CONFIG
        }
        if ema:
            save_dict['ema_state_dict'] = ema.ema.state_dict()
        torch.save(save_dict, save_dir / 'last.pt')
        
        # Early stopping
        if early_stopping and early_stopping(-metrics['loss']):
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"\n⏱️  Time: {elapsed/60:.1f} minutes")
    print(f"📉 Best loss: {best_loss:.4f}")
    print(f"💾 Weights saved: {save_dir}")
    if ema:
        print(f"   (includes EMA weights)")
    print(f"\n🚀 To run inference:")
    print(f"   python inference.py --weights \"{save_dir / 'best.pt'}\" --source image.jpg")
    
    # Save training history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return save_dir


if __name__ == '__main__':
    main()
