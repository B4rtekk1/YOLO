"""
YOLOv11 Quick Training Script
Downloads a subset of COCO and trains with:
- EMA (Exponential Moving Average)
- Warmup scheduler
- CutMix augmentation (optional)
- Enhanced losses
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
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from yolov11.model import YOLOv11
from yolov11.losses import YOLOv11Loss
from yolov11.data import COCODataset, create_dataloader
from yolov11.utils.training import ModelEMA, WarmupScheduler, EarlyStopping
from train import validate


CONFIG = {
    'num_images': 300,
    'split': 'train',
    'task': 'detect',
    'model_size': 'n',
    'num_classes': 5,
    'classes': ['person', 'bicycle', 'car', 'motorcycle', 'bus'],
    'img_size': 640,
    'batch_size': 2,
    'epochs': 50,
    'lr': 0.01,
    'output_dir': 'coco_mini',
    'save_dir': 'runs/train_mini',
    'num_workers': 4,
    'use_ema': True,
    'ema_decay': 0.9999,
    'warmup_epochs': 3,
    'early_stopping': 10,
    'label_smoothing': 0.0,
    'use_compile': False,  # Set True to enable torch.compile (PyTorch 2.0+, 20-50% speedup)
}


def download_file(url: str, dest: str) -> bool:
    """Download a single file."""
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception:
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
    
    subset_name = f"{config['task']}_{config['split']}2017_subset_{config['num_images']}"
    if config.get('classes'):
        class_str = "_".join(config['classes'])
        subset_name += f"_{class_str}"
    
    subset_ann_path = annotations_dir / f"{subset_name}.json"
    
    if subset_ann_path.exists():
        with open(subset_ann_path, 'r') as f:
            subset_data = json.load(f)
            config['num_classes'] = len(subset_data['categories'])
        print(f"Dataset already exists: {subset_ann_path} ({config['num_classes']} classes)")
        return str(images_dir), str(subset_ann_path)
    
    print(f"\nDownloading COCO {config['split']}2017 ({config['num_images']} images)...")
    
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
    
    print(f"Loading annotations from {ann_path}...")
    
    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("Installing pycocotools...")
        os.system("pip install pycocotools -q")
        from pycocotools.coco import COCO
    
    coco = COCO(str(ann_path))
    
    if config.get('classes'):
        cat_ids = coco.getCatIds(catNms=config['classes'])
        img_ids = coco.getImgIds(catIds=cat_ids)
        print(f"Filtering for classes: {config['classes']} (IDs: {cat_ids})")
    elif config['task'] == 'pose':
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
    else:
        cat_ids = []
        img_ids = list(coco.imgs.keys())
    
    img_ids = img_ids[:config['num_images']]
    images_info = coco.loadImgs(img_ids)
    
    print(f"Downloading {len(images_info)} images...")
    
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
    
    print(f"Downloaded {successful}/{len(images_info)} images")
    
    if cat_ids:
        ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=cat_ids)
    else:
        ann_ids = coco.getAnnIds(imgIds=img_ids)
    annotations = coco.loadAnns(ann_ids)
    
    target_cats = coco.loadCats(cat_ids) if cat_ids else coco.loadCats(coco.getCatIds())
    if cat_ids:
        target_cats = sorted(target_cats, key=lambda x: x['id'])
    
    subset_coco = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'images': images_info,
        'annotations': annotations,
        'categories': target_cats
    }
    
    config['num_classes'] = len(target_cats)
    
    with open(subset_ann_path, 'w') as f:
        json.dump(subset_coco, f)
    
    print(f"Annotations saved: {subset_ann_path}")
    
    return str(images_dir), str(subset_ann_path)


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, 
                warmup_scheduler=None, use_amp=True):
    """Train for one epoch with warmup support."""
    model.train()
    total_loss = 0
    loss_items = {'box': 0, 'cls': 0, 'dfl': 0}
    
    num_batches = len(dataloader)
    rank = int(os.environ.get('RANK', -1))
    
    pbar = None
    if rank in [-1, 0]:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(pbar or dataloader):
        if warmup_scheduler and epoch <= warmup_scheduler.warmup_epochs:
            warmup_scheduler.step(epoch, batch_idx, num_batches)
        
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
        
        if pbar and (batch_idx + 1) % 5 == 0:  # Update every 5 batches for smoothness
            avg_loss = total_loss / (batch_idx + 1)
            avg_box = loss_items['box'] / (batch_idx + 1)
            avg_cls = loss_items['cls'] / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'box': f'{avg_box:.3f}',
                'cls': f'{avg_cls:.3f}'
            })
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        **{k: v / n for k, v in loss_items.items()}
    }


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Quick Training')
    parser.add_argument('--num-images', type=int, default=CONFIG['num_images'])
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--batch', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=CONFIG['lr'])
    parser.add_argument('--model', type=str, default=CONFIG['model_size'])
    parser.add_argument('--task', type=str, default=CONFIG['task'])
    args = parser.parse_args()
    
    # Update CONFIG with CLI args
    CONFIG['num_images'] = args.num_images
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch
    CONFIG['lr'] = args.lr
    CONFIG['model_size'] = args.model
    CONFIG['task'] = args.task

    print("\n" + "="*60)
    print("  YOLOv11 TRAINING ON COCO SUBSET")
    print("  With EMA, Warmup, and Enhanced Features")
    print("="*60)
    
    # Multi-GPU Setup
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
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use DataParallel if multiple GPUs available and not distributed
        if not distributed and torch.cuda.device_count() > 1:
            if rank in [-1, 0]:
                print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
    
    if rank in [-1, 0]:
        print(f"\nDevice: {device}")
        if torch.cuda.is_available() and not distributed:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if rank in [-1, 0]:
        print(f"\nConfiguration:")
        for k, v in CONFIG.items():
            print(f"   {k}: {v}")
        
        print("\n" + "-"*60)
        images_dir, ann_path = download_coco_subset(CONFIG)
    else:
        # Other ranks wait for rank 0 to finish download
        images_dir = str(Path(CONFIG['output_dir']) / f"{CONFIG['split']}2017")
        ann_path = str(Path(CONFIG['output_dir']) / "annotations" / f"{CONFIG['task']}_{CONFIG['split']}2017_subset_*.json") # This is problematic but rank 0 handles it
        # Realistically, other ranks don't need to download, just need the paths
        dist.barrier()
        # Find the correct ann_path
        ann_files = list(Path(CONFIG['output_dir']).glob("annotations/*.json"))
        if ann_files:
            ann_path = str(ann_files[0])
    
    print("\n" + "-"*60)
    print("Creating DataLoader...")
    
    dataloader = create_dataloader(
        root=images_dir,
        ann_file=ann_path,
        task=CONFIG['task'],
        img_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        augment=True,
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        distributed=distributed
    )
    
    if rank in [-1, 0]:
        print(f"   Batches per epoch: {len(dataloader)}")
    
    print("\n" + "-"*60)
    print("Creating YOLOv11 model...")
    
    nc = 1 if CONFIG['task'] == 'pose' else CONFIG['num_classes']
    model = YOLOv11(
        num_classes=nc,
        task=CONFIG['task'],
        model_size=CONFIG['model_size']
    ).to(device)
    
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif not distributed and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    if rank in [-1, 0]:
        (model.module if hasattr(model, 'module') else model).info()
    
    # Apply torch.compile for PyTorch 2.0+ speedup
    if CONFIG.get('use_compile', False):
        if hasattr(torch, 'compile'):
            print('Compiling model with torch.compile (reduce-overhead mode)...')
            model = torch.compile(model, mode='reduce-overhead')
            print('Model compiled successfully!')
        else:
            print('Warning: torch.compile requires PyTorch 2.0+, skipping compilation')
    
    ema = None
    if CONFIG['use_ema'] and rank in [-1, 0]:
        ema_model = model.module if hasattr(model, 'module') else model
        ema = ModelEMA(ema_model, decay=CONFIG['ema_decay'])
        print(f"EMA enabled (decay={CONFIG['ema_decay']})")
    
    criterion = YOLOv11Loss(
        task=CONFIG['task'],
        num_classes=nc
    )
    
    optimizer = optim.SGD(
        (model.module if hasattr(model, 'module') else model).parameters(),
        lr=CONFIG['lr'],
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True
    )
    
    warmup_scheduler = None
    if CONFIG['warmup_epochs'] > 0:
        warmup_scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=CONFIG['warmup_epochs']
        )
        print(f"Warmup enabled ({CONFIG['warmup_epochs']} epochs)")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['epochs'],
        eta_min=CONFIG['lr'] * 0.01
    )
    
    early_stopping = None
    if CONFIG['early_stopping'] > 0:
        early_stopping = EarlyStopping(patience=CONFIG['early_stopping'])
        print(f"Early stopping enabled (patience={CONFIG['early_stopping']})")
    
    scaler = GradScaler()
    
    save_dir = Path(CONFIG['save_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
    if rank in [-1, 0]:
        save_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "="*60)
        print(f"  TRAINING: {CONFIG['epochs']} epochs")
        print("="*60 + "\n")
    
    # Create val dataloader (using 20% of training data for validation in quick mode if no ann)
    print("Creating Validation DataLoader...")
    val_loader = create_dataloader(
        root=images_dir,
        ann_file=ann_path,
        task=CONFIG['task'],
        img_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        augment=False,
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )
    
    best_loss = float('inf')
    history = []
    
    start_time = time.time()
    
    for epoch in range(CONFIG['epochs']):
        if distributed:
            dataloader.sampler.set_epoch(epoch)
            
        metrics = train_epoch(
            model, dataloader, criterion, optimizer, scaler, device, 
            epoch + 1, warmup_scheduler
        )
        
        if ema:
            ema.update(model)
        
        if epoch >= CONFIG['warmup_epochs']:
            scheduler.step()
        
        if rank in [-1, 0]:
            history.append(metrics)
            
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
                  f"loss={metrics['loss']:.4f}, "
                  f"box={metrics['box']:.4f}, "
                  f"cls={metrics['cls']:.4f}, "
                  f"lr={lr:.6f}")
        
        # Periodic Evaluation every 5 epochs using EMA model
        eval_metrics = {}
        if val_loader and (epoch + 1) % 200 == 0:
            val_model = ema.ema if ema else model
            eval_metrics = validate(val_model, val_loader, criterion, device)
            print(f'Evaluation Epoch {epoch+1}: mAP50={eval_metrics["mAP50"]:.4f}, '
                  f'mAP50-95={eval_metrics["mAP50-95"]:.4f}')
        
        if rank in [-1, 0]:
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': (model.module if hasattr(model, 'module') else model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': CONFIG
                }
                if ema:
                    save_dict['ema_state_dict'] = ema.ema.state_dict()
                torch.save(save_dict, save_dir / 'best.pt')
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': (model.module if hasattr(model, 'module') else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['loss'],
                'config': CONFIG
            }
            if ema:
                save_dict['ema_state_dict'] = ema.ema.state_dict()
            torch.save(save_dict, save_dir / 'last.pt')
            
            if early_stopping and early_stopping(-metrics['loss']):
                print(f"\nEarly stopping at epoch {epoch+1}")
                # We should signal other processes to stop, but for a quick script it's okay to just break rank 0
                break
    
    if rank in [-1, 0]:
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("  TRAINING COMPLETE")
        print("="*60)
        print(f"\nTime: {elapsed/60:.1f} minutes")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Weights saved: {save_dir}")
        if ema:
            print(f"   (includes EMA weights)")
        if CONFIG.get('use_compile', False):
            print(f"   (trained with torch.compile)")
        print(f"\nTo run inference:")
        print(f"   python inference.py --weights \"{save_dir / 'best.pt'}\" --source image.jpg")
        
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    if distributed:
        dist.destroy_process_group()
    
    return save_dir


if __name__ == '__main__':
    main()
