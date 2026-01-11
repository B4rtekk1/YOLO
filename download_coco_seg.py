"""
COCO Instance Segmentation Dataset Downloader
Downloads and prepares COCO instance segmentation dataset for training
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

# COCO URLs
COCO_URLS = {
    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
    'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

# File sizes
FILE_SIZES = {
    'train2017': '18GB',
    'val2017': '1GB',
    'annotations': '252MB'
}


class DownloadProgressBar(tqdm):
    """tqdm progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path, desc: str = None):
    """Download file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists():
        print(f'  {dest.name} already exists, skipping...')
        return
    
    print(f'  Downloading {desc or dest.name}...')
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dest.name) as t:
        urlretrieve(url, dest, reporthook=t.update_to)


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract zip file with progress."""
    print(f'  Extracting {zip_path.name}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc='Extracting'):
            zip_ref.extract(member, dest_dir)


def create_mini_dataset(ann_file: Path, output_file: Path, max_images: int = 5000):
    """
    Create a mini version of the dataset for quick testing.
    
    Args:
        ann_file: Full annotation file
        output_file: Output file for mini dataset
        max_images: Maximum number of images to include
    """
    print(f'  Creating mini dataset ({max_images} images)...')
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Select first N images
    selected_images = data['images'][:max_images]
    selected_image_ids = {img['id'] for img in selected_images}
    
    # Filter annotations
    selected_anns = [ann for ann in data['annotations'] 
                     if ann['image_id'] in selected_image_ids]
    
    mini_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data['categories'],
        'images': selected_images,
        'annotations': selected_anns
    }
    
    with open(output_file, 'w') as f:
        json.dump(mini_data, f)
    
    print(f'  Created: {len(selected_images)} images, {len(selected_anns)} annotations')


def download_coco_seg(output_dir: str = 'coco_seg', download_images: bool = True, mini: bool = False):
    """
    Download and prepare COCO Instance Segmentation dataset.
    
    Args:
        output_dir: Output directory
        download_images: Whether to download images
        mini: Create mini dataset for testing
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print('=' * 60)
    print('COCO Instance Segmentation Dataset Downloader')
    print('=' * 60)
    print(f'\nOutput directory: {output_path.absolute()}')
    print(f'\nDataset sizes:')
    for name, size in FILE_SIZES.items():
        print(f'  - {name}: ~{size}')
    print()
    
    # Download annotations
    print('\n[1/3] Downloading annotations...')
    ann_zip = output_path / 'annotations_trainval2017.zip'
    download_file(COCO_URLS['annotations'], ann_zip, 'annotations (~252MB)')
    
    # Extract annotations
    if not (output_path / 'annotations').exists():
        extract_zip(ann_zip, output_path)
    
    # Create mini dataset if requested
    if mini:
        print('\n[2/3] Creating mini datasets...')
        train_ann = output_path / 'annotations' / 'instances_train2017.json'
        train_mini = output_path / 'annotations' / 'instances_train2017_mini.json'
        if train_ann.exists() and not train_mini.exists():
            create_mini_dataset(train_ann, train_mini, max_images=5000)
        
        val_ann = output_path / 'annotations' / 'instances_val2017.json'
        val_mini = output_path / 'annotations' / 'instances_val2017_mini.json'
        if val_ann.exists() and not val_mini.exists():
            create_mini_dataset(val_ann, val_mini, max_images=1000)
    else:
        print('\n[2/3] Skipping mini dataset creation')
    
    if download_images:
        # Download training images
        print('\n[3/3] Downloading images...')
        train_zip = output_path / 'train2017.zip'
        download_file(COCO_URLS['train2017'], train_zip, 'train2017 (~18GB)')
        
        if not (output_path / 'train2017').exists():
            extract_zip(train_zip, output_path)
        
        # Download validation images
        val_zip = output_path / 'val2017.zip'
        download_file(COCO_URLS['val2017'], val_zip, 'val2017 (~1GB)')
        
        if not (output_path / 'val2017').exists():
            extract_zip(val_zip, output_path)
    else:
        print('\n[3/3] Skipping image download')
    
    # Print summary
    print('\n' + '=' * 60)
    print('Download Complete!')
    print('=' * 60)
    print(f'\nDataset structure:')
    print(f'  {output_path}/')
    print(f'  ├── train2017/           # 118k training images')
    print(f'  ├── val2017/             # 5k validation images')
    print(f'  └── annotations/')
    print(f'      ├── instances_train2017.json      # Full training (860k instances)')
    print(f'      ├── instances_val2017.json        # Full validation')
    if mini:
        print(f'      ├── instances_train2017_mini.json # Mini training (5k images)')
        print(f'      └── instances_val2017_mini.json   # Mini validation (1k images)')
    
    print(f'\nTo train YOLOv11-Seg:')
    ann_suffix = '_mini' if mini else ''
    print(f'  python train.py \\')
    print(f'      --task segment \\')
    print(f'      --model m \\')
    print(f'      --data {output_path}/train2017 \\')
    print(f'      --ann {output_path}/annotations/instances_train2017{ann_suffix}.json \\')
    print(f'      --batch 16 \\')
    print(f'      --epochs 100')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download COCO Instance Segmentation Dataset')
    parser.add_argument('--output', type=str, default='coco_seg', help='Output directory')
    parser.add_argument('--no-images', action='store_true', help='Skip image download')
    parser.add_argument('--mini', action='store_true', help='Create mini dataset for testing')
    args = parser.parse_args()
    
    download_coco_seg(args.output, download_images=not args.no_images, mini=args.mini)


if __name__ == '__main__':
    main()
