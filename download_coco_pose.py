"""
COCO Keypoints (Pose) Dataset Downloader
Downloads and prepares COCO keypoints dataset for pose estimation training
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

# COCO Keypoints URLs
COCO_URLS = {
    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
    'val2017': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

# File sizes for progress display
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


def filter_person_annotations(ann_file: Path, output_file: Path):
    """
    Filter annotations to keep only person instances with keypoints.
    This significantly reduces memory usage during training.
    """
    print(f'  Filtering person annotations from {ann_file.name}...')
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Keep only person category (id=1)
    person_cat = [cat for cat in data['categories'] if cat['name'] == 'person']
    
    # Filter annotations: keep only person with keypoints
    person_anns = []
    valid_image_ids = set()
    
    for ann in tqdm(data['annotations'], desc='Filtering'):
        if ann['category_id'] == 1 and 'keypoints' in ann:
            # Check if at least some keypoints are visible
            kpts = ann['keypoints']
            num_visible = sum(1 for i in range(2, len(kpts), 3) if kpts[i] > 0)
            if num_visible >= 5:  # At least 5 visible keypoints
                person_anns.append(ann)
                valid_image_ids.add(ann['image_id'])
    
    # Filter images to keep only those with valid annotations
    filtered_images = [img for img in data['images'] if img['id'] in valid_image_ids]
    
    # Create filtered dataset
    filtered_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': person_cat,
        'images': filtered_images,
        'annotations': person_anns
    }
    
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f)
    
    print(f'  Filtered: {len(person_anns)} person annotations in {len(filtered_images)} images')


def download_coco_pose(output_dir: str = 'coco_pose', download_images: bool = True):
    """
    Download and prepare COCO Keypoints dataset.
    
    Args:
        output_dir: Output directory
        download_images: Whether to download images (set False if you already have them)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print('=' * 60)
    print('COCO Keypoints (Pose) Dataset Downloader')
    print('=' * 60)
    print(f'\nOutput directory: {output_path.absolute()}')
    print(f'\nDataset sizes:')
    for name, size in FILE_SIZES.items():
        print(f'  - {name}: ~{size}')
    print()
    
    # Download annotations first (always needed)
    print('\n[1/4] Downloading annotations...')
    ann_zip = output_path / 'annotations_trainval2017.zip'
    download_file(COCO_URLS['annotations'], ann_zip, 'annotations (~252MB)')
    
    # Extract annotations
    if not (output_path / 'annotations').exists():
        extract_zip(ann_zip, output_path)
    
    # Filter person keypoint annotations
    print('\n[2/4] Preparing keypoint annotations...')
    
    # Training annotations
    train_ann = output_path / 'annotations' / 'person_keypoints_train2017.json'
    train_filtered = output_path / 'annotations' / 'person_keypoints_train2017_filtered.json'
    if train_ann.exists() and not train_filtered.exists():
        filter_person_annotations(train_ann, train_filtered)
    
    # Validation annotations
    val_ann = output_path / 'annotations' / 'person_keypoints_val2017.json'
    val_filtered = output_path / 'annotations' / 'person_keypoints_val2017_filtered.json'
    if val_ann.exists() and not val_filtered.exists():
        filter_person_annotations(val_ann, val_filtered)
    
    if download_images:
        # Download training images
        print('\n[3/4] Downloading training images...')
        train_zip = output_path / 'train2017.zip'
        download_file(COCO_URLS['train2017'], train_zip, 'train2017 (~18GB)')
        
        if not (output_path / 'train2017').exists():
            extract_zip(train_zip, output_path)
        
        # Download validation images
        print('\n[4/4] Downloading validation images...')
        val_zip = output_path / 'val2017.zip'
        download_file(COCO_URLS['val2017'], val_zip, 'val2017 (~1GB)')
        
        if not (output_path / 'val2017').exists():
            extract_zip(val_zip, output_path)
    else:
        print('\n[3/4] Skipping image download (--no-images flag)')
        print('[4/4] Skipping image download')
    
    # Print summary
    print('\n' + '=' * 60)
    print('Download Complete!')
    print('=' * 60)
    print(f'\nDataset structure:')
    print(f'  {output_path}/')
    print(f'  ├── train2017/           # Training images')
    print(f'  ├── val2017/             # Validation images')
    print(f'  └── annotations/')
    print(f'      ├── person_keypoints_train2017.json')
    print(f'      ├── person_keypoints_train2017_filtered.json  # Person-only')
    print(f'      ├── person_keypoints_val2017.json')
    print(f'      └── person_keypoints_val2017_filtered.json    # Person-only')
    
    print(f'\nTo train YOLOv11-Pose:')
    print(f'  python train.py \\')
    print(f'      --task pose \\')
    print(f'      --model m \\')
    print(f'      --data {output_path}/train2017 \\')
    print(f'      --ann {output_path}/annotations/person_keypoints_train2017_filtered.json \\')
    print(f'      --batch 32 \\')
    print(f'      --epochs 100')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download COCO Keypoints Dataset')
    parser.add_argument('--output', type=str, default='coco_pose', help='Output directory')
    parser.add_argument('--no-images', action='store_true', help='Skip image download (annotations only)')
    args = parser.parse_args()
    
    download_coco_pose(args.output, download_images=not args.no_images)


if __name__ == '__main__':
    main()
