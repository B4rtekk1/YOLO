"""
COCO Dataset Downloader
Downloads a subset of COCO 2017 images and annotations
"""

import os
import json
import argparse
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


COCO_BASE_URL = "http://images.cocodataset.org"
ANNOTATIONS_URL = f"{COCO_BASE_URL}/annotations/annotations_trainval2017.zip"

ANNOTATION_FILES = {
    'detect': 'instances_{split}2017.json',
    'segment': 'instances_{split}2017.json',
    'pose': 'person_keypoints_{split}2017.json'
}


def download_file(url: str, dest: str) -> bool:
    """Download a single file."""
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def verify_image(path):
    """Verify if image file is valid."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


def download_image(args):
    """Download a single image (for ThreadPoolExecutor)."""
    img_info, dest_dir, base_url = args
    filename = img_info['file_name']
    url = f"{base_url}/{filename}"
    dest_path = dest_dir / filename
    
    # Check if file exists AND is valid
    if dest_path.exists():
        if verify_image(dest_path):
            return True
        else:
            # Remove corrupted file and re-download
            dest_path.unlink()
    
    success = download_file(url, str(dest_path))
    
    # Verify after download
    if success and not verify_image(dest_path):
        dest_path.unlink()
        return False
    
    return success


def download_coco_subset(
    output_dir: str,
    num_images: int = 100,
    split: str = 'val',
    task: str = 'detect',
    num_workers: int = 8
):
    """
    Download a subset of COCO dataset.
    
    Args:
        output_dir: Target directory
        num_images: Number of images to download
        split: 'train' or 'val'
        task: 'detect', 'segment', or 'pose'
        num_workers: Number of download threads
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / f"{split}2017"
    annotations_dir = output_dir / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    ann_filename = ANNOTATION_FILES[task].format(split=split)
    ann_path = annotations_dir / ann_filename
    
    print(f"\nDownloading COCO {split}2017 annotations for {task}...")
    
    try:
        from pycocotools.coco import COCO
        
        if not ann_path.exists():
            print(f"Downloading: {ann_filename}")
            ann_url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            
            print("Downloading annotation archive (~250MB)...")
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
            print("Annotations downloaded!")
    
    except ImportError:
        print("pycocotools not installed. Installing...")
        os.system("pip install pycocotools")
        from pycocotools.coco import COCO
    
    print(f"\nLoading annotations from {ann_path}...")
    coco = COCO(str(ann_path))
    
    if task == 'pose':
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
    else:
        img_ids = list(coco.imgs.keys())
    
    img_ids = img_ids[:num_images]
    print(f"Selected {len(img_ids)} images to download")
    
    images_info = coco.loadImgs(img_ids)
    
    print(f"\nDownloading {len(images_info)} images...")
    base_url = f"{COCO_BASE_URL}/{split}2017"
    
    download_args = [(img, images_dir, base_url) for img in images_info]
    
    successful = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_image, arg): arg for arg in download_args}
        
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                pbar.update(1)
    
    print(f"Downloaded {successful}/{len(images_info)} images")
    
    subset_ann_path = annotations_dir / f"{task}_{split}2017_subset_{num_images}.json"
    
    print(f"\nCreating subset annotation file...")
    
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
    
    print(f"Annotations saved: {subset_ann_path}")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Images: {images_dir}")
    print(f"Annotations: {subset_ann_path}")
    print(f"Image count: {successful}")
    print(f"Annotation count: {len(annotations)}")
    print(f"\nReady for training!")
    print(f"\npython train.py --task {task} --model s \\")
    print(f"    --data \"{images_dir}\" \\")
    print(f"    --ann \"{subset_ann_path}\" \\")
    print(f"    --epochs 50 --batch 8")
    
    return str(images_dir), str(subset_ann_path)


def main():
    parser = argparse.ArgumentParser(description='Download COCO dataset subset')
    parser.add_argument('--output', type=str, default='coco', help='Output directory')
    parser.add_argument('--num-images', type=int, default=100, help='Number of images to download')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Data split')
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'segment', 'pose'], help='Task type')
    parser.add_argument('--workers', type=int, default=8, help='Number of threads')
    args = parser.parse_args()
    
    download_coco_subset(
        output_dir=args.output,
        num_images=args.num_images,
        split=args.split,
        task=args.task,
        num_workers=args.workers
    )


if __name__ == '__main__':
    main()
