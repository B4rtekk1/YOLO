"""
COCO Dataset Downloader
Pobiera określoną liczbę obrazów i annotacji z COCO 2017
"""

import os
import json
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# COCO URLs
COCO_BASE_URL = "http://images.cocodataset.org"
ANNOTATIONS_URL = f"{COCO_BASE_URL}/annotations/annotations_trainval2017.zip"

# Mapowanie tasków na pliki annotacji
ANNOTATION_FILES = {
    'detect': 'instances_{split}2017.json',
    'segment': 'instances_{split}2017.json',
    'pose': 'person_keypoints_{split}2017.json'
}


def download_file(url: str, dest: str) -> bool:
    """Pobierz pojedynczy plik."""
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_image(args):
    """Pobierz jeden obraz (dla ThreadPoolExecutor)."""
    img_info, dest_dir, base_url = args
    filename = img_info['file_name']
    url = f"{base_url}/{filename}"
    dest_path = dest_dir / filename
    
    if dest_path.exists():
        return True
    
    return download_file(url, str(dest_path))


def download_coco_subset(
    output_dir: str,
    num_images: int = 100,
    split: str = 'val',
    task: str = 'detect',
    num_workers: int = 8
):
    """
    Pobierz podzbiór COCO.
    
    Args:
        output_dir: Katalog docelowy
        num_images: Liczba obrazów do pobrania
        split: 'train' lub 'val'
        task: 'detect', 'segment', lub 'pose'
        num_workers: Liczba wątków do pobierania
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / f"{split}2017"
    annotations_dir = output_dir / "annotations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Nazwa pliku annotacji
    ann_filename = ANNOTATION_FILES[task].format(split=split)
    ann_path = annotations_dir / ann_filename
    
    # Musimy najpierw pobrać pełne annotacje (lub użyć API)
    print(f"\n📥 Pobieranie annotacji COCO {split}2017 dla {task}...")
    
    # Użyj COCO API do pobrania annotacji
    try:
        from pycocotools.coco import COCO
        
        # URL do pojedynczego pliku annotacji
        ann_url = f"http://images.cocodataset.org/annotations/{ann_filename.replace(f'{split}2017', f'{split}2017')}"
        
        # Alternatywnie - bezpośredni URL
        if not ann_path.exists():
            print(f"Pobieranie: {ann_filename}")
            # Pełne annotacje są w zipie, więc pobierzemy mini wersję
            ann_url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            
            import zipfile
            import tempfile
            
            print("Pobieranie archiwum annotacji (~250MB)...")
            zip_path = output_dir / "annotations.zip"
            
            # Pobieranie z paskiem postępu
            with tqdm(unit='B', unit_scale=True, desc="annotations") as pbar:
                def reporthook(count, block_size, total_size):
                    if pbar.total is None and total_size > 0:
                        pbar.total = total_size
                    pbar.update(block_size)
                
                urllib.request.urlretrieve(ann_url, str(zip_path), reporthook)
            
            print("Rozpakowywanie...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(output_dir)
            
            zip_path.unlink()
            print("✅ Annotacje pobrane!")
    
    except ImportError:
        print("⚠️ pycocotools nie zainstalowane. Instaluję...")
        os.system("pip install pycocotools")
        from pycocotools.coco import COCO
    
    # Wczytaj annotacje
    print(f"\n📂 Wczytywanie annotacji z {ann_path}...")
    coco = COCO(str(ann_path))
    
    # Dla pose - wybierz tylko obrazy z osobami
    if task == 'pose':
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
    else:
        img_ids = list(coco.imgs.keys())
    
    # Ogranicz do żądanej liczby
    img_ids = img_ids[:num_images]
    print(f"📸 Wybrano {len(img_ids)} obrazów do pobrania")
    
    # Pobierz informacje o obrazach
    images_info = coco.loadImgs(img_ids)
    
    # Pobierz obrazy równolegle
    print(f"\n📥 Pobieranie {len(images_info)} obrazów...")
    base_url = f"{COCO_BASE_URL}/{split}2017"
    
    download_args = [(img, images_dir, base_url) for img in images_info]
    
    successful = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_image, arg): arg for arg in download_args}
        
        with tqdm(total=len(futures), desc="Pobieranie") as pbar:
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                pbar.update(1)
    
    print(f"✅ Pobrano {successful}/{len(images_info)} obrazów")
    
    # Stwórz plik annotacji tylko dla pobranych obrazów
    subset_ann_path = annotations_dir / f"{task}_{split}2017_subset_{num_images}.json"
    
    print(f"\n📝 Tworzenie pliku annotacji dla podzbioru...")
    
    # Zbierz annotacje dla wybranych obrazów
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    annotations = coco.loadAnns(ann_ids)
    
    # Stwórz nowy plik COCO
    subset_coco = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'images': images_info,
        'annotations': annotations,
        'categories': coco.dataset.get('categories', [])
    }
    
    with open(subset_ann_path, 'w') as f:
        json.dump(subset_coco, f)
    
    print(f"✅ Annotacje zapisane: {subset_ann_path}")
    
    # Podsumowanie
    print(f"\n{'='*50}")
    print(f"📊 PODSUMOWANIE")
    print(f"{'='*50}")
    print(f"Obrazy: {images_dir}")
    print(f"Annotacje: {subset_ann_path}")
    print(f"Liczba obrazów: {successful}")
    print(f"Liczba annotacji: {len(annotations)}")
    print(f"\n🚀 Gotowe do trenowania!")
    print(f"\npython train.py --task {task} --model s \\")
    print(f"    --data \"{images_dir}\" \\")
    print(f"    --ann \"{subset_ann_path}\" \\")
    print(f"    --epochs 50 --batch 8")
    
    return str(images_dir), str(subset_ann_path)


def main():
    parser = argparse.ArgumentParser(description='Pobierz podzbiór COCO dataset')
    parser.add_argument('--output', type=str, default='coco', help='Katalog wyjściowy')
    parser.add_argument('--num-images', type=int, default=100, help='Liczba obrazów do pobrania')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Split danych')
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'segment', 'pose'], help='Typ zadania')
    parser.add_argument('--workers', type=int, default=8, help='Liczba wątków')
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
