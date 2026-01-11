import json
import os
from pathlib import Path

def create_kaggle_notebook():
    files_to_include = [
        ('yolov11/__init__.py', 'yolov11/__init__.py'),
        ('yolov11/blocks.py', 'yolov11/blocks.py'),
        ('yolov11/backbone.py', 'yolov11/backbone.py'),
        ('yolov11/neck.py', 'yolov11/neck.py'),
        ('yolov11/head.py', 'yolov11/head.py'),
        ('yolov11/model.py', 'yolov11/model.py'),
        ('yolov11/losses/__init__.py', 'yolov11/losses/__init__.py'),
        ('yolov11/losses/box_loss.py', 'yolov11/losses/box_loss.py'),
        ('yolov11/losses/cls_loss.py', 'yolov11/losses/cls_loss.py'),
        ('yolov11/losses/seg_loss.py', 'yolov11/losses/seg_loss.py'),
        ('yolov11/losses/pose_loss.py', 'yolov11/losses/pose_loss.py'),
        ('yolov11/losses/enhanced_loss.py', 'yolov11/losses/enhanced_loss.py'),
        ('yolov11/losses/combined_loss.py', 'yolov11/losses/combined_loss.py'),
        ('yolov11/utils/__init__.py', 'yolov11/utils/__init__.py'),
        ('yolov11/utils/nms.py', 'yolov11/utils/nms.py'),
        ('yolov11/utils/metrics.py', 'yolov11/utils/metrics.py'),
        ('yolov11/utils/training.py', 'yolov11/utils/training.py'),
        ('yolov11/utils/visualization.py', 'yolov11/utils/visualization.py'),
        ('yolov11/utils/pruning.py', 'yolov11/utils/pruning.py'),
        ('yolov11/utils/export.py', 'yolov11/utils/export.py'),
        ('yolov11/data/__init__.py', 'yolov11/data/__init__.py'),
        ('yolov11/data/augmentations.py', 'yolov11/data/augmentations.py'),
        ('yolov11/data/dataset.py', 'yolov11/data/dataset.py'),
        ('train.py', 'train.py'),
        ('download_coco.py', 'download_coco.py'),
        ('download_coco_pose.py', 'download_coco_pose.py'),
        ('download_coco_seg.py', 'download_coco_seg.py'),
        ('inference.py', 'inference.py'),
    ]

    cells = []
    
    # Title and Intro
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# YOLOv11: Full Project for Kaggle\n",
            "\n",
            "This notebook recreates the entire YOLOv11 project structure and allows for training and inference.\n",
            "\n",
            "### Setup Instructions:\n",
            "1. Run all cells to create the file structure.\n",
            "2. **Training**: All commands starting with `!` are meant to be run directly in code cells (not in a terminal).\n",
            "3. Ensure you have enabled **GPU T4 x2** in the Kaggle Accelerator settings."
        ]
    })

    # Installation
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["!pip install pycocotools -q"]
    })

    # GPU Check
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "print(f'CUDA available: {torch.cuda.is_available()}')\n",
            "print(f'GPU count: {torch.cuda.device_count()}')\n",
            "for i in range(torch.cuda.device_count()):\n",
            "    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')"
        ]
    })

    # Directory Creation
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["!mkdir -p yolov11/losses yolov11/utils yolov11/data"]
    })

    # Add File Cells
    for target_path, source_path in files_to_include:
        if not os.path.exists(source_path):
            print(f"Warning: {source_path} not found, skipping...")
            continue
            
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        cell_source = [f"%%writefile {target_path}\n"]
        cell_source.extend([line + '\n' for line in content.splitlines()])
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell_source
        })

    # Data Preparation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Preparation\n",
            "\n",
            "First, we need to download a subset of the COCO dataset. We'll use `download_coco.py` to do this automatically.\n",
            "This will create a `coco_mini` directory with 100 images."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Download COCO subset (100 images)\n",
            "!python download_coco.py --output coco_mini --num-images 100 --split val"
        ]
    })

    # Multi-GPU Training Instruction
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Distributed Training (Multi-GPU)\n",
            "\n",
            "Once data is downloaded, launch the training using `torchrun` to use both GPUs."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Launch training with 2 GPUs\n",
            "# Note: We specify the annotation file explicitly for the subset\n",
            "!torchrun --nproc_per_node=2 train.py --data coco_mini --ann coco_mini/annotations/detect_val2017_subset_100.json --epochs 50 --batch 16"
        ]
    })

    # Infernece Example
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Inference"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run inference on an image\n",
            "import os, glob\n",
            "runs = glob.glob('runs/train/20*')\n",
            "if runs:\n",
            "    latest_run = max(runs, key=os.path.getmtime)\n",
            "    weights_path = os.path.join(latest_run, 'last.pt')\n",
            "    print(f'Latest weights: {weights_path}')\n",
            "    # Example command to run inference (uncomment to run):\n",
            "    # !python inference.py --weights {weights_path} --source coco_mini/val2017/000000000139.jpg\n",
            "else:\n",
            "    print('No training runs found.')"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open('YOLOv11_Kaggle.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print("Notebook generated: YOLOv11_Kaggle.ipynb")

if __name__ == '__main__':
    create_kaggle_notebook()
