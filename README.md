# YOLOv8 From Scratch

Implementacja YOLOv8 od zera w PyTorch, wspierająca:

- **Detection** - wykrywanie obiektów (80 klas COCO)
- **Segmentation** - segmentacja instancyjna
- **Pose Estimation** - wykrywanie szkieletu człowieka (17 keypoints COCO)

## 📁 Struktura projektu

```
YOLO/
├── yolov8/
│   ├── __init__.py          # Package exports
│   ├── blocks.py             # Basic blocks: Conv, C2f, SPPF, DFL
│   ├── backbone.py           # CSPDarknet backbone
│   ├── neck.py               # PANet/FPN neck
│   ├── head.py               # Detection/Segmentation/Pose heads
│   ├── model.py              # Main YOLOv8 model
│   ├── losses/               # Loss functions
│   │   ├── box_loss.py       # CIoU, DFL losses
│   │   ├── cls_loss.py       # Classification losses
│   │   ├── seg_loss.py       # Segmentation losses
│   │   ├── pose_loss.py      # OKS, keypoint losses
│   │   └── combined_loss.py  # Task-Aligned Assigner
│   ├── data/                 # Data loading
│   │   ├── dataset.py        # COCO/YOLO format loaders
│   │   └── augmentations.py  # Mosaic, MixUp, etc.
│   └── utils/                # Utilities
│       ├── nms.py            # Non-Maximum Suppression
│       ├── metrics.py        # mAP, OKS metrics
│       └── visualization.py  # Drawing functions
├── train.py                  # Training script
├── inference.py              # Inference script
└── requirements.txt          # Dependencies
```

## 🚀 Instalacja

```bash
pip install -r requirements.txt
```

## 🎯 Użycie

### Tworzenie modelu

```python
from yolov8 import YOLOv8

# Detection
model = YOLOv8(num_classes=80, task='detect', model_size='s')

# Segmentation
model = YOLOv8(num_classes=80, task='segment', model_size='s')

# Pose Estimation
model = YOLOv8(num_classes=1, task='pose', model_size='s')

# Forward pass
import torch
x = torch.randn(1, 3, 640, 640)
outputs = model(x)
model.info()
```

### Trenowanie

```bash
# Detection na COCO
python train.py --task detect --model s --data /path/to/coco/images \
    --ann /path/to/annotations.json --epochs 100 --batch 16

# Segmentation
python train.py --task segment --model s --data /path/to/data \
    --ann annotations_seg.json --epochs 100

# Pose Estimation
python train.py --task pose --model s --data /path/to/data \
    --ann annotations_pose.json --epochs 100
```

### Inference

```bash
# Na obrazie
python inference.py --weights runs/train/best.pt --source image.jpg --task detect

# Na video
python inference.py --weights runs/train/best.pt --source video.mp4 --task detect

# Webcam
python inference.py --weights runs/train/best.pt --source 0 --task detect
```

## 📐 Architektura YOLOv8

```
Input (640x640x3)
    │
    ▼
┌─────────────────────────────────────────────────┐
│                  BACKBONE (CSPDarknet)          │
├─────────────────────────────────────────────────┤
│  Stem → Stage1 → Stage2 → Stage3 → Stage4       │
│  Conv    C2f      C2f      C2f     C2f+SPPF    │
│                     │        │         │        │
│                    P3       P4        P5        │
│                 (80x80)  (40x40)   (20x20)      │
└─────────────────────────────────────────────────┘
                     │        │         │
                     ▼        ▼         ▼
┌─────────────────────────────────────────────────┐
│                  NECK (PANet)                   │
├─────────────────────────────────────────────────┤
│  FPN (Top-Down):    P5 → N4 → N3               │
│  PAN (Bottom-Up):   N3 → N4 → N5               │
└─────────────────────────────────────────────────┘
                     │        │         │
                     ▼        ▼         ▼
┌─────────────────────────────────────────────────┐
│              HEAD (Decoupled)                   │
├─────────────────────────────────────────────────┤
│  Classification branch: Conv → Conv → Pred     │
│  Regression branch:     Conv → Conv → Pred     │
│  + Mask coefficients (segment)                 │
│  + Keypoints (pose)                            │
└─────────────────────────────────────────────────┘
```

## 📊 Rozmiary modeli

| Model    | Params  | FLOPs   | Input |
|----------|---------|---------|-------|
| YOLOv8n  | ~3.2M   | ~8.7G   | 640   |
| YOLOv8s  | ~11.2M  | ~28.6G  | 640   |
| YOLOv8m  | ~25.9M  | ~78.9G  | 640   |
| YOLOv8l  | ~43.7M  | ~165.2G | 640   |
| YOLOv8x  | ~68.2M  | ~257.8G | 640   |

## 🔧 Kluczowe komponenty

### Bloki budulcowe

- **Conv**: Conv2D + BatchNorm + SiLU
- **C2f**: Cross Stage Partial with 2 convolutions (szybsza wersja C3)
- **SPPF**: Spatial Pyramid Pooling Fast
- **DFL**: Distribution Focal Loss dla regresji box

### Loss Functions

- **CIoU Loss**: Complete IoU dla regresji bounding box
- **DFL**: Distribution Focal Loss
- **BCE/Focal/Varifocal**: Klasyfikacja
- **OKS Loss**: Object Keypoint Similarity dla pose

### Augmentacje

- **Mosaic**: Łączy 4 obrazy w jeden
- **MixUp**: Mieszanie dwóch obrazów
- **RandomHSV**: Modyfikacja barw
- **RandomFlip**: Odbicie lustrzane

## 📚 COCO Keypoints (17 punktów)

```
Keypoint ID | Nazwa         | Połączenie
------------|---------------|------------
0           | nose          | 
1           | left_eye      | 0-1
2           | right_eye     | 0-2
3           | left_ear      | 1-3
4           | right_ear     | 2-4
5           | left_shoulder | 5-6, 5-7, 5-11
6           | right_shoulder| 6-8, 6-12
7           | left_elbow    | 7-9
8           | right_elbow   | 8-10
9           | left_wrist    | 
10          | right_wrist   | 
11          | left_hip      | 11-12, 11-13
12          | right_hip     | 12-14
13          | left_knee     | 13-15
14          | right_knee    | 14-16
15          | left_ankle    | 
16          | right_ankle   | 
```

## 📝 Licencja

MIT License
