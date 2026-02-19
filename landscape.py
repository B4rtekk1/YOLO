import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import copy

from yolov11.model import YOLOv11
from yolov11.losses import YOLOv11Loss
from yolov11.data import create_dataloader

def get_params(model):
    """Get all model parameters as a single flat vector."""
    return torch.cat([p.view(-1) for p in model.parameters()])

def set_params(model, theta):
    """Set model parameters from a flat vector."""
    idx = 0
    for p in model.parameters():
        length = p.numel()
        p.data.copy_(theta[idx : idx + length].view(p.shape))
        idx += length

def normalize_direction(direction, params):
    """Normalize direction relative to weight scale (filter-wise normalization)."""
    direction = direction.to(params.device)
    # Simple vector normalization for optimization
    return direction * (params.norm() / direction.norm())

def plot_landscape():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = 'saves/last.pt'
    data_path = 'coco_mini'
    grid_res = 100  # Grid resolution (e.g., 50x50 = 2500 loss calculations)
    steps = 0.5    # Perturbation range (-0.5 to 0.5)

    # 2. Load model and checkpoint
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Extract configuration from checkpoint
    cfg = ckpt.get('config', {})
    model_size = cfg.get('model_size', 's')
    num_classes = cfg.get('num_classes', 80)
    task = cfg.get('task', 'detect')

    model = YOLOv11(num_classes=num_classes, task=task, model_size=model_size).to(device)
    
    # Safe weight loading
    sd = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    
    model.load_state_dict(sd)
    model.eval()

    criterion = YOLOv11Loss(task='detect', num_classes=80)

    # 3. Prepare data (small batch for speed)
    dataloader = create_dataloader(
        root=data_path,
        ann_file=str(Path(data_path) / "annotations/instances_val2017.json"),
        batch_size=4,
        img_size=640,
        augment=False
    )
    images, targets = next(iter(dataloader))
    images = images.to(device)
    for k in targets: targets[k] = targets[k].to(device)

    # 4. Generate random directions (d1 and d2)
    current_params = get_params(model).detach()
    d1 = torch.randn_like(current_params)
    d2 = torch.randn_like(current_params)

    # Gram-Schmidt Orthogonalization
    d1 = d1 / d1.norm()
    d2 = d2 - torch.dot(d1, d2) * d1
    d2 = d2 / d2.norm()

    # Filter-wise normalization
    d1 = normalize_direction(d1, current_params)
    d2 = normalize_direction(d2, current_params)

    # 5. Calculate baseline point (center of the plot)
    with torch.no_grad():
        outputs = model(images)
        base_loss, base_loss_items = criterion(outputs, targets)
        z_base = base_loss.item()
        
    print(f"\n[BASELINE DATA]")
    print(f"Total Loss: {z_base:.4f}")
    print(f"Box Loss:   {base_loss_items.get('loss_box', 0):.4f}")
    print(f"Cls Loss:   {base_loss_items.get('loss_cls', 0):.4f}")
    print("-" * 20)

    # 6. Calculate loss grid
    x = np.linspace(-steps, steps, grid_res)
    y = np.linspace(-steps, steps, grid_res)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((grid_res, grid_res))

    print(f"Generating Total Loss Landscape on {grid_res}x{grid_res} grid...")
    for i in tqdm(range(grid_res)):
        for j in range(grid_res):
            alpha = X[i, j]
            beta = Y[i, j]
            
            # New weights: W' = W + alpha*d1 + beta*d2
            new_params = current_params + alpha * d1 + beta * d2
            set_params(model, new_params)
            
            with torch.no_grad():
                outputs = model(images)
                loss, _ = criterion(outputs, targets)
                Z[i, j] = loss.item()

    # Restore baseline weights
    set_params(model, current_params)

    # 7. Plotting
    fig = plt.figure(figsize=(14, 8))
    
    # 3D View
    ax = fig.add_subplot(121, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none', alpha=0.8)
    
    # Add current position marker in 3D
    ax.scatter([0], [0], [z_base], color='red', s=100, label=f'Current Model (Loss: {z_base:.3f})', edgecolors='white', zorder=10)
    
    ax.set_title(f'YOLOv11 Total Loss Landscape (3D)\nBase Box: {base_loss_items.get("loss_box", 0):.3f} | Base Cls: {base_loss_items.get("loss_cls", 0):.3f}')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.legend()
    
    # Contour View
    ax2 = fig.add_subplot(122)
    cp = ax2.contourf(X, Y, Z, levels=30, cmap='terrain')
    fig.colorbar(cp)
    ax2.set_title(f'Total Loss Contours\n(Red X = Current Weights)')
    ax2.plot(0, 0, 'rx', markersize=12, markeredgewidth=2)

    plt.tight_layout()
    plt.savefig('loss_landscape.png', dpi=300)
    print("Generated: loss_landscape_new.png")
    # plt.show()

if __name__ == "__main__":
    plot_landscape()