"""
yolov11.utils.training – Training Helper Utilities
===================================================

Provides auxiliary classes that improve training stability and final accuracy:

* :class:`ModelEMA` – Exponential Moving Average of model weights.  The EMA
  model is used for evaluation and saved as ``best.pt``; it typically yields
  0.1-0.5 mAP improvement over the raw model.
* :class:`WarmupScheduler` – Linear LR warm-up for the first N epochs to
  prevent early divergence with large batch sizes.
* :class:`CosineAnnealingWarmRestarts` – Cosine LR schedule with periodic
  restarts (SGDR) for escaping local minima.
* :class:`ProgressiveResizing` – Gradually increases image resolution during
  training for faster early convergence and better final accuracy.
* :class:`EarlyStopping` – Halts training when validation metric stagnates.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import copy
import math


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model weights.
    
    Maintains a shadow copy of model weights that is updated using
    exponential moving average, which typically leads to better
    generalization and more stable training.
    
    Usage:
        ema = ModelEMA(model)
        for epoch in range(epochs):
            train(model)
            ema.update(model)
        # Use ema.ema for evaluation
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: float = 2000,
        updates: int = 0
    ):
        """
        Args:
            model: Model to track
            decay: EMA decay factor (higher = slower updates)
            tau: Time constant for decay warmup
            updates: Initial update count
        """
        self.ema: nn.Module = copy.deepcopy(model).eval()
        self.updates = updates
        self.decay = decay
        self.tau = tau
        
        # Disable gradients for EMA model
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    def update(self, model: nn.Module):
        """Update the EMA shadow weights from the current model parameters.

        Uses a warmup-adjusted decay so that early updates have a smaller
        decay factor (i.e. the EMA tracks the model more closely at the start
        of training when the weights are still changing rapidly):

        .. code-block:: text

            d = decay * (1 - exp(-updates / tau))
            ema_weight = d * ema_weight + (1 - d) * model_weight

        Args:
            model: The live training model (may be wrapped in DDP/DataParallel).
        """
        with torch.no_grad():
            self.updates += 1
            
            # Decay with warmup
            d = self.decay * (1 - math.exp(-self.updates / self.tau))
            
            # Use unwrapped model state dict to avoid DDP prefixes
            model_to_update = model.module if hasattr(model, 'module') else model
            msd = model_to_update.state_dict()
            
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
    
    def update_attr(self, model: nn.Module, include=(), exclude=('process_group', 'reducer')):
        """Update EMA attributes."""
        for k, v in model.__dict__.items():
            if (len(include) > 0 and k not in include) or k.startswith('_') or k in exclude:
                continue
            setattr(self.ema, k, v)


class WarmupScheduler:
    """
    Learning rate warmup scheduler.
    
    Gradually increases learning rate from 0 to base_lr during warmup epochs.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 3,
        warmup_bias_lr: float = 0.1,
        warmup_momentum: float = 0.8
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum
        
        # Store initial values
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.base_momentums = [pg.get('momentum', 0.9) for pg in optimizer.param_groups]
    
    def step(self, epoch: float, batch_idx: int, num_batches: int):
        """Update learning rate based on warmup progress."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            progress = (epoch * num_batches + batch_idx) / (self.warmup_epochs * num_batches)
            
            for i, pg in enumerate(self.optimizer.param_groups):
                pg['lr'] = self.base_lrs[i] * progress
                if 'momentum' in pg:
                    pg['momentum'] = self.warmup_momentum + (self.base_momentums[i] - self.warmup_momentum) * progress


class CosineAnnealingWarmRestarts:
    """
    Cosine annealing with warm restarts.
    
    Learning rate follows cosine curve with periodic restarts.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 1e-6
    ):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.T_cur = 0
        self.T_i = T_0
    
    def step(self, epoch: int):
        """Update learning rate."""
        if epoch >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
        
        # Cosine annealing
        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = self.eta_min + (self.base_lrs[i] - self.eta_min) * \
                       (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        
        self.T_cur += 1


class ProgressiveResizing:
    """
    Progressive resizing strategy for training.
    
    Starts with smaller images and gradually increases size,
    which can speed up training and improve generalization.
    """
    
    def __init__(
        self,
        start_size: int = 320,
        end_size: int = 640,
        epochs: int = 100,
        milestones: Optional[list] = None
    ):
        self.start_size = start_size
        self.end_size = end_size
        self.epochs = epochs
        self.milestones = milestones or [0.3, 0.6, 0.8]  # Fraction of epochs
    
    def get_size(self, epoch: int) -> int:
        """Get image size for current epoch."""
        progress = epoch / self.epochs
        
        # Find current milestone
        sizes = [self.start_size]
        step = (self.end_size - self.start_size) // len(self.milestones)
        
        for i, m in enumerate(self.milestones):
            if progress >= m:
                sizes.append(self.start_size + step * (i + 1))
        
        return min(sizes[-1], self.end_size)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_score: Validation metric (higher is better)
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


if __name__ == "__main__":
    # Test utilities
    from yolov11.model import YOLOv11
    
    print("Testing training utilities...")
    
    # Test EMA
    model = YOLOv11(num_classes=80, model_size='n')
    ema = ModelEMA(model)
    
    # Simulate training update
    ema.update(model)
    print(f"EMA updates: {ema.updates}")
    
    # Test warmup scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    warmup = WarmupScheduler(optimizer, warmup_epochs=3)
    
    # Simulate warmup
    warmup.step(epoch=0, batch_idx=50, num_batches=100)
    print(f"LR after warmup step: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Test progressive resizing
    resizer = ProgressiveResizing(start_size=320, end_size=640, epochs=100)
    for epoch in [0, 30, 60, 90]:
        size = resizer.get_size(epoch)
        print(f"Epoch {epoch}: image size = {size}")
    
    print("All utilities OK!")
