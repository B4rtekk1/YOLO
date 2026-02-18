"""
Simple SORT-like Tracker for YOLOv11
Uses IoU-based matching and Kalman filter for smoothing
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter + 1e-6)


def iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Calculate IoU matrix between two sets of boxes."""
    n = len(boxes1)
    m = len(boxes2)
    iou_matrix = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = iou(boxes1[i], boxes2[j])
    
    return iou_matrix


class Track:
    """Single object track."""
    
    _next_id = 1
    
    def __init__(self, box: np.ndarray, score: float, label: int):
        self.id = Track._next_id
        Track._next_id += 1
        
        self.box = box.copy()
        self.score = score
        self.label = label
        
        # Smoothing with exponential moving average
        self.smoothed_box = box.copy()
        self.alpha = 0.7  # Smoothing factor (higher = less smoothing)
        
        # Track state
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        
        # History for velocity estimation
        self.history = deque(maxlen=10)
        self.history.append(box.copy())
    
    def update(self, box: np.ndarray, score: float, label: int):
        """Update track with new detection."""
        self.box = box.copy()
        self.score = score
        self.label = label
        
        # Smooth the box
        self.smoothed_box = self.alpha * box + (1 - self.alpha) * self.smoothed_box
        
        self.hits += 1
        self.time_since_update = 0
        self.history.append(box.copy())
    
    def predict(self):
        """Predict next position - simplified, no velocity drift."""
        self.age += 1
        self.time_since_update += 1
        # Removed velocity prediction - it was causing box drift
    
    def get_state(self) -> Tuple[np.ndarray, float, int, int]:
        """Get current state: (smoothed_box, score, label, track_id)."""
        return self.smoothed_box, self.score, self.label, self.id


class SimpleTracker:
    """
    Simple IoU-based tracker with smoothing.
    
    Usage:
        tracker = SimpleTracker()
        for frame in video:
            detections = model.predict(frame)
            tracked = tracker.update(detections['boxes'], detections['scores'], detections['labels'])
            # tracked contains: boxes, scores, labels, track_ids
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3
    ):
        """
        Args:
            iou_threshold: Minimum IoU for matching
            max_age: Maximum frames to keep track without update
            min_hits: Minimum hits before track is confirmed
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
    
    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Update tracker with new detections.
        
        Args:
            boxes: Detection boxes (N, 4) in xyxy format
            scores: Detection scores (N,)
            labels: Detection labels (N,)
        
        Returns:
            Dict with smoothed boxes, scores, labels, and track_ids
        """
        # Predict new positions for existing tracks
        for track in self.tracks:
            track.predict()
        
        if len(boxes) == 0:
            # No detections - just return existing tracks
            self._remove_dead_tracks()
            return self._get_output()
        
        if len(self.tracks) == 0:
            # No existing tracks - create new ones
            for box, score, label in zip(boxes, scores, labels):
                self.tracks.append(Track(box, score, label))
            return self._get_output()
        
        # Calculate IoU matrix
        track_boxes = np.array([t.box for t in self.tracks])
        iou_matrix = iou_batch(track_boxes, boxes)
        
        # Greedy matching (simple but effective)
        matched_tracks = set()
        matched_dets = set()
        
        # Sort by IoU descending
        rows, cols = np.unravel_index(np.argsort(iou_matrix.ravel())[::-1], iou_matrix.shape)
        
        for r, c in zip(rows, cols):
            if r in matched_tracks or c in matched_dets:
                continue
            if iou_matrix[r, c] < self.iou_threshold:
                break
            
            # Match found
            self.tracks[r].update(boxes[c], scores[c], labels[c])
            matched_tracks.add(r)
            matched_dets.add(c)
        
        # Create new tracks for unmatched detections
        for d in range(len(boxes)):
            if d not in matched_dets:
                self.tracks.append(Track(boxes[d], scores[d], labels[d]))
        
        # Remove dead tracks
        self._remove_dead_tracks()
        
        return self._get_output()
    
    def _remove_dead_tracks(self):
        """Remove tracks that haven't been updated for too long."""
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update < self.max_age
        ]
    
    def _get_output(self) -> Dict[str, np.ndarray]:
        """Get output for confirmed tracks."""
        confirmed = [t for t in self.tracks if t.hits >= self.min_hits]
        
        if len(confirmed) == 0:
            return {
                'boxes': np.zeros((0, 4)),
                'scores': np.zeros(0),
                'labels': np.zeros(0, dtype=int),
                'track_ids': np.zeros(0, dtype=int)
            }
        
        boxes = []
        scores = []
        labels = []
        track_ids = []
        
        for track in confirmed:
            box, score, label, tid = track.get_state()
            boxes.append(box)
            scores.append(score)
            labels.append(label)
            track_ids.append(tid)
        
        return {
            'boxes': np.array(boxes),
            'scores': np.array(scores),
            'labels': np.array(labels, dtype=int),
            'track_ids': np.array(track_ids, dtype=int)
        }
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        Track._next_id = 1
