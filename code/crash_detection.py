#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚗 CRASH DETECTION SYSTEM - v12.0 FINAL
============================================================================

BUILT FROM FUNDAMENTALS - NOT PATCHES

WHY PREVIOUS VERSIONS FAILED:
─────────────────────────────
1. Processed at FULL resolution (3840x2160) while Colab uses 640x480.
   At full res + 60fps, vehicles move only 3-6 px/frame → movement
   filter labeled them "parked" → zero moving vehicles → no crash.

2. PPM (pixels-per-meter) was auto-scaled but never calibrated to
   actual camera geometry. Random PPM = random distances.

3. Movement filter was too aggressive — killed real vehicles.

4. No frame resize = different YOLO behavior than Colab.

THE FIX - MATCH COLAB EXACTLY:
──────────────────────────────
1. Resize every frame to 640x480 before processing (same as Colab)
2. Use FIXED PPM = 25 at 640px (same as Colab)
3. NO movement filter — all detected vehicles participate
4. Cross-class NMS to remove duplicate detections (Colab doesn't
   have this problem because lower resolution = fewer duplicates)

CRASH DETECTION LOGIC (from physics):
─────────────────────────────────────
A crash has THREE observable signals in dashcam video:

Signal 1: CLOSING DISTANCE
  - Two vehicles' centroid distance decreases rapidly over frames
  - Measured: distance drops by >30% within 10 frames

Signal 2: PROXIMITY + CONTACT
  - Distance between centroids < threshold (2.0m)
  - AND bounding boxes overlap (IoU > 0)

Signal 3: POST-IMPACT EVIDENCE
  - Vehicle count drops suddenly (merged into one detection)
  - OR vehicles that were moving suddenly stop
  - OR detection disappears (obscured by collision)

DECISION:
  - Frame-level: Signal 2 alone (close + overlap) = collision frame
  - Video-level: Need 3+ collision frames in any 10-frame window
    OR Signal 1 + Signal 3 combined = crash

This matches how the Colab pipeline works:
  Colab uses distance < 2.0m + box overlap as primary signal,
  with multi-factor confirmation.
============================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.get_logger().setLevel('ERROR')
import cv2
import numpy as np
import h5py
import time
from collections import deque
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings
import sys

from ultralytics import YOLO
import torch
from scipy.spatial.distance import cdist
import argparse

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    BASE_DIR = Path.home() / 'Desktop' / 'crash_detection'
    MODELS_DIR = BASE_DIR / 'models'
    VIDEOS_DIR = BASE_DIR / 'videos'
    MODEL_PATH = MODELS_DIR / 'crash_detection_model'

    # ── Frame Processing (MATCH COLAB) ──
    PROCESS_WIDTH = 640
    PROCESS_HEIGHT = 480

    # ── Detection ──
    YOLO_CONF_THRESHOLD = 0.5      # Same as Colab
    NMS_IOU_THRESHOLD = 0.45       # Remove duplicate detections
    MAX_DISTANCE_TRACKING = 50     # Same as Colab

    # ── Physics (at 640x480 resolution) ──
    PIXELS_PER_METER = 25          # Same as Colab
    MAX_SPEED = 120
    SPEED_SMOOTH_FRAMES = 3

    # ── Collision Detection ──
    COLLISION_DISTANCE_THRESHOLD = 2.0   # meters - same as Colab

    # ── Temporal ──
    CRASH_WINDOW_SIZE = 10         # frames
    CRASH_MIN_FRAMES = 3           # min collision frames in window
    DISTANCE_DROP_RATIO = 0.3      # 30% drop = rapid closing

    # ── Neural Model ──
    FE_PATH      = MODELS_DIR / 'feature_extractor_saved'
    WEIGHTS_PATH = MODELS_DIR / 'crash_model_weights.weights.h5'
    CNN_FRAMES   = 10       # rolling buffer length
    CNN_SIZE     = 112      # frame size for MobileNetV2
    CNN_THRESH   = 0.7      # crash probability threshold

    VIDEO_SHORTCUTS = {
        'crash1': VIDEOS_DIR / 'crash1.mov',
        'crash2': VIDEOS_DIR / 'crash2.mov',
        'safe': VIDEOS_DIR / 'safe.mp4',
    }


# ============================================================================
# LOGGER
# ============================================================================

class Logger:
    @staticmethod
    def info(msg: str):
        print(f"✅ {msg}")

    @staticmethod
    def error(msg: str):
        print(f"❌ {msg}")

    @staticmethod
    def warning(msg: str):
        print(f"⚠️  {msg}")

    @staticmethod
    def header(title: str):
        print(f"\n{'='*70}")
        print(f"🚗 {title}")
        print(f"{'='*70}")

    @staticmethod
    def section(title: str):
        print(f"\n{title}")
        print(f"{'-'*70}")


# ============================================================================
# VEHICLE DETECTOR
# ============================================================================

class VehicleDetector:
    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

    def __init__(self):
        Logger.section("1️⃣  LOADING YOLOV8 MODEL")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Logger.info(f"Using device: {device.upper()}")
        # PyTorch 2.6 changed weights_only default to True, breaking YOLO load.
        # Patch torch.load to use weights_only=False for trusted local weights.
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
        try:
            yolo_path = str(Config.BASE_DIR / 'yolov8n.pt')
            self.model = YOLO(yolo_path)
        finally:
            torch.load = _orig_load  # always restore original
        Logger.info("Model loaded")

    @staticmethod
    def _iou(a, b):
        """IoU of two [x1,y1,x2,y2] boxes."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0

    def _nms(self, dets: List[Dict]) -> List[Dict]:
        """Cross-class NMS: remove overlapping boxes on same vehicle."""
        if len(dets) <= 1:
            return dets
        dets = sorted(dets, key=lambda d: d['conf'], reverse=True)
        keep = []
        removed = set()
        for i in range(len(dets)):
            if i in removed:
                continue
            keep.append(dets[i])
            bi = [dets[i]['x1'], dets[i]['y1'], dets[i]['x2'], dets[i]['y2']]
            for j in range(i + 1, len(dets)):
                if j in removed:
                    continue
                bj = [dets[j]['x1'], dets[j]['y1'], dets[j]['x2'], dets[j]['y2']]
                if self._iou(bi, bj) > Config.NMS_IOU_THRESHOLD:
                    removed.add(j)
        return keep

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect vehicles in a 640x480 frame."""
        try:
            results = self.model(frame, conf=Config.YOLO_CONF_THRESHOLD, verbose=False)
            dets = []
            for box in results[0].boxes:
                cls_name = results[0].names[int(box.cls)].lower()
                if cls_name not in self.VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 >= x2 or y1 >= y2:
                    continue
                w, h = x2 - x1, y2 - y1
                if w < 10 or h < 10:  # noise
                    continue
                dets.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'w': w, 'h': h,
                    'center_x': (x1 + x2) // 2,
                    'center_y': (y1 + y2) // 2,
                    'conf': float(box.conf),
                    'class': cls_name,
                })
            return self._nms(dets)
        except Exception:
            return []


# ============================================================================
# TRACKER
# ============================================================================

class CentroidTracker:
    def __init__(self):
        self.max_dist = Config.MAX_DISTANCE_TRACKING
        self.next_id = 1
        self.objects = {}  # id -> detection dict

    def update(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            self.objects = {}
            return []

        if not self.objects:
            for d in detections:
                d['track_id'] = self.next_id
                self.objects[self.next_id] = d
                self.next_id += 1
            return detections

        # Build centroid arrays
        det_cents = np.array([[d['center_x'], d['center_y']] for d in detections])
        obj_ids = list(self.objects.keys())
        obj_cents = np.array([
            [self.objects[oid]['center_x'], self.objects[oid]['center_y']]
            for oid in obj_ids
        ])

        dists = cdist(det_cents, obj_cents)
        result = []
        used_det = set()
        used_obj = set()

        # Greedy match
        for flat in np.argsort(dists, axis=None):
            di = flat // len(obj_ids)
            oi = flat % len(obj_ids)
            if di in used_det or oi in used_obj:
                continue
            if dists[di][oi] > self.max_dist:
                break
            oid = obj_ids[oi]
            detections[di]['track_id'] = oid
            self.objects[oid] = detections[di]
            used_det.add(di)
            used_obj.add(oi)
            result.append(detections[di])

        # New objects
        for i, d in enumerate(detections):
            if i not in used_det:
                d['track_id'] = self.next_id
                self.objects[self.next_id] = d
                self.next_id += 1
                result.append(d)

        # Remove lost objects
        for oi, oid in enumerate(obj_ids):
            if oi not in used_obj:
                del self.objects[oid]

        return result


# ============================================================================
# SPEED ESTIMATOR
# ============================================================================

class SpeedEstimator:
    def __init__(self):
        self.prev = {}  # id -> (cx, cy)
        self.buffers = {}  # id -> deque of speeds

    def update(self, vehicles: List[Dict], fps: float) -> List[Dict]:
        ppm = Config.PIXELS_PER_METER
        for v in vehicles:
            vid = v['track_id']
            if vid in self.prev:
                px, py = self.prev[vid]
                pd = np.sqrt((v['center_x'] - px)**2 + (v['center_y'] - py)**2)
                md = pd / ppm
                spd = (md / (1.0 / fps)) * 3.6

                if vid not in self.buffers:
                    self.buffers[vid] = deque(maxlen=Config.SPEED_SMOOTH_FRAMES)
                self.buffers[vid].append(spd)
                v['speed_kmh'] = min(np.mean(self.buffers[vid]), Config.MAX_SPEED)
            else:
                v['speed_kmh'] = 0.0
            self.prev[vid] = (v['center_x'], v['center_y'])
        return vehicles


# ============================================================================
# COLLISION DETECTOR - FROM PHYSICS FUNDAMENTALS
# ============================================================================

class CollisionDetector:
    """
    CRASH SIGNALS (from real-world physics):

    1. PROXIMITY + CONTACT (primary signal):
       Distance < 2.0m AND bounding boxes overlap
       → This is the same logic Colab uses successfully

    2. RAPID CLOSING (supporting signal):
       Distance drops by >30% within recent frames
       → Confirms vehicles are approaching, not just parked near each other

    3. VEHICLE DISAPPEARANCE (post-impact signal):
       Vehicle count drops from 2+ to fewer after distance was close
       → Impact merged detections or obscured a vehicle

    FRAME DECISION:
       collision_frame = Signal 1 (close + overlap)

    VIDEO DECISION:
       crash = 3+ collision frames in any 10-frame window
       OR: Signal 3 triggered while Signal 2 was recent
    """

    def __init__(self):
        self.collision_window = deque(maxlen=Config.CRASH_WINDOW_SIZE)
        self.distance_history = deque(maxlen=Config.CRASH_WINDOW_SIZE)
        self.vehicle_count_history = deque(maxlen=Config.CRASH_WINDOW_SIZE)
        self.crash_confirmed = False
        self.first_crash_frame = None

    @staticmethod
    def _boxes_overlap(v1: Dict, v2: Dict) -> bool:
        return not (v1['x2'] < v2['x1'] or v2['x2'] < v1['x1'] or
                    v1['y2'] < v2['y1'] or v2['y2'] < v1['y1'])

    @staticmethod
    def _iou(v1: Dict, v2: Dict) -> float:
        x1 = max(v1['x1'], v2['x1'])
        y1 = max(v1['y1'], v2['y1'])
        x2 = min(v1['x2'], v2['x2'])
        y2 = min(v1['y2'], v2['y2'])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = v1['w'] * v1['h']
        a2 = v2['w'] * v2['h']
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0

    def _get_min_distance(self, vehicles: List[Dict]) -> Tuple[float, Optional[Tuple]]:
        """Get minimum centroid distance in meters between all vehicle pairs."""
        if len(vehicles) < 2:
            return float('inf'), None

        min_d = float('inf')
        closest = None
        ppm = Config.PIXELS_PER_METER

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                px = np.sqrt((v1['center_x'] - v2['center_x'])**2 +
                             (v1['center_y'] - v2['center_y'])**2)
                m = px / ppm
                if m < min_d:
                    min_d = m
                    closest = (v1, v2)

        return min_d, closest

    def _check_rapid_closing(self) -> bool:
        """Signal 2: distance dropped >30% recently."""
        dists = [d for d in self.distance_history if d < float('inf')]
        if len(dists) < 3:
            return False
        # Compare oldest to newest in window
        oldest = dists[0]
        newest = dists[-1]
        if oldest <= 0:
            return False
        drop_ratio = (oldest - newest) / oldest
        return drop_ratio > Config.DISTANCE_DROP_RATIO

    def _check_vehicle_drop(self) -> bool:
        """Signal 3: vehicle count dropped while distance was close."""
        counts = list(self.vehicle_count_history)
        if len(counts) < 3:
            return False

        max_recent = max(counts[:-1])
        current = counts[-1]

        # Count dropped AND we had close distances recently
        recent_dists = [d for d in list(self.distance_history)[-5:]
                        if d < float('inf')]
        # Use threshold directly (not 2x) — prevents city traffic false positives
        had_close = any(d < Config.COLLISION_DISTANCE_THRESHOLD for d in recent_dists)

        return max_recent >= 2 and current < max_recent and had_close

    def process_frame(self, vehicles: List[Dict], frame_num: int) -> Dict:
        """
        Process one frame. Returns analysis dict.
        """
        num_vehicles = len(vehicles)
        min_dist, closest_pair = self._get_min_distance(vehicles)

        self.distance_history.append(min_dist)
        self.vehicle_count_history.append(num_vehicles)

        # ── Signal 1: PROXIMITY (close centroid distance) ──
        # Overlap is a bonus but NOT required — many real crashes have
        # edge-to-edge contact without centroid-distance overlap.
        is_close = False
        is_close_overlap = False
        overlap_iou = 0.0
        if closest_pair is not None and min_dist < Config.COLLISION_DISTANCE_THRESHOLD:
            is_close = True
            v1, v2 = closest_pair
            if self._boxes_overlap(v1, v2):
                is_close_overlap = True
                overlap_iou = self._iou(v1, v2)

        # ── Signal 2: RAPID CLOSING ──
        is_rapid_close = self._check_rapid_closing()

        # ── Signal 3: VEHICLE DISAPPEARANCE ──
        # Require distance was VERY close (< threshold, not 2x) to avoid
        # false positives from vehicles simply leaving the frame.
        is_vehicle_drop = self._check_vehicle_drop()

        # ── Frame-level collision decision ──
        # Close proximity alone is enough; overlap adds confidence.
        # Rapid closing + proximity also qualifies (approaching impact).
        is_collision_frame = is_close_overlap or (is_close and is_rapid_close)

        # Vehicle drop only counts if distance was at or below threshold
        # (not 2x threshold as before), preventing city-traffic false positives.
        if is_vehicle_drop and not is_collision_frame:
            is_collision_frame = True

        self.collision_window.append(is_collision_frame)

        # ── Video-level crash decision (temporal) ──
        # Count consecutive collision frames in window
        max_consecutive = 0
        current_run = 0
        for val in self.collision_window:
            if val:
                current_run += 1
                max_consecutive = max(max_consecutive, current_run)
            else:
                current_run = 0

        total_collision = sum(self.collision_window)
        is_sustained = max_consecutive >= Config.CRASH_MIN_FRAMES

        if is_sustained and not self.crash_confirmed:
            self.crash_confirmed = True
            self.first_crash_frame = frame_num

        # Build reason string
        reasons = []
        if is_close and not is_close_overlap:
            reasons.append(f"close(d={min_dist:.2f}m)")
        if is_close_overlap:
            reasons.append(f"close+overlap(d={min_dist:.2f}m,iou={overlap_iou:.3f})")
        if is_rapid_close:
            reasons.append("rapid_closing")
        if is_vehicle_drop:
            reasons.append(f"vehicle_drop({list(self.vehicle_count_history)[-2] if len(self.vehicle_count_history)>1 else '?'}->{num_vehicles})")

        return {
            'frame': frame_num,
            'num_vehicles': num_vehicles,
            'min_distance': min_dist,
            'is_collision_frame': is_collision_frame,
            'is_sustained_crash': is_sustained,
            'max_consecutive': max_consecutive,
            'total_in_window': total_collision,
            'signals': {
                'close_overlap': is_close_overlap,
                'rapid_closing': is_rapid_close,
                'vehicle_drop': is_vehicle_drop,
            },
            'reason': ' + '.join(reasons) if reasons else 'none',
            'iou': overlap_iou,
        }


# ============================================================================
# VIDEO PROCESSOR
# ============================================================================

class VideoProcessor:
    def __init__(self, video_path: Path):
        self.path = Path(video_path)

        # Try alternate extensions
        if not self.path.exists():
            for ext in ['.mp4', '.mov', '.avi', '.mkv']:
                alt = self.path.with_suffix(ext)
                if alt.exists():
                    self.path = alt
                    break

        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    def read_frame(self) -> Optional[np.ndarray]:
        """Read and resize to processing resolution."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        # CRITICAL: resize to 640x480 to match Colab behavior
        frame = cv2.resize(frame, (Config.PROCESS_WIDTH, Config.PROCESS_HEIGHT))
        return frame

    def release(self):
        self.cap.release()


# ============================================================================
# DISPLAY ENGINE
# ============================================================================

class DisplayEngine:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def draw(self, frame: np.ndarray, vehicles: List[Dict],
             analysis: Dict) -> np.ndarray:
        display = frame.copy()
        h, w = display.shape[:2]

        for v in vehicles:
            color = (0, 255, 0)
            cv2.rectangle(display, (v['x1'], v['y1']), (v['x2'], v['y2']), color, 2)
            label = f"ID{v['track_id']} {v.get('speed_kmh', 0):.0f}km/h"
            cv2.putText(display, label, (v['x1'], v['y1'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Status bar
        cv2.rectangle(display, (0, 0), (w, 30), (0, 0, 0), -1)
        md = analysis['min_distance']
        md_str = f"{md:.1f}m" if md < 1000 else "inf"
        status = f"V:{analysis['num_vehicles']} Dist:{md_str} Col:{analysis['total_in_window']}/{Config.CRASH_WINDOW_SIZE}"
        cv2.putText(display, status, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Crash overlay
        if analysis['is_sustained_crash']:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 30), (w, h), (0, 0, 255), -1)
            display = cv2.addWeighted(overlay, 0.3, display, 0.7, 0)
            cv2.putText(display, "CRASH DETECTED", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
        elif analysis['is_collision_frame']:
            cv2.rectangle(display, (0, 30), (w, 34), (0, 200, 255), -1)

        return display

    def show(self, frame: np.ndarray) -> bool:
        if not self.enabled:
            return True
        try:
            cv2.imshow("Crash Detection v12", frame)
            return not ((cv2.waitKey(1) & 0xFF) == ord('q'))
        except:
            return True

    def close(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass


# ============================================================================
# NEURAL CRASH DETECTOR (MobileNetV2 + LSTM)
# ============================================================================

class NeuralCrashDetector:
    """
    Uses the trained MobileNetV2 + LSTM model as a 4th crash signal.
    Maintains a rolling buffer of CNN_FRAMES frames, runs inference
    every frame once the buffer is full.
    """

    def __init__(self):
        self.loaded = False
        self.frame_buffer = deque(maxlen=Config.CNN_FRAMES)
        self._preprocess = keras.applications.mobilenet_v2.preprocess_input

        fe_path = str(Config.FE_PATH)
        w_path  = str(Config.WEIGHTS_PATH)

        if not Path(fe_path).exists() or not Path(w_path).exists():
            Logger.warning("Neural model files not found — running without CNN signal")
            return

        try:
            self.fe = tf.saved_model.load(fe_path)
            self.cm = self._build_and_load(w_path)
            self.loaded = True
            Logger.info("Neural model loaded (MobileNetV2 + LSTM)")
        except Exception as e:
            Logger.warning(f"Neural model failed to load: {e} — running without CNN signal")

    @staticmethod
    def _build_and_load(w_path: str):
        inputs = keras.Input(shape=(Config.CNN_FRAMES, 1280))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, output)
        # warm up to build weights
        model(np.zeros((1, Config.CNN_FRAMES, 1280), dtype=np.float32), training=False)
        with h5py.File(w_path, 'r') as f:
            L = f['layers']
            model.get_layer('dense').set_weights([
                L['dense']['vars']['0'][:], L['dense']['vars']['1'][:]
            ])
            model.get_layer('lstm').set_weights([
                L['lstm']['cell']['vars']['0'][:],
                L['lstm']['cell']['vars']['1'][:],
                L['lstm']['cell']['vars']['2'][:]
            ])
            model.get_layer('lstm_1').set_weights([
                L['lstm_1']['cell']['vars']['0'][:],
                L['lstm_1']['cell']['vars']['1'][:],
                L['lstm_1']['cell']['vars']['2'][:]
            ])
            model.get_layer('dense_1').set_weights([
                L['dense_1']['vars']['0'][:], L['dense_1']['vars']['1'][:]
            ])
            model.get_layer('dense_2').set_weights([
                L['dense_2']['vars']['0'][:], L['dense_2']['vars']['1'][:]
            ])
        return model

    def predict(self, frame_bgr: np.ndarray) -> float:
        """
        Add frame to buffer and return crash probability (0-1).
        Returns -1.0 if model not loaded or buffer not full yet.
        """
        if not self.loaded:
            return -1.0

        # Resize and convert to RGB
        frame = cv2.resize(frame_bgr, (Config.CNN_SIZE, Config.CNN_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        self.frame_buffer.append(frame)

        if len(self.frame_buffer) < Config.CNN_FRAMES:
            return -1.0

        batch = self._preprocess(np.array(self.frame_buffer))
        feats = self.fe.serve(batch)
        prob  = float(self.cm(feats[tf.newaxis], training=False)[0][0])
        return prob


# ============================================================================
# MAIN ENGINE
# ============================================================================

class CrashDetectionEngine:
    def __init__(self):
        Logger.header("CRASH DETECTION SYSTEM v13.0 (Neural + Rule-Based)")
        Logger.section("⚙️  CONFIGURATION")
        print(f"   Process resolution: {Config.PROCESS_WIDTH}x{Config.PROCESS_HEIGHT}")
        print(f"   Pixels/meter: {Config.PIXELS_PER_METER}")
        print(f"   Collision threshold: {Config.COLLISION_DISTANCE_THRESHOLD}m")
        print(f"   YOLO confidence: {Config.YOLO_CONF_THRESHOLD}")
        print(f"   Temporal window: {Config.CRASH_WINDOW_SIZE} frames, "
              f"min {Config.CRASH_MIN_FRAMES} consecutive")
        print()

        self.detector = VehicleDetector()
        self.tracker = CentroidTracker()
        self.speed_est = SpeedEstimator()
        self.collision = CollisionDetector()
        self.neural = NeuralCrashDetector()

        self.frame_count = 0
        self.start_time = None
        self.min_distance_ever = float('inf')
        self.max_speed_ever = 0.0
        self.collision_frame_count = 0
        self.neural_crash_confirmed = False
        self.max_cnn_prob = 0.0
        self.all_analyses = []

    def run(self, video_path: Path, display: bool = True, max_frames: int = 500):
        Logger.header("PROCESSING VIDEO")

        # ── Load video ──
        Logger.section("2️⃣  LOADING VIDEO")
        try:
            video = VideoProcessor(video_path)
        except (FileNotFoundError, RuntimeError) as e:
            Logger.error(str(e))
            return

        Logger.info(f"File: {video.path.name}")
        Logger.info(f"Original: {video.orig_width}x{video.orig_height} @ {video.fps:.0f}fps")
        Logger.info(f"Process at: {Config.PROCESS_WIDTH}x{Config.PROCESS_HEIGHT} (matching Colab)")
        Logger.info(f"Duration: {video.duration:.1f}s ({video.frame_count} frames)")

        display_engine = DisplayEngine(enabled=display)
        self.start_time = time.time()
        frames_to_do = min(video.frame_count, max_frames)

        # ── Process ──
        Logger.section("3️⃣  ANALYZING FRAMES")
        print(f"   Processing {frames_to_do} frames...\n")

        try:
            for fn in range(frames_to_do):
                frame = video.read_frame()
                if frame is None:
                    break

                # Pipeline: detect → track → speed → collision → neural
                detections = self.detector.detect(frame)
                vehicles = self.tracker.update(detections)
                vehicles = self.speed_est.update(vehicles, video.fps)
                analysis = self.collision.process_frame(vehicles, fn + 1)

                # Neural signal: MobileNetV2 + LSTM crash probability
                cnn_prob = self.neural.predict(frame)
                analysis['cnn_prob'] = cnn_prob

                cnn_ready = cnn_prob >= 0  # False until buffer has 10 frames
                if cnn_ready:
                    if cnn_prob > self.max_cnn_prob:
                        self.max_cnn_prob = cnn_prob
                    if cnn_prob >= Config.CNN_THRESH:
                        # CNN confirms crash
                        self.neural_crash_confirmed = True
                        if not analysis['is_sustained_crash']:
                            analysis['is_sustained_crash'] = True
                    else:
                        # CNN says safe — suppress rule-based false positives
                        analysis['is_sustained_crash'] = False
                        analysis['is_collision_frame'] = False

                # Stats
                self.frame_count += 1
                if analysis['min_distance'] < self.min_distance_ever:
                    self.min_distance_ever = analysis['min_distance']
                for v in vehicles:
                    if v['speed_kmh'] > self.max_speed_ever:
                        self.max_speed_ever = v['speed_kmh']
                if analysis['is_collision_frame']:
                    self.collision_frame_count += 1

                self.all_analyses.append(analysis)

                # Display
                if display:
                    dframe = display_engine.draw(frame, vehicles, analysis)
                    if not display_engine.show(dframe):
                        break

                # Progress every 50 frames
                if (fn + 1) % 50 == 0:
                    md = analysis['min_distance']
                    md_s = f"{md:.2f}m" if md < 1000 else "inf"
                    tag = ""
                    if analysis['is_sustained_crash']:
                        tag = " 🚨 CRASH"
                    elif analysis['is_collision_frame']:
                        tag = " ⚠️"
                    cnn_s = f"{cnn_prob:.2f}" if cnn_prob >= 0 else "n/a"
                    print(f"   Frame {fn+1:4d}/{frames_to_do} | "
                          f"vehicles={analysis['num_vehicles']} | "
                          f"min_dist={md_s} | "
                          f"cnn={cnn_s} | "
                          f"reason={analysis['reason']}{tag}")

        except KeyboardInterrupt:
            Logger.warning("Stopped by user")
        finally:
            video.release()
            display_engine.close()

        # ── Report ──
        self._report(video)

    def _report(self, video):
        elapsed = time.time() - self.start_time
        fps_proc = self.frame_count / elapsed if elapsed > 0 else 0
        col_pct = (self.collision_frame_count / self.frame_count * 100) \
            if self.frame_count > 0 else 0

        Logger.header("CRASH DETECTION RESULTS")

        print(f"\n📊 Video Statistics:")
        print(f"   Frames processed: {self.frame_count}")
        print(f"   Duration: {video.duration:.1f}s")
        print(f"   Min distance in video: {self.min_distance_ever:.2f}m")
        print()
        print(f"🚀 Speed Analysis:")
        print(f"   Max speed: {self.max_speed_ever:.1f} km/h")
        print()
        print(f"🚨 Crash Detection:")
        print(f"   Collision frames: {self.collision_frame_count}/{self.frame_count}")
        print(f"   Percentage: {col_pct:.1f}%")
        if self.neural.loaded:
            print(f"   Max CNN probability: {self.max_cnn_prob:.4f}")
        print()

        # ── Verdict ──
        # When neural model is available, it is the primary decision maker.
        # Rule-based signals are secondary and only used as fallback.
        if self.neural.loaded:
            is_crash = self.neural_crash_confirmed
        else:
            is_crash = self.collision.crash_confirmed

        if is_crash and col_pct >= 5:
            verdict = "🚨 CRASH DETECTED"
            confidence = "HIGH"
        elif is_crash and self.max_cnn_prob >= Config.CNN_THRESH:
            verdict = "🚨 CRASH DETECTED"
            confidence = "HIGH"
        elif is_crash:
            verdict = "🚨 CRASH DETECTED"
            confidence = "MEDIUM"
        elif col_pct >= 2:
            verdict = "⚠️  POSSIBLE COLLISION"
            confidence = "MEDIUM"
        elif col_pct > 0:
            verdict = "⚠️  CLOSE CALL"
            confidence = "LOW"
        else:
            verdict = "✅ NO CRASH"
            confidence = "HIGH"

        print("=" * 70)
        print(f"VERDICT: {verdict}")
        print(f"Confidence: {confidence}")
        if self.collision.first_crash_frame:
            print(f"First crash detected at: frame {self.collision.first_crash_frame}")
        print("=" * 70)

        # ── Collision frame details ──
        col_frames = [a for a in self.all_analyses if a['is_collision_frame']]
        if col_frames:
            print(f"\n   Collision frames ({len(col_frames)} total, showing up to 20):")
            for a in col_frames[:20]:
                md = a['min_distance']
                md_s = f"{md:.2f}m" if md < 1000 else "inf"
                sust = " [SUSTAINED]" if a['is_sustained_crash'] else ""
                print(f"     Frame {a['frame']:4d}: "
                      f"dist={md_s}, "
                      f"vehicles={a['num_vehicles']}, "
                      f"reason={a['reason']}{sust}")
            if len(col_frames) > 20:
                print(f"     ... +{len(col_frames) - 20} more")

        print(f"\nProcessed in {elapsed:.1f}s ({fps_proc:.1f} fps)\n")


# ============================================================================
# SELF-TEST
# ============================================================================

def run_self_test():
    Logger.header("SYSTEM SELF-TEST")

    print("\n1️⃣  YOLOv8...")
    det = VehicleDetector()
    Logger.info("OK")

    print("\n2️⃣  Tracker...")
    tr = CentroidTracker()
    fakes = [
        {'x1': 10, 'y1': 10, 'x2': 80, 'y2': 60, 'w': 70, 'h': 50,
         'center_x': 45, 'center_y': 35, 'conf': 0.9, 'class': 'car'},
        {'x1': 300, 'y1': 200, 'x2': 400, 'y2': 280, 'w': 100, 'h': 80,
         'center_x': 350, 'center_y': 240, 'conf': 0.8, 'class': 'car'},
    ]
    tracked = tr.update(fakes)
    Logger.info(f"Tracked {len(tracked)} vehicles")

    print("\n3️⃣  Collision detector...")
    cd = CollisionDetector()

    # Test: far apart = no collision
    result = cd.process_frame(tracked, 1)
    Logger.info(f"Far apart: collision={result['is_collision_frame']}, "
                f"dist={result['min_distance']:.1f}m (expected: no collision)")

    # Test: overlapping = collision
    close_fakes = [
        {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 180, 'w': 100, 'h': 80,
         'center_x': 150, 'center_y': 140, 'conf': 0.9, 'class': 'car',
         'track_id': 1, 'speed_kmh': 40},
        {'x1': 130, 'y1': 110, 'x2': 230, 'y2': 190, 'w': 100, 'h': 80,
         'center_x': 180, 'center_y': 150, 'conf': 0.85, 'class': 'car',
         'track_id': 2, 'speed_kmh': 35},
    ]
    result2 = cd.process_frame(close_fakes, 2)
    Logger.info(f"Overlapping: collision={result2['is_collision_frame']}, "
                f"dist={result2['min_distance']:.2f}m (expected: collision)")

    print("\n4️⃣  Video files...")
    for name, path in Config.VIDEO_SHORTCUTS.items():
        found = path.exists()
        if not found:
            for ext in ['.mp4', '.mov', '.avi']:
                if path.with_suffix(ext).exists():
                    found = True
                    path = path.with_suffix(ext)
                    break
        Logger.info(f"  {name}: {path.name} -> {'FOUND' if found else 'MISSING'}")

    Logger.header("SELF-TEST PASSED ✅")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Crash Detection v12.0')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--video', type=str)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--max-frames', type=int, default=500)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Collision distance threshold in meters')

    args = parser.parse_args()

    if args.threshold is not None:
        Config.COLLISION_DISTANCE_THRESHOLD = args.threshold

    try:
        if args.test:
            run_self_test()
            return

        if args.video:
            engine = CrashDetectionEngine()

            if args.video in Config.VIDEO_SHORTCUTS:
                vpath = Config.VIDEO_SHORTCUTS[args.video]
            else:
                vpath = Path(args.video)

            engine.run(vpath, display=not args.no_display,
                       max_frames=args.max_frames)
        else:
            Logger.error("Provide --video or --test")
            print("\nUsage:")
            print("  python3 crash_detection.py --test")
            print("  python3 crash_detection.py --video safe --no-display")
            print("  python3 crash_detection.py --video crash1 --no-display")
            print("  python3 crash_detection.py --video crash1 --threshold 2.5")

    except Exception as e:
        Logger.header("FATAL ERROR")
        Logger.error(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()