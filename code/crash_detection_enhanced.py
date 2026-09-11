#!/usr/bin/env python3
"""
	Enhanced version — adds dashcam ego zone, webcam live mode, video recording

CRASH DETECTION ENHANCED v16.0
==============================
 - Neural model (MobileNetV2 + LSTM) as primary signal
 - Live Camera: real-time crash detection with webcam
 - Speed overlay on all frames
 - Fault Detection: which car caused the crash
 - Crash Recording: saves crash frames to files
 - CNN gates final verdict (no false positives from rule-based signals)

Run: python3 code/crash_detection_enhanced.py --video crash2
Run: python3 code/crash_detection_enhanced.py --video safe --no-display
Run: python3 code/crash_detection_enhanced.py --video crash2 --save-output
Run: python3 code/crash_detection_enhanced.py --camera
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import h5py
import time
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict
import warnings
import sys
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.get_logger().setLevel('ERROR')

import torch
from ultralytics import YOLO
from scipy.spatial.distance import cdist
warnings. filterwarnings('ignore')



# ============================================================================
# CONFIG — Single Source of Truth (SSOT)
# ============================================================================

class Config:
    BASE_DIR   = Path(__file__).parent.resolve().parent
    VIDEOS_DIR = BASE_DIR / 'videos'
    MODELS_DIR = BASE_DIR / 'models'
    YOLO_PATH  = BASE_DIR / 'yolov8n.pt'
    OUTPUT_DIR = BASE_DIR / 'crash_outputs'

    # ── Camera Calibration (OV5647 at 640×480 estimates) ──
    H_CAM       = 1.25     # camera height above ground (meters)
    PITCH_DEG   = 2.0      # camera tilt below horizon (degrees)
    FX          = 460.0    # focal length X (pixels)
    FY          = 460.0    # focal length Y (pixels)
    CX_IMG      = 320.0    # principal point X (image_width / 2)
    CY_IMG      = 240.0    # principal point Y (image_height / 2)

    # ── Detection ──
    YOLO_CONF      = 0.5
    CONF_NEW_TRACK = 0.50   # min confidence to create new track
    NMS_IOU        = 0.45
    TRACK_MAX_DIST = 80
    MIN_BOX_AREA   = 2000
    MIN_VEHICLE_AR = 0.4    # min aspect ratio (w/h) for vehicles

    # ── Speed ──
    MAX_SPEED      = 180.0  # km/h cap (highway compatible)
    CAMERA_FPS     = 30     # fallback when measured FPS unavailable

    # ── Collision (TTC-based) ──
    TTC_CRITICAL   = 1.2    # seconds — collision imminent
    DIST_CONTACT   = 1.5    # meters — physical contact threshold
    CLOSING_MIN    = 0.5    # m/s — min closing speed to compute TTC
    CRASH_WIN      = 10     # frame window for temporal check
    CRASH_MIN_FR   = 3      # min consecutive collision frames

    # ── Kalman Filter Tuning ──
    KF_Q_VAR       = 0.5    # process noise intensity
    KF_R_X         = 0.15   # measurement noise X (lateral)
    KF_R_Z         = 0.4    # measurement noise Z (longitudinal)
    KF_INIT_COV    = 5.0    # initial covariance

    # ── Tracking ──
    TRACK_TTL_SEC  = 2.0    # seconds to keep lost tracks alive

    # ── Neural model ──
    FE_PATH      = MODELS_DIR / 'feature_extractor_saved'
    WEIGHTS_PATH = MODELS_DIR / 'crash_model_weights.weights.h5'
    CNN_FRAMES   = 10
    CNN_SIZE     = 112
    CNN_THRESH   = 0.80

    # ── ROI mask (fraction of frame height) ──
    ROI_SKY_CUT  = 0.40    # mask out top 40% (sky)
    ROI_HOOD_CUT = 0.90    # mask out bottom 10% (hood)

    VIDEO_SHORTCUTS = {
        'crash1': VIDEOS_DIR / 'crash1.mov',
        'crash2': VIDEOS_DIR / 'crash2.mov',
        'safe'  : VIDEOS_DIR / 'safe.mp4',
    }

    @classmethod
    def set_resolution(cls, width, height):
        """Update principal point for current resolution."""
        cls.CX_IMG = width / 2.0
        cls.CY_IMG = height / 2.0

    @classmethod
    def validate(cls):
        """Runtime validation — call once at startup."""
        assert 0.3 < cls.H_CAM < 4.0, f"Camera height {cls.H_CAM}m out of range"
        assert -15 < cls.PITCH_DEG < 15, f"Pitch {cls.PITCH_DEG}° out of range"
        assert cls.FX > 50, f"Focal length {cls.FX} too small"


# ============================================================================
# LOGGER
# ============================================================================

class Log:
    @staticmethod
    def ok(m):   print(f"✅ {m}")
    @staticmethod
    def err(m):  print(f"❌ {m}")
    @staticmethod
    def warn(m): print(f"⚠️  {m}")
    @staticmethod
    def head(m):
        print(f"\n{'='*70}\n🚗 {m}\n{'='*70}")
    @staticmethod
    def sec(m):
        print(f"\n{m}\n{'-'*70}")


# ============================================================================
# NEURAL CRASH DETECTOR
# ============================================================================

class NeuralCrashDetector:
    def __init__(self):
        self.loaded = False
        self.frame_buffer = deque(maxlen=Config.CNN_FRAMES)
        self._pre = keras.applications.mobilenet_v2.preprocess_input

        if not Config.FE_PATH.exists() or not Config.WEIGHTS_PATH.exists():
            Log.warn("Neural model files not found — CNN signal disabled")
            return
        try:
            self.fe = tf.saved_model.load(str(Config.FE_PATH))
            self.cm = self._build_and_load(str(Config.WEIGHTS_PATH))
            self.loaded = True
            Log.ok("Neural model loaded (MobileNetV2 + LSTM)")
        except Exception as e:
            Log.warn(f"Neural model failed: {e}")

    @staticmethod
    def _build_and_load(w_path):
        inp = keras.Input(shape=(Config.CNN_FRAMES, 1280))
        x = layers.Dense(256, activation='relu')(inp)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inp, out)
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
        if not self.loaded:
            return -1.0
        frame = cv2.resize(frame_bgr, (Config.CNN_SIZE, Config.CNN_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) < Config.CNN_FRAMES:
            return -1.0
        batch = self._pre(np.array(self.frame_buffer))
        feats = self.fe.serve(batch)
        return float(self.cm(feats[tf.newaxis], training=False)[0][0])


# ============================================================================
# YOLO DETECTOR
# ============================================================================

class Detector:
    VEHICLES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
    OBJECTS  = {'person', 'dog', 'cat', 'backpack', 'suitcase',
                'chair', 'bench', 'traffic light', 'stop sign'}

    def __init__(self):
        # PyTorch 2.6 fix: patch torch.load to allow YOLO weights
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
        try:
            yolo_path = str(Config.YOLO_PATH) if Config.YOLO_PATH.exists() else 'yolov8n.pt'
            self.model = YOLO(yolo_path)
        finally:
            torch.load = _orig
        Log.ok("YOLO loaded")

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / (ua + 1e-6)

    def detect(self, frame) -> List[Dict]:
        try:
            results = self.model(frame, conf=Config.YOLO_CONF, verbose=False)
        except Exception:
            return []
        raw = []
        for box in results[0].boxes:
            cls = results[0].names[int(box.cls)].lower()
            is_vehicle = cls in self.VEHICLES
            is_object  = cls in self.OBJECTS
            if not (is_vehicle or is_object):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            min_area = Config.MIN_BOX_AREA if is_vehicle else 400
            if w < 15 or h < 15 or w * h < min_area:
                continue
            raw.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'w': w, 'h': h,
                'cx': (x1 + x2) // 2, 'cy': (y1 + y2) // 2,
                'conf': float(box.conf), 'cls': cls,
                'is_vehicle': is_vehicle,
            })
        # NMS
        raw.sort(key=lambda d: d['conf'], reverse=True)
        keep, killed = [], set()
        for i in range(len(raw)):
            if i in killed:
                continue
            keep.append(raw[i])
            bi = [raw[i]['x1'], raw[i]['y1'], raw[i]['x2'], raw[i]['y2']]
            for j in range(i + 1, len(raw)):
                if j not in killed:
                    bj = [raw[j]['x1'], raw[j]['y1'], raw[j]['x2'], raw[j]['y2']]
                    if self._iou(bi, bj) > Config.NMS_IOU:
                        killed.add(j)
        return keep


def scene_validate_vehicles(vehicles: List[Dict]) -> List[Dict]:
    """
    Post-detection filter: removes implausible 'vehicle' detections.
    Prevents indoor objects (faces, shirts, furniture) from being
    classified as vehicles by checking geometric plausibility.
    """
    validated = []
    for v in vehicles:
        if not v.get('is_vehicle', False):
            validated.append(v)  # keep non-vehicle objects as-is
            continue
        # Check aspect ratio: real cars are wider than tall
        ar = v['w'] / max(v['h'], 1)
        if ar < Config.MIN_VEHICLE_AR:
            continue  # skip tall/narrow detections (people, poles)
        # Check confidence threshold for new vehicle detections
        if v.get('conf', 0) < Config.CONF_NEW_TRACK:
            continue
        validated.append(v)
    return validated


# ============================================================================
# TRACKER
# ============================================================================

class Tracker:
    def __init__(self):
        self.next_id = 1
        self.objects = {}

    def update(self, dets: List[Dict]) -> List[Dict]:
        if not dets:
            self.objects = {}
            return []
        if not self.objects:
            for d in dets:
                d['id'] = self.next_id
                self.objects[self.next_id] = d
                self.next_id += 1
            return dets

        det_c = np.array([[d['cx'], d['cy']] for d in dets])
        oids  = list(self.objects.keys())
        obj_c = np.array([[self.objects[o]['cx'], self.objects[o]['cy']] for o in oids])
        D = cdist(det_c, obj_c)
        result, used_d, used_o = [], set(), set()

        for flat in np.argsort(D, axis=None):
            di, oi = flat // len(oids), flat % len(oids)
            if di in used_d or oi in used_o:
                continue
            if D[di][oi] > Config.TRACK_MAX_DIST:
                break
            tid = oids[oi]
            dets[di]['id'] = tid
            self.objects[tid] = dets[di]
            used_d.add(di); used_o.add(oi)
            result.append(dets[di])

        for i, d in enumerate(dets):
            if i not in used_d:
                d['id'] = self.next_id
                self.objects[self.next_id] = d
                self.next_id += 1
                result.append(d)

        for oi, oid in enumerate(oids):
            if oi not in used_o:
                del self.objects[oid]
        return result



# ============================================================================
# GROUND PROJECTION & UTILITIES (Steps 2, 4)
# ============================================================================

def pixel_to_ground(x_center: float, y_bottom: float) -> tuple:
    """
    Projects tire contact point (x_center, y_bottom) in pixel space to
    3D camera coordinates (X, Z) in meters on the ground plane.

    Math:
        angle_v = arctan((y_bottom - cy) / fy)
        Z = h_cam / tan(pitch + angle_v)
        X = (x_center - cx) × Z / fx
    """
    pitch_rad = np.radians(Config.PITCH_DEG)
    angle_v = np.arctan2(y_bottom - Config.CY_IMG, Config.FY)
    denom = np.tan(pitch_rad + angle_v)
    if denom <= 0.001:
        Z = 100.0  # at or above horizon
    else:
        Z = Config.H_CAM / denom
    X = (x_center - Config.CX_IMG) * Z / Config.FX
    return float(X), float(Z)


def apply_road_roi(frame: np.ndarray) -> np.ndarray:
    """Masks out sky (top) and hood (bottom) before YOLO inference."""
    h, w = frame.shape[:2]
    mask = np.zeros_like(frame)
    y_top = int(h * Config.ROI_SKY_CUT)
    y_bot = int(h * Config.ROI_HOOD_CUT)
    pts = np.array([
        [0, y_top], [w, y_top], [w, y_bot], [0, y_bot]
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    return cv2.bitwise_and(frame, mask)


def compute_ttc(pos_a, vel_a, pos_b, vel_b) -> float:
    """
    Time-To-Collision between two vehicles using 3D vector projection.
    closing_speed = -dot(p_rel, v_rel) / ||p_rel||
    TTC = ||p_rel|| / closing_speed
    """
    p_rel = np.asarray(pos_b, dtype=float) - np.asarray(pos_a, dtype=float)
    v_rel = np.asarray(vel_b, dtype=float) - np.asarray(vel_a, dtype=float)
    distance = np.linalg.norm(p_rel)
    if distance < 0.1:
        return 0.0  # already in contact
    closing_speed = -np.dot(p_rel, v_rel) / distance
    if closing_speed > Config.CLOSING_MIN:
        return distance / closing_speed
    return float('inf')  # separating or parallel


# ============================================================================
# KALMAN FILTER (Step 3 — 4D state tracker per vehicle)
# ============================================================================

class VehicleKF:
    """Tracks one vehicle: state = [X, Z, Vx, Vz] in meters & m/s."""

    def __init__(self, init_x: float, init_z: float, init_time: float):
        self.x = np.array([[init_x], [init_z], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(4) * Config.KF_INIT_COV
        self.last_time = init_time
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.diag([Config.KF_R_X, Config.KF_R_Z])

    def predict_and_update(self, meas_x: float, meas_z: float, current_time: float):
        dt = current_time - self.last_time
        dt = max(0.001, min(dt, 1.0))
        self.last_time = current_time

        F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1]
        ], dtype=np.float64)

        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q = np.array([
            [dt4/4,   0,     dt3/2, 0    ],
            [0,       dt4/4, 0,     dt3/2],
            [dt3/2,   0,     dt2,   0    ],
            [0,       dt3/2, 0,     dt2  ]
        ], dtype=np.float64) * Config.KF_Q_VAR

        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Update
        z = np.array([[meas_x], [meas_z]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return tuple(self.x.flatten())  # (X, Z, Vx, Vz)


# ============================================================================
# SPEED & 3D STATE ESTIMATOR (replaces old SpeedEstimator)
# ============================================================================

class SpeedEstimator:
    """
    Projects each vehicle to ground plane and tracks with Kalman Filter.
    Outputs: X_m, Z_m, Vx_ms, Vz_ms, speed (km/h) per vehicle.
    """

    def __init__(self):
        self.trackers = {}      # vid -> VehicleKF
        self.last_time = None

    def update(self, vehicles: List[Dict], nominal_fps: float) -> List[Dict]:
        now = time.perf_counter()
        if self.last_time is not None:
            dt = now - self.last_time
            dt = max(0.001, min(dt, 1.0))
        else:
            dt = 1.0 / nominal_fps
        self.last_time = now

        current_ids = set()

        for v in vehicles:
            vid = v['id']
            current_ids.add(vid)

            # Anchor to tire contact point (y_bottom = y2)
            x_center = (v['x1'] + v['x2']) / 2.0
            y_bottom = float(v['y2'])
            X_m, Z_m = pixel_to_ground(x_center, y_bottom)

            if vid not in self.trackers:
                self.trackers[vid] = VehicleKF(X_m, Z_m, now)
                v['X_m'], v['Z_m'] = X_m, Z_m
                v['Vx_ms'], v['Vz_ms'] = 0.0, 0.0
                v['speed'] = 0.0
            else:
                X, Z, Vx, Vz = self.trackers[vid].predict_and_update(X_m, Z_m, now)
                v['X_m'], v['Z_m'] = X, Z
                v['Vx_ms'], v['Vz_ms'] = Vx, Vz
                v['speed'] = min(float(np.sqrt(Vx**2 + Vz**2) * 3.6),
                                 Config.MAX_SPEED)

        # TTL-based cleanup: keep lost tracks for TRACK_TTL_SEC
        stale = [vid for vid, kf in self.trackers.items()
                 if vid not in current_ids and
                    (now - kf.last_time) > Config.TRACK_TTL_SEC]
        for sid in stale:
            del self.trackers[sid]

        return vehicles


# ============================================================================
# RULE-BASED COLLISION (TTC-based, replaces old pixel/PPM approach)
# ============================================================================

class RuleCollision:
    def __init__(self):
        self.dist_hist  = deque(maxlen=Config.CRASH_WIN)
        self.count_hist = deque(maxlen=Config.CRASH_WIN)
        self.col_win    = deque(maxlen=Config.CRASH_WIN)

    def process(self, vehicles: List[Dict]):
        n = len(vehicles)
        self.count_hist.append(n)

        min_dist = float('inf')
        min_ttc  = float('inf')
        is_col_frame = False

        for i in range(n):
            for j in range(i + 1, n):
                a, b = vehicles[i], vehicles[j]

                # 3D metric distance
                dX = a.get('X_m', 0) - b.get('X_m', 0)
                dZ = a.get('Z_m', 0) - b.get('Z_m', 0)
                dist_3d = np.sqrt(dX**2 + dZ**2)

                if dist_3d < min_dist:
                    min_dist = dist_3d

                # TTC via vector-projected closing speed
                ttc = compute_ttc(
                    [a.get('X_m', 0), a.get('Z_m', 0)],
                    [a.get('Vx_ms', 0), a.get('Vz_ms', 0)],
                    [b.get('X_m', 0), b.get('Z_m', 0)],
                    [b.get('Vx_ms', 0), b.get('Vz_ms', 0)]
                )
                if ttc < min_ttc:
                    min_ttc = ttc

                # Collision condition: physical contact OR imminent TTC
                contact = dist_3d < Config.DIST_CONTACT
                imminent = (min_ttc < Config.TTC_CRITICAL)

                if contact or imminent:
                    # Confirm with 2D bounding box overlap
                    if not (a['x2'] < b['x1'] or b['x2'] < a['x1'] or
                            a['y2'] < b['y1'] or b['y2'] < a['y1']):
                        is_col_frame = True

        self.dist_hist.append(min_dist)

        # Vehicle drop signal (kept from original)
        counts = list(self.count_hist)
        if len(counts) >= 3 and not is_col_frame:
            recent_close = any(
                d < Config.DIST_CONTACT
                for d in list(self.dist_hist)[-5:]
                if d < float('inf')
            )
            if (max(counts[:-1], default=0) >= 2 and
                counts[-1] < max(counts[:-1], default=0) and
                recent_close):
                is_col_frame = True

        self.col_win.append(is_col_frame)

        # Sustained check (original logic — verified correct)
        consec = cur = 0
        for v in self.col_win:
            cur = cur + 1 if v else 0
            consec = max(consec, cur)

        return {
            'is_col_frame': is_col_frame,
            'is_sustained': consec >= Config.CRASH_MIN_FR,
            'min_dist': min_dist,
            'min_ttc': min_ttc,
        }


# ============================================================================
# DISPLAY
# ============================================================================

class Display:
    def __init__(self, on=True):
        self.on = on

    def draw(self, frame, vehicles, is_crash, cnn_prob, min_dist, min_ttc=None):
        out = frame.copy()
        h, w = out.shape[:2]

        vehicle_colors = [(0, 255, 0), (255, 100, 0), (0, 100, 255), (255, 255, 0), (0, 255, 255)]
        object_color   = (180, 180, 180)
        max_spd = 0.0
        for v in vehicles:
            is_veh = v.get('is_vehicle', True)
            color  = vehicle_colors[v['id'] % len(vehicle_colors)] if is_veh else object_color
            thickness = 3 if is_veh else 2
            cv2.rectangle(out, (v['x1'], v['y1']), (v['x2'], v['y2']), color, thickness)
            if is_veh:
                spd = v.get('speed', 0)
                max_spd = max(max_spd, spd)
                label = f"ID{v['id']} {spd:.0f}km/h"
            else:
                label = f"{v['cls']}"
            cv2.putText(out, label,
                        (v['x1'], max(v['y1'] - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        n_veh = sum(1 for v in vehicles if v.get('is_vehicle', True))
        n_obj = len(vehicles) - n_veh


        # Info bar at top
        bar = np.zeros((90, w, 3), dtype=np.uint8)
        cnn_s = f"{cnn_prob:.2f}" if cnn_prob >= 0 else "warming up"
        spd_s = f"{max_spd:.0f}km/h" if n_veh else "--"
        ttc_s = f"{min_ttc:.1f}s" if min_ttc is not None and min_ttc < 999 else "--"
        md_s = f"{min_dist:.2f}m" if min_dist < 999 else "inf"
        cv2.putText(bar, f"Vehicles:{n_veh}  Objects:{n_obj}  |  Dist:{md_s}  TTC:{ttc_s}  Speed:{spd_s}  |  CNN:{cnn_s}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

        if is_crash:
            cv2.rectangle(bar, (0, 62), (w, 90), (0, 0, 220), -1)
            cv2.putText(bar, "  CRASH DETECTED  ",
                        (w // 2 - 130, 84),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        return cv2.vconcat([bar, out])

    def show(self, frame, title="Crash Detection v17.0 — 3D Perception") -> bool:
        if not self.on:
            return True
        try:
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            return True
        except Exception:
            return True

    def close(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ============================================================================
# MAIN ENGINE
# ============================================================================

class CrashDetectionEnhanced:
    def __init__(self, save_output=False):
        Log.head("CRASH DETECTION v17.0 — 3D PERCEPTION + DEPTH + BEV")
        Config.validate()
        Config.OUTPUT_DIR.mkdir(exist_ok=True)

        self.detector    = Detector()
        self.neural      = NeuralCrashDetector()
        self.save_output = save_output


    def _make_pipeline(self):
        return Tracker(), SpeedEstimator(), RuleCollision()

    def run_video(self, video_path, display=True, max_frames=500):
        from pathlib import Path as P
        path = P(video_path)
        if not path.exists():
            for ext in ['.mp4', '.mov', '.avi', '.mkv']:
                alt = path.with_suffix(ext)
                if alt.exists():
                    path = alt
                    break
        if not path.exists():
            Log.err(f"Not found: {video_path}"); return

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            Log.err(f"Cannot open: {path}"); return

        fps    = cap.get(cv2.CAP_PROP_FPS) or Config.CAMERA_FPS
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Config.set_resolution(orig_w, orig_h)
        dur    = total / fps

        Log.head(f"PROCESSING: {path.name}")
        Log.ok(f"Resolution: {orig_w}x{orig_h} @ {fps:.0f}fps")
        Log.ok(f"Duration: {dur:.1f}s ({total} frames)")

        tracker, speed_est, rule_col = self._make_pipeline()
        disp = Display(on=display)

        frame_count = crash_frames = 0
        neural_confirmed = False
        max_cnn = 0.0
        min_dist_ever = float('inf')
        max_speed_ever = 0.0
        first_fault = None
        output_dir = Config.OUTPUT_DIR / path.stem
        if self.save_output:
            output_dir.mkdir(exist_ok=True)

        todo = min(total, max_frames)
        print(f"\nProcessing {todo} frames... (press Q to quit)\n")
        t0 = time.time()

        try:
            for fn in range(todo):
                ret, frame = cap.read()
                if not ret:
                    break

                roi_frame = apply_road_roi(frame)
                dets    = self.detector.detect(roi_frame)
                dets    = scene_validate_vehicles(dets)
                tracked = tracker.update(dets)
                tracked = speed_est.update(tracked, fps)
                vehs    = [v for v in tracked if v.get('is_vehicle', True)]

                rule    = rule_col.process(vehs)
                cnn     = self.neural.predict(frame)
                fh, fw  = frame.shape[:2]

                if cnn >= 0:
                    max_cnn = max(max_cnn, cnn)

                # CNN gates the verdict, require >=2 vehicles.
                # NOTE: this threshold has no derivation - see README bug B5.
                if cnn >= 0:
                    is_crash = (cnn >= Config.CNN_THRESH) and (len(vehs) >= 2)
                    if is_crash:
                        neural_confirmed = True
                else:
                    is_crash = rule['is_sustained'] and (len(vehs) >= 2)

                frame_count += 1
                if is_crash:
                    crash_frames += 1
                    if self.save_output and len(os.listdir(output_dir)) < 20:
                        cv2.imwrite(str(output_dir / f"crash_{fn+1:04d}.jpg"), frame)

                min_dist_ever  = min(min_dist_ever, rule['min_dist'])
                max_speed_ever = max(max_speed_ever,
                                     max((v.get('speed', 0) for v in vehs), default=0))

                if display:
                    df = disp.draw(
                        frame, tracked, is_crash, cnn, rule['min_dist'],
                        min_ttc=rule.get('min_ttc')
                    )
                    if not disp.show(df):
                        break

                if (fn + 1) % 50 == 0:
                    md = rule['min_dist']
                    md_s = f"{md:.2f}m" if md < 999 else "inf"
                    cnn_s = f"{cnn:.2f}" if cnn >= 0 else "n/a"
                    tag = " CRASH" if is_crash else ""
                    print(f"   Frame {fn+1:4d}/{todo} | vehicles={len(vehs)} | "
                          f"dist={md_s} | cnn={cnn_s}{tag}")

        except KeyboardInterrupt:
            Log.warn("Stopped by user")
        except Exception as e:
            Log.warn(f"Stopped: {e}")
        finally:
            cap.release()
            disp.close()
            self._report(path.name, frame_count, crash_frames,
                         neural_confirmed, max_cnn, min_dist_ever,
                         max_speed_ever, time.time() - t0, first_fault)

    def run_camera(self, display=True, record=False):
        Log.head("LIVE CAMERA MODE")
        Log.ok("Press Q to quit")
        if record:
            Log.ok("Recording ON — output will be saved to crash_outputs/")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            Log.err("Cannot open webcam"); return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Config.set_resolution(cam_w, cam_h)
        fps = cap.get(cv2.CAP_PROP_FPS) or Config.CAMERA_FPS

        tracker, speed_est, rule_col = self._make_pipeline()
        disp = Display(on=display)

        frame_count = crash_frames = 0
        neural_confirmed = False
        max_cnn = 0.0
        min_dist_ever = float('inf')
        max_speed_ever = 0.0
        first_fault = None
        writer = None
        t0 = time.time()

        print("\nCamera active — press Q to stop\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                roi_frame = apply_road_roi(frame)
                dets    = self.detector.detect(roi_frame)
                dets    = scene_validate_vehicles(dets)
                tracked = tracker.update(dets)
                tracked = speed_est.update(tracked, fps)
                vehs    = [v for v in tracked if v.get('is_vehicle', True)]

                rule    = rule_col.process(vehs)
                cnn     = self.neural.predict(frame)
                fh, fw  = frame.shape[:2]

                if cnn >= 0:
                    max_cnn = max(max_cnn, cnn)

                if cnn >= 0:
                    is_crash = (cnn >= Config.CNN_THRESH) and (len(vehs) >= 2)
                    if is_crash:
                        neural_confirmed = True
                else:
                    is_crash = rule['is_sustained'] and (len(vehs) >= 2)

                frame_count += 1
                if is_crash:
                    crash_frames += 1

                min_dist_ever  = min(min_dist_ever, rule['min_dist'])
                max_speed_ever = max(max_speed_ever,
                                     max((v.get('speed', 0) for v in vehs), default=0))

                df = disp.draw(
                    frame, tracked, is_crash, cnn, rule['min_dist'],
                    min_ttc=rule.get('min_ttc')
                )

                # Init video writer on first annotated frame
                if record and writer is None:
                    Config.OUTPUT_DIR.mkdir(exist_ok=True)
                    ts   = time.strftime("%Y%m%d_%H%M%S")
                    path = Config.OUTPUT_DIR / f"recording_{ts}.mp4"
                    h_df, w_df = df.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(path), fourcc, fps, (w_df, h_df))
                    Log.ok(f"Recording to: {path}")

                if record and writer is not None:
                    writer.write(df)

                if not disp.show(df, title="Live Camera — Crash Detection"):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            disp.close()
            if writer is not None:
                writer.release()
                Log.ok("Recording saved.")
            self._report("webcam", frame_count, crash_frames,
                         neural_confirmed, max_cnn, min_dist_ever,
                         max_speed_ever, time.time() - t0, first_fault)

    def _report(self, name, frame_count, crash_frames,
                neural_confirmed, max_cnn, min_dist_ever,
                max_speed_ever, elapsed, fault_info=None):
        fps_proc = frame_count / elapsed if elapsed > 0 else 0
        crash_pct = (crash_frames / frame_count * 100) if frame_count > 0 else 0

        Log.head("RESULTS")
        print(f"\n📹 Source: {name}")
        print(f"\n📊 Statistics:")
        print(f"   Frames processed : {frame_count}")
        print(f"   Processing time  : {elapsed:.1f}s ({fps_proc:.1f} fps)")
        print(f"\n📏 Measurements:")
        md = min_dist_ever
        print(f"   Closest approach : {md:.2f}m" if md < 999 else "   Closest approach: inf")
        print(f"   Max speed        : {max_speed_ever:.1f} km/h")
        print(f"\n🚨 Crash Detection:")
        print(f"   Crash frames     : {crash_frames}/{frame_count} ({crash_pct:.1f}%)")
        print(f"   Max CNN prob     : {max_cnn:.4f}")
        print(f"\n{'='*70}")

        if self.neural.loaded:
            is_crash = neural_confirmed and (crash_pct >= 1.5)
        else:
            is_crash = crash_pct >= 5.0

        if is_crash and max_cnn >= Config.CNN_THRESH:
            print(f"VERDICT: 🚨 CRASH DETECTED")
            print(f"Confidence: HIGH (CNN={max_cnn:.2f})")
        elif is_crash:
            print(f"VERDICT: 🚨 CRASH DETECTED")
            print(f"Confidence: MEDIUM")
        elif crash_pct >= 2.0:
            print(f"VERDICT: ⚠️  POSSIBLE COLLISION")
            print(f"Confidence: LOW")
        else:
            print(f"VERDICT: ✅ NO CRASH")
            print(f"Confidence: HIGH")

        if is_crash and fault_info:
            print(f"\n⚠️  Fault Detection:")
            print(f"   Collision type : {fault_info.get('collision_type', 'unknown')}")
            print(f"   At fault       : {fault_info['at_fault']}")
            print(f"   Reason         : {fault_info['reason']}")
            v2 = fault_info['v2']
            print(f"   V{fault_info['v1']} speed      : {fault_info['v1_speed']:.0f} km/h")
            if v2 != 'ego':
                print(f"   V{v2} speed      : {fault_info['v2_speed']:.0f} km/h")

        print(f"{'='*70}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Crash Detection v16.0 Enhanced')
    parser.add_argument('--video',      type=str,  help='Video name (crash1, crash2, safe) or path')
    parser.add_argument('--camera',     action='store_true', help='Use live webcam')
    parser.add_argument('--record',     action='store_true', help='Record annotated webcam video to file')
    parser.add_argument('--save-output',action='store_true', help='Save crash frames as images')
    parser.add_argument('--no-display', action='store_true', help='Suppress video window')
    parser.add_argument('--max-frames', type=int,  default=500, help='Max frames to process')
    args = parser.parse_args()

    engine = CrashDetectionEnhanced(save_output=args.save_output)

    if args.camera:
        engine.run_camera(display=not args.no_display, record=args.record)
    elif args.video:
        vpath = Config.VIDEO_SHORTCUTS.get(args.video, Path(args.video))
        engine.run_video(vpath, display=not args.no_display, max_frames=args.max_frames)
    else:
        Log.err("Provide --video or --camera")
        print("\nUsage:")
        print("  python3 code/crash_detection_enhanced.py --video crash2")
        print("  python3 code/crash_detection_enhanced.py --video safe --no-display")
        print("  python3 code/crash_detection_enhanced.py --video crash2 --save-output")
        print("  python3 code/crash_detection_enhanced.py --camera")


if __name__ == '__main__':
    main()