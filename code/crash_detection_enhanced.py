#!/usr/bin/env python3
"""
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
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    BASE_DIR   = Path(__file__).parent.resolve().parent
    VIDEOS_DIR = BASE_DIR / 'videos'
    MODELS_DIR = BASE_DIR / 'models'
    YOLO_PATH  = BASE_DIR / 'yolov8n.pt'
    OUTPUT_DIR = BASE_DIR / 'crash_outputs'

    YOLO_CONF      = 0.5
    NMS_IOU        = 0.45
    TRACK_MAX_DIST = 80
    MIN_BOX_AREA   = 2000
    PPM            = 25       # pixels per meter (fallback)
    MAX_SPEED      = 120
    SPEED_SMOOTH   = 3
    CAMERA_FPS     = 30

    COLLISION_DIST = 2.0      # meters
    CRASH_WIN      = 10       # frame window for rule-based
    CRASH_MIN_FR   = 3        # min consecutive collision frames

    # Neural model
    FE_PATH      = MODELS_DIR / 'feature_extractor_saved'
    WEIGHTS_PATH = MODELS_DIR / 'crash_model_weights.weights.h5'
    CNN_FRAMES   = 10
    CNN_SIZE     = 112
    CNN_THRESH   = 0.80

    # Dashcam ego zone (fraction of frame dimensions)
    EGO_ZONE_W   = 0.20   # width  = 20% of frame
    EGO_ZONE_H   = 0.09   # height =  9% of frame
    EGO_ZONE_Y   = 0.96   # bottom edge at 96% down the frame

    VIDEO_SHORTCUTS = {
        'crash1': VIDEOS_DIR / 'crash1.mov',
        'crash2': VIDEOS_DIR / 'crash2.mov',
        'safe'  : VIDEOS_DIR / 'safe.mp4',
    }


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
# SPEED & DISTANCE
# ============================================================================

class SpeedEstimator:
    def __init__(self):
        self.prev = {}
        self.bufs = {}

    def update(self, vehicles: List[Dict], fps: float) -> List[Dict]:
        for v in vehicles:
            vid = v['id']
            if vid in self.prev:
                px, py = self.prev[vid]
                pd  = np.sqrt((v['cx'] - px)**2 + (v['cy'] - py)**2)
                spd = (pd / Config.PPM) / (1.0 / fps) * 3.6
                if vid not in self.bufs:
                    self.bufs[vid] = deque(maxlen=Config.SPEED_SMOOTH)
                self.bufs[vid].append(spd)
                v['speed'] = min(float(np.mean(self.bufs[vid])), Config.MAX_SPEED)
            else:
                v['speed'] = 0.0
            self.prev[vid] = (v['cx'], v['cy'])
        return vehicles


# ============================================================================
# RULE-BASED COLLISION
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
        is_close = is_overlap = False

        for i in range(n):
            for j in range(i + 1, n):
                a, b = vehicles[i], vehicles[j]
                pd = np.sqrt((a['cx'] - b['cx'])**2 + (a['cy'] - b['cy'])**2)
                md = pd / Config.PPM
                if md < min_dist:
                    min_dist = md
                if md < Config.COLLISION_DIST:
                    is_close = True
                    if not (a['x2'] < b['x1'] or b['x2'] < a['x1'] or
                            a['y2'] < b['y1'] or b['y2'] < a['y1']):
                        is_overlap = True

        self.dist_hist.append(min_dist)

        # Rapid closing signal
        dists = [d for d in self.dist_hist if d < float('inf')]
        is_rapid = False
        if len(dists) >= 3 and dists[0] > 0:
            is_rapid = (dists[0] - dists[-1]) / dists[0] > 0.3

        # Vehicle drop signal (strict — within threshold only)
        counts = list(self.count_hist)
        is_drop = False
        if len(counts) >= 3:
            close_recent = any(d < Config.COLLISION_DIST for d in list(self.dist_hist)[-5:] if d < float('inf'))
            if max(counts[:-1], default=0) >= 2 and counts[-1] < max(counts[:-1], default=0) and close_recent:
                is_drop = True

        is_col_frame = (is_close and is_overlap) or (is_close and is_rapid)
        if is_drop and not is_col_frame:
            is_col_frame = True

        self.col_win.append(is_col_frame)

        # Sustained check
        consec = cur = 0
        for v in self.col_win:
            cur = cur + 1 if v else 0
            consec = max(consec, cur)

        return {
            'is_col_frame': is_col_frame,
            'is_sustained': consec >= Config.CRASH_MIN_FR,
            'min_dist': min_dist,
        }


# ============================================================================
# FAULT DETECTOR
# ============================================================================

class FaultDetector:
    def __init__(self):
        self.pos_hist = {}   # id -> deque of (cx, cy)

    def _update(self, vehicles):
        for v in vehicles:
            if v['id'] not in self.pos_hist:
                self.pos_hist[v['id']] = deque(maxlen=25)
            self.pos_hist[v['id']].append((v['cx'], v['cy']))

    def _displacement(self, vid, lookback=8):
        """Total pixel displacement over last N frames — more reliable than speed."""
        h = self.pos_hist.get(vid)
        if not h or len(h) < 2:
            return 0.0
        n = min(lookback, len(h))
        return float(np.linalg.norm(np.array(h[-1]) - np.array(h[-n])))

    def _was_stationary(self, vid, frames=15, threshold=10):
        """Was this vehicle barely moving for the last N frames (i.e. parked/stopped)?"""
        h = self.pos_hist.get(vid)
        if not h or len(h) < 3:
            return False
        pts = list(h)[-min(frames, len(h)):]
        total = sum(np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i]))
                    for i in range(len(pts) - 1))
        return total < threshold

    def _direction_vector(self, vid, lookback=8):
        """Normalised direction of movement over last N frames."""
        h = self.pos_hist.get(vid)
        if not h or len(h) < lookback:
            return None
        vec = np.array(h[-1]) - np.array(h[-lookback])
        mag = np.linalg.norm(vec)
        return vec / mag if mag > 3 else None

    def _collision_type(self, a, b):
        da = self._displacement(a['id'])
        db = self._displacement(b['id'])

        # Both barely moved → post-crash stop, can't determine
        if da < 5 and db < 5:
            return 'stationary'

        va = self._direction_vector(a['id'])
        vb = self._direction_vector(b['id'])

        # One was parked before crash
        a_parked = self._was_stationary(a['id'])
        b_parked = self._was_stationary(b['id'])
        if a_parked and not b_parked:
            return 'rear-end-b'   # B moved into parked A
        if b_parked and not a_parked:
            return 'rear-end-a'   # A moved into parked B

        if va is None or vb is None:
            # Fallback: whichever moved more is the aggressor
            if da > db + 15:
                return 'rear-end-a'
            if db > da + 15:
                return 'rear-end-b'
            return 'unknown'

        ab = np.array([b['cx'] - a['cx'], b['cy'] - a['cy']], dtype=float)
        ab_norm = ab / (np.linalg.norm(ab) + 1e-6)

        a_toward = float(np.dot(va, ab_norm))
        b_toward = float(np.dot(vb, -ab_norm))

        if a_toward > 0.45 and b_toward > 0.45:
            return 'head-on'
        if a_toward > 0.45 and b_toward < 0.15:
            return 'rear-end-a'
        if b_toward > 0.45 and a_toward < 0.15:
            return 'rear-end-b'
        return 'side'

    @staticmethod
    def _motion_label(disp):
        if disp < 6:   return "stopped"
        if disp < 20:  return "moving slowly"
        if disp < 45:  return "moving fast"
        return "moving very fast"

    def analyze(self, vehicles: List[Dict]) -> Optional[Dict]:
        if len(vehicles) < 2:
            return None
        self._update(vehicles)

        pairs = []
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                d = np.sqrt((a['cx'] - b['cx'])**2 + (a['cy'] - b['cy'])**2)
                pairs.append((d, a, b))
        _, a, b = min(pairs, key=lambda x: x[0])

        da   = self._displacement(a['id'])
        db   = self._displacement(b['id'])
        la   = self._motion_label(da)
        lb   = self._motion_label(db)
        ctype = self._collision_type(a, b)

        if ctype == 'rear-end-a':
            fault  = f"V{a['id']}"
            reason = f"V{a['id']} hit V{b['id']} from behind ({la})"
        elif ctype == 'rear-end-b':
            fault  = f"V{b['id']}"
            reason = f"V{b['id']} hit V{a['id']} from behind ({lb})"
        elif ctype == 'head-on':
            fault  = 'shared'
            reason = f"Head-on — both vehicles approaching each other"
        elif ctype == 'stationary':
            fault  = 'unclear'
            reason = "Both vehicles stationary — unable to determine fault"
        else:
            if da > db + 15:
                fault  = f"V{a['id']}"
                reason = f"V{a['id']} crossed into V{b['id']} ({la})"
            elif db > da + 15:
                fault  = f"V{b['id']}"
                reason = f"V{b['id']} crossed into V{a['id']} ({lb})"
            else:
                fault  = 'unclear'
                reason = f"Side impact between V{a['id']} and V{b['id']}"

        return {'at_fault': fault, 'reason': reason, 'collision_type': ctype,
                'v1': a['id'], 'v2': b['id'],
                'v1_speed': a.get('speed', 0), 'v2_speed': b.get('speed', 0)}


# ============================================================================
# DASHCAM EGO ZONE
# ============================================================================

class EgoZone:
    """Fixed zone at bottom-center representing the dashcam car's front bumper."""

    @staticmethod
    def bounds(fw, fh):
        """Returns (x1, y1, x2, y2) of the ego zone."""
        zw = int(fw * Config.EGO_ZONE_W)
        zh = int(fh * Config.EGO_ZONE_H)
        cx = fw // 2
        y2 = int(fh * Config.EGO_ZONE_Y)
        return cx - zw // 2, y2 - zh, cx + zw // 2, y2

    @staticmethod
    def overlaps(vx1, vy1, vx2, vy2, ex1, ey1, ex2, ey2) -> bool:
        return not (vx2 < ex1 or vx1 > ex2 or vy2 < ey1 or vy1 > ey2)

    @staticmethod
    def check(vehicles, fw, fh):
        """Return list of vehicles whose bbox overlaps the ego zone."""
        ex1, ey1, ex2, ey2 = EgoZone.bounds(fw, fh)
        return [v for v in vehicles
                if v.get('is_vehicle', True) and
                EgoZone.overlaps(v['x1'], v['y1'], v['x2'], v['y2'], ex1, ey1, ex2, ey2)]

    @staticmethod
    def fault(v, fw):
        """Direction the offending vehicle came from."""
        spd = v.get('speed', 0)
        spd_s = f"{spd:.0f} km/h" if spd < Config.MAX_SPEED else "high speed"
        cx_v   = v['cx']
        cx_ego = fw // 2
        margin = fw * 0.12
        if cx_v < cx_ego - margin:
            direction = "from the left"
        elif cx_v > cx_ego + margin:
            direction = "from the right"
        else:
            direction = "head-on"
        return {
            'at_fault'      : f"V{v['id']}",
            'reason'        : f"V{v['id']} struck your car {direction} ({spd_s})",
            'collision_type': f'dashcam-{direction.replace(" ", "-")}',
            'v1'            : v['id'],
            'v2'            : 'ego',
            'v1_speed'      : spd,
            'v2_speed'      : 0,
        }


# ============================================================================
# DISPLAY
# ============================================================================

class Display:
    def __init__(self, on=True):
        self.on = on

    def draw(self, frame, vehicles, is_crash, cnn_prob, min_dist,
             fault_info=None, dashcam=False, ego_hits=None):
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

        # --- Permanent dashcam ego zone ---
        if dashcam:
            ex1, ey1, ex2, ey2 = EgoZone.bounds(w, h)
            hit_ids = {v['id'] for v in (ego_hits or [])}
            zone_color = (0, 0, 255) if ego_hits else (0, 200, 255)
            cv2.rectangle(out, (ex1, ey1), (ex2, ey2), zone_color, 3)
            cv2.putText(out, "YOUR CAR",
                        (ex1, ey1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone_color, 2)
            # Draw line from each hitting vehicle to ego zone
            for v in (ego_hits or []):
                cx_v, cy_v = v['cx'], v['cy']
                cx_e = (ex1 + ex2) // 2
                cy_e = (ey1 + ey2) // 2
                cv2.line(out, (cx_v, cy_v), (cx_e, cy_e), (0, 0, 255), 2)

        # Info bar at top
        bar = np.zeros((90, w, 3), dtype=np.uint8)
        n_veh = sum(1 for v in vehicles if v.get('is_vehicle', True))
        n_obj = len(vehicles) - n_veh
        cnn_s = f"{cnn_prob:.2f}" if cnn_prob >= 0 else "warming up"
        spd_s = f"{max_spd:.0f}km/h" if n_veh else "--"
        if dashcam:
            mode_s = "DASHCAM"
            cv2.putText(bar, f"[{mode_s}] Vehicles:{n_veh}  Objects:{n_obj}  |  Speed:{spd_s}  |  CNN:{cnn_s}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 200, 255), 2)
        else:
            md_s = f"{min_dist:.2f}m" if min_dist < 999 else "inf"
            cv2.putText(bar, f"Vehicles:{n_veh}  Objects:{n_obj}  |  Dist:{md_s}  Speed:{spd_s}  |  CNN:{cnn_s}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
        if fault_info:
            cv2.putText(bar, f"Fault: {fault_info['reason'][:60]}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

        if is_crash:
            cv2.rectangle(bar, (0, 62), (w, 90), (0, 0, 220), -1)
            cv2.putText(bar, "  CRASH DETECTED  ",
                        (w // 2 - 130, 84),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        return cv2.vconcat([bar, out])

    def show(self, frame, title="Crash Detection v16.0") -> bool:
        if not self.on:
            return True
        try:
            cv2.imshow(title, frame)
            return not ((cv2.waitKey(1) & 0xFF) == ord('q'))
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
    def __init__(self, save_output=False, dashcam=False):
        Log.head("CRASH DETECTION v16.0 — ENHANCED")
        Config.OUTPUT_DIR.mkdir(exist_ok=True)

        self.detector    = Detector()
        self.neural      = NeuralCrashDetector()
        self.save_output = save_output
        self.dashcam     = dashcam
        if dashcam:
            Log.ok("Dashcam mode ON — ego zone active")

    def _make_pipeline(self):
        return Tracker(), SpeedEstimator(), RuleCollision(), FaultDetector()

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

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur    = total / fps

        Log.head(f"PROCESSING: {path.name}")
        Log.ok(f"Resolution: {orig_w}x{orig_h} @ {fps:.0f}fps")
        Log.ok(f"Duration: {dur:.1f}s ({total} frames)")

        tracker, speed_est, rule_col, fault_det = self._make_pipeline()
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

                dets    = self.detector.detect(frame)
                tracked = tracker.update(dets)
                tracked = speed_est.update(tracked, fps)
                vehs    = [v for v in tracked if v.get('is_vehicle', True)]
                rule    = rule_col.process(vehs)
                cnn     = self.neural.predict(frame)
                fh, fw  = frame.shape[:2]

                if cnn >= 0:
                    max_cnn = max(max_cnn, cnn)

                ego_hits   = []
                fault_info = None

                if self.dashcam:
                    # Dashcam: ego zone is the only crash trigger
                    ego_hits = EgoZone.check(vehs, fw, fh)
                    is_crash = len(ego_hits) > 0
                    if is_crash:
                        neural_confirmed = True
                        fault_info = EgoZone.fault(ego_hits[0], fw)
                        if first_fault is None:
                            first_fault = fault_info
                else:
                    # Standard: CNN gates verdict, require ≥2 vehicles
                    if cnn >= 0:
                        if (cnn >= Config.CNN_THRESH) and (len(vehs) >= 2):
                            neural_confirmed = True
                            is_crash = True
                        else:
                            is_crash = False
                    else:
                        is_crash = rule['is_sustained'] and (len(vehs) >= 2)

                    if is_crash and len(vehs) >= 2:
                        fault_info = fault_det.analyze(vehs)
                        if fault_info and first_fault is None:
                            first_fault = fault_info

                frame_count += 1
                if is_crash:
                    crash_frames += 1
                    if self.save_output and len(os.listdir(output_dir)) < 20:
                        cv2.imwrite(str(output_dir / f"crash_{fn+1:04d}.jpg"), frame)

                min_dist_ever  = min(min_dist_ever, rule['min_dist'])
                max_speed_ever = max(max_speed_ever,
                                     max((v.get('speed', 0) for v in vehs), default=0))

                if display:
                    df = disp.draw(frame, tracked, is_crash, cnn, rule['min_dist'],
                                   fault_info, dashcam=self.dashcam, ego_hits=ego_hits)
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
        fps = Config.CAMERA_FPS

        tracker, speed_est, rule_col, fault_det = self._make_pipeline()
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

                dets    = self.detector.detect(frame)
                tracked = tracker.update(dets)
                tracked = speed_est.update(tracked, fps)
                vehs    = [v for v in tracked if v.get('is_vehicle', True)]
                rule    = rule_col.process(vehs)
                cnn     = self.neural.predict(frame)
                fh, fw  = frame.shape[:2]

                if cnn >= 0:
                    max_cnn = max(max_cnn, cnn)

                ego_hits   = []
                fault_info = None

                if self.dashcam:
                    # Dashcam mode: ego zone is the only crash trigger
                    ego_hits = EgoZone.check(vehs, fw, fh)
                    is_crash = len(ego_hits) > 0
                    if is_crash:
                        neural_confirmed = True
                        fault_info = EgoZone.fault(ego_hits[0], fw)
                        if first_fault is None:
                            first_fault = fault_info
                else:
                    if cnn >= 0:
                        is_crash = (cnn >= Config.CNN_THRESH) and (len(vehs) >= 2)
                        if is_crash:
                            neural_confirmed = True
                    else:
                        is_crash = rule['is_sustained'] and (len(vehs) >= 2)

                    if is_crash and len(vehs) >= 2:
                        fault_info = fault_det.analyze(vehs)
                        if fault_info and first_fault is None:
                            first_fault = fault_info

                frame_count += 1
                if is_crash:
                    crash_frames += 1

                min_dist_ever  = min(min_dist_ever, rule['min_dist'])
                max_speed_ever = max(max_speed_ever,
                                     max((v.get('speed', 0) for v in vehs), default=0))

                df = disp.draw(frame, tracked, is_crash, cnn, rule['min_dist'],
                               fault_info, dashcam=self.dashcam, ego_hits=ego_hits)

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
    parser.add_argument('--dashcam',    action='store_true', help='Dashcam mode: ego zone + CNN-only crash detection')
    parser.add_argument('--record',     action='store_true', help='Record annotated webcam video to file')
    parser.add_argument('--save-output',action='store_true', help='Save crash frames as images')
    parser.add_argument('--no-display', action='store_true', help='Suppress video window')
    parser.add_argument('--max-frames', type=int,  default=500, help='Max frames to process')
    args = parser.parse_args()

    engine = CrashDetectionEnhanced(save_output=args.save_output, dashcam=args.dashcam)

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
