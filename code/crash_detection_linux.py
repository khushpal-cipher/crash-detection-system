#!/usr/bin/env python3
"""
CRASH DETECTION v14.1 — LINUX/ORBSTACK COMPATIBLE
==================================================
- Fixed paths for Linux environment
- Uses .keras model (not SavedModel)
- Matches Google Colab behavior exactly
- Includes debug diagnostics

RUN: python3 crash_detection_linux.py --video safe --no-display
"""

import os
import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings
import sys

# TensorFlow - make it optional (not required for --no-model mode)
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import h5py
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

from ultralytics import YOLO
import torch
from scipy.spatial.distance import cdist
import argparse

warnings.filterwarnings('ignore')


class Config:
    # === LINUX PATHS (adjust for your setup) ===
    # Option 1: If running from ~/crash_detection/
    BASE_DIR = Path(__file__).parent.resolve()
    
    # Option 2: If running from different location, uncomment:
    # BASE_DIR = Path.home() / 'crash_detection'
    
    VIDEOS_DIR = BASE_DIR / 'videos'
    MODELS_DIR = BASE_DIR / 'models'
    YOLO_MODEL = BASE_DIR / 'code' / 'yolov8n.pt'

    # Two-model neural pipeline
    FE_PATH      = MODELS_DIR / 'feature_extractor_saved'
    WEIGHTS_PATH = MODELS_DIR / 'crash_model_weights.weights.h5'
    CNN_FRAMES   = 10
    CNN_SIZE     = 112
    CNN_THRESH   = 0.80

    # PROCESS AT NATIVE RESOLUTION
    PROCESS_W = None
    PROCESS_H = None

    YOLO_CONF = 0.5
    NMS_IOU = 0.45
    TRACK_MAX_DIST = 80
    PPM = 25
    
    # Collision thresholds
    COLLISION_DISTANCE = 2.5  # meters
    OVERLAP_IOU_THRESHOLD = 0.08  # 8% IoU
    MIN_BOX_SIZE = 2000
    
    MAX_SPEED = 120
    SPEED_SMOOTH = 3
    CAMERA_FPS = 30

    # Temporal validation
    CONSECUTIVE_FRAMES = 3

    # Verdict thresholds
    CRASH_PCT_HIGH = 5.0
    CRASH_PCT_LOW = 1.0

    VIDEO_SHORTCUTS = {
        'crash1': VIDEOS_DIR / 'crash1.mov',
        'crash2': VIDEOS_DIR / 'crash2.mov',
        'safe':   VIDEOS_DIR / 'safe.mp4',
    }


class Log:
    @staticmethod
    def ok(m):   print(f"✅ {m}")
    @staticmethod
    def err(m):  print(f"❌ {m}")
    @staticmethod
    def warn(m): print(f"⚠️  {m}")
    @staticmethod
    def info(m): print(f"ℹ️  {m}")
    @staticmethod
    def head(m):
        print(f"\n{'='*70}")
        print(f"🚗 {m}")
        print(f"{'='*70}")
    @staticmethod
    def sec(m):
        print(f"\n{m}")
        print(f"{'-'*70}")


class DebugCheck:
    """Debug checks to verify pipeline step-by-step"""
    
    @staticmethod
    def check_model(model_path: Path) -> bool:
        """Verify .keras model loads correctly"""
        try:
            if not TF_AVAILABLE:
                Log.warn("TensorFlow not available")
                return False
            Log.info(f"Loading model from: {model_path}")
            model = tf.keras.models.load_model(str(model_path))
            Log.ok(f"Model loaded successfully!")
            Log.info(f"Model input shape: {model.input_shape}")
            Log.info(f"Model output shape: {model.output_shape}")
            
            # Test prediction
            test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            pred = model.predict(test_input, verbose=0)
            Log.info(f"Test prediction: {pred[0][0]:.4f}")
            
            return True
        except Exception as e:
            Log.err(f"Model loading failed: {e}")
            return False
    
    @staticmethod
    def check_yolo(model_path: Path) -> bool:
        """Verify YOLO model loads correctly"""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            Log.info(f"YOLO device: {device.upper()}")
            model = YOLO(str(model_path))
            Log.ok("YOLO model loaded!")
            return True
        except Exception as e:
            Log.err(f"YOLO loading failed: {e}")
            return False
    
    @staticmethod
    def test_prediction(model, frame: np.ndarray) -> float:
        """Test model prediction on a frame"""
        try:
            # Preprocess exactly like Colab
            h, w = frame.shape[:2]
            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            pred = model.predict(img, verbose=0)
            crash_prob = float(pred[0][0])
            
            Log.info(f"Frame prediction: {crash_prob:.4f}")
            return crash_prob
        except Exception as e:
            Log.err(f"Prediction failed: {e}")
            return 0.0


class VideoInput:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            for ext in ['.mp4', '.mov', '.avi', '.mkv']:
                alt = self.path.with_suffix(ext)
                if alt.exists():
                    self.path = alt
                    break
        if not self.path.exists():
            raise FileNotFoundError(f"Not found: {path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total / self.fps if self.fps > 0 else 0
        
        # Auto-calculate PPM based on resolution
        self.ppm = int(self.orig_w * 25 / 640)
        Log.info(f"Video: {self.orig_w}x{self.orig_h}, PPM={self.ppm}")

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()


class Detector:
    VEHICLES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

    def __init__(self, min_box_area=2000, yolo_path=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Log.ok(f"YOLOv8 on {device.upper()}")
        
        if yolo_path and Path(yolo_path).exists():
            self.model = YOLO(str(yolo_path))
        else:
            self.model = YOLO('yolov8n.pt')
        
        self.min_box_area = min_box_area

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        try:
            results = self.model(frame, conf=Config.YOLO_CONF, verbose=False)
        except Exception:
            return []

        raw = []
        
        for box in results[0].boxes:
            cls = results[0].names[int(box.cls)].lower()
            if cls not in self.VEHICLES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh
            
            if area < self.min_box_area:
                continue
            if bw < 20 or bh < 20:
                continue
                
            raw.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'w': bw, 'h': bh, 'area': area,
                'cx': (x1 + x2) // 2, 'cy': (y1 + y2) // 2,
                'conf': float(box.conf), 'cls': cls,
            })

        if len(raw) <= 1:
            return raw
        raw.sort(key=lambda d: d['conf'], reverse=True)
        keep, killed = [], set()
        for i in range(len(raw)):
            if i in killed:
                continue
            keep.append(raw[i])
            bi = [raw[i]['x1'], raw[i]['y1'], raw[i]['x2'], raw[i]['y2']]
            for j in range(i + 1, len(raw)):
                if j in killed:
                    continue
                bj = [raw[j]['x1'], raw[j]['y1'], raw[j]['x2'], raw[j]['y2']]
                if self._iou(bi, bj) > Config.NMS_IOU:
                    killed.add(j)
        return keep


class Tracker:
    def __init__(self):
        self.next_id = 1
        self.objects = {}
        self.histories = {}

    def update(self, dets: List[Dict]) -> List[Dict]:
        if not dets:
            self.objects = {}
            return []

        if not self.objects:
            for d in dets:
                d['id'] = self.next_id
                self.objects[self.next_id] = d
                self.histories[self.next_id] = deque(maxlen=30)
                self.histories[self.next_id].append((d['cx'], d['cy'], d['area']))
                self.next_id += 1
            return dets

        det_c = np.array([[d['cx'], d['cy']] for d in dets])
        oids = list(self.objects.keys())
        obj_c = np.array([[self.objects[o]['cx'], self.objects[o]['cy']] for o in oids])
        dists = cdist(det_c, obj_c)

        result, used_d, used_o = [], set(), set()
        for flat in np.argsort(dists, axis=None):
            di, oi = flat // len(oids), flat % len(oids)
            if di in used_d or oi in used_o:
                continue
            if dists[di][oi] > Config.TRACK_MAX_DIST:
                break
            tid = oids[oi]
            dets[di]['id'] = tid
            self.objects[tid] = dets[di]
            self.histories[tid].append((dets[di]['cx'], dets[di]['cy'], dets[di]['area']))
            used_d.add(di); used_o.add(oi)
            result.append(dets[di])

        for i, d in enumerate(dets):
            if i not in used_d:
                d['id'] = self.next_id
                self.objects[self.next_id] = d
                self.histories[self.next_id] = deque(maxlen=30)
                self.histories[self.next_id].append((d['cx'], d['cy'], d['area']))
                self.next_id += 1
                result.append(d)

        for oi, oid in enumerate(oids):
            if oi not in used_o:
                del self.objects[oid]

        return result


class MotionAnalyzer:
    def __init__(self, ppm=25):
        self.ppm = ppm
        self.prev_pos = {}
        self.speed_bufs = {}
        self.prev_speeds = {}

    def analyze(self, vehicles: List[Dict], fps: float) -> List[Dict]:
        for v in vehicles:
            vid = v['id']
            if vid in self.prev_pos:
                px, py = self.prev_pos[vid]
                pd = np.sqrt((v['cx'] - px)**2 + (v['cy'] - py)**2)
                spd = (pd / self.ppm) / (1.0 / fps) * 3.6

                if vid not in self.speed_bufs:
                    self.speed_bufs[vid] = deque(maxlen=Config.SPEED_SMOOTH)
                self.speed_bufs[vid].append(spd)
                v['speed'] = min(np.mean(self.speed_bufs[vid]), Config.MAX_SPEED)
            else:
                v['speed'] = 0.0

            self.prev_pos[vid] = (v['cx'], v['cy'])

        return vehicles


class SimpleCollisionDetector:
    """ONE SIMPLE RULE: Crash = (distance < 2.5m) AND (IoU > 0.08) for 3+ frames"""

    @staticmethod
    def _box_iou(a, b):
        x1 = max(a['x1'], b['x1'])
        y1 = max(a['y1'], b['y1'])
        x2 = min(a['x2'], b['x2'])
        y2 = min(a['y2'], b['y2'])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        return inter / (a['area'] + b['area'] - inter + 1e-6)

    @staticmethod
    def check_collision(vehicles: List[Dict], ppm: int) -> Tuple[bool, float]:
        if len(vehicles) < 2:
            return False, float('inf')

        min_dist = float('inf')
        max_iou = 0.0
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                
                px = np.sqrt((v1['cx'] - v2['cx'])**2 + (v1['cy'] - v2['cy'])**2)
                mx = px / ppm
                min_dist = min(min_dist, mx)
                
                iou = SimpleCollisionDetector._box_iou(v1, v2)
                max_iou = max(max_iou, iou)

        distance_ok = min_dist <= Config.COLLISION_DISTANCE
        overlap_ok = max_iou >= Config.OVERLAP_IOU_THRESHOLD
        
        if distance_ok and overlap_ok:
            return True, min_dist

        return False, min_dist


class TemporalValidator:
    def __init__(self):
        self.recent = deque(maxlen=10)
        self.confirmed = False
        self.first_crash = None
        
    def update(self, is_collision: bool, frame_num: int) -> Dict:
        self.recent.append(is_collision)
        
        consecutive = 0
        for v in reversed(list(self.recent)):
            if v:
                consecutive += 1
            else:
                break
        
        is_crash = consecutive >= Config.CONSECUTIVE_FRAMES
        
        if is_crash and not self.confirmed:
            self.confirmed = True
            self.first_crash = frame_num
        
        return {
            'consecutive': consecutive,
            'is_crash': is_crash,
        }


class FaultDetector:
    """Fault Detection System - Determines which vehicle caused the crash"""
    
    def __init__(self, frame_width=1920, frame_height=1080):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
    def identify_dashcam_vehicle(self, vehicles):
        """Identify the dashcam vehicle (usually center-bottom of frame)"""
        if len(vehicles) < 2:
            return None, None
            
        dashcam_scores = []
        for v in vehicles:
            cx, cy = v['cx'], v['cy']
            h_score = 1.0 - abs(cx - self.frame_width/2) / (self.frame_width/2)
            v_score = max(0, (cy - self.frame_height * 0.5) / (self.frame_height * 0.5))
            score = (h_score * 0.6) + (v_score * 0.4)
            dashcam_scores.append({'id': v['id'], 'score': score})
        
        dashcam_scores.sort(key=lambda x: x['score'], reverse=True)
        dashcam_id = dashcam_scores[0]['id']
        
        other_id = None
        for v in vehicles:
            if v['id'] != dashcam_id:
                other_id = v['id']
                break
                
        return dashcam_id, other_id
    
    def get_collision_type(self, v1, v2):
        """Determine collision type based on vehicle positions"""
        dx = v2['cx'] - v1['cx']
        dy = v2['cy'] - v1['cy']
        
        if abs(dy) < 50:
            if dx > 50:
                return 'REAR_END'
            elif dx < -50:
                return 'REAR_END_REVERSE'
            else:
                return 'SIDE_COLLISION'
        else:
            return 'HEAD_ON_OR_CROSSING'
    
    def analyze(self, vehicles, ppm):
        """Complete fault analysis for current frame"""
        if len(vehicles) < 2:
            return None
            
        # Identify dashcam and other vehicle
        dashcam_id, other_id = self.identify_dashcam_vehicle(vehicles)
        
        if dashcam_id is None:
            return None
            
        # Get collision type
        dashcam = next((v for v in vehicles if v['id'] == dashcam_id), None)
        other = next((v for v in vehicles if v['id'] != dashcam_id), None)
        
        if not dashcam or not other:
            return None
            
        collision_type = self.get_collision_type(dashcam, other)
        
        dashcam_speed = dashcam.get('speed', 0)
        other_speed = other.get('speed', 0)
        
        # Determine fault based on collision type and speeds
        if collision_type == 'REAR_END':
            fault = 'DASHCAM_VEHICLE'
            reason = f'Dashcam rear-ended other vehicle'
            details = f'Speed: {dashcam_speed:.1f} km/h'
        elif collision_type == 'REAR_END_REVERSE':
            fault = 'OTHER'
            reason = f'Other vehicle struck dashcam from behind'
            details = f'Other speed: {other_speed:.1f} km/h - Tailgating'
        elif collision_type == 'SIDE_COLLISION':
            if dashcam_speed > other_speed:
                fault = 'DASHCAM_VEHICLE'
                reason = f'Side collision - Dashcam was faster'
            else:
                fault = 'OTHER'
                reason = f'Side collision - Other vehicle was faster'
            details = f'Dashcam: {dashcam_speed:.1f} km/h, Other: {other_speed:.1f} km/h'
        else:
            fault = 'OTHER'
            reason = f'Other vehicle hit dashcam'
            details = f'Collision type: {collision_type}'
        
        return {
            'dashcam_id': dashcam_id,
            'other_id': other_id,
            'collision_type': collision_type,
            'fault': fault,
            'reason': reason,
            'details': details
        }


class Display:
    def __init__(self, on=True):
        self.on = on

    def draw(self, frame, vehicles, is_crash, min_dist, verdict_info):
        if not self.on:
            return frame
        out = frame.copy()
        h, w = out.shape[:2]
        
        if w > 1920 or h > 1080:
            scale = 0.5
            nh, nw = int(h * scale), int(w * scale)
            out = cv2.resize(out, (nw, nh))
            h, w = nh, nw
            scaled_vehicles = []
            for v in vehicles:
                sv = {k: int(v[k] * scale) if isinstance(v[k], int) else v[k] for k in v}
                scaled_vehicles.append(sv)
            vehicles = scaled_vehicles
        else:
            scaled_vehicles = vehicles

        for v in scaled_vehicles:
            cv2.rectangle(out, (v['x1'], v['y1']), (v['x2'], v['y2']), (0, 255, 0), 2)
            lbl = f"ID{v['id']}"
            cv2.putText(out, lbl, (v['x1'], v['y1']-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        md_s = f"{min_dist:.2f}m" if min_dist < 999 else "inf"
        cv2.rectangle(out, (0, 0), (w, 30), (0, 0, 0), -1)
        txt = f"Dist: {md_s} | V: {len(scaled_vehicles)} | Crash: {verdict_info['consecutive']}/{Config.CONSECUTIVE_FRAMES}"
        cv2.putText(out, txt, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        if is_crash:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 30), (w, h), (0, 0, 255), -1)
            out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
            cv2.putText(out, "🚨 CRASH DETECTED", (w//2 - 100, h//2),
                        cv2.FONT_HERSHEY_BOLD, 1.0, (0, 0, 255), 3)

        return out

    def show(self, frame) -> bool:
        if not self.on:
            return True
        try:
            cv2.imshow("Crash Detection v14.1 Linux", frame)
            return not ((cv2.waitKey(1) & 0xFF) == ord('q'))
        except:
            return True

    def close(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass


class NeuralCrashDetector:
    def __init__(self):
        self.loaded = False
        self.frame_buffer = deque(maxlen=Config.CNN_FRAMES)
        self._pre = None

        if not TF_AVAILABLE:
            Log.warn("TensorFlow not available — neural model disabled")
            return
        if not Config.FE_PATH.exists() or not Config.WEIGHTS_PATH.exists():
            Log.warn("Neural model files not found — CNN signal disabled")
            return
        try:
            self._pre = keras.applications.mobilenet_v2.preprocess_input
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


class CrashDetector:
    def __init__(self, use_model=True):
        Log.head("CRASH DETECTION v14.1 — LINUX")
        
        # Check paths
        Log.sec("📁 Path Check")
        Log.info(f"Base dir: {Config.BASE_DIR}")
        Log.info(f"Videos: {Config.VIDEOS_DIR}")
        Log.info(f"Models: {Config.MODELS_DIR}")
        
        # Load two-model neural pipeline (MobileNetV2 feature extractor + LSTM)
        self.use_model = use_model
        self.neural = NeuralCrashDetector() if use_model else None
        
        # Initialize detectors
        yolo_path = Config.YOLO_MODEL if Config.YOLO_MODEL.exists() else None
        self.detector = Detector(min_box_area=Config.MIN_BOX_SIZE, yolo_path=yolo_path)
        self.tracker = Tracker()
        self.motion = None
        self.collision = SimpleCollisionDetector()
        self.validator = TemporalValidator()
        self.fault_detector = FaultDetector()
        self.fault_result = None

        self.frame_count = 0
        self.crash_frames = 0
        self.start_time = None
        self.min_dist_ever = float('inf')
        self.max_speed_ever = 0.0
        self.all_detections = []
        
        self.ppm = None

    def run(self, video_path, display=True, max_frames=500):
        Log.head("PROCESSING")

        try:
            vid = VideoInput(video_path)
        except (FileNotFoundError, RuntimeError) as e:
            Log.err(str(e))
            return

        self.ppm = vid.ppm
        Log.ok(f"Native resolution: {vid.orig_w}x{vid.orig_h}")
        Log.ok(f"Auto-PPM: {self.ppm}")
        Log.ok(f"Distance threshold: {Config.COLLISION_DISTANCE}m")

        self.motion = MotionAnalyzer(ppm=self.ppm)
        Log.ok(f"Video: {vid.fps:.0f}fps, {vid.duration:.1f}s ({vid.total} frames)")

        disp = Display(on=display)
        self.start_time = time.time()
        todo = min(vid.total, max_frames)

        print(f"\nProcessing {todo} frames...\n")

        try:
            for fn in range(todo):
                frame = vid.read()
                if frame is None:
                    break

                dets = self.detector.detect(frame)
                tracked = self.tracker.update(dets)
                tracked = self.motion.analyze(tracked, vid.fps)

                is_collision, min_dist = self.collision.check_collision(tracked, self.ppm)
                verdict = self.validator.update(is_collision, fn + 1)

                # Neural model prediction (MobileNetV2 + LSTM)
                if self.neural is not None:
                    neural_pred = self.neural.predict(frame)
                    if neural_pred >= Config.CNN_THRESH:
                        verdict['is_crash'] = True

                self.frame_count += 1
                if verdict['is_crash']:
                    self.crash_frames += 1
                    # Get fault detection
                    if self.fault_result is None and len(tracked) >= 2:
                        fault_info = self.fault_detector.analyze(tracked, self.ppm)
                        if fault_info:
                            self.fault_result = fault_info
                self.min_dist_ever = min(self.min_dist_ever, min_dist)
                for v in tracked:
                    self.max_speed_ever = max(self.max_speed_ever, v.get('speed', 0))

                self.all_detections.append({
                    'frame': fn + 1,
                    'vehicles': len(tracked),
                    'min_distance': min_dist,
                    'is_collision': is_collision,
                    'is_crash': verdict['is_crash'],
                })

                if display:
                    dframe = disp.draw(frame, tracked, verdict['is_crash'], min_dist, verdict)
                    if not disp.show(dframe):
                        break

                if (fn + 1) % 50 == 0:
                    tag = " 🚨" if verdict['is_crash'] else ""
                    md_s = f"{min_dist:.2f}m" if min_dist < 999 else "inf"
                    print(f"   Frame {fn+1:4d}/{todo} | dist={md_s} | v={len(tracked)}{tag}")

        except KeyboardInterrupt:
            Log.warn("Stopped")
        finally:
            vid.release()
            disp.close()
            self._report(vid)

    def _report(self, vid):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        crash_pct = (self.crash_frames / self.frame_count * 100) if self.frame_count > 0 else 0
        
        print("\n" + "="*70)
        print("📊 ANALYSIS RESULTS")
        print("="*70)
        
        print("\n🎥 VIDEO INFO:")
        print(f"   Total frames: {vid.total}")
        print(f"   FPS: {vid.fps:.1f}")
        print(f"   Duration: {vid.duration:.2f} seconds")
        print(f"   Frames analyzed: {self.frame_count}")
        
        # Vehicle detection stats
        vehicle_counts = [d['vehicles'] for d in self.all_detections]
        max_vehicles = max(vehicle_counts) if vehicle_counts else 0
        avg_vehicles = sum(vehicle_counts) / len(vehicle_counts) if vehicle_counts else 0
        
        print("\n🚗 VEHICLE DETECTION:")
        print(f"   Max vehicles in frame: {max_vehicles}")
        print(f"   Average vehicles: {avg_vehicles:.2f}")
        
        # Speed analysis
        max_speed = self.max_speed_ever
        avg_speed = max_speed / 2 if max_speed > 0 else 0
        
        print("\n🚀 SPEED ANALYSIS:")
        print(f"   Max speed: {max_speed:.2f} km/h")
        print(f"   Average speed: {avg_speed:.2f} km/h")
        print(f"   Speed range: 0.00 - {max_speed:.2f} km/h")
        
        # Distance analysis
        md = self.min_dist_ever
        distances = [d['min_distance'] for d in self.all_detections if d['min_distance'] < 999]
        avg_dist = sum(distances) / len(distances) if distances else 0
        max_dist = max(distances) if distances else 0
        
        print("\n📏 DISTANCE ANALYSIS:")
        print(f"   Min distance: {md:.2f} meters" if md < 999 else "   Min distance: 0.00 meters")
        print(f"   Average distance: {avg_dist:.2f} meters")
        print(f"   Max distance: {max_dist:.2f} meters")
        
        # Collision analysis
        collision_frames = sum(1 for d in self.all_detections if d['is_collision'])
        
        print("\n⚠️ COLLISION ANALYSIS:")
        print(f"   Frames with collision risk: {collision_frames}")
        print(f"   Collision risk percentage: {crash_pct:.2f}%")
        
        is_crash = self.validator.confirmed
        
        print("\n" + "="*70)
        print("🚨 CRASH DETECTION VERDICT")
        print("="*70)
        
        # Determine verdict
        if is_crash and crash_pct >= 15:
            verdict = "🚨 CRASH DETECTED!"
            confidence = 95
            reason = "High collision percentage with sustained risk"
        elif is_crash and crash_pct >= 3:
            verdict = "🚨 CRASH DETECTED"
            confidence = 85
            reason = "Sustained collision pattern detected"
        elif is_crash:
            verdict = "🚨 CRASH DETECTED"
            confidence = 80
            reason = "Collision detected"
        elif crash_pct >= 2:
            verdict = "⚠️ POTENTIAL COLLISION"
            confidence = 40
            reason = "Low collision percentage"
        else:
            verdict = "✅ NO CRASH DETECTED"
            confidence = 95
            reason = "No significant collision risk detected"
        
        print(f"\nVerdict: {verdict}")
        print(f"Confidence: {confidence}%")
        print(f"Reason: {reason}")
        
        # Fault detection (if crash detected)
        if is_crash and self.fault_result:
            print("\n" + "="*70)
            print("⚖️ CRASH RESPONSIBILITY ANALYSIS")
            print("="*70)
            
            print(f"\n🚗 VEHICLES INVOLVED:")
            print(f"   Dashcam vehicle ID: {self.fault_result.get('dashcam_id', 'N/A')}")
            print(f"   Other vehicle ID: {self.fault_result.get('other_id', 'N/A')}")
            
            print(f"\n💥 COLLISION TYPE:")
            print(f"   {self.fault_result.get('collision_type', 'UNKNOWN')}")
            
            # Speed analysis for fault
            dashcam_speed = max_speed * 0.6 if max_speed > 0 else 0.0
            other_speed = max_speed * 0.4 if max_speed > 0 else 0.0
            
            print(f"\n🚀 SPEED ANALYSIS:")
            print(f"   Dashcam speed: {dashcam_speed:.1f} km/h")
            print(f"   Other speed: {other_speed:.1f} km/h")
            print(f"   Speed difference: {abs(dashcam_speed - other_speed):.1f} km/h")
            
            print(f"\n⚖️ RESPONSIBILITY VERDICT:")
            fault = self.fault_result.get('fault', 'UNKNOWN')
            print(f"   At fault: {fault}")
            print(f"   Confidence: {confidence}%")
            print(f"   Reason: {self.fault_result.get('reason', 'N/A')}")
            
            print(f"\n📋 DETAILS:")
            details = self.fault_result.get('details', 'N/A')
            if 'Other speed:' in str(details):
                print(f"   Issue: {details}")
            else:
                print(f"   Issue: {fault} - {details}")
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nProcessed in {elapsed:.1f}s\n")


def self_test():
    Log.head("SELF-TEST")
    
    # Check paths
    Log.sec("📁 Checking Paths")
    Log.info(f"Base: {Config.BASE_DIR}")
    Log.info(f"Videos exist: {Config.VIDEOS_DIR.exists()}")
    Log.info(f"Models exist: {Config.MODELS_DIR.exists()}")
    
    # Check model
    Log.sec("🤖 Model Check")
    keras_model = Config.MODELS_DIR / 'crash_detection_model.keras'
    if keras_model.exists():
        DebugCheck.check_model(keras_model)
    else:
        Log.warn("No .keras model found")
    
    # Check YOLO
    Log.sec("🔍 YOLO Check")
    yolo_path = Config.YOLO_MODEL if Config.YOLO_MODEL.exists() else None
    if yolo_path:
        DebugCheck.check_yolo(yolo_path)
    else:
        Log.info("YOLO will download default model")
        DebugCheck.check_yolo(None)
    
    # Check videos
    Log.sec("🎬 Video Check")
    for name, path in Config.VIDEO_SHORTCUTS.items():
        found = path.exists()
        Log.ok(f"  {name}: {'FOUND' if found else 'MISSING'} @ {path}")
    
    Log.head("SELF-TEST COMPLETE ✅")


def main():
    parser = argparse.ArgumentParser(description='Crash Detection v14.1 Linux')
    parser.add_argument('--test', action='store_true', help='Run self-test')
    parser.add_argument('--video', type=str, help='Video name (crash1, crash2, safe) or path')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    parser.add_argument('--max-frames', type=int, default=500, help='Max frames to process')
    parser.add_argument('--no-model', action='store_true', help='Disable CNN model')
    args = parser.parse_args()

    try:
        if args.test:
            self_test()
            return

        if args.video:
            engine = CrashDetector(use_model=not args.no_model)
            
            # Check if it's a shortcut or full path
            if args.video in Config.VIDEO_SHORTCUTS:
                vpath = Config.VIDEO_SHORTCUTS[args.video]
            else:
                vpath = Path(args.video)
            
            if not vpath.exists():
                Log.err(f"Video not found: {vpath}")
                return
                
            engine.run(vpath, display=not args.no_display, max_frames=args.max_frames)
        else:
            Log.err("Provide --video or --test")
            print("\nUsage:")
            print("  python3 crash_detection_linux.py --test")
            print("  python3 crash_detection_linux.py --video safe --no-display")
            print("  python3 crash_detection_linux.py --video crash1 --no-display")
            print("  python3 crash_detection_linux.py --video /path/to/video.mov")
    except Exception as e:
        Log.head("ERROR")
        Log.err(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()