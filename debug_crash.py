# 🔍 CRASH DETECTION DEBUGGING SCRIPT
# Run this to see EXACTLY why crashes are not detected

"""
This script processes a crash video and prints detailed debug info
for every single frame to identify the exact failure point.

Usage:
    python3 debug_crash.py --video ~/Desktop/crash_detection/videos/crash1.mov
"""

import tensorflow as tf
import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
import torch
from scipy.spatial.distance import cdist
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

PIXELS_PER_METER = 40
MAX_SPEED = 120
SPEED_SMOOTH_FRAMES = 3
COLLISION_DISTANCE_THRESHOLD = 1.5
CRASH_THRESHOLD_HIGH = 0.65
CRASH_THRESHOLD_MEDIUM = 0.40
MODEL_PATH = Path.home() / 'Desktop' / 'crash_detection' / 'models' / 'crash_detection_model'


# ============================================================================
# MINIMAL IMPLEMENTATIONS FOR DEBUGGING
# ============================================================================

class VehicleDetector:
    def __init__(self):
        print("Loading YOLOv8...")
        self.model = YOLO('yolov8n.pt')
    
    def detect_vehicles(self, frame, conf_threshold=0.5):
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []
        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            conf = float(detection.conf)
            cls_id = int(detection.cls)
            class_name = results[0].names[cls_id]
            
            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
            if class_name.lower() in vehicle_classes:
                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'w': x2 - x1, 'h': y2 - y1,
                    'conf': conf, 'class': class_name,
                    'center_x': (x1 + x2) // 2,
                    'center_y': (y1 + y2) // 2
                })
        return detections


class SimpleCentroidTracker:
    def __init__(self, max_distance=50):
        self.max_distance = max_distance
        self.next_id = 1
        self.tracked_vehicles = {}
        self.vehicle_histories = {}
    
    def track(self, detections, frame_num=0):
        if len(detections) == 0:
            return []
        
        current_centroids = np.array([[det['center_x'], det['center_y']] for det in detections])
        
        if len(self.tracked_vehicles) == 0:
            for i, det in enumerate(detections):
                vid = self.next_id
                self.next_id += 1
                det['track_id'] = vid
                self.tracked_vehicles[vid] = det
                self.vehicle_histories[vid] = [{'x': det['center_x'], 'y': det['center_y'], 'frame': frame_num}]
            return detections
        
        prev_ids = list(self.tracked_vehicles.keys())
        prev_centroids = np.array([
            [self.tracked_vehicles[vid]['center_x'], self.tracked_vehicles[vid]['center_y']]
            for vid in prev_ids
        ])
        
        distances = cdist(current_centroids, prev_centroids)
        matched_detections = []
        used_prev_ids = set()
        
        for i, det in enumerate(detections):
            min_distance_idx = np.argmin(distances[i])
            min_distance = distances[i][min_distance_idx]
            
            if min_distance < self.max_distance and min_distance_idx not in used_prev_ids:
                prev_id = prev_ids[min_distance_idx]
                det['track_id'] = prev_id
                self.tracked_vehicles[prev_id] = det
                self.vehicle_histories[prev_id].append({
                    'x': det['center_x'], 'y': det['center_y'], 'frame': frame_num
                })
                used_prev_ids.add(min_distance_idx)
                matched_detections.append(det)
            else:
                vid = self.next_id
                self.next_id += 1
                det['track_id'] = vid
                self.tracked_vehicles[vid] = det
                self.vehicle_histories[vid] = [{'x': det['center_x'], 'y': det['center_y'], 'frame': frame_num}]
                matched_detections.append(det)
        
        for i, prev_id in enumerate(prev_ids):
            if i not in used_prev_ids:
                del self.tracked_vehicles[prev_id]
        
        return matched_detections


class SpeedEstimator:
    def __init__(self, fps=30, pixels_per_meter=40):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.prev_positions = {}
        self.speed_history = {}
    
    def estimate_speed(self, tracked_vehicles):
        vehicles_with_speed = []
        for vehicle in tracked_vehicles:
            track_id = vehicle['track_id']
            curr_x = vehicle['center_x']
            curr_y = vehicle['center_y']
            
            if track_id in self.prev_positions:
                prev_x, prev_y = self.prev_positions[track_id]
                pixel_distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                meter_distance = pixel_distance / self.pixels_per_meter
                time_per_frame = 1.0 / self.fps
                speed_ms = meter_distance / time_per_frame
                speed_kmh = speed_ms * 3.6
                speed_kmh = min(max(0, speed_kmh), MAX_SPEED)
                
                if track_id not in self.speed_history:
                    self.speed_history[track_id] = deque(maxlen=SPEED_SMOOTH_FRAMES)
                self.speed_history[track_id].append(speed_kmh)
                smoothed_speed = np.mean(list(self.speed_history[track_id]))
                vehicle['speed_kmh'] = smoothed_speed
            else:
                vehicle['speed_kmh'] = 0
            
            self.prev_positions[track_id] = (curr_x, curr_y)
            vehicles_with_speed.append(vehicle)
        return vehicles_with_speed


class DistanceEstimator:
    def __init__(self, pixels_per_meter=40):
        self.pixels_per_meter = pixels_per_meter
    
    def estimate_distance(self, vehicles):
        distances = []
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1 = vehicles[i]
                v2 = vehicles[j]
                pixel_dist = np.sqrt((v1['center_x'] - v2['center_x'])**2 + 
                                    (v1['center_y'] - v2['center_y'])**2)
                meter_dist = pixel_dist / self.pixels_per_meter
                distances.append({
                    'v1': v1['track_id'],
                    'v2': v2['track_id'],
                    'pixels': pixel_dist,
                    'meters': meter_dist
                })
        return distances


class CrashDetectionModel:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(str(model_path))
    
    def predict(self, frame):
        try:
            processed = cv2.resize(frame, (224, 224))
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            processed = processed.astype(np.float32) / 255.0
            batch = np.expand_dims(processed, axis=0)
            
            try:
                prediction = self.model.predict(batch, verbose=0)
            except:
                prediction = self.model(batch, training=False)
            
            if hasattr(prediction, 'numpy'):
                prediction = prediction.numpy()
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]
            if prediction.ndim == 2:
                crash_prob = float(prediction[0][0])
            elif prediction.ndim == 1:
                crash_prob = float(prediction[0])
            else:
                crash_prob = float(prediction)
            
            return min(max(0.0, crash_prob), 1.0)
        except Exception as e:
            return 0.0


# ============================================================================
# DEBUG COLLISION DETECTOR
# ============================================================================

class DebugCollisionDetector:
    """Collision detector with FULL debug output"""
    
    def __init__(self, distance_threshold=1.5):
        self.distance_threshold = distance_threshold
        self.prev_data = None
    
    def detect_collision(self, tracked_vehicles, distances):
        print(f"  [COLLISION CHECK] {len(tracked_vehicles)} vehicles, {len(distances)} distances")
        
        if len(tracked_vehicles) < 1 or len(distances) == 0:
            print(f"    ✗ Not enough data")
            return False, 0
        
        factors = 0
        
        # FACTOR 1: Distance below threshold
        if len(distances) > 0:
            min_distance = min([d['meters'] for d in distances])
            print(f"    Min distance: {min_distance:.2f}m (threshold: {self.distance_threshold}m)")
            if min_distance < self.distance_threshold:
                factors += 1
                print(f"      ✓ FACTOR 1: Distance below threshold")
            else:
                print(f"      ✗ Factor 1: Distance above threshold")
        
        # FACTOR 2: Distance decreasing
        if len(distances) > 0 and self.prev_data:
            min_distance = min([d['meters'] for d in distances])
            prev_dist = self.prev_data
            distance_change = min_distance - prev_dist
            print(f"    Distance change: {distance_change:.3f}m/frame")
            if distance_change < -0.1:
                factors += 1
                print(f"      ✓ FACTOR 2: Distance decreasing rapidly")
            else:
                print(f"      ✗ Factor 2: Distance not decreasing")
        
        # FACTOR 3: Bounding box OVERLAP - CRITICAL
        overlap_found = False
        for i in range(len(tracked_vehicles)):
            for j in range(i + 1, len(tracked_vehicles)):
                v1 = tracked_vehicles[i]
                v2 = tracked_vehicles[j]
                
                # Check overlap in both X and Y
                x_overlap = not (v1['x2'] < v2['x1'] or v2['x2'] < v1['x1'])
                y_overlap = not (v1['y2'] < v2['y1'] or v2['y2'] < v1['y1'])
                
                print(f"    V{v1['track_id']} vs V{v2['track_id']}:")
                print(f"      X: [{v1['x1']},{v1['x2']}] vs [{v2['x1']},{v2['x2']}] → {x_overlap}")
                print(f"      Y: [{v1['y1']},{v1['y2']}] vs [{v2['y1']},{v2['y2']}] → {y_overlap}")
                
                if x_overlap and y_overlap:
                    factors += 1
                    overlap_found = True
                    print(f"      ✓ FACTOR 3: BOUNDING BOXES OVERLAP!")
        
        if not overlap_found:
            print(f"      ✗ Factor 3: No overlap")
        
        # FACTOR 4: Speed drop
        print(f"    Speed check: {[v['speed_kmh'] for v in tracked_vehicles]}")
        
        # Store for next frame
        if len(distances) > 0:
            self.prev_data = min([d['meters'] for d in distances])
        
        print(f"    TOTAL FACTORS: {factors}/4")
        is_collision = factors >= 2  # Lowered threshold
        
        if is_collision:
            print(f"  ✓✓✓ COLLISION DETECTED ✓✓✓")
        else:
            print(f"  ✗✗✗ No collision ✗✗✗")
        
        return is_collision, factors


# ============================================================================
# MAIN DEBUG LOOP
# ============================================================================

def debug_video(video_path, max_frames=200):
    print(f"\n{'='*80}")
    print(f"🔍 DEBUGGING CRASH VIDEO: {video_path}")
    print(f"{'='*80}\n")
    
    # Initialize
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    detector = VehicleDetector()
    tracker = SimpleCentroidTracker()
    speed_estimator = SpeedEstimator(fps=fps)
    distance_estimator = DistanceEstimator()
    collision_detector = DebugCollisionDetector()
    model = CrashDetectionModel(MODEL_PATH)
    
    frame_num = 0
    crash_detected = False
    
    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\n{'─'*80}")
        print(f"FRAME {frame_num}")
        print(f"{'─'*80}")
        
        # Process
        detections = detector.detect_vehicles(frame)
        tracked = tracker.track(detections, frame_num)
        with_speed = speed_estimator.estimate_speed(tracked)
        distances = distance_estimator.estimate_distance(with_speed)
        is_collision, factors = collision_detector.detect_collision(with_speed, distances)
        model_confidence = model.predict(frame)
        
        print(f"\n  [DETECTION] Found {len(tracked)} vehicles")
        for v in with_speed:
            print(f"    V{v['track_id']}: {v['speed_kmh']:.1f}km/h at ({v['center_x']}, {v['center_y']})")
        
        print(f"\n  [DISTANCES]")
        for d in distances:
            print(f"    V{d['v1']} <-> V{d['v2']}: {d['meters']:.2f}m ({d['pixels']:.0f}px)")
        
        print(f"\n  [MODEL PREDICTION] {model_confidence:.2f} (threshold high: {CRASH_THRESHOLD_HIGH}, medium: {CRASH_THRESHOLD_MEDIUM})")
        
        # CRASH DECISION LOGIC
        print(f"\n  [CRASH DECISION]")
        if model_confidence > CRASH_THRESHOLD_HIGH:
            is_crash = True
            print(f"    ✓ Model confidence {model_confidence:.2f} > HIGH threshold {CRASH_THRESHOLD_HIGH}")
        elif is_collision and model_confidence > CRASH_THRESHOLD_MEDIUM:
            is_crash = True
            print(f"    ✓ Collision + Model {model_confidence:.2f} > MEDIUM threshold {CRASH_THRESHOLD_MEDIUM}")
        else:
            is_crash = False
            reason = []
            if model_confidence <= CRASH_THRESHOLD_HIGH:
                reason.append(f"model {model_confidence:.2f} <= {CRASH_THRESHOLD_HIGH}")
            if not is_collision or model_confidence <= CRASH_THRESHOLD_MEDIUM:
                reason.append(f"collision={is_collision}, model={model_confidence:.2f} <= {CRASH_THRESHOLD_MEDIUM}")
            print(f"    ✗ {', '.join(reason)}")
        
        if is_crash:
            print(f"\n  🚨🚨🚨 CRASH DETECTED 🚨🚨🚨")
            crash_detected = True
        
        frame_num += 1
    
    cap.release()
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {'CRASH DETECTED ✓' if crash_detected else 'NO CRASH ✗'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--max-frames', type=int, default=200)
    args = parser.parse_args()
    
    debug_video(args.video, args.max_frames)
