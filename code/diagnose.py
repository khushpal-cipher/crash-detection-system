#!/usr/bin/env python3
"""
🔍 CRASH DETECTION DIAGNOSTIC TOOL
====================================
Tests each pipeline stage independently to find exactly where the problem is.

Run: python3 diagnose.py --video crash1
     python3 diagnose.py --video safe
     python3 diagnose.py --video crash1 --save-frames

This will tell you:
  1. Is YOLO detecting vehicles correctly?
  2. Is the tracker assigning consistent IDs?
  3. Are distances calculated correctly?
  4. Are speeds realistic?
  5. Does the CNN model output anything useful?
  6. Does the scoring logic respond correctly?
"""

import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque
import argparse
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# ── Paths ──
BASE = Path.home() / 'Desktop' / 'crash_detection'
VIDEOS = BASE / 'videos'
MODELS = BASE / 'models'
DIAG_DIR = BASE / 'diagnostics'

SHORTCUTS = {
    'crash1': VIDEOS / 'crash1.mov',
    'crash2': VIDEOS / 'crash2.mov',
    'safe': VIDEOS / 'safe.mp4',
}

PROCESS_W, PROCESS_H = 640, 480
PPM = 25


def find_video(name):
    if name in SHORTCUTS:
        p = SHORTCUTS[name]
    else:
        p = Path(name)
    if p.exists():
        return p
    for ext in ['.mp4', '.mov', '.avi', '.mkv']:
        alt = p.with_suffix(ext)
        if alt.exists():
            return alt
    return None


def header(title):
    print(f"\n{'='*70}")
    print(f"🔍 {title}")
    print(f"{'='*70}")


def ok(msg):
    print(f"  ✅ {msg}")


def fail(msg):
    print(f"  ❌ {msg}")


def info(msg):
    print(f"  ℹ️  {msg}")


def warn(msg):
    print(f"  ⚠️  {msg}")


# ═══════════════════════════════════════════════════════════════
# TEST 1: VIDEO READING
# ═══════════════════════════════════════════════════════════════

def test_video(path):
    header("TEST 1: VIDEO INPUT")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"Cannot open {path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok(f"File: {path.name}")
    ok(f"Original resolution: {w}x{h}")
    ok(f"FPS: {fps}")
    ok(f"Frames: {total}")
    ok(f"Duration: {total/fps:.1f}s")

    ret, frame = cap.read()
    if not ret:
        fail("Cannot read first frame")
        cap.release()
        return None

    resized = cv2.resize(frame, (PROCESS_W, PROCESS_H))
    ok(f"Resized to: {PROCESS_W}x{PROCESS_H}")
    ok(f"Frame dtype: {resized.dtype}, shape: {resized.shape}")

    cap.release()
    return {'fps': fps, 'total': total, 'orig_w': w, 'orig_h': h}


# ═══════════════════════════════════════════════════════════════
# TEST 2: YOLO DETECTION
# ═══════════════════════════════════════════════════════════════

def test_detection(path, save_frames=False):
    header("TEST 2: YOLO VEHICLE DETECTION")

    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    ok("YOLOv8 loaded")

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sample frames evenly across the video
    sample_indices = np.linspace(0, total - 1, min(20, total), dtype=int)
    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

    frame_results = []

    if save_frames:
        os.makedirs(DIAG_DIR, exist_ok=True)

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_640 = cv2.resize(frame, (PROCESS_W, PROCESS_H))
        results = model(frame_640, conf=0.5, verbose=False)

        detections = []
        all_classes = []
        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls)].lower()
            all_classes.append(cls_name)
            if cls_name in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': (x1+x2)//2, 'cy': (y1+y2)//2,
                    'w': x2-x1, 'h': y2-y1,
                    'conf': float(box.conf),
                    'cls': cls_name,
                })

        frame_results.append({
            'frame': int(idx),
            'time': idx / fps,
            'vehicles': len(detections),
            'all_objects': len(all_classes),
            'classes': list(set(all_classes)),
            'detections': detections,
        })

        # Save annotated frame
        if save_frames:
            annotated = frame_640.copy()
            for d in detections:
                cv2.rectangle(annotated, (d['x1'], d['y1']), (d['x2'], d['y2']),
                              (0, 255, 0), 2)
                cv2.putText(annotated, f"{d['cls']} {d['conf']:.2f}",
                            (d['x1'], d['y1']-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1)
            cv2.putText(annotated, f"Frame {idx} | Vehicles: {len(detections)}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(str(DIAG_DIR / f"det_frame_{int(idx):04d}.jpg"), annotated)

    cap.release()

    # Report
    veh_counts = [r['vehicles'] for r in frame_results]

    print(f"\n  Sampled {len(frame_results)} frames across the video:\n")
    print(f"  {'Frame':>6} {'Time':>6} {'Vehicles':>9} {'All Objects':>12} {'Classes'}")
    print(f"  {'-'*60}")
    for r in frame_results:
        t = f"{r['time']:.1f}s"
        cls_str = ', '.join(r['classes'][:5])
        print(f"  {r['frame']:6d} {t:>6} {r['vehicles']:>9} {r['all_objects']:>12} {cls_str}")

    print()
    ok(f"Min vehicles in any frame: {min(veh_counts)}")
    ok(f"Max vehicles in any frame: {max(veh_counts)}")
    ok(f"Average vehicles: {np.mean(veh_counts):.1f}")

    if max(veh_counts) < 2:
        warn("YOLO never detects 2+ vehicles simultaneously!")
        warn("This means distance/collision logic can NEVER trigger.")
        warn("Possible causes:")
        warn("  - Video shows only 1 vehicle (dashcam car not detected)")
        warn("  - Vehicles too small or occluded at 640x480")
        warn("  - YOLO confidence too high (try lowering to 0.3)")
    elif min(veh_counts) == 0:
        info("Some frames have 0 vehicles — may miss crash moment")

    if save_frames:
        ok(f"Saved annotated frames to {DIAG_DIR}/")

    # Check for duplicate detections (overlapping boxes)
    print(f"\n  Checking for duplicate/overlapping detections...")
    dup_count = 0
    for r in frame_results:
        dets = r['detections']
        for i in range(len(dets)):
            for j in range(i+1, len(dets)):
                a, b = dets[i], dets[j]
                # Check IoU
                ix1 = max(a['x1'], b['x1'])
                iy1 = max(a['y1'], b['y1'])
                ix2 = min(a['x2'], b['x2'])
                iy2 = min(a['y2'], b['y2'])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1) * (iy2-iy1)
                    union = a['w']*a['h'] + b['w']*b['h'] - inter
                    iou = inter / union if union > 0 else 0
                    if iou > 0.4:
                        dup_count += 1
    if dup_count > 0:
        warn(f"Found {dup_count} overlapping detection pairs (IoU > 0.4)")
        warn("Cross-class NMS is needed to remove duplicates")
    else:
        ok("No duplicate detections found")

    return frame_results


# ═══════════════════════════════════════════════════════════════
# TEST 3: DISTANCE CALCULATION
# ═══════════════════════════════════════════════════════════════

def test_distances(frame_results):
    header("TEST 3: DISTANCE CALCULATION")

    print(f"\n  Pairwise distances for frames with 2+ vehicles:\n")
    print(f"  {'Frame':>6} {'Pair':>12} {'Pixel Dist':>11} {'Meter Dist':>11} {'Overlap?':>9}")
    print(f"  {'-'*55}")

    any_close = False
    min_dist_ever = float('inf')

    for r in frame_results:
        dets = r['detections']
        if len(dets) < 2:
            continue

        for i in range(len(dets)):
            for j in range(i+1, len(dets)):
                a, b = dets[i], dets[j]
                px_dist = np.sqrt((a['cx']-b['cx'])**2 + (a['cy']-b['cy'])**2)
                m_dist = px_dist / PPM

                # Check overlap
                overlap = not (a['x2'] < b['x1'] or b['x2'] < a['x1'] or
                               a['y2'] < b['y1'] or b['y2'] < a['y1'])

                min_dist_ever = min(min_dist_ever, m_dist)
                if m_dist < 5.0:
                    any_close = True

                print(f"  {r['frame']:6d} "
                      f"{'V'+str(i)+'-V'+str(j):>12} "
                      f"{px_dist:>10.1f}px "
                      f"{m_dist:>10.2f}m "
                      f"{'YES' if overlap else 'no':>9}")

    print()
    if min_dist_ever < 999:
        ok(f"Minimum distance across all samples: {min_dist_ever:.2f}m")
    else:
        warn("No frames with 2+ vehicles — cannot calculate distances")

    if any_close:
        ok("Found frames with close vehicles (< 5m)")
    else:
        if min_dist_ever < 999:
            warn(f"No vehicles closer than 5m (min was {min_dist_ever:.2f}m)")
            warn("At PPM={PPM}, this means closest vehicles were "
                 f"{int(min_dist_ever * PPM)}px apart")
            info("If this is a crash video, the crash moment might be between sampled frames")
            info("Or the vehicles might not both be detected at impact")


# ═══════════════════════════════════════════════════════════════
# TEST 4: CNN MODEL CHECK
# ═══════════════════════════════════════════════════════════════

def test_model(path):
    header("TEST 4: CNN CRASH MODEL")

    model_path = MODELS / 'crash_detection_model'

    if not model_path.exists():
        warn(f"Model directory not found: {model_path}")
        info("This is OK — the pipeline works without the CNN model")
        info("The CNN model was trained on averaged frames and is unreliable on Mac")
        return

    try:
        import tensorflow as tf
        ok(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        warn("TensorFlow not installed")
        return

    # Try loading
    try:
        model = tf.keras.models.load_model(str(model_path))
        ok(f"Model loaded from {model_path}")
        ok(f"Input shape: {model.input_shape}")
        ok(f"Output shape: {model.output_shape}")
        ok(f"Parameters: {model.count_params():,}")
    except Exception as e:
        fail(f"Model loading failed: {e}")
        info("Try loading with: model = tf.saved_model.load(path)")
        try:
            model = tf.saved_model.load(str(model_path))
            ok("Loaded with tf.saved_model.load (serving format)")
            info("This is a serving-only model — no .predict() method")
            info("Must use: model.signatures['serving_default'](input)")
        except Exception as e2:
            fail(f"Also failed with tf.saved_model.load: {e2}")
            return
        return

    # Test with sample frames from video
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_indices = np.linspace(0, total - 1, 10, dtype=int)
    predictions = []

    print(f"\n  Testing model on 10 sample frames:\n")
    print(f"  {'Frame':>6} {'Prediction':>12} {'Assessment'}")
    print(f"  {'-'*40}")

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Preprocess exactly like training
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        try:
            pred = model.predict(img, verbose=0)
            val = float(pred[0][0])
        except Exception:
            try:
                pred = model(img, training=False)
                val = float(pred.numpy()[0][0])
            except Exception as e:
                fail(f"Prediction failed: {e}")
                val = -1

        predictions.append(val)

        assess = ""
        if val < 0:
            assess = "ERROR"
        elif val == 0.0:
            assess = "⚠️ Always zero (broken)"
        elif 0.0 < val < 0.3:
            assess = "Low (no crash)"
        elif 0.3 <= val < 0.7:
            assess = "Medium (uncertain)"
        else:
            assess = "High (crash)"

        print(f"  {int(idx):6d} {val:>12.6f} {assess}")

    cap.release()

    # Diagnosis
    print()
    if all(p == 0.0 for p in predictions if p >= 0):
        fail("MODEL OUTPUTS 0.0 FOR EVERY FRAME")
        info("Known issue: SavedModel exported from Colab doesn't work on Mac")
        info("Root cause: model.export() in Keras 3.x creates serving-only format")
        info("Solution: Model is NOT used in crash decision — pipeline handles it")
    elif all(abs(p - predictions[0]) < 0.001 for p in predictions if p >= 0):
        warn(f"Model outputs same value ({predictions[0]:.6f}) for every frame")
        info("Model is not discriminating between frames")
    elif max(predictions) - min(predictions) < 0.1:
        warn(f"Model output range very narrow: {min(predictions):.4f} - {max(predictions):.4f}")
        info("Model is barely responding to different inputs")
    else:
        ok(f"Model output range: {min(predictions):.4f} - {max(predictions):.4f}")
        ok("Model is producing varied outputs (may be usable)")


# ═══════════════════════════════════════════════════════════════
# TEST 5: SCORING LOGIC (simulate crash scenario)
# ═══════════════════════════════════════════════════════════════

def test_scoring():
    header("TEST 5: SCORING LOGIC VALIDATION")

    print("\n  Simulating scenarios to verify scoring responds correctly:\n")

    # Simulate: two vehicles far apart
    print("  Scenario A: Two vehicles 15m apart, no movement")
    vehicles_far = [
        {'id': 1, 'cx': 100, 'cy': 200, 'x1': 60, 'y1': 170, 'x2': 140, 'y2': 230,
         'w': 80, 'h': 60, 'area': 4800, 'speed': 0, 'accel': 0, 'decel_ratio': 0},
        {'id': 2, 'cx': 475, 'cy': 200, 'x1': 435, 'y1': 170, 'x2': 515, 'y2': 230,
         'w': 80, 'h': 60, 'area': 4800, 'speed': 0, 'accel': 0, 'decel_ratio': 0},
    ]
    px_dist = np.sqrt((100-475)**2) 
    m_dist = px_dist / PPM
    print(f"    Distance: {px_dist:.0f}px = {m_dist:.1f}m")
    print(f"    Expected: score ≈ 0.0 (safe)")

    # Simulate: two vehicles overlapping with deceleration
    print("\n  Scenario B: Two vehicles overlapping, decelerating")
    vehicles_crash = [
        {'id': 1, 'cx': 300, 'cy': 200, 'x1': 250, 'y1': 170, 'x2': 350, 'y2': 230,
         'w': 100, 'h': 60, 'area': 6000, 'speed': 10, 'accel': -30, 'decel_ratio': 0.75},
        {'id': 2, 'cx': 330, 'cy': 210, 'x1': 280, 'y1': 180, 'x2': 380, 'y2': 240,
         'w': 100, 'h': 60, 'area': 6000, 'speed': 5, 'accel': -25, 'decel_ratio': 0.83},
    ]
    px_dist2 = np.sqrt((300-330)**2 + (200-210)**2)
    m_dist2 = px_dist2 / PPM
    print(f"    Distance: {px_dist2:.0f}px = {m_dist2:.1f}m")
    print(f"    Boxes overlap: YES")
    print(f"    Deceleration: 75-83%")
    print(f"    Expected: score > 0.4 (crash)")

    # Simulate: one vehicle only
    print("\n  Scenario C: Only 1 vehicle")
    vehicles_one = [
        {'id': 1, 'cx': 300, 'cy': 200, 'x1': 250, 'y1': 170, 'x2': 350, 'y2': 230,
         'w': 100, 'h': 60, 'area': 6000, 'speed': 40, 'accel': 0, 'decel_ratio': 0},
    ]
    print(f"    Expected: score = 0.0 (can't crash alone)")

    # Now actually run them through the scorer if crash_detection.py exists
    code_path = BASE / 'code' / 'crash_detection.py'
    if not code_path.exists():
        info(f"Cannot run scorer — {code_path} not found")
        info("Copy crash_detection.py to ~/Desktop/crash_detection/code/ first")
        return

    # Try to import the scorer
    try:
        sys.path.insert(0, str(BASE / 'code'))

        # We need to test the CollisionScorer from the actual code
        # Import it dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("cd", str(code_path))
        cd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cd)

        if hasattr(cd, 'CollisionScorer'):
            scorer = cd.CollisionScorer()

            # Test scenario A
            res_a = scorer.score_frame(vehicles_far)
            score_a = res_a['score']
            status_a = "✅ PASS" if score_a < 0.1 else "❌ FAIL"
            print(f"\n  Result A: score={score_a:.4f} {status_a}")
            for k, v in res_a['signals'].items():
                if v > 0.01:
                    print(f"    {k}: {v:.4f}")

            # Reset scorer for scenario B
            scorer2 = cd.CollisionScorer()
            # Feed a few "approaching" frames first to build distance history
            for dist_step in [8.0, 5.0, 3.0, 2.0]:
                far_vehicles = [
                    {'id': 1, 'cx': 300, 'cy': 200, 'x1': 250, 'y1': 170, 'x2': 350, 'y2': 230,
                     'w': 100, 'h': 60, 'area': 6000, 'speed': 40, 'accel': 0, 'decel_ratio': 0},
                    {'id': 2, 'cx': int(300 + dist_step * PPM), 'cy': 200,
                     'x1': int(250 + dist_step * PPM), 'y1': 170,
                     'x2': int(350 + dist_step * PPM), 'y2': 230,
                     'w': 100, 'h': 60, 'area': 6000, 'speed': 35, 'accel': -2, 'decel_ratio': 0.05},
                ]
                scorer2.score_frame(far_vehicles)

            res_b = scorer2.score_frame(vehicles_crash)
            score_b = res_b['score']
            status_b = "✅ PASS" if score_b > 0.3 else "❌ FAIL"
            print(f"\n  Result B: score={score_b:.4f} {status_b}")
            for k, v in res_b['signals'].items():
                if v > 0.01:
                    print(f"    {k}: {v:.4f}")

            # Test scenario C
            scorer3 = cd.CollisionScorer()
            res_c = scorer3.score_frame(vehicles_one)
            score_c = res_c['score']
            status_c = "✅ PASS" if score_c < 0.05 else "❌ FAIL"
            print(f"\n  Result C: score={score_c:.4f} {status_c}")

        else:
            warn("CollisionScorer class not found in crash_detection.py")
            info("The scoring logic test requires the v13 code")

    except Exception as e:
        warn(f"Could not import scoring logic: {e}")
        info("This test requires crash_detection.py v13 in ~/Desktop/crash_detection/code/")


# ═══════════════════════════════════════════════════════════════
# TEST 6: FULL PIPELINE ON KEY FRAMES
# ═══════════════════════════════════════════════════════════════

def test_pipeline_frames(path, save_frames=False):
    header("TEST 6: FRAME-BY-FRAME PIPELINE TRACE")

    from ultralytics import YOLO
    from scipy.spatial.distance import cdist

    model = YOLO('yolov8n.pt')
    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if save_frames:
        os.makedirs(DIAG_DIR, exist_ok=True)

    print(f"\n  Processing EVERY frame (up to 300) with full pipeline trace...\n")
    print(f"  {'Frame':>6} {'Vehicles':>9} {'MinDist':>9} {'MaxSpeed':>9} {'Overlap':>8} {'Notes'}")
    print(f"  {'-'*65}")

    prev_positions = {}
    notable_frames = []
    max_todo = min(total, 300)

    for fn in range(max_todo):
        ret, frame = cap.read()
        if not ret:
            break

        frame_640 = cv2.resize(frame, (PROCESS_W, PROCESS_H))
        results = model(frame_640, conf=0.5, verbose=False)

        dets = []
        for box in results[0].boxes:
            cls = results[0].names[int(box.cls)].lower()
            if cls not in vehicle_classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < 10 or h < 10:
                continue
            dets.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': (x1+x2)//2, 'cy': (y1+y2)//2,
                'w': w, 'h': h,
            })

        # Distance
        min_dist = float('inf')
        has_overlap = False
        if len(dets) >= 2:
            for i in range(len(dets)):
                for j in range(i+1, len(dets)):
                    a, b = dets[i], dets[j]
                    pd = np.sqrt((a['cx']-b['cx'])**2 + (a['cy']-b['cy'])**2)
                    md = pd / PPM
                    min_dist = min(min_dist, md)
                    # overlap check
                    if not (a['x2'] < b['x1'] or b['x2'] < a['x1'] or
                            a['y2'] < b['y1'] or b['y2'] < a['y1']):
                        has_overlap = True

        # Speed (simple, from centroid displacement)
        max_speed = 0
        for i, d in enumerate(dets):
            key = f"{fn}_{i}"
            # Use closest previous position
            for prev_key, (px, py) in list(prev_positions.items()):
                pd = np.sqrt((d['cx']-px)**2 + (d['cy']-py)**2)
                if pd < 50:  # close enough to be same vehicle
                    spd = (pd / PPM) / (1.0 / fps) * 3.6
                    max_speed = max(max_speed, min(spd, 120))

        # Store current positions
        prev_positions = {}
        for i, d in enumerate(dets):
            prev_positions[f"{fn}_{i}"] = (d['cx'], d['cy'])

        # Determine if notable
        notes = []
        if min_dist < 3.0 and min_dist < 999:
            notes.append(f"CLOSE({min_dist:.1f}m)")
        if has_overlap:
            notes.append("OVERLAP")
        if len(dets) == 0:
            notes.append("NO_VEHICLES")

        is_notable = bool(notes) or (fn % 25 == 0)

        if is_notable:
            md_s = f"{min_dist:.2f}m" if min_dist < 999 else "inf"
            note_str = " ".join(notes) if notes else ""
            marker = "  >>>" if notes else "     "
            print(f"{marker}{fn:4d} {len(dets):>9} {md_s:>9} {max_speed:>8.0f} "
                  f"{'YES' if has_overlap else 'no':>8} {note_str}")

            if notes:
                notable_frames.append({
                    'frame': fn, 'vehicles': len(dets),
                    'min_dist': min_dist, 'overlap': has_overlap,
                    'notes': notes,
                })

        # Save notable frames
        if save_frames and notes:
            annotated = frame_640.copy()
            for d in dets:
                cv2.rectangle(annotated, (d['x1'], d['y1']), (d['x2'], d['y2']),
                              (0, 255, 0), 2)
            label = f"F{fn} V={len(dets)} D={min_dist:.1f}m {' '.join(notes)}"
            cv2.putText(annotated, label, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(DIAG_DIR / f"notable_{fn:04d}.jpg"), annotated)

    cap.release()

    # Summary
    print(f"\n  {'─'*50}")
    if notable_frames:
        close_frames = [f for f in notable_frames if any('CLOSE' in n for n in f['notes'])]
        overlap_frames = [f for f in notable_frames if f['overlap']]

        ok(f"Notable frames found: {len(notable_frames)}")
        if close_frames:
            ok(f"Frames with close vehicles (< 3m): {len(close_frames)}")
            closest = min(close_frames, key=lambda f: f['min_dist'])
            ok(f"Closest approach: {closest['min_dist']:.2f}m at frame {closest['frame']}")
        else:
            warn("No frames with vehicles closer than 3m")

        if overlap_frames:
            ok(f"Frames with box overlap: {len(overlap_frames)}")
        else:
            warn("No frames with overlapping bounding boxes")

        if not close_frames and not overlap_frames:
            warn("DIAGNOSIS: No close vehicles or overlaps detected")
            warn("The crash may not be visible at 640x480 with YOLO")
            warn("Or the crash involves only 1 detected vehicle (dashcam + other)")
    else:
        warn("No notable frames found")

    if save_frames:
        ok(f"Saved notable frames to {DIAG_DIR}/")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Crash Detection Diagnostics')
    parser.add_argument('--video', required=True, help='Video file or shortcut')
    parser.add_argument('--save-frames', action='store_true',
                        help='Save annotated diagnostic frames')
    parser.add_argument('--skip-model', action='store_true',
                        help='Skip CNN model test')
    parser.add_argument('--only', type=str, default=None,
                        help='Run only specific test (1-6)')
    args = parser.parse_args()

    path = find_video(args.video)
    if path is None:
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)

    header(f"DIAGNOSTICS FOR: {path.name}")
    print(f"  This will test each pipeline stage independently.")
    print(f"  Look for ❌ and ⚠️  markers to find the problem.\n")

    tests = args.only.split(',') if args.only else ['1','2','3','4','5','6']

    vid_info = None
    det_results = None

    if '1' in tests:
        vid_info = test_video(path)

    if '2' in tests:
        det_results = test_detection(path, save_frames=args.save_frames)

    if '3' in tests and det_results:
        test_distances(det_results)

    if '4' in tests and not args.skip_model:
        test_model(path)

    if '5' in tests:
        test_scoring()

    if '6' in tests:
        test_pipeline_frames(path, save_frames=args.save_frames)

    header("DIAGNOSIS COMPLETE")
    print()
    print("  What to look for:")
    print("  ─────────────────")
    print("  Test 2: Does YOLO detect 2+ vehicles at the crash moment?")
    print("         If not → crash is invisible to the pipeline")
    print("  Test 3: Do any vehicle pairs get close (< 3m)?")
    print("         If not → distance threshold will never trigger")
    print("  Test 4: Does CNN model output anything other than 0.0?")
    print("         If not → model is broken (known issue, OK to ignore)")
    print("  Test 5: Does scoring logic respond correctly to test scenarios?")
    print("         If not → bug in crash_detection.py scoring code")
    print("  Test 6: At the actual crash frame, are vehicles detected + close?")
    print("         This is the definitive test")
    print()
    print(f"  If --save-frames was used, check {DIAG_DIR}/ for visual evidence.")
    print()


if __name__ == '__main__':
    main()
