# 🚗 Crash Detection System v17.0 — 3D Perception Pipeline

Real-time crash detection using monocular camera video. Combines deep learning (MobileNetV2 + LSTM), YOLOv8 object detection, 3D pinhole camera projection, Kalman filter state estimation, monocular depth estimation (MiDaS), and Time-To-Collision (TTC) physics to detect, classify, and attribute fault in vehicle collisions.

Works on dashcam footage, recorded video files, and live webcam feeds.

---

## What It Does

| Input | Output |
|-------|--------|
| Dashcam / video / webcam frame | ✅ **NO CRASH** or 🚨 **CRASH DETECTED** |
| Two or more vehicles in frame | 3D metric distance, speed (km/h), TTC (seconds) |
| Crash detected | Fault attribution: who hit whom, from what direction |
| Every frame | BEV (Bird's-Eye-View) top-down minimap + optional depth heatmap |

### Verified Test Results

```
safe.mp4  → ✅ NO CRASH  | CNN=0.79 | Closest=5.69m  | 0/500 crash frames
crash1.mov → 🚨 CRASH     | CNN=1.00 | Closest=0.83m  | 21% crash frames
crash2.mov → 🚨 CRASH     | CNN=1.00 | Closest=0.34m  | 14% crash frames
webcam     → ✅ NO CRASH  | No false triggers
```

---

## Key Features

- **3D Ground Projection** — converts pixel coordinates to real-world meters using pinhole camera geometry ($Z = h_{cam} / \tan(\theta + \arctan((y - c_y) / f_y))$)
- **4D Kalman Filter** — tracks each vehicle's state `[X, Z, Vx, Vz]` in metric space for smooth, noise-resistant speed estimation
- **Time-To-Collision (TTC)** — vector-projected closing speed between vehicle pairs; crash triggered when TTC < 1.2 seconds
- **MiDaS Depth Estimation** — Intel's pre-trained monocular depth network produces dense per-pixel depth maps, calibrated to meters using ground-plane anchoring
- **BEV Minimap** — real-time top-down view showing all tracked vehicles, velocity arrows, distance arcs (5/10/20/40m), and TTC danger zones
- **Neural Crash Classifier** — MobileNetV2 feature extractor + LSTM sequence model trained on dashcam crash footage (CNN probability 0.0–1.0)
- **YOLOv8 Detection & Tracking** — detects cars, trucks, buses, motorcycles, bicycles, pedestrians, and common objects with centroid-based multi-object tracking
- **Dashcam Ego Zone** — semi-transparent reference box representing your car's front bumper; any vehicle entering it triggers a crash alert
- **Fault Detection** — identifies which vehicle caused the crash using 3D metric velocity vectors and trajectory history (rear-end, head-on, side impact)
- **Scene Validation** — filters implausible detections (indoor objects, wrong aspect ratio) to prevent false positives
- **ROI Masking** — masks sky (top 40%) and hood (bottom 10%) before detection to reduce noise

---

## System Architecture

```
Video Frame
    │
    ├──────────────────────────────────────────────────────────┐
    ▼                                                          ▼
┌─────────────┐                                    ┌───────────────────┐
│ apply_road_ │                                    │   MiDaS v3.1      │
│ roi() mask  │                                    │   Depth Estimator │
└──────┬──────┘                                    │   (MPS GPU)       │
       ▼                                           └────────┬──────────┘
┌─────────────┐     ┌──────────────────────────┐            │
│  YOLOv8n    │────▶│  Tracker (centroid)       │            │
│  Detection  │     │  + scene_validate()       │            │
└─────────────┘     └──────────┬───────────────┘            │
                               │                             │
                    ┌──────────▼───────────────┐            │
                    │  SpeedEstimator           │◀───────────┘
                    │  pixel_to_ground() → KF   │  depth-enhanced
                    │  [X, Z, Vx, Vz] per car   │  metric projection
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │  RuleCollision (TTC)      │
                    │  compute_ttc() vector     │
                    │  3D metric distance       │
                    └──────────┬───────────────┘
                               │
┌─────────────┐                │
│ MobileNetV2 │     ┌──────────▼───────────────┐
│ (SavedModel)│────▶│  LSTM Sequence Model     │
│ 1280 feats  │     │  Input(10,1280)→sigmoid  │
└─────────────┘     └──────────┬───────────────┘
                               │  CNN prob 0.0–1.0
                    ┌──────────▼───────────────┐     ┌──────────────┐
                    │  Final Verdict           │────▶│ BEV Renderer │
                    │  CNN ≥ 0.80 + ≥2 vehs    │     │ top-down map │
                    │  + FaultDetector 3D       │     └──────────────┘
                    └──────────────────────────┘
```

---

## Quick Start

### 1. Create a virtual environment (Python 3.10 recommended)

```bash
python3.10 -m venv crash_env
source crash_env/bin/activate
```

### 2. Install dependencies

```bash
pip install opencv-python ultralytics torch torchvision scipy h5py numpy tensorflow timm
```

> **macOS Apple Silicon:** Use `tensorflow-macos` instead of `tensorflow`. Also install `timm` for MiDaS depth estimation.

### 3. Run on a video

```bash
python code/crash_detection_enhanced.py --video /path/to/dashcam.mp4
```

### 4. Run live webcam

```bash
python code/crash_detection_enhanced.py --camera --dashcam
```

---

## Usage

### All Commands

```bash
# Video file (full path or shortcut name)
python code/crash_detection_enhanced.py --video crash1
python code/crash_detection_enhanced.py --video /path/to/any_video.mp4

# Live webcam
python code/crash_detection_enhanced.py --camera

# Dashcam mode (ego zone active)
python code/crash_detection_enhanced.py --camera --dashcam

# Record annotated session to MP4
python code/crash_detection_enhanced.py --camera --dashcam --record

# Headless mode (no GUI window)
python code/crash_detection_enhanced.py --video crash1 --no-display

# Save crash frames as images
python code/crash_detection_enhanced.py --video crash1 --save-output

# Limit frame count
python code/crash_detection_enhanced.py --video crash1 --max-frames 200
```

### Keyboard Controls (During Playback)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `B` | Toggle BEV minimap on/off |
| `D` | Toggle depth heatmap on/off |

### All Flags

| Flag | Description |
|------|-------------|
| `--video <name/path>` | Process a video file (supports shortcuts: `crash1`, `crash2`, `safe`) |
| `--camera` | Use live webcam feed |
| `--dashcam` | Enable ego zone + CNN-only crash detection |
| `--record` | Save annotated webcam session to MP4 |
| `--save-output` | Save crash frames as JPEG images |
| `--no-display` | Run without opening a GUI window |
| `--max-frames N` | Limit processing to N frames |

---

## Project Structure

```
crash-detection-system/
├── code/
│   ├── crash_detection_enhanced.py   # Main pipeline (v17.0)
│   ├── crash_detection.py            # Standalone version (synced constants)
│   ├── crash_detection_linux.py      # Linux/Raspberry Pi version
│   ├── depth_estimator.py            # MiDaS monocular depth module
│   ├── bev_renderer.py              # BEV top-down minimap renderer
│   └── diagnose.py                  # Diagnostic utilities
├── models/
│   ├── crash_model_weights.weights.h5  # Trained LSTM weights (6.7 MB)
│   └── feature_extractor_saved/        # MobileNetV2 SavedModel (19 MB)
├── camera_detect.py                 # Standalone webcam script
├── debug_crash.py                   # Debug/testing utilities
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## How the Math Works

### 3D Ground Projection

Every detected vehicle's tire contact point `(x_center, y_bottom)` is projected to real-world coordinates:

```
angle_v = arctan((y_bottom - cy) / fy)
Z = h_cam / tan(pitch + angle_v)       ← depth in meters
X = (x_center - cx) × Z / fx           ← lateral offset in meters
```

### Kalman Filter (per vehicle)

Each vehicle maintains a 4D state vector `[X, Z, Vx, Vz]` updated every frame:

```
State:       x = [X, Z, Vx, Vz]ᵀ
Transition:  F = [[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]]
Measurement: z = [X_measured, Z_measured]ᵀ
```

### Time-To-Collision

```
p_rel = pos_b - pos_a
v_rel = vel_b - vel_a
closing_speed = -dot(p_rel, v_rel) / ||p_rel||
TTC = ||p_rel|| / closing_speed
```

Crash condition: `TTC < 1.2s` AND bounding boxes overlap AND confirmed across 3+ consecutive frames.

---

## Configuration

All parameters are controlled via the `Config` class in `crash_detection_enhanced.py` (Single Source of Truth):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `H_CAM` | 1.25m | Camera height above ground |
| `PITCH_DEG` | 2.0° | Camera tilt below horizon |
| `FX`, `FY` | 460px | Focal length |
| `TTC_WARN` | 2.5s | Time-to-collision warning threshold |
| `TTC_CRITICAL` | 1.2s | Collision imminent threshold |
| `DIST_CONTACT` | 1.5m | Physical contact distance |
| `CNN_THRESH` | 0.80 | Neural network crash probability threshold |
| `MAX_SPEED` | 180 km/h | Speed cap (highway compatible) |
| `KF_Q_VAR` | 0.5 | Kalman filter process noise |

---

## Model Training

The neural crash classifier was trained in **Google Colab** (TF 2.19 / Keras 3) on dashcam crash footage.

### Pipeline
1. **Feature extraction** — MobileNetV2 (ImageNet weights) processes each frame into a 1280-dim vector
2. **Sequence building** — 10 evenly-spaced frames per video saved as `(10, 1280)` numpy arrays
3. **LSTM classifier** — `Dense(256) → LSTM(128) → LSTM(64) → Dense(1, sigmoid)`
4. **Training** — EarlyStopping on val_AUC, ReduceLROnPlateau, class weights for imbalanced data

### Compatibility Fix
Colab uses Keras 3 (TF 2.19); local Mac runs Keras 2 (TF 2.13). Direct `.keras` loading fails.
**Solution:** MobileNetV2 exported as `SavedModel` via `model.export()`, LSTM weights saved via `model.save_weights()` and loaded manually through `h5py`.

---

## Dashcam Mode

When `--dashcam` is active, a semi-transparent **EGO** box appears at the bottom-center when vehicles are present.

- Any vehicle whose bounding box overlaps the ego zone → **CRASH DETECTED**
- CNN also fires independently — works for wall/barrier impacts where YOLO can't detect the obstacle
- Ego zone is hidden when no vehicles are in frame (prevents false "YOUR CAR" labels indoors)
- Fault reason shows direction: `head-on`, `from the left`, `from the right`

---

## Fault Detection

On crash detection, the system analyses the two closest vehicles using 3D metric velocity vectors:

| Collision Type | How It's Determined |
|----------------|-------------------|
| Rear-end | One vehicle's Vz significantly higher than the other |
| Head-on | Both vehicles have opposing velocity vectors |
| Side impact | Lateral velocity (Vx) dominates the relative motion |
| Stationary hit | One vehicle speed > 5 km/h, other < 2 km/h |
| Wall/barrier | CNN triggered without second vehicle (dashcam mode) |

---

## Edge-Case Handling

| Scenario | How It's Handled |
|----------|-----------------|
| Indoor / webcam false positives | Scene validator filters by aspect ratio + confidence |
| Sky / hood detections | ROI mask blacks out top 40% and bottom 10% |
| Noisy speed readings | Kalman filter smooths velocity estimates |
| Brief occlusions | Track TTL keeps lost vehicles alive for 2 seconds |
| Single-frame glitches | Temporal gate requires 3/5 consecutive danger frames |
| High-resolution video (4K) | Ego zone clamped to max 250×80px regardless of resolution |

---

## Performance

Tested on MacBook Air (Apple Silicon M1, Python 3.10, TF 2.15):

| Video | Resolution | CNN | Verdict | FPS |
|-------|-----------|-----|---------|-----|
| crash1.mov | 3408×1910 | 0.9998 | 🚨 CRASH | ~8 |
| crash2.mov | 3408×1910 | 0.9998 | 🚨 CRASH | ~8 |
| safe.mp4 | 3840×2160 | 0.7914 | ✅ NO CRASH | ~7.5 |
| webcam | 640×480 | — | ✅ NO CRASH | ~9 |

> MiDaS depth estimation adds ~30ms per frame. Toggle it off with `D` key for faster processing.

---

## Compatibility

| Platform | Status |
|----------|--------|
| macOS Apple Silicon (M1/M2/M3) | ✅ Tested — MPS GPU acceleration |
| macOS Intel | ✅ Compatible (CPU inference) |
| Linux (Ubuntu, Raspberry Pi OS 64-bit) | ✅ Compatible |
| Windows | Should work (untested) |

---

## Version History

| Version | Changes |
|---------|---------|
| **v17.0** | 3D pinhole projection, Kalman filter, TTC collision, MiDaS depth, BEV renderer, scene validator, ego zone fixes |
| v16.0 | Neural crash detector, dashcam mode, fault detection, speed overlay |
| v15.0 | MobileNetV2 + LSTM classifier, video recording |
| v14.0 | YOLOv8 tracking, rule-based collision |

---

## License

MIT