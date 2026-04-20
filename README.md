# Crash Detection System

A real-time car crash detection system using a **MobileNetV2 + LSTM neural network** for visual crash recognition, combined with **YOLOv8** vehicle tracking. Supports dashcam mode with an ego zone for collision detection, fault analysis, live webcam feed, and full video recording.

---

## Features

- **Neural crash detection** — MobileNetV2 extracts features from each frame; an LSTM learns crash patterns across 10-frame sequences
- **YOLOv8 vehicle tracking** — detects and tracks cars, trucks, buses, motorcycles, bicycles, pedestrians, and common objects
- **Dashcam ego zone** — a permanent reference box representing your car's front bumper; any vehicle entering it triggers a crash alert
- **Wall / barrier crash support** — in dashcam mode the CNN fires without needing a second vehicle detected
- **Fault detection** — identifies which vehicle caused the crash using trajectory analysis (rear-end, head-on, side impact)
- **Speed overlay** — live speed label on each tracked vehicle
- **Video recording** — saves the full annotated webcam session to an MP4 file
- **Human & object detection** — detects people and objects alongside vehicles (no false crash triggers from overlapping people)

---

## System Architecture

```
Video Frame
    │
    ▼
┌─────────────┐     ┌──────────────────────────────────┐
│  YOLOv8n    │────▶│  Tracker + Speed Estimator       │
│  Detection  │     │  (centroid tracking, km/h labels) │
└─────────────┘     └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │  Rule-based Collision Detector    │
                    │  (distance, overlap, trajectory)  │
                    └──────────────┬───────────────────┘
                                   │
┌─────────────┐                    │
│ MobileNetV2 │     ┌──────────────▼───────────────────┐
│ (SavedModel)│────▶│  LSTM Sequence Model             │
│ 1280 feats  │     │  Input(10,1280)→LSTM→sigmoid     │
└─────────────┘     └──────────────┬───────────────────┘
                                   │  CNN prob 0.0–1.0
                    ┌──────────────▼───────────────────┐
                    │  Final Verdict                    │
                    │  Dashcam: ego zone OR CNN ≥ 0.80  │
                    │  Standard: CNN ≥ 0.80 + ≥2 vehs  │
                    └──────────────────────────────────┘
```

---

## Model Training

The model was trained in **Google Colab** (TF 2.19 / Keras 3) on dashcam crash footage.

### Pipeline
1. **Feature extraction** — MobileNetV2 (ImageNet weights) processes each frame into a 1280-dim vector
2. **Sequence building** — 10 evenly-spaced frames per video saved as `(10, 1280)` numpy arrays
3. **LSTM classifier** — `Dense(256) → LSTM(128) → LSTM(64) → Dense(1, sigmoid)`
4. **Training** — EarlyStopping on val_AUC, ReduceLROnPlateau, class weights for imbalanced data

### Compatibility fix
Colab uses Keras 3 (TF 2.19); local Mac runs Keras 2 (TF 2.13). Direct `.keras` loading fails.  
**Solution used:**
- MobileNetV2 exported as `SavedModel` format via `model.export()` → loaded with `tf.saved_model.load()`
- LSTM weights saved with `model.save_weights()` → manually loaded via `h5py` and assigned layer by layer

---

## Requirements

- Python 3.9+
- TensorFlow 2.13.1
- OpenCV
- PyTorch + Ultralytics (YOLOv8)
- SciPy, h5py, NumPy

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
crash-detection-system/
├── code/
│   ├── crash_detection_enhanced.py   # Main script (dashcam + webcam + video)
│   └── crash_detection.py            # Basic version
├── models/
│   ├── crash_model_weights.weights.h5  # Trained LSTM weights (6.7 MB)
│   └── feature_extractor_saved/        # MobileNetV2 SavedModel (19 MB)
├── requirements.txt
├── LICENSE
└── README.md
```

> **Note:** `yolov8n.pt` is automatically downloaded by Ultralytics on first run. Test videos are not included due to size.

---

## Usage

### Test on a video file

```bash
python3 code/crash_detection_enhanced.py --video /path/to/video.mp4
```

### Live webcam (standard mode)

```bash
python3 code/crash_detection_enhanced.py --camera
```

### Dashcam mode (ego zone active)

```bash
python3 code/crash_detection_enhanced.py --camera --dashcam
```

### Dashcam mode with video recording

```bash
python3 code/crash_detection_enhanced.py --camera --dashcam --record
```

### All flags

| Flag | Description |
|------|-------------|
| `--video <name/path>` | Process a video file |
| `--camera` | Use live webcam |
| `--dashcam` | Enable ego zone + CNN-only crash detection |
| `--record` | Save annotated webcam session to MP4 |
| `--save-output` | Save crash frames as JPEG images |
| `--no-display` | Run without opening a window |
| `--max-frames N` | Limit processing to N frames |

---

## Dashcam Mode

When `--dashcam` is active, a permanent **cyan box** appears at the bottom centre of the frame representing your car's front bumper.

- Any vehicle whose bounding box overlaps the ego zone → **CRASH DETECTED**
- CNN also fires independently — works for **wall and barrier impacts** where YOLO cannot detect the obstacle
- Fault reason shows direction: `head-on`, `from the left`, `from the right`

```
┌─────────────────────────────────────┐
│  [DASHCAM] Vehicles:2  Speed:54km/h │  ← info bar
├─────────────────────────────────────┤
│                                     │
│    [car box]         [car box]      │
│                                     │
│          ┌──────────┐               │
│          │ YOUR CAR │               │  ← permanent ego zone
│          └──────────┘               │
└─────────────────────────────────────┘
```

---

## Fault Detection

On crash detection, the system analyses the two closest vehicles using 25-frame position history:

| Collision type | Verdict |
|----------------|---------|
| One vehicle moving into a stationary one | `VX hit VY from behind` |
| Both vehicles moving toward each other | `Head-on — both approaching` |
| One vehicle crossing into another's path | `VX crossed into VY` |
| Wall / barrier (dashcam mode) | `CNN detected crash impact` |

Motion is described as **stopped / moving slowly / moving fast / moving very fast** rather than unreliable km/h values (speed accuracy depends on camera calibration).

---

## Performance

Tested on Mac (Apple Silicon, Python 3.9, TF 2.13):

| Video | CNN prob | Verdict |
|-------|----------|---------|
| crash1.mov (real crash) | 0.9998 | ✅ CRASH DETECTED |
| crash2.mov (real crash) | 0.9998 | ✅ CRASH DETECTED |
| safe.mp4 (city traffic) | 0.0001 | ✅ NO CRASH |

Processing speed: ~13 fps on CPU.

---

## Known Limitations

- Speed values (km/h) are uncalibrated — PPM (pixels per metre) is hardcoded; accurate speed requires camera calibration
- CNN buffer is 10 frames — at 60 fps a very brief crash moment may partially miss the window
- YOLO does not detect walls or barriers — dashcam mode relies on CNN alone for those scenarios
- Trained on a limited dataset — more diverse crash footage would improve generalisation

---

## Compatibility

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon / Intel) | ✅ Tested |
| Linux (Ubuntu, Raspberry Pi OS 64-bit) | ✅ Compatible |
| Windows | Should work (untested) |