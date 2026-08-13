#!/usr/bin/env python3
"""
Camera Crash Detection for Raspberry Pi (v3 — picamera2)
=========================================================
Uses picamera2 (native libcamera) + YOLO + fault detection
No TensorFlow needed, no libcamerify wrapper needed!

Run:  python camera_detect.py
      python camera_detect.py --dashcam
      python camera_detect.py --no-display
"""
import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.distance import cdist
import argparse

# Use picamera2 for native camera access
from picamera2 import Picamera2

# ── Config ──
YOLO_MODEL    = Path(__file__).parent / 'yolov8n.pt'
YOLO_CONF     = 0.5
NMS_IOU       = 0.45
MIN_BOX_AREA  = 2000
TRACK_DIST    = 80
PPM           = 25       # DEPRECATED: enhanced.py uses pinhole projection
COLLISION_M   = 1.5     # synced with enhanced.py DIST_CONTACT
OVERLAP_IOU   = 0.08
CONSEC_FRAMES = 3
VEHICLES      = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

# Ego zone (dashcam mode)
EGO_ZONE_W = 0.20
EGO_ZONE_H = 0.09
EGO_ZONE_Y = 0.96

print("Loading YOLOv8n...")
yolo = YOLO(str(YOLO_MODEL) if YOLO_MODEL.exists() else 'yolov8n.pt')
print("YOLO ready")

# ── Tracker state ──
next_id = 1
objects = {}
prev_pos = {}
speed_bufs = {}
crash_window = deque(maxlen=10)


# ====================================================================
# DETECTION + TRACKING + SPEED
# ====================================================================

def detect(frame):
    results = yolo(frame, conf=YOLO_CONF, verbose=False)
    raw = []
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)].lower()
        if cls not in VEHICLES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        if w * h < MIN_BOX_AREA or w < 20 or h < 20:
            continue
        raw.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'w': w, 'h': h, 'area': w * h,
            'cx': (x1 + x2) // 2, 'cy': (y1 + y2) // 2,
            'conf': float(box.conf), 'cls': cls,
        })
    return raw


def track(dets):
    global next_id, objects
    if not dets:
        objects = {}
        return []
    if not objects:
        for d in dets:
            d['id'] = next_id
            objects[next_id] = d
            next_id += 1
        return dets

    det_c = np.array([[d['cx'], d['cy']] for d in dets])
    oids = list(objects.keys())
    obj_c = np.array([[objects[o]['cx'], objects[o]['cy']] for o in oids])
    dists = cdist(det_c, obj_c)

    result, used_d, used_o = [], set(), set()
    for flat in np.argsort(dists, axis=None):
        di, oi = flat // len(oids), flat % len(oids)
        if di in used_d or oi in used_o:
            continue
        if dists[di][oi] > TRACK_DIST:
            break
        tid = oids[oi]
        dets[di]['id'] = tid
        objects[tid] = dets[di]
        used_d.add(di)
        used_o.add(oi)
        result.append(dets[di])

    for i, d in enumerate(dets):
        if i not in used_d:
            d['id'] = next_id
            objects[next_id] = d
            next_id += 1
            result.append(d)

    for oi, oid in enumerate(oids):
        if oi not in used_o:
            del objects[oid]

    return result


def calc_speed(vehs, fps):
    for v in vehs:
        vid = v['id']
        if vid in prev_pos:
            px, py = prev_pos[vid]
            pd = np.sqrt((v['cx'] - px)**2 + (v['cy'] - py)**2)
            spd = (pd / PPM) / (1.0 / fps) * 3.6
            if vid not in speed_bufs:
                speed_bufs[vid] = deque(maxlen=3)
            speed_bufs[vid].append(spd)
            v['speed'] = min(np.mean(speed_bufs[vid]), 120)
        else:
            v['speed'] = 0.0
        prev_pos[vid] = (v['cx'], v['cy'])
    return vehs


def check_collision(vehs):
    if len(vehs) < 2:
        return False, float('inf')
    min_dist = float('inf')
    max_iou = 0.0
    for i in range(len(vehs)):
        for j in range(i + 1, len(vehs)):
            a, b = vehs[i], vehs[j]
            px = np.sqrt((a['cx'] - b['cx'])**2 + (a['cy'] - b['cy'])**2)
            min_dist = min(min_dist, px / PPM)
            ix1 = max(a['x1'], b['x1'])
            iy1 = max(a['y1'], b['y1'])
            ix2 = min(a['x2'], b['x2'])
            iy2 = min(a['y2'], b['y2'])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                max_iou = max(max_iou, inter / (a['area'] + b['area'] - inter + 1e-6))
    return (min_dist <= COLLISION_M and max_iou >= OVERLAP_IOU), min_dist


# ====================================================================
# FAULT DETECTOR — Real dot-product trajectory analysis
# ====================================================================

class FaultDetector:
    def __init__(self):
        self.pos_hist = {}

    def _update(self, vehicles):
        for v in vehicles:
            if v['id'] not in self.pos_hist:
                self.pos_hist[v['id']] = deque(maxlen=25)
            self.pos_hist[v['id']].append((v['cx'], v['cy']))

    def _displacement(self, vid, lookback=8):
        h = self.pos_hist.get(vid)
        if not h or len(h) < 2:
            return 0.0
        n = min(lookback, len(h))
        return float(np.linalg.norm(np.array(h[-1]) - np.array(h[-n])))

    def _was_stationary(self, vid, frames=15, threshold=10):
        h = self.pos_hist.get(vid)
        if not h or len(h) < 3:
            return False
        pts = list(h)[-min(frames, len(h)):]
        total = sum(np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i]))
                    for i in range(len(pts) - 1))
        return total < threshold

    def _direction_vector(self, vid, lookback=8):
        h = self.pos_hist.get(vid)
        if not h or len(h) < lookback:
            return None
        vec = np.array(h[-1]) - np.array(h[-lookback])
        mag = np.linalg.norm(vec)
        return vec / mag if mag > 3 else None

    def _collision_type(self, a, b):
        da = self._displacement(a['id'])
        db = self._displacement(b['id'])

        if da < 5 and db < 5:
            return 'stationary'

        va = self._direction_vector(a['id'])
        vb = self._direction_vector(b['id'])

        a_parked = self._was_stationary(a['id'])
        b_parked = self._was_stationary(b['id'])
        if a_parked and not b_parked:
            return 'rear-end-b'
        if b_parked and not a_parked:
            return 'rear-end-a'

        if va is None or vb is None:
            if da > db + 15:
                return 'rear-end-a'
            if db > da + 15:
                return 'rear-end-b'
            return 'unknown'

        # Dot product analysis
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

    def analyze(self, vehicles):
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

        da    = self._displacement(a['id'])
        db    = self._displacement(b['id'])
        la    = self._motion_label(da)
        lb    = self._motion_label(db)
        ctype = self._collision_type(a, b)

        if ctype == 'rear-end-a':
            fault  = "V{}".format(a['id'])
            reason = "V{} hit V{} from behind ({})".format(a['id'], b['id'], la)
        elif ctype == 'rear-end-b':
            fault  = "V{}".format(b['id'])
            reason = "V{} hit V{} from behind ({})".format(b['id'], a['id'], lb)
        elif ctype == 'head-on':
            fault  = 'shared'
            reason = "Head-on - both vehicles approaching each other"
        elif ctype == 'stationary':
            fault  = 'unclear'
            reason = "Both vehicles stationary - unable to determine fault"
        else:
            if da > db + 15:
                fault  = "V{}".format(a['id'])
                reason = "V{} crossed into V{} ({})".format(a['id'], b['id'], la)
            elif db > da + 15:
                fault  = "V{}".format(b['id'])
                reason = "V{} crossed into V{} ({})".format(b['id'], a['id'], lb)
            else:
                fault  = 'unclear'
                reason = "Side impact between V{} and V{}".format(a['id'], b['id'])

        return {
            'at_fault': fault, 'reason': reason, 'collision_type': ctype,
            'v1': a['id'], 'v2': b['id'],
            'v1_speed': a.get('speed', 0), 'v2_speed': b.get('speed', 0),
        }


# ====================================================================
# EGO ZONE — Dashcam mode
# ====================================================================

class EgoZone:
    @staticmethod
    def bounds(fw, fh):
        zw = int(fw * EGO_ZONE_W)
        zh = int(fh * EGO_ZONE_H)
        cx = fw // 2
        y2 = int(fh * EGO_ZONE_Y)
        return cx - zw // 2, y2 - zh, cx + zw // 2, y2

    @staticmethod
    def overlaps(vx1, vy1, vx2, vy2, ex1, ey1, ex2, ey2):
        return not (vx2 < ex1 or vx1 > ex2 or vy2 < ey1 or vy1 > ey2)

    @staticmethod
    def check(vehicles, fw, fh):
        ex1, ey1, ex2, ey2 = EgoZone.bounds(fw, fh)
        return [v for v in vehicles
                if EgoZone.overlaps(v['x1'], v['y1'], v['x2'], v['y2'],
                                    ex1, ey1, ex2, ey2)]

    @staticmethod
    def fault(v, fw):
        spd = v.get('speed', 0)
        spd_s = "{:.0f} km/h".format(spd)
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
            'at_fault': "V{}".format(v['id']),
            'reason': "V{} struck your car {} ({})".format(v['id'], direction, spd_s),
            'collision_type': "dashcam-{}".format(direction.replace(" ", "-")),
            'v1': v['id'], 'v2': 'ego',
            'v1_speed': spd, 'v2_speed': 0,
        }


# ====================================================================
# DISPLAY
# ====================================================================

def draw(frame, vehs, is_crash, min_dist, fault_info=None, dashcam=False, ego_hits=None):
    h, w = frame.shape[:2]

    if dashcam:
        ex1, ey1, ex2, ey2 = EgoZone.bounds(w, h)
        ego_color = (0, 0, 255) if ego_hits else (0, 200, 255)
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), ego_color, 2)
        cv2.putText(frame, "YOUR CAR", (ex1 + 5, ey2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ego_color, 1)

    for v in vehs:
        color = (0, 0, 255) if is_crash else (0, 255, 0)
        cv2.rectangle(frame, (v['x1'], v['y1']), (v['x2'], v['y2']), color, 2)
        spd = v.get('speed', 0)
        lbl = "ID{} {:.0f}km/h".format(v['id'], spd)
        cv2.putText(frame, lbl, (v['x1'], v['y1'] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    md = "{:.1f}m".format(min_dist) if min_dist < 999 else "inf"
    cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 0), -1)
    mode = "[DASHCAM]" if dashcam else "[SURVEILLANCE]"
    info = "{} Vehicles: {} | Dist: {}".format(mode, len(vehs), md)
    cv2.putText(frame, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if is_crash:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 35), (w, h), (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        cv2.putText(frame, "CRASH DETECTED!", (w // 2 - 150, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if fault_info:
            y_off = h // 2 + 20
            cv2.putText(frame, "At fault: {}".format(fault_info['at_fault']),
                        (w // 2 - 150, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Type: {}".format(fault_info['collision_type']),
                        (w // 2 - 150, y_off + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


# ====================================================================
# MAIN — uses picamera2 for reliable frame capture
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Pi Camera Crash Detection v3')
    parser.add_argument('--no-display', action='store_true', help='Headless mode')
    parser.add_argument('--dashcam', action='store_true', help='Dashcam mode with ego zone')
    parser.add_argument('--record', action='store_true', help='Save output video')
    args = parser.parse_args()

    fault_det = FaultDetector()
    first_fault = None

    print("")
    if args.dashcam:
        print("Mode: DASHCAM (ego zone active)")
    else:
        print("Mode: SURVEILLANCE (multi-vehicle)")

    # ── Initialize picamera2 ──
    print("Opening camera via picamera2...")
    picam = Picamera2()
    cam_config = picam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam.configure(cam_config)
    picam.start()
    time.sleep(1)  # Let camera warm up
    print("Camera active - press Ctrl+C to stop")
    print("")

    fps = 30
    frame_count = 0
    crash_frames = 0
    max_speed = 0.0
    min_dist_ever = float('inf')
    t0 = time.time()
    writer = None

    try:
        while True:
            # Capture frame from picamera2 (returns RGB numpy array directly)
            frame_rgb = picam.capture_array()

            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            fh, fw = frame.shape[:2]
            dets = detect(frame)
            tracked = track(dets)
            tracked = calc_speed(tracked, fps)

            ego_hits = []
            fault_info = None

            if args.dashcam:
                ego_hits = EgoZone.check(tracked, fw, fh)
                is_crash = len(ego_hits) > 0
                if is_crash and ego_hits:
                    fault_info = EgoZone.fault(ego_hits[0], fw)
                    if first_fault is None:
                        first_fault = fault_info
                min_dist = float('inf')
            else:
                is_col, min_dist = check_collision(tracked)
                crash_window.append(is_col)
                consec = 0
                for v in reversed(list(crash_window)):
                    if v:
                        consec += 1
                    else:
                        break
                is_crash = consec >= CONSEC_FRAMES

                if is_crash and len(tracked) >= 2:
                    fault_info = fault_det.analyze(tracked)
                    if fault_info and first_fault is None:
                        first_fault = fault_info
                else:
                    fault_det._update(tracked)

            frame_count += 1
            if is_crash:
                crash_frames += 1
            min_dist_ever = min(min_dist_ever, min_dist)
            for v in tracked:
                max_speed = max(max_speed, v.get('speed', 0))

            # Print stats every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - t0
                real_fps = frame_count / elapsed
                md = "{:.1f}m".format(min_dist) if min_dist < 999 else "inf"
                tag = " CRASH!" if is_crash else ""
                print("  Frame {} | {:.1f}fps | v={} | dist={}{}".format(
                    frame_count, real_fps, len(tracked), md, tag))

            if not args.no_display:
                out = draw(frame.copy(), tracked, is_crash, min_dist,
                           fault_info=fault_info, dashcam=args.dashcam,
                           ego_hits=ego_hits)

                if args.record and writer is None:
                    h_out, w_out = out.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    rec_path = str(Path(__file__).parent / 'pi_recording.mp4')
                    writer = cv2.VideoWriter(rec_path, fourcc, fps, (w_out, h_out))
                    print("Recording to: {}".format(rec_path))
                if writer is not None:
                    writer.write(out)

                cv2.imshow("Pi Crash Detection v3", out)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        picam.stop()
        picam.close()
        if writer is not None:
            writer.release()
            print("Recording saved.")
        cv2.destroyAllWindows()

        elapsed = time.time() - t0
        real_fps = frame_count / elapsed if elapsed > 0 else 0
        crash_pct = (crash_frames / frame_count * 100) if frame_count > 0 else 0
        md = "{:.2f}m".format(min_dist_ever) if min_dist_ever < 999 else "inf"

        print("")
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print("  Frames      : {}".format(frame_count))
        print("  Time        : {:.1f}s".format(elapsed))
        print("  FPS         : {:.1f}".format(real_fps))
        print("  Max speed   : {:.1f} km/h".format(max_speed))
        print("  Min distance: {}".format(md))
        print("  Crash frames: {}/{} ({:.1f}%)".format(
            crash_frames, frame_count, crash_pct))

        print("")
        if crash_pct >= 5:
            print("VERDICT: CRASH DETECTED")
        elif crash_pct >= 2:
            print("VERDICT: POSSIBLE COLLISION")
        else:
            print("VERDICT: NO CRASH")

        if first_fault:
            print("")
            print("-" * 60)
            print("FAULT ANALYSIS (dot-product trajectory)")
            print("-" * 60)
            print("  At fault       : {}".format(first_fault['at_fault']))
            print("  Collision type : {}".format(first_fault['collision_type']))
            print("  Reason         : {}".format(first_fault['reason']))
            print("  V{} speed      : {:.0f} km/h".format(
                first_fault['v1'], first_fault['v1_speed']))
            if first_fault['v2'] != 'ego':
                print("  V{} speed      : {:.0f} km/h".format(
                    first_fault['v2'], first_fault['v2_speed']))

        print("=" * 60)


if __name__ == '__main__':
    main()
