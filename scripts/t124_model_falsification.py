#!/usr/bin/env python3
"""
T1/T2/B1 falsification on the three local videos - the exact clips CNN_THRESH=0.80 was fitted to.
Feature extractor: models/feature_extractor_saved (as deployed).
LSTM head: numpy forward pass straight from crash_model_weights.weights.h5,
           because the local TF (2.15/Keras 2.15) CANNOT read a Keras 3 file. That is bug R2.
"""
import os, sys, json, numpy as np, cv2, h5py
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

ROOT = os.path.dirname(os.path.abspath(__file__))
FE   = os.path.join(ROOT, "models/feature_extractor_saved")
W    = os.path.join(ROOT, "models/crash_model_weights.weights.h5")
VID  = os.path.join(ROOT, "videos")
SIZE, SEQ, MAXF = 112, 10, 900
TRAIN_STRIDE_S = 49/9/10.0          # CCD: linspace(0,49,10) over 50 frames @10fps

# ---------- LSTM head in numpy ----------
def sigm(x): return 1.0/(1.0+np.exp(-x))
def lstm(x, Wk, Uk, b):
    T, u = x.shape[0], Uk.shape[0]
    h = np.zeros(u); c = np.zeros(u); out = np.empty((T,u))
    for t in range(T):
        z = x[t] @ Wk + h @ Uk + b
        i, f, g, o = sigm(z[:u]), sigm(z[u:2*u]), np.tanh(z[2*u:3*u]), sigm(z[3*u:])
        c = f*c + i*g; h = o*np.tanh(c); out[t] = h
    return out

class Head:
    def __init__(self, path):
        with h5py.File(path,'r') as f:
            L = f['layers']
            g = lambda n,i: np.array(L[n]['vars'][str(i)])
            self.d0 = (g('dense',0),   g('dense',1))
            self.l1 = (np.array(L['lstm']['cell']['vars']['0']),
                       np.array(L['lstm']['cell']['vars']['1']),
                       np.array(L['lstm']['cell']['vars']['2']))
            self.l2 = (np.array(L['lstm_1']['cell']['vars']['0']),
                       np.array(L['lstm_1']['cell']['vars']['1']),
                       np.array(L['lstm_1']['cell']['vars']['2']))
            self.d1 = (g('dense_1',0), g('dense_1',1))
            self.d2 = (g('dense_2',0), g('dense_2',1))
    def __call__(self, seq):                       # seq (10,1280)
        x = np.maximum(seq @ self.d0[0] + self.d0[1], 0)
        x = lstm(x, *self.l1)
        x = lstm(x, *self.l2)[-1]
        x = np.maximum(x @ self.d1[0] + self.d1[1], 0)
        return float(sigm(x @ self.d2[0] + self.d2[1])[0])

# ---------- load ----------
print("loading feature extractor ...", flush=True)
fe = tf.saved_model.load(FE)
serve = fe.signatures['serving_default'] if 'serving_default' in fe.signatures else None
def feats(batch):
    t = tf.constant(batch, tf.float32)
    if hasattr(fe, 'serve'): r = fe.serve(t)
    elif serve is not None:  r = list(serve(t).values())[0]
    else:                    r = fe(t)
    return np.array(r)
head = Head(W)
print("head params:", sum(a.size for p in (head.d0,head.l1,head.l2,head.d1,head.d2) for a in p), flush=True)

pre = tf.keras.applications.mobilenet_v2.preprocess_input
results = {}
for name in ("crash1.mov","crash2.mov","safe.mp4"):
    path = os.path.join(VID, name)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames=[]
    while len(frames) < MAXF:
        ok, fr = cap.read()
        if not ok: break
        fr = cv2.resize(fr,(SIZE,SIZE)); frames.append(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
    cap.release()
    n = len(frames)
    print(f"\n{name}: {tot} frames @ {fps:.1f} fps -> decoded {n}", flush=True)
    F = np.empty((n,1280), np.float32)
    for i in range(0, n, 64):
        F[i:i+64] = feats(pre(np.asarray(frames[i:i+64], np.float32)))
    del frames

    stride = max(1, int(round(fps*TRAIN_STRIDE_S)))
    rng = np.random.default_rng(0)
    A=B=C=D=None
    A = [head(F[i:i+SEQ])                      for i in range(0, n-SEQ, 5)]                       # deployed
    B = [head(F[i:i+SEQ][rng.permutation(SEQ)])for i in range(0, n-SEQ, 5)]                       # shuffled
    C = [head(F[np.arange(i,i+SEQ*stride,stride)]) for i in range(0, n-SEQ*stride, 5)] if n>SEQ*stride else []
    D = [head(np.repeat(F[i:i+1],SEQ,0))       for i in range(0, n-SEQ, 5)]                       # single frame x10
    st = lambda v: dict(n=len(v), mean=round(float(np.mean(v)),4), max=round(float(np.max(v)),4),
                        frac_over_080=round(float(np.mean(np.array(v)>=0.80)),4)) if len(v) else {}
    results[name] = dict(fps=round(fps,1), decoded=n, train_matched_stride_frames=stride,
                         A_deployed_consecutive=st(A), B_temporally_shuffled=st(B),
                         C_training_matched_stride=st(C), D_single_frame_tiled=st(D))
    print(json.dumps(results[name], indent=2), flush=True)

os.makedirs(os.path.join(ROOT,"runs/falsification"), exist_ok=True)
with open(os.path.join(ROOT,"runs/falsification/T124_local_videos.json"),"w") as f:
    json.dump(results,f,indent=2)
print("\nwrote runs/falsification/T124_local_videos.json")
