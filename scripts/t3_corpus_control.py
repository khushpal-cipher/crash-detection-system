#!/usr/bin/env python3
"""
T3 - corpus control: evaluate the existing CCD-trained model (crash_model_weights.weights.h5)
on Nexar test-public, where positive and negative clips share one corpus (no BDD100K vs YouTube
split, no source leakage by construction of the benchmark). Labels from data/nexar/solution.csv.

Reuses the exact NumPy LSTM head + SavedModel feature extractor from t124_model_falsification.py,
because the local TF (2.15/Keras 2.15) in the old env cannot read the Keras 3 weights file (bug R2).
Run inside ~/envs/crashdet (TF 2.19 / Keras 3).

Scoring matches the deployed decision path (code/crash_detection_enhanced.py): a sliding window of
Config.CNN_FRAMES=10 CONSECUTIVE decoded frames (native fps, no stride-sampling) fed through the
LSTM head; a clip's score is the max over all windows, mirroring `cnn >= CNN_THRESH` being checked
every frame during live inference.
"""
import os, sys, csv, json, time, numpy as np, cv2, h5py
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE   = os.path.join(ROOT, "models/feature_extractor_saved")
W    = os.path.join(ROOT, "models/crash_model_weights.weights.h5")
NEXAR = os.path.join(ROOT, "data/nexar")
SIZE, SEQ, MAXF = 112, 10, 900
THRESH = 0.80

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
pre = tf.keras.applications.mobilenet_v2.preprocess_input

def score_clip(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while len(frames) < MAXF:
        ok, fr = cap.read()
        if not ok: break
        fr = cv2.resize(fr, (SIZE, SIZE))
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    n = len(frames)
    if n < SEQ:
        return None, n, fps
    F = np.empty((n, 1280), np.float32)
    for i in range(0, n, 64):
        F[i:i+64] = feats(pre(np.asarray(frames[i:i+64], np.float32)))
    scores = [head(F[i:i+SEQ]) for i in range(0, n - SEQ + 1)]
    return float(np.max(scores)), n, fps

def main():
    # ---------- labels ----------
    labels = {}
    with open(os.path.join(NEXAR, "solution.csv")) as f:
        for row in csv.DictReader(f):
            if row["Usage"] == "Public":
                labels[row["id"]] = int(row["target"])

    records = []
    t0 = time.time()
    for split, y in (("positive", 1), ("negative", 0)):
        d = os.path.join(NEXAR, "test-public", split)
        files = sorted(f for f in os.listdir(d) if f.endswith(".mp4"))
        for i, fn in enumerate(files):
            clip_id = os.path.splitext(fn)[0]
            label = labels.get(clip_id, y)  # fall back to directory if id missing from solution.csv
            try:
                score, n, fps = score_clip(os.path.join(d, fn))
            except Exception as e:
                print(f"FAILED {split}/{fn}: {e}", flush=True)
                continue
            if score is None:
                print(f"SKIPPED {split}/{fn}: only {n} frames decoded (<{SEQ})", flush=True)
                continue
            records.append(dict(id=clip_id, split=split, label=label, score=round(score,4),
                                 decoded_frames=n, fps=round(fps,1)))
            if (i+1) % 25 == 0:
                elapsed = time.time()-t0
                print(f"{split}: {i+1}/{len(files)}  ({elapsed:.0f}s elapsed)", flush=True)

    os.makedirs(os.path.join(ROOT,"runs/falsification"), exist_ok=True)
    with open(os.path.join(ROOT,"runs/falsification/T3_corpus_control.json"),"w") as f:
        json.dump(records, f, indent=2)

    # ---------- metrics ----------
    y_true = np.array([r["label"] for r in records])
    y_score = np.array([r["score"] for r in records])
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    pred = (y_score >= THRESH).astype(int)
    tp = int(((pred==1)&(y_true==1)).sum()); fp = int(((pred==1)&(y_true==0)).sum())
    fn_ = int(((pred==0)&(y_true==1)).sum()); tn = int(((pred==0)&(y_true==0)).sum())
    fpr = fp/(fp+tn) if (fp+tn) else float("nan")
    tpr = tp/(tp+fn_) if (tp+fn_) else float("nan")

    summary = dict(n_clips=len(records), n_positive=int(y_true.sum()), n_negative=int((1-y_true).sum()),
                    roc_auc=round(float(auc),4), average_precision=round(float(ap),4),
                    threshold=THRESH, tp=tp, fp=fp, fn=fn_, tn=tn,
                    fpr_at_threshold=round(fpr,4), tpr_at_threshold=round(tpr,4),
                    elapsed_seconds=round(time.time()-t0,1))
    print(json.dumps(summary, indent=2))

    with open(os.path.join(ROOT,"runs/falsification/T3_corpus_control.md"),"w") as f:
        f.write("# T3 - Corpus control (Nexar test-public)\n\n")
        f.write(f"Model: `models/crash_model_weights.weights.h5` (CCD-trained). ")
        f.write(f"Benchmark: Nexar test-public, {summary['n_clips']} clips "
                f"({summary['n_positive']} positive / {summary['n_negative']} negative), "
                f"positives and negatives from the same corpus (no source leakage by construction).\n\n")
        f.write(f"**ROC-AUC: {summary['roc_auc']}**\n\n**AP: {summary['average_precision']}**\n\n")
        f.write(f"At the deployed threshold {THRESH}: TP={tp} FP={fp} FN={fn_} TN={tn}, "
                f"TPR={summary['tpr_at_threshold']}, FPR={summary['fpr_at_threshold']}\n\n")
        f.write(f"Runtime: {summary['elapsed_seconds']}s.\n")

    print("\nwrote runs/falsification/T3_corpus_control.json and .md")

if __name__ == "__main__":
    main()
