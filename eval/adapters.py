"""Model adapters — the model-agnostic interface README §41 Phase 4 task 5 asks for.

An adapter is anything with:
    .name                      -> str, goes into metrics.json
    .score(clip_path) -> float in [0,1], or None if the clip is unscorable

That is the whole contract. No base class, no registry: three models, duck typing is enough.

Deliberately NOT the adapter's job:
  * labels            -- benchmark.load_labels(), from solution.csv
  * clip durations    -- benchmark.durations(), ONE canonical table for every model, so
                         FP/hour denominators are identical across adapters and therefore
                         comparable. A per-adapter denominator would be a silent bug.
  * metrics           -- benchmark.evaluate()
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AlwaysNegative:
    """The trivial baseline README §31 requires in every report.

    At realistic prevalence this is a genuinely strong competitor on FP/hour -- it scores
    exactly 0 false alarms per hour. If a real model cannot beat it, there is no product.
    """

    name = "always-negative"

    def score(self, clip_path):
        return 0.0


class CachedScores:
    """Replays scores already committed to a results JSON.

    Used for the retired MobileNetV2+LSTM. Its scores were produced by
    `scripts/t3_corpus_control.py` over 527.6 s; that script generated committed evidence
    and progress.md §12 forbids touching it, so re-deriving the same numbers through a
    reimplementation would risk changing a published result to gain nothing. This replays
    the recorded scores through the shared metric path instead -- which is what "one code
    path" is actually for: identical metrics, identical splits, identical denominators.
    """

    def __init__(self, name, json_path, score_field="score"):
        self.name = name
        with open(json_path) as f:
            self._scores = {r["id"]: float(r[score_field]) for r in json.load(f)}

    def score(self, clip_path):
        clip_id = os.path.splitext(os.path.basename(clip_path))[0]
        return self._scores.get(clip_id)


class BadasOpen:
    """BADAS-Open (V-JEPA2 ViT-L), scored through the vendored upstream pipeline.

    Wraps rather than reimplements: upstream already does the sliding window, the
    VJEPA2VideoProcessor preprocessing, and the temperature-2.0 scaling that matches their
    training. `predict()` returns one probability per frame, P(class 1) = P(collision),
    with NaN for the first `frame_count` frames (no future to predict from yet).

    Clip reduction is **np.nanmax**, not builtin `max`. Upstream's cli.py uses `max(...)`,
    which returns NaN whenever a NaN is encountered first -- a latent bug on every clip,
    since their own windowing guarantees leading NaNs (progress.md U-B5).

    skip_predictor=True drops ~25% of the compute and leaves `last_hidden_state` bit-identical,
    because upstream's forward() discards the predictor output anyway (README §41 Phase 5,
    gate note reason 3). Default False = faithful to the shipped code path.
    """

    MODEL_NAME = "facebook/vjepa2-vitl-fpc16-256-ssv2"

    def __init__(self, device="mps", stride=1, frame_count=16, img_size=224,
                 target_fps=8.0, checkpoint=None, skip_predictor=False,
                 save_frames_dir=None):
        self.name = f"badas-open(stride={stride},fps={target_fps},img={img_size})"
        self.stride = stride
        self._ckpt = checkpoint or os.path.join(
            ROOT, "models", "badas", "weights", "badas_open.pth"
        )
        self._cfg = dict(device=device, stride=stride, frame_count=frame_count,
                         img_size=img_size, target_fps=target_fps,
                         skip_predictor=skip_predictor)
        self._model = None
        # ponytail: per-frame scores from the 18h sweep were discarded (one nanmax per
        # clip). Saving them here is free (~1.6 MB/667 clips) and is what unlocks
        # mean-vs-max reduction and t_start/t_peak/t_end without re-running the model.
        self._save_frames_dir = save_frames_dir
        if save_frames_dir:
            os.makedirs(save_frames_dir, exist_ok=True)

    def load(self):
        """Deferred so constructing an adapter stays free. ~8.5 s once the HF cache is warm."""
        import sys
        sys.path.insert(0, os.path.join(ROOT, "vendor", "badas-open"))
        from badas.models.vjepa import VJEPAModel

        c = self._cfg
        self._model = VJEPAModel(
            model_name=self.MODEL_NAME,
            checkpoint_path=self._ckpt,
            frame_count=c["frame_count"],
            img_size=c["img_size"],
            window_stride=c["stride"],
            target_fps=c["target_fps"],
            use_sliding_window=True,
            device=c["device"],
            skip_predictor=c["skip_predictor"],
        )
        self._model.load()
        return self

    def score(self, clip_path):
        if self._model is None:
            self.load()
        per_frame = np.asarray(self._model.predict(clip_path), dtype=float)
        if self._save_frames_dir:
            clip_id = os.path.splitext(os.path.basename(clip_path))[0]
            np.savez(os.path.join(self._save_frames_dir, f"{clip_id}.npz"),
                     scores=per_frame, target_fps=self._cfg["target_fps"],
                     stride=self._cfg["stride"], frame_count=self._cfg["frame_count"])
        if per_frame.size == 0 or np.all(np.isnan(per_frame)):
            return None
        return float(np.nanmax(per_frame))
