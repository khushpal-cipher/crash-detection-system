"""Score an arbitrary directory of clips and keep the per-frame traces. Gate 3's plumbing.

R1's gates 1 and 2 both live inside Nexar. Gate 3 is the only test that leaves it, and it
needs traces from a corpus whose crashes are NOT truncated to the clip end. This produces
them, and nothing else.

WHY THIS IS NOT A FLAG ON eval/run_baselines.py (D42). The approved plan originally said to
add `--clips-dir` there. `eval/benchmark.py::run()` has four separate label/Nexar couplings
-- `labels[cid]`, `hours(neg_ids, table)`, `evaluate()` needing both classes, and
`load_metadata()` -- plus `clip_paths()` hard-wired to data/nexar/test-public. It is also
the function ALL FOUR regression guards exercise. Threading a no-label mode through it
would risk this project's honesty mechanism to serve a job that needs no metrics at all.
Gate 3a needs *traces*, not AP. So this shares no code with it.

NO LABELS, NO METRICS, BY DESIGN. Not an omission: the vendored consensus annotations are
collision-TIMING files, so a directory scored here is positives-only and AP is undefined on
one class. Gate 3b's AP test runs on DAD's full 466-clip test split, where the 301
negatives actually exist, and is a separate job.

The adapter is reused COMPLETELY UNCHANGED. `BadasOpen(save_frames_dir=...)` already writes
the exact .npz format `eval/timing.py::load_traces_abs` reads, so the analysis side needs no
new reader either.

DENSE STRIDE-1 IS MANDATORY FOR GATE 3a (D41). NEW_PLAN.md §7.2's tail-scoring trick (score
only the final ~8 windows, ~10x cheaper) is what makes most of that plan fit this hardware,
but 3a asks WHERE IN THE CLIP THE PEAK FALLS. A tail-only run would presuppose the answer.
Hence stride 1 over the whole clip, and hence the cost.

COST IS MEASURED, NOT ASSERTED (D40). The ~167 s/clip from the Nexar sweep does not transfer
between corpora -- per-clip cost scales with clip duration, and DADA's clips are longer and
variable. Run `--limit 20` first and read the s/clip this prints. Stop condition from D40:
a pilot implying > 18 h for 221 clips means re-plan onto the Studio or subsample stratified
on `Time-of-collision`, NEVER on score.

Resumable, because a 10 h run on a laptop or a 2-4 h Studio chunk will be interrupted: every
clip is appended to <out>/scores.jsonl as it lands and ids already there are skipped.

    PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i \
      ~/envs/badas/bin/python scripts/score_external.py \
          --clips-dir data/dada2000/test --out runs/gate3a --limit 20
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.adapters import BadasOpen  # noqa: E402

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")


def already_scored(log_path):
    """Clip ids present in a previous run's log. Malformed lines are ignored rather than
    fatal -- a truncated final line is what an interrupted run leaves behind, and refusing
    to resume because of it would throw away the hours the log represents."""
    done = set()
    if not os.path.exists(log_path):
        return done
    with open(log_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--out", required=True, help="gets scores.jsonl and frames/*.npz")
    ap.add_argument("--limit", type=int, default=None, help="pilot: score only N clips")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=1,
                    help="1 = dense. Gate 3a REQUIRES dense (D41); do not raise it there.")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        _self_check()
        return

    clips = sorted(f for f in os.listdir(a.clips_dir) if f.endswith(VIDEO_EXT))
    if a.limit:
        clips = clips[:a.limit]
    frames_dir = os.path.join(a.out, "frames")
    log_path = os.path.join(a.out, "scores.jsonl")
    os.makedirs(a.out, exist_ok=True)
    done = already_scored(log_path)
    todo = [c for c in clips if os.path.splitext(c)[0] not in done]

    print(f"clips: {len(clips)}  already scored: {len(done & {os.path.splitext(c)[0] for c in clips})}"
          f"  to score: {len(todo)}")
    if a.stride != 1:
        print(f"  ⚠️  stride={a.stride} is NOT dense. Gate 3a needs the full trace (D41).")
    if not todo:
        print("nothing to do")
        return

    model = BadasOpen(device=a.device, stride=a.stride, save_frames_dir=frames_dir).load()
    log = open(log_path, "a")
    t0 = time.time()
    for n, name in enumerate(todo, 1):
        clip_id = os.path.splitext(name)[0]
        t1 = time.time()
        try:
            s = model.score(os.path.join(a.clips_dir, name))
            rec = {"id": clip_id, "score": s} if s is not None else \
                  {"id": clip_id, "reason": "adapter returned None"}
        except Exception as e:  # one unreadable clip must not end a 10 h run
            rec = {"id": clip_id, "reason": f"{type(e).__name__}: {e}"}
        rec["adapter"] = model.name
        rec["elapsed_s"] = round(time.time() - t1, 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        os.fsync(log.fileno())  # hours of work; an fsync per ~167 s clip is free
        mean = (time.time() - t0) / n
        print(f"  {n}/{len(todo)}  {clip_id}  {rec['elapsed_s']}s  "
              f"(mean {mean:.1f} s/clip, {mean * len(todo) / 3600:.1f} h for this run)",
              flush=True)
    log.close()

    mean = (time.time() - t0) / len(todo)
    print(f"\nMEASURED {mean:.1f} s/clip over {len(todo)} clips -- this is the number to plan "
          f"with, not the Nexar sweep's ~167 s/clip (D40).")
    print(f"  extrapolated: 221 DADA clips = {mean * 221 / 3600:.1f} h   "
          f"466 DAD clips = {mean * 466 / 3600:.1f} h")
    if mean * 221 / 3600 > 18:
        print("  🔴 STOP CONDITION HIT (D40): > 18 h for 221 clips. Re-plan onto the Studio "
              "or subsample stratified on Time-of-collision -- never on score.")
    print(f"\ntraces in {frames_dir}/ -- read them with "
          f"eval.timing.load_traces_abs('{frames_dir}')")


def _self_check():
    """Resume logic only. The scoring path is BadasOpen, which is already exercised by the
    committed 667-clip sweep and by scripts/badas_smoke.py -- re-testing it here would spend
    MPS time to re-prove someone else's test."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "scores.jsonl")
        assert already_scored(log) == set(), "missing log must read as empty, not raise"
        print("ok  missing log resumes from scratch")

        with open(log, "w") as f:
            f.write(json.dumps({"id": "a", "score": 0.9}) + "\n")
            f.write(json.dumps({"id": "b", "reason": "adapter returned None"}) + "\n")
            f.write('{"id": "c", "sco')  # what an interrupted run actually leaves
        done = already_scored(log)
        assert done == {"a", "b"}, done
        print("ok  truncated final line ignored, earlier clips still resumed")

        # A skipped clip is one the adapter will not be asked to score again. This is the
        # whole point: without it a chunked Studio run re-does every clip each session.
        clips = ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
        todo = [c for c in clips if os.path.splitext(c)[0] not in done]
        assert todo == ["c.mp4", "d.mp4"], todo
        print("ok  only unscored clips are queued")

        # A clip that FAILED is recorded and therefore skipped, not retried forever.
        assert "b" in done, "a failed clip must not be retried on every resume"
        print("ok  failed clips are not retried in a loop")

    print("PASS")


if __name__ == "__main__":
    main()
