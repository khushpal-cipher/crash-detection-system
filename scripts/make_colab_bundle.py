"""Build the tiny bundle Colab needs to run the EXACT committed scoring code, with checksums.

WHY THIS EXISTS AND WHY IT IS NOT `git clone`. The repo has 10 unpushed commits. A clone of
origin/main would silently lack scripts/score_external.py, eval/fp_rate.py and
scripts/verify_manifest.py -- every piece session 10-14 built. Colab would then run *older*
code, produce plausible numbers, and nothing would raise an error. Pushing to fix that is not
available either: the user has deferred the push across six sessions.

So the code travels as a bundle, and it travels WITH per-file SHA-256 so the notebook can prove
that what arrived is what left. That is GATE B. Every corpus-acquisition failure this project
has had -- DADA's mismatched ids, the JPEG-named-PNG frames, the split zip, the wrong committed
notebook -- came from an unchecked assumption at exactly this kind of seam.

WHAT IS DELIBERATELY NOT IN THE BUNDLE:
  * badas_open.pth (3.98 GB) -- fetched on Colab from the ungated Apache-2.0 mirror
    getnexar/BADAS-Open and checked against its own committed sha256 (GATE A). Shipping 4 GB
    through Drive to re-verify it on arrival buys nothing.
  * Nexar clips -- already uploaded to the user's Drive.
  * eval/benchmark.py, timing.py, calibration.py, fp_rate.py -- the analysis side stays on the
    Mac. Colab produces TRACES; every number this project reports is computed locally by
    committed code. Sending analysis code to Colab would invite a second, divergent code path.

    ~/envs/badas/bin/python scripts/make_colab_bundle.py --self-check
    ~/envs/badas/bin/python scripts/make_colab_bundle.py --out /tmp/colab_bundle.tar.gz
"""

import argparse
import json
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.verify_manifest import sha256  # noqa: E402  -- one hashing implementation, not two

# The exact files Colab needs to score. Nothing else, so the bundle cannot quietly grow into a
# second copy of the repo that drifts from this one.
MEMBERS = [
    "eval/adapters.py",                          # BadasOpen -- the scoring path, unmodified
    "scripts/score_external.py",                 # the resumable runner, unmodified
    "runs/baselines/badas-open/scores.jsonl",     # GATE C's MPS reference (667 committed scores)
]
TREES = ["vendor/badas-open"]                    # the vendored upstream package, 26 files


def collect(root=ROOT):
    """-> sorted list of repo-relative paths. Trees are walked, never globbed loosely."""
    paths = list(MEMBERS)
    for tree in TREES:
        for dirpath, _, filenames in os.walk(os.path.join(root, tree)):
            if "__pycache__" in dirpath:
                continue
            for fn in filenames:
                if fn.endswith(".pyc"):
                    continue
                paths.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return sorted(paths)


def manifest(paths, root=ROOT):
    """Nested shape, matching scripts/verify_manifest.py so one reader serves both."""
    clips = {}
    for rel in paths:
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            raise SystemExit(f"missing {rel} -- bundle would be incomplete, refusing to build")
        clips[rel] = {"sha256": sha256(full), "bytes": os.path.getsize(full)}
    return {"n": len(clips), "failed": [], "clips": clips}


def build(out_path, root=ROOT):
    paths = collect(root)
    man = manifest(paths, root)
    man_json = json.dumps(man, indent=2)

    with tarfile.open(out_path, "w:gz") as tar:
        for rel in paths:
            tar.add(os.path.join(root, rel), arcname=rel)
        info = tarfile.TarInfo("BUNDLE_MANIFEST.json")
        info.size = len(man_json.encode())
        tar.addfile(info, __import__("io").BytesIO(man_json.encode()))

    return paths, man, out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "colab_bundle.tar.gz"))
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return _self_check()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    paths, man, out = build(a.out)
    size = os.path.getsize(out)
    print(f"bundled {len(paths)} files -> {out}  ({size / 1024:.0f} KiB)")
    print(f"bundle sha256  {sha256(out)}")
    print(f"\n{'file':<52} {'bytes':>9}")
    for rel in paths[:6]:
        print(f"  {rel:<50} {man['clips'][rel]['bytes']:>9,}")
    print(f"  ... and {len(paths) - 6} more (vendor/badas-open)")
    print("\nNext: upload this file to MyDrive/crash_detection_colab/, then run "
          "scripts/colab_comma2k19.ipynb §1.")


def _self_check():
    import tempfile

    paths = collect()
    assert "eval/adapters.py" in paths, "the scoring path must be in the bundle"
    assert "scripts/score_external.py" in paths, "the runner must be in the bundle"
    assert "runs/baselines/badas-open/scores.jsonl" in paths, "GATE C's reference must be in"
    assert any(p.startswith("vendor/badas-open/badas/models/vjepa.py") for p in paths), paths[:5]
    print(f"ok  bundle lists {len(paths)} files including the scoring path and GATE C reference")

    assert not any("__pycache__" in p or p.endswith(".pyc") for p in paths), "compiled junk"
    print("ok  no __pycache__ or .pyc -- bundle is source only")

    # The analysis side must NOT travel. Numbers are produced locally by committed code, and a
    # second copy on Colab is how two divergent code paths start.
    for forbidden in ("eval/benchmark.py", "eval/timing.py", "eval/fp_rate.py",
                      "eval/calibration.py"):
        assert forbidden not in paths, f"{forbidden} must stay on the Mac"
    print("ok  analysis modules excluded -- Colab makes traces, the Mac makes numbers")

    man = manifest(paths)
    assert man["n"] == len(paths)
    # bytes >= 0, not > 0: the package's three namespace __init__.py files are legitimately
    # empty, and an empty file still has a digest that a truncation would change.
    assert all(len(r["sha256"]) == 64 and r["bytes"] >= 0 for r in man["clips"].values())
    empty = [p for p, r in man["clips"].items() if r["bytes"] == 0]
    assert all(p.endswith("__init__.py") for p in empty), f"unexpected empty file: {empty}"
    print(f"ok  manifest carries a 64-char digest and a byte length for all {man['n']} files "
          f"({len(empty)} legitimately-empty __init__.py)")

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "b.tar.gz")
        _, man2, _ = build(out, ROOT)
        with tarfile.open(out) as tar:
            names = set(tar.getnames())
        assert "BUNDLE_MANIFEST.json" in names, "GATE B has nothing to check against"
        assert names - {"BUNDLE_MANIFEST.json"} == set(paths), "tar contents != manifest"
        print("ok  tar contains exactly the manifested files plus BUNDLE_MANIFEST.json")

        # A tampered file must be caught by digest, not merely by size -- the same failure
        # verify_manifest.py exists to catch on the video side.
        rel = "eval/adapters.py"
        assert man2["clips"][rel]["sha256"] == sha256(os.path.join(ROOT, rel))
        print("ok  digests are of the real files (spot-checked against the working tree)")

    print("PASS")


if __name__ == "__main__":
    main()
