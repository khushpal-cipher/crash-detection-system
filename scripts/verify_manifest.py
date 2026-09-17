"""Verify a downloaded clip directory against the SHA-256 manifest built beside it.

WHY THIS GATES GATE 3a. A truncated mp4 still decodes. ffmpeg will happily hand back the
frames it got, the scorer will produce a trace, and gate 3a will return a confident verdict
on a corpus that is quietly not the one that was built. Nothing anywhere downstream would
notice. The 220 DADA clips took ~2 h of Colab work and crossed a network; the manifest is
the only thing that proves what landed is byte-identical to what was made.

Byte length is checked too, not just the digest. It costs nothing and it names the common
failure -- a short file -- before the hash is even computed.

Manifest shape, written by scripts/colab_dada_extract.ipynb cell B:
    {"fps": 30, "n": 220, "failed": [...], "clips": {"10_068.mp4": {"sha256": ..., "bytes": ...}}}
NOTE the nesting: an earlier, abandoned version of that notebook wrote a FLAT {name: {...}}
manifest, so a reader that assumes flat will fail on the real file.

    ~/envs/badas/bin/python scripts/verify_manifest.py --self-check
    ~/envs/badas/bin/python scripts/verify_manifest.py \
        --dir data/dada2000/gate3a --expect 220
"""

import argparse
import hashlib
import json
import os
import sys


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(block), b""):
            h.update(blk)
    return h.hexdigest()


def verify(clips_dir, expect=None, manifest_name="manifest.json"):
    """-> (ok_count, problems). A problem is (name, reason); an empty list means verified."""
    with open(os.path.join(clips_dir, manifest_name)) as f:
        man = json.load(f)
    clips = man["clips"] if "clips" in man else man  # tolerate the flat shape
    problems, ok = [], 0

    for name, rec in sorted(clips.items()):
        path = os.path.join(clips_dir, name)
        if not os.path.exists(path):
            problems.append((name, "MISSING"))
            continue
        size = os.path.getsize(path)
        if size != rec["bytes"]:
            # Named before hashing: a short file is the download failure we expect, and
            # saying "1.2 MiB of 3.4 MiB" is more useful than "digest mismatch".
            problems.append((name, f"SIZE {size} != {rec['bytes']} (truncated download)"))
            continue
        if sha256(path) != rec["sha256"]:
            problems.append((name, "SHA-256 MISMATCH (same length, different bytes)"))
            continue
        ok += 1

    if expect is not None and len(clips) != expect:
        problems.append(("<manifest>", f"lists {len(clips)} clips, expected {expect}"))
    return ok, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--expect", type=int, default=None, help="required clip count, e.g. 220")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return _self_check()
    if not a.dir:
        ap.error("need --dir or --self-check")

    ok, problems = verify(a.dir, a.expect)
    print(f"verified {ok} clips in {a.dir}")
    if problems:
        print(f"\n🔴 {len(problems)} PROBLEM(S) -- DO NOT SCORE THIS DIRECTORY:")
        for name, why in problems:
            print(f"  {name:20s} {why}")
        print("\nRe-download the named clips from MyDrive/dada2000_gate3a and run this again.")
        sys.exit(1)
    print("PASS -- byte-identical to the manifest. Safe to score.")


def _self_check():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        payload = {"good.mp4": b"\x00" * 4096, "short.mp4": b"\x01" * 2048,
                   "swapped.mp4": b"\x02" * 1024, "gone.mp4": b"\x03" * 512}
        clips = {}
        for name, data in payload.items():
            with open(os.path.join(d, name), "wb") as f:
                f.write(data)
            clips[name] = {"sha256": sha256(os.path.join(d, name)), "bytes": len(data)}
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump({"fps": 30, "n": len(clips), "failed": [], "clips": clips}, f)

        ok, problems = verify(d, expect=4)
        assert (ok, problems) == (4, []), (ok, problems)
        print("ok  an intact directory verifies clean")

        # Truncate one: still a valid file, still decodes, wrong content. The whole point.
        with open(os.path.join(d, "short.mp4"), "wb") as f:
            f.write(b"\x01" * 1000)
        # Same length, different bytes -- a size-only check would pass this.
        with open(os.path.join(d, "swapped.mp4"), "wb") as f:
            f.write(b"\xff" * 1024)
        os.remove(os.path.join(d, "gone.mp4"))

        ok, problems = verify(d, expect=4)
        by_name = dict(problems)
        assert ok == 1, ok
        assert "truncated" in by_name["short.mp4"], by_name
        assert "MISMATCH" in by_name["swapped.mp4"], by_name
        assert by_name["gone.mp4"] == "MISSING", by_name
        print("ok  truncated / same-length-different-bytes / missing are each caught")

        ok, problems = verify(d, expect=220)
        assert any(n == "<manifest>" for n, _ in problems), problems
        print("ok  a wrong clip count is a problem even if every file verifies")

    print("PASS")


if __name__ == "__main__":
    main()
