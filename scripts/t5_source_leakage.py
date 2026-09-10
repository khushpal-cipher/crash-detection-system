#!/usr/bin/env python3
"""T5 - source leakage quantification. Stdlib only."""
import re, random, collections, statistics, pathlib

ROOT = pathlib.Path(__file__).parent
rows = []
for line in (ROOT/"data/Crash-1500.txt").read_text().splitlines():
    if not line.strip(): continue
    m = re.match(r'^(\d+),\[(.*?)\],(.*)$', line.strip())
    vid, labels, rest = m.group(1), m.group(2), m.group(3).split(',')
    binl = [int(x) for x in labels.split(',')]
    startframe, ytid, timing, weather, ego = rest[0], rest[1], rest[2], rest[3], rest[4]
    rows.append(dict(vid=vid, binl=binl, start=int(startframe), yt=ytid,
                     timing=timing, weather=weather, ego=ego))
print(f"parsed {len(rows)} crash clips\n")

# ---------- group structure ----------
g = collections.Counter(r['yt'] for r in rows)
sizes = collections.Counter(g.values())
print("=== T5a: youtubeID group structure (POSITIVES) ===")
print(f"  1500 crash clips come from {len(g)} distinct YouTube source videos")
print(f"  mean clips per source video: {1500/len(g):.2f}   max: {max(g.values())}")
print("  group-size histogram (clips per source video -> how many sources):")
for k in sorted(sizes): print(f"    {k:>3} clip(s): {sizes[k]:>4} sources")
multi = sum(v for k,v in g.items() if v>1)
print(f"  clips belonging to a MULTI-CLIP source: {multi}/1500 = {multi/1500:.1%}\n")

# ---------- Monte Carlo over random 80/20 clip splits (seed-independent) ----------
print("=== T5b: leakage under RANDOM 80/20 clip split (the Colab method) ===")
ids = [r['yt'] for r in rows]
leaked_groups, leaked_clips = [], []
for trial in range(200):
    rnd = random.Random(trial)
    idx = list(range(1500)); rnd.shuffle(idx)
    val = set(idx[:300])                      # 300 crash clips in val (stratified 20%)
    byg = collections.defaultdict(lambda: [0,0])
    for i,y in enumerate(ids): byg[y][1 if i in val else 0] += 1
    lg = [y for y,(tr,va) in byg.items() if tr>0 and va>0]
    leaked_groups.append(len(lg))
    leaked_clips.append(sum(sum(byg[y]) for y in lg))
print(f"  source videos split ACROSS train and val : {statistics.mean(leaked_groups):.0f} "
      f"(min {min(leaked_groups)}, max {max(leaked_groups)}) out of {len(g)}")
print(f"  crash clips implicated by that leakage    : {statistics.mean(leaked_clips):.0f}/1500 "
      f"= {statistics.mean(leaked_clips)/1500:.1%}")
vc = statistics.mean(leaked_clips)*0.2
print(f"  => roughly {vc:.0f} of the 300 val crash clips ({vc/300:.0%}) have a sibling clip,")
print( "     cut from the SAME YouTube video, sitting in the training set.\n")

# ---------- official split ----------
print("=== T5c: does the OFFICIAL CCD split avoid this? ===")
def load(p):
    out=set()
    for l in (ROOT/p).read_text().splitlines():
        l=l.strip()
        if l.startswith('positive/'): out.add(l.split('/')[1].split('.')[0])
    return out
otr, ote = load("data/train.txt"), load("data/test.txt")
print(f"  official train positives: {len(otr)}   official test positives: {len(ote)}")
byv = {r['vid']: r['yt'] for r in rows}
gtr = {byv[v] for v in otr if v in byv}; gte = {byv[v] for v in ote if v in byv}
both = gtr & gte
print(f"  source videos appearing in BOTH official train and test: {len(both)}")
print(f"  => the official split is {'ALSO NOT source-grouped' if both else 'source-grouped'}\n")

# ---------- discarded metadata ----------
print("=== T5d: metadata that was available and discarded ===")
for f in ('timing','weather','ego'):
    print(f"  {f:8}: {dict(collections.Counter(r[f] for r in rows))}")
on = [next((i for i,b in enumerate(r['binl']) if b), None) for r in rows]
on = [x for x in on if x is not None]
print(f"  accident onset frame (of 50): mean {statistics.mean(on):.1f}, "
      f"median {statistics.median(on)}, min {min(on)}, max {max(on)}")
print(f"  => on average {statistics.mean(on)/50:.0%} of every 'crash' clip contains NO accident,")
print( "     yet every sampled frame was trained with label=1.")
sampled = [round(i*49/9) for i in range(10)]
pre = statistics.mean(sum(1 for s in sampled if s < o)/10 for o in on)
print(f"  uniform 10-frame sampling picks frames {sampled}")
print(f"  => {pre:.0%} of the frames the model saw for a positive clip are PRE-accident.")
