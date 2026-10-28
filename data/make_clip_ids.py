"""Write the clip-ID list for a set of VidGen-1M latent shards, without captions.

  python data/make_clip_ids.py --shards <dir with the cap_*.json files from training/encode_vidgen.py> --split train --out clip_ids_train.csv
  python data/make_clip_ids.py --shards <held-out dir> --split held_out --out clip_ids_held_out.csv

Each cap_<zip>_<part>.json holds one record per encoded window: {"vid", "start", "caption"}. Only `vid` (VidGen-1M's clip
name: a YouTube ID plus a scene index), the window start in seconds and the zip number are written. Captions are never
written: they come from VidGen-1M (made with VILA, a non-commercial model); get them from VidGen-1M itself.
Rows follow the shard order the trainer loads (sorted file names), so row i is the trainer's window i.
"""
import argparse, csv, glob, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True); ap.add_argument("--split", required=True, choices=["train", "held_out"])
ap.add_argument("--out", required=True); ap.add_argument("--expect", type=int, default=0, help="fail unless this many rows")
a = ap.parse_args()

n = 0
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["vid", "start_s", "zip", "split"])
    # same order as training/train_t2v.py Shards (sorted lat_*.npy; the cap_*.json files sort identically)
    for cp in sorted(glob.glob(os.path.join(a.shards, "cap_*.json"))):
        m = re.match(r"cap_(\d+)", os.path.basename(cp))
        for r in json.load(open(cp)):
            w.writerow([r["vid"], round(float(r["start"]), 3), int(m.group(1)) if m else "", a.split]); n += 1
print(a.out, n, "windows")
assert not a.expect or n == a.expect, f"expected {a.expect} rows, got {n}"
