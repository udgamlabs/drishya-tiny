"""Extract a small SkyTimelapse subset from the Hugging Face zip (range reads, no full download) for local tests.
Usage: python data/fetch_sky_subset.py --train 24 --test 8 --out data/sky_subset"""
import argparse, os, zipfile
from huggingface_hub import HfFileSystem
ap = argparse.ArgumentParser(); ap.add_argument("--train", type=int, default=24); ap.add_argument("--test", type=int, default=8)
ap.add_argument("--out", default="data/sky_subset"); a = ap.parse_args()
fs = HfFileSystem()
with fs.open("datasets/maxin-cn/SkyTimelapse/sky_timelapse.zip", "rb", block_size=2**22) as f:
    z = zipfile.ZipFile(f)
    names = z.namelist()
    for split, k in (("sky_train", a.train), ("sky_test", a.test)):
        leaves = sorted({n.rsplit("/", 1)[0] for n in names if f"/{split}/" in n and n.lower().endswith(".jpg")})
        # spread the picks over the split; keep clips with >= 33 frames
        cnt = {}
        for n in names:
            if f"/{split}/" in n and n.lower().endswith(".jpg"):
                cnt[n.rsplit("/", 1)[0]] = cnt.get(n.rsplit("/", 1)[0], 0) + 1
        good = [l for l in leaves if cnt[l] >= 33]
        pick = good[:: max(1, len(good) // k)][:k]
        for l in pick:
            for n in names:
                if n.startswith(l + "/") and n.lower().endswith(".jpg"):
                    z.extract(n, a.out)
        print(split, "clips", len(pick), "frames", sum(cnt[l] for l in pick), flush=True)
