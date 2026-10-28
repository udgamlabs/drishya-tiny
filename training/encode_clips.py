"""Encode SkyTimelapse windows into Wan2.2 latent shards (+ throughput log). Stage 1 step 2.

  python encode_clips.py --root /workspace/data/sky_timelapse --split sky_train --out /workspace/lat/train \
      --frames 17 --stride 8 --max_minutes 40
  python encode_clips.py ... --bench   # fp32 vs bf16 speed + round-trip PSNR on 16 windows -> bench.json, then exit

Writes <out>/lat_00000.npy ... (N,48,T',16,16) float16 normalised latents, <out>/index.json (clip dir, start),
<out>/encode_log.json (windows, windows/s, load vs GPU seconds, stop reason).
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from sky import list_clips, train_windows, load_frames  # noqa: E402
from vae import VAE  # noqa: E402


class Windows(Dataset):
    def __init__(self, clips, wins, n, size):
        self.clips, self.wins, self.n, self.size = clips, wins, n, size

    def __len__(self):
        return len(self.wins)

    def __getitem__(self, i):
        ci, s = self.wins[i]
        return load_frames(self.clips[ci][1][s:s + self.n], self.size), i


def psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="sky_train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0, help="first window start (Stage 2 images: 24 = frames between windows)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--shard", type=int, default=2048)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench_n", type=int, default=16)
    ap.add_argument("--shuffle_seed", type=int, default=0, help="-1 = keep sorted order")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clips = list_clips(os.path.join(a.root, a.split), min_frames=max(a.frames, 17))  # same clips as the video windows
    wins = train_windows(clips, a.frames, a.stride, a.offset)
    if a.shuffle_seed >= 0:  # seeded random order, so a time cap still samples every clip evenly
        import random
        random.Random(a.shuffle_seed).shuffle(wins)
    if a.max_windows:
        wins = wins[: a.max_windows]
    print(json.dumps({"clips": len(clips), "windows": len(wins)}), flush=True)

    if a.bench:  # pre-registered rule: use bf16 only if >= 1.3x faster and round-trip PSNR within 0.1 dB of fp32
        x = torch.stack([load_frames(clips[ci][1][s:s + a.frames], a.size) for ci, s in wins[: a.bench_n]])
        res = {}
        for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            v = VAE(a.device, dt)
            v.encode(x[: a.batch]); sync(a.device)  # warm-up
            t0 = time.time(); zs = [v.encode(x[i:i + a.batch]) for i in range(0, a.bench_n, a.batch)]; sync(a.device)
            te = time.time() - t0
            t0 = time.time(); ys = [v.decode(z) for z in zs]; sync(a.device)
            td = time.time() - t0
            y = torch.cat(ys)[:, : a.frames]
            res[name] = dict(enc_clips_per_s=a.bench_n / te, dec_clips_per_s=a.bench_n / td, psnr=float(np.mean([psnr(x[i], y[i]) for i in range(a.bench_n)])),
                             peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9 if a.device == "cuda" else None)
            del v
            if a.device == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        res["use_bf16"] = bool(res["bf16"]["enc_clips_per_s"] >= 1.3 * res["fp32"]["enc_clips_per_s"]
                               and res["fp32"]["psnr"] - res["bf16"]["psnr"] <= 0.1)
        json.dump(res, open(os.path.join(a.out, "bench.json"), "w"), indent=1)
        print(json.dumps(res), flush=True)
        return

    vae = VAE(a.device, torch.float32 if a.dtype == "fp32" else torch.bfloat16)
    dl = DataLoader(Windows(clips, wins, a.frames, a.size), batch_size=a.batch, num_workers=a.workers,
                    shuffle=False, drop_last=False, prefetch_factor=4 if a.workers else None)
    buf, idx, n_done, shard_i, gpu_s, t0, stop = [], [], 0, 0, 0.0, time.time(), "all windows"
    t_wait = time.time()
    load_s = 0.0

    def flush():
        nonlocal buf, shard_i
        if buf:
            np.save(os.path.join(a.out, f"lat_{shard_i:05d}.npy"), np.concatenate(buf).astype(np.float16))
            shard_i += 1; buf = []

    for x, ids in dl:
        load_s += time.time() - t_wait
        g0 = time.time()
        z = vae.encode(x); sync(a.device)
        gpu_s += time.time() - g0
        buf.append(z.cpu().numpy()); idx += [dict(clip=os.path.relpath(clips[wins[i][0]][0], a.root), start=wins[i][1]) for i in ids.tolist()]
        n_done += x.shape[0]
        if sum(b.shape[0] for b in buf) >= a.shard:
            flush()
        if n_done % (a.batch * 100) == 0:
            el = time.time() - t0
            print(json.dumps({"windows": n_done, "per_s": round(n_done / el, 2), "gpu_s": round(gpu_s, 1),
                              "load_wait_s": round(load_s, 1), "elapsed_min": round(el / 60, 2)}), flush=True)
        if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
            stop = f"time cap {a.max_minutes} min"; break
        t_wait = time.time()
    flush()
    el = time.time() - t0
    log = dict(split=a.split, frames=a.frames, stride=a.stride, dtype=a.dtype, windows=n_done, windows_total=len(wins),
               clips=len(clips), per_s=n_done / el, gpu_per_s=n_done / max(gpu_s, 1e-9), load_wait_s=load_s,
               minutes=el / 60, stop=stop, shards=shard_i)
    json.dump(idx, open(os.path.join(a.out, "index.json"), "w"))
    json.dump(log, open(os.path.join(a.out, "encode_log.json"), "w"), indent=1)
    print(json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
