"""Stage 3 data: stream VidGen-1M zips -> Wan2.2 latents + captions, zip by zip (download next zip while encoding).

  python encode_vidgen.py --zips 0-599 --out /workspace/lat/train --windows 2 --frames 17 --fps 8 --max_minutes 600
  python encode_vidgen.py --zips 2040-2047 --out /workspace/lat/val --windows 1 --real_frames /workspace/val_real.npy

Each clip gives up to --windows windows of --frames frames at --fps (17 frames at 8 fps = 2 s): window 0 starts at
0.5 s, window 1 in the middle of the clip (only if the clip is long enough), so the two never overlap.
Output shards: lat_XXXXX.npy (N,48,T',16,16) float16 normalised latents + cap_XXXXX.json (captions, clip ids, starts).
A zip marked done (<out>/done/<zip>) is skipped on restart; shards are named lat_<zip>_<part>, so several pods or a
resumed pod never collide. --real_frames also saves the 16-frame uint8 clips at
256px for the held-out FVD statistics.
"""
import argparse, json, os, shutil, sys, threading, time, zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from videoio import probe, load_clip  # noqa: E402
from vae import VAE  # noqa: E402

REPO = "Fudan-FUXI/VIDGEN-1M"


def _one_thread(_):
    torch.set_num_threads(1)
    if "VIDEO_DECODE_THREADS" not in os.environ:
        os.environ["VIDEO_DECODE_THREADS"] = "1"


class Win(Dataset):
    def __init__(self, items, frames, fps, size):
        self.items, self.frames, self.fps, self.size = items, frames, fps, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, start = self.items[i]
        try:
            x = load_clip(path, n=self.frames, fps=self.fps, size=self.size, start=start, fast=True)
        except Exception:
            x = None
        if x is None:
            x = torch.zeros(self.frames, 3, self.size, self.size, dtype=torch.uint8); i = -1
        return x, i


def windows(paths, frames, fps, nwin):
    need = (frames - 1) / fps + 0.6
    out = []
    for p in paths:
        try:
            src_fps, n, _, _ = probe(p)
            dur = n / src_fps if src_fps else 0
        except Exception:
            continue
        if dur >= 0.5 + need:
            out.append((p, 0.5))
        if nwin > 1 and dur >= 2 * need + 1.0:
            out.append((p, max(0.5 + need, dur / 2 - need / 2)))
    return out


def parse_range(s):
    a, b = s.split("-") if "-" in s else (s, s)
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/workspace/zips")
    ap.add_argument("--windows", type=int, default=2); ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--fps", type=float, default=8.0); ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=4096); ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"]); ap.add_argument("--device", default="cuda")
    ap.add_argument("--real_frames", default="", help="also save 16-frame uint8 clips here (held-out FVD set)")
    ap.add_argument("--local_dir", default="", help="TEST: use local mp4s instead of downloading zips")
    ap.add_argument("--clip_offset", type=int, default=0); ap.add_argument("--clip_limit", type=int, default=0)
    ap.add_argument("--guard_per_s", type=float, default=0, help="stop (exit 3) if windows/s over the first group is below this")
    ap.add_argument("--group", type=int, default=4, help="zips per loader (amortises worker start-up)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(os.path.join(a.out, "done"), exist_ok=True); os.makedirs(a.work, exist_ok=True)
    caps = {}
    if not a.local_dir:
        from huggingface_hub import hf_hub_download
        cj = hf_hub_download(REPO, "VidGen_1M_video_caption.json", repo_type="dataset", local_dir=a.work)
    else:
        cj = os.path.join(os.path.dirname(a.local_dir.rstrip("/")), "VidGen_1M_video_caption.json")
    for r in json.load(open(cj)):
        caps[r["vid"]] = r["caption"]
    vae = VAE(a.device, torch.float32 if a.dtype == "fp32" else torch.bfloat16)
    zips = parse_range(a.zips)
    buf_l, buf_c, real = [], [], []
    cur_zip, part = None, 0
    t0, n_done, gpu_s, stop = time.time(), 0, 0.0, "all zips"
    log = open(os.path.join(a.out, "encode_log.jsonl"), "a")

    def fetch(k):  # download + extract one zip
        d = os.path.join(a.work, f"z{k}")
        if a.local_dir:
            return a.local_dir
        from huggingface_hub import hf_hub_download
        z = hf_hub_download(REPO, f"VidGen_video_{k}.zip", repo_type="dataset", local_dir=a.work)
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
        os.remove(z)
        return d

    def flush():  # shards are named by zip, so pods/restarts never collide: lat_<zip>_<part>.npy + cap_<zip>_<part>.json
        nonlocal part, buf_l, buf_c
        if buf_l:
            np.save(os.path.join(a.out, f"lat_{cur_zip:04d}_{part:02d}.npy"), np.concatenate(buf_l).astype(np.float16))
            json.dump(buf_c, open(os.path.join(a.out, f"cap_{cur_zip:04d}_{part:02d}.json"), "w"))
            part += 1; buf_l, buf_c = [], []

    todo = [k for k in zips if not os.path.exists(os.path.join(a.out, "done", str(k)))]
    groups = [todo[i:i + a.group] for i in range(0, len(todo), a.group)]  # one loader per group of zips (worker start-up
    nxt = {}                                                                # cost is paid once per ~4 zips, not per zip)

    def fetch_group(g):
        for k in g:
            nxt[k] = fetch(k)

    th = None
    if groups:
        th = threading.Thread(target=fetch_group, args=(groups[0],)); th.start()
    from concurrent.futures import ThreadPoolExecutor
    for gi, g in enumerate(groups):
        th.join()
        dirs = {k: nxt.pop(k) for k in g}
        if gi + 1 < len(groups):  # prefetch the next group while this one encodes
            th = threading.Thread(target=fetch_group, args=(groups[gi + 1],)); th.start()
        items, zip_of = [], []
        for k in g:
            d = dirs[k]
            paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(".mp4"))
            paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in caps]
            if a.clip_limit:  # tests / held-out splits of a local folder
                paths = paths[a.clip_offset: a.clip_offset + a.clip_limit]
            with ThreadPoolExecutor(32) as ex:  # clip-length checks in parallel
                its = [w for ws in ex.map(lambda p: windows([p], a.frames, a.fps, a.windows), paths) for w in ws]
            items += its; zip_of += [k] * len(its)
        dl = DataLoader(Win(items, a.frames, a.fps, a.size), batch_size=a.batch, num_workers=a.workers,
                        prefetch_factor=4 if a.workers else None, worker_init_fn=_one_thread)  # no oversubscription
        gt = time.time(); per_zip = {k: ([], []) for k in g}
        for x, idx in dl:
            keep = idx >= 0
            if keep.sum() == 0:
                continue
            x, idx = x[keep], idx[keep]
            g0 = time.time()
            with torch.no_grad():
                z = vae.encode(x).cpu().numpy()
            gpu_s += time.time() - g0
            for row, j in enumerate(idx.tolist()):
                p, st = items[j]; vid = os.path.splitext(os.path.basename(p))[0]
                L, Cp = per_zip[zip_of[j]]; L.append(z[row:row + 1]); Cp.append({"vid": vid, "start": st, "caption": caps[vid]})
            if a.real_frames and len(real) < 2048:
                real += [v[:16] for v in x[: 2048 - len(real)]]
            n_done += len(idx)
        for k in g:  # one shard per zip (names never collide across pods), then its done marker
            L, Cp = per_zip[k]
            if L:
                np.save(os.path.join(a.out, f"lat_{k:04d}_00.npy"), np.concatenate(L).astype(np.float16))
                json.dump(Cp, open(os.path.join(a.out, f"cap_{k:04d}_00.json"), "w"))
            if not a.local_dir:
                shutil.rmtree(dirs[k], ignore_errors=True)
            open(os.path.join(a.out, "done", str(k)), "w").write(str(len(L)))
        rec = {"zips": g, "windows": sum(len(per_zip[k][0]) for k in g), "group_s": round(time.time() - gt, 1), "total": n_done,
               "per_s": round(n_done / (time.time() - t0), 2), "gpu_per_s": round(n_done / max(gpu_s, 1e-9), 2),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if a.guard_per_s and gi == 0 and rec["per_s"] < a.guard_per_s:
            print(json.dumps({"STOP": "SLOW_HOST", "per_s": rec["per_s"], "need": a.guard_per_s}), flush=True); sys.exit(3)
        if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
            stop = f"time cap {a.max_minutes} min"; break
    flush()
    if a.real_frames and real:
        np.save(a.real_frames, torch.stack(real).numpy())
    summ = {"windows": n_done, "zips_done": len(os.listdir(os.path.join(a.out, "done"))), "minutes": round((time.time() - t0) / 60, 1),
            "per_s": round(n_done / max(time.time() - t0, 1e-9), 2), "stop": stop,
            "shards": len([f for f in os.listdir(a.out) if f.startswith("lat_")])}
    json.dump(summ, open(os.path.join(a.out, "encode_summary.json"), "w"), indent=1); print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
