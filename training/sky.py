"""SkyTimelapse (Latte authors' copy, HF maxin-cn/SkyTimelapse) frame-folder loading.

Layout after unzip: <root>/sky_train/<video>/<video>_<k>/*.jpg and <root>/sky_test/... ; each leaf folder is one clip.
Frames are centre-cropped to a square and resized to 256px ("center-crop-resize-256", as in Latte's evaluation).

- `list_clips(split_dir, min_frames)` -> sorted list of (clip_dir, [frame paths]).
- `train_windows(clips, n, stride)` -> (clip index, start) windows of n consecutive frames.
- `load_frames(paths, size)` -> (T, 3, size, size) uint8.
- `fvd_real_clips(clips, n_clips, frames, seed, mirror)` -> the StyleGAN-V / Latte real set: the first `n_clips` clips
  (sorted) that have >= `frames` frames, one random window of `frames` consecutive frames each, random h-flip if mirror.
"""
import glob, os, random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def list_clips(split_dir, min_frames=1):
    out = []
    for d in sorted(glob.glob(os.path.join(split_dir, "*", "*"))):
        if not os.path.isdir(d):
            continue
        fr = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))
        if len(fr) >= min_frames:
            out.append((d, [os.path.join(d, f) for f in fr]))
    return out


def train_windows(clips, n=17, stride=8, offset=0):
    """offset > 0 picks frames between the stride-spaced windows (Stage 2: unseen frames as images)."""
    return [(ci, s) for ci, (_, fr) in enumerate(clips) for s in range(offset, len(fr) - n + 1, stride)]


def load_frames(paths, size=256):
    ims = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2)  # (T,3,H,W)
    H, W = x.shape[-2:]
    m = min(H, W)
    x = x[..., (H - m) // 2:(H - m) // 2 + m, (W - m) // 2:(W - m) // 2 + m]
    x = F.interpolate(x.float(), size=(size, size), mode="bicubic", antialias=True, align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8)


def fvd_real_clips(clips, n_clips=2048, frames=16, seed=0, mirror=True, size=256):
    """Yields (T,3,H,W) uint8 clips; deterministic for a given seed."""
    rng = random.Random(seed)
    eligible = [c for c in clips if len(c[1]) >= frames][:n_clips]
    for _, fr in eligible:
        s = rng.randint(0, len(fr) - frames)
        x = load_frames(fr[s:s + frames], size)
        if mirror and rng.random() < 0.5:
            x = x.flip(-1)
        yield x


def windows_cycle(clips, n_clips, frames=16, seed=0, size=256):
    """Held-out reference set with more clips than videos: cycles through clips with fresh random windows."""
    rng = random.Random(seed)
    eligible = [c for c in clips if len(c[1]) >= frames]
    for i in range(n_clips):
        fr = eligible[i % len(eligible)][1]
        s = rng.randint(0, len(fr) - frames)
        yield load_frames(fr[s:s + frames], size)


def fvd_real_specs(clips, n_clips=2048, frames=16, seed=0, mirror=True):
    """Same selection as fvd_real_clips, as (frame paths, flip) specs for parallel loading."""
    rng = random.Random(seed)
    out = []
    for _, fr in [c for c in clips if len(c[1]) >= frames][:n_clips]:
        s = rng.randint(0, len(fr) - frames)
        out.append((fr[s:s + frames], bool(mirror and rng.random() < 0.5)))
    return out


def windows_cycle_specs(clips, n_clips, frames=16, seed=0):
    rng = random.Random(seed)
    eligible = [c for c in clips if len(c[1]) >= frames]
    out = []
    for i in range(n_clips):
        fr = eligible[i % len(eligible)][1]
        s = rng.randint(0, len(fr) - frames)
        out.append((fr[s:s + frames], False))
    return out


def load_specs(specs, size=256, workers=16):
    from concurrent.futures import ThreadPoolExecutor

    def one(sp):
        x = load_frames(sp[0], size)
        return x.flip(-1) if sp[1] else x
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, specs))
