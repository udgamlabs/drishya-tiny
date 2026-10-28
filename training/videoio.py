"""Clip loading for Drishya Tiny: decode an mp4, take `n` frames at ~`fps`, centre-crop square, resize to `size`.

Returns uint8 tensors (T, 3, size, size); `to_vae` maps them to (3, T, H, W) floats in [-1, 1].
"""
import math
import os

import av
import torch
import torch.nn.functional as F


def probe(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 25)
        n = s.frames or int(float(c.duration or 0) / 1e6 * fps)
        return fps, n, s.width, s.height


def load_clip(path, n=33, fps=16.0, size=256, start=0.0, seek=True, fast=False):
    """Frame i of the output is the source frame nearest to start + i/fps seconds. None if the clip is too short.
    seek=True jumps to the keyframe before `start` instead of decoding from 0 (Stage 3: mid-clip windows)."""
    with av.open(path) as c:
        s = c.streams.video[0]
        if os.environ.get("VIDEO_DECODE_THREADS", "auto") == "auto":
            s.thread_type = "AUTO"
        else:  # many loader processes: one decode thread each (avoids oversubscription)
            s.thread_count = int(os.environ["VIDEO_DECODE_THREADS"])
        src_fps = float(s.average_rate or s.guessed_rate or 25)
        want = [round((start + i / fps) * src_fps) for i in range(n)]
        if seek and start > 0.5 and s.time_base:
            c.seek(int((start - 0.25) / s.time_base), stream=s, backward=True, any_frame=False)
        frames, k = [], 0
        for fr in c.decode(s):
            if fr.pts is None:
                continue
            i = round(float(fr.pts * s.time_base) * src_fps)  # source frame index from the timestamp
            if k < n and want[k] <= i:
                if fast:  # Stage 3: libswscale scales the short side to `size` during RGB conversion (~3x cheaper)
                    sc = size / min(fr.width, fr.height)
                    img = torch.from_numpy(fr.to_ndarray(format="rgb24", width=max(size, round(fr.width * sc)),
                                                         height=max(size, round(fr.height * sc)), interpolation="AREA"))
                else:
                    img = torch.from_numpy(fr.to_ndarray(format="rgb24"))
            while k < n and want[k] < i:  # a wanted frame was skipped (should not happen after a backward seek)
                frames.append(frames[-1] if frames else img); k += 1
            while k < n and want[k] == i:
                frames.append(img)
                k += 1
            if k == n:
                break
    if len(frames) < n:
        return None
    x = torch.stack(frames).permute(0, 3, 1, 2)  # (T,3,H,W) uint8
    H, W = x.shape[-2:]
    m = min(H, W)
    x = x[..., (H - m) // 2:(H - m) // 2 + m, (W - m) // 2:(W - m) // 2 + m]
    if x.shape[-1] == size:  # fast path: already scaled
        return x.contiguous()
    x = F.interpolate(x.float(), size=(size, size), mode="bicubic", antialias=True, align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8)


def to_vae(x_u8):
    """(T,3,H,W) uint8 -> (3,T,H,W) float in [-1,1]."""
    return x_u8.permute(1, 0, 2, 3).float() / 127.5 - 1.0


def from_vae(x):
    """(3,T,H,W) float in [-1,1] -> (T,3,H,W) uint8."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 0, 2, 3)


def psnr(a_u8, b_u8):
    mse = (a_u8.float() - b_u8.float()).pow(2).mean().item()
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
