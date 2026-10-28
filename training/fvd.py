"""Frechet Video Distance with the StyleGAN-V / Latte protocol.

Matches stylegan-v/src/metrics/frechet_video_distance.py (commit on master, read 2026-10-18):
- detector: the torchscript I3D at https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt
  (a reproduction of the tf.hub Kinetics-400 I3D used by the original FVD);
- input: uint8-range videos (B, 3, T, H, W); detector called with rescale=True, resize=True, return_features=True;
- stats: mean and population covariance (divide by N) of the 400-d features; distance via scipy.linalg.sqrtm.
The paper protocol for 256px is 2,048 clips x 16 frames on each side; fewer clips bias FVD upward.
"""
import numpy as np
import scipy.linalg
import torch

DETECTOR_KW = dict(rescale=True, resize=True, return_features=True)


def load_detector(path, device="cpu"):
    return torch.jit.load(path, map_location="cpu").eval().to(device)


@torch.no_grad()
def features(det, videos, device="cpu", batch=8):
    """videos: iterable of (T,3,H,W) uint8 tensors -> (N, 400) float64 numpy."""
    out, buf = [], []

    def flush():
        x = torch.stack(buf).permute(0, 2, 1, 3, 4).contiguous().to(device)  # (B,3,T,H,W) uint8
        out.append(det(x, **DETECTOR_KW).float().cpu().double().numpy())  # MPS has no float64: cast on CPU
        buf.clear()

    for v in videos:
        buf.append(v)
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    return np.concatenate(out)


def stats(f):
    mu = f.mean(0)
    return mu, (f.T @ f) / f.shape[0] - np.outer(mu, mu)


def frechet(s1, s2):
    (m1, c1), (m2, c2) = s1, s2
    s = scipy.linalg.sqrtm(c1 @ c2)  # scipy >= 1.16 dropped disp=; older versions return (s, err) only with disp=False
    s = s[0] if isinstance(s, tuple) else s
    return float(np.real(np.square(m1 - m2).sum() + np.trace(c1 + c2 - 2 * s)))


def fvd(f_real, f_gen):
    return frechet(stats(f_real), stats(f_gen))


def motion(v):
    """Mean absolute change between consecutive frames, in [0,1]. v: (T,3,H,W) uint8. A still video scores 0.
    FVD under-weights motion (P0 check), so every FVD is reported next to motion(gen) / motion(real)."""
    x = v.float() / 255.0
    return (x[1:] - x[:-1]).abs().mean().item()
