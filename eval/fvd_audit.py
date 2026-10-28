"""FVD harness audit: does our FVD harness (training/fvd.py) compute the same number as StyleGAN-V's official code?

Builds 16-frame 256x256 PNG clips from local SkyTimelapse frames (data/fetch_sky_subset.py gives a small subset) (non-overlapping windows; centre-crop to the short
side, then PIL bicubic resize to 256, the usual Sky preprocessing), splits them into two halves A and B, and computes:
  1. FVD(A, B) and FVD(A, still(B)) with the official StyleGAN-V code (metric fvd2048_16f, run unchanged except two
     shims: CPU instead of cuda, and scipy>=1.16's sqrtm, which dropped disp=; the I3D file is our sha-checked copy of
     the official torchscript detector instead of a fresh Dropbox download);
  2. the same two numbers with our harness on the same PNG frames.
Pass = they agree to within 0.5% (float rounding only). Writes audit_result.json to --out.
StyleGAN-V is not included here (it has no licence file); clone https://github.com/universome/stylegan-v yourself.

  python eval/fvd_audit.py --stylegan <stylegan-v clone> --work <scratch dir> --sky <sky_timelapse dir> --i3d i3d_torchscript.pt
"""
import argparse, json, os, sys, glob

import numpy as np
from PIL import Image
import torch
import multiprocessing as mp
mp.set_start_method("fork", force=True)  # macOS defaults to spawn; the official loader uses 3 workers (Linux = fork)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import fvd as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stylegan", required=True); ap.add_argument("--work", required=True)
ap.add_argument("--sky", required=True, help="SkyTimelapse root (folders of frames)")
ap.add_argument("--i3d", required=True, help="the StyleGAN-V I3D torchscript detector")
ap.add_argument("--out", default="audit_result.json")
ap.add_argument("--max_per_half", type=int, default=2048)
a = ap.parse_args()


def prep(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((256, 256), Image.BICUBIC)


# 1. build the PNG clip folders (once)
A, B, BS = (os.path.join(a.work, x) for x in ("A", "B", "B_still"))
if not os.path.exists(os.path.join(a.work, "built.json")):
    clips = []
    for d in sorted(glob.glob(os.path.join(a.sky, "*", "*", "*"))):
        fs = sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith((".jpg", ".png")))
        clips += [fs[i:i + 16] for i in range(0, len(fs) - 15, 16)]
    half = min(len(clips) // 2, a.max_per_half)
    for k, c in enumerate(clips[: 2 * half]):
        dst = A if k % 2 == 0 else B
        od = os.path.join(dst, f"clip{k:05d}"); os.makedirs(od, exist_ok=True)
        ims = [prep(p) for p in c]
        for j, im in enumerate(ims):
            im.save(os.path.join(od, f"{j:03d}.png"))
        if dst == B:
            sd = os.path.join(BS, f"clip{k:05d}"); os.makedirs(sd, exist_ok=True)
            for j in range(16):
                ims[0].save(os.path.join(sd, f"{j:03d}.png"))
    json.dump({"clips_per_half": half}, open(os.path.join(a.work, "built.json"), "w"))
half = json.load(open(os.path.join(a.work, "built.json")))["clips_per_half"]
print("clips per half", half, flush=True)

# 2. official StyleGAN-V code
sys.path.insert(0, a.stylegan); sys.path.insert(0, os.path.join(a.stylegan, "src"))  # "src.*" and the "training.*" class names
import scipy.linalg  # noqa: E402
_sqrtm = scipy.linalg.sqrtm
def sqrtm_compat(x, disp=True, **kw):  # scipy >= 1.16 removed disp; the official call passes disp=False and unpacks 2 values
    s = _sqrtm(x, **kw); s = s[0] if isinstance(s, tuple) else s
    return (s, 0.0) if disp is False else s
scipy.linalg.sqrtm = sqrtm_compat
from omegaconf import OmegaConf  # noqa: E402
from src import dnnlib  # noqa: E402
from src.metrics import metric_main, metric_utils  # noqa: E402
_det = torch.jit.load(a.i3d, map_location="cpu").eval()
metric_utils.get_feature_detector = lambda url, device=None, num_gpus=1, rank=0, verbose=False: _det


def official(real, fake):
    cfg = OmegaConf.create({"max_num_frames": 10000})
    dk = dnnlib.EasyDict(class_name="training.dataset.VideoFramesFolderDataset", path=real, cfg=cfg, xflip=False, resolution=256, use_labels=False)
    gk = dnnlib.EasyDict(class_name="training.dataset.VideoFramesFolderDataset", path=fake, cfg=cfg, xflip=False, resolution=256, use_labels=False)
    r = metric_main.calc_metric(metric="fvd2048_16f", dataset_kwargs=dk, gen_dataset_kwargs=gk, generator_as_dataset=True,
                                num_gpus=1, rank=0, device=torch.device("cpu"), progress=metric_utils.ProgressMonitor(verbose=False),
                                cache=False, num_runs=1)
    return float(r["results"]["fvd2048_16f"])


# 3. our harness on the same frames
def load_dir(root):
    return [torch.from_numpy(np.stack([np.array(Image.open(f)) for f in sorted(glob.glob(os.path.join(d, "*.png")))])).permute(0, 3, 1, 2).contiguous()
            for d in sorted(glob.glob(os.path.join(root, "clip*")))]


det = F.load_detector(a.i3d, "cpu")
FC = os.path.join(a.work, "ours_features.npz")
if os.path.exists(FC):
    z = np.load(FC); fA, fB, fS = z["A"], z["B"], z["S"]
else:
    fA = F.features(det, load_dir(A)); fB = F.features(det, load_dir(B)); fS = F.features(det, load_dir(BS))
    np.savez(FC, A=fA, B=fB, S=fS)
ours = {"A_vs_B": F.fvd(fA, fB), "A_vs_still": F.fvd(fA, fS)}
print("ours", ours, flush=True)
off = {"A_vs_B": official(A, B), "A_vs_still": official(A, BS)}
print("official", off, flush=True)
rel = {k: abs(ours[k] - off[k]) / max(abs(off[k]), 1e-9) for k in ours}
res = {"clips_per_half": half, "official": off, "ours": ours, "rel_diff": rel, "pass": all(v < 0.005 for v in rel.values()),
       "note": "Equivalence of the computation on identical frames; not the published Sky benchmark numbers (local subset, ~32 source videos)."}
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
