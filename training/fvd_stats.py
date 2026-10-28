"""Real-video FVD statistics and reference baselines for SkyTimelapse (Stage 1 step 3).

  python fvd_stats.py --root /workspace/data/sky_timelapse --i3d /workspace/data/i3d_torchscript.pt --out realstats.npz

- train_real: StyleGAN-V / Latte real set (first 2,048 train clips with >= 16 frames, one random 16-frame window each,
  random h-flip). Generated videos are scored against this.
- test_real: 2,048 windows cycled over the 224 held-out test clips (real-vs-real reference: the best achievable).
- test_static: the same test windows frozen on their first frame (a "perfect appearance, zero motion" baseline that a
  model must beat; P0 showed FVD alone under-weights motion).
Writes the I3D features, per-clip motion, and refs.json with the reference FVDs.
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import fvd as F_  # noqa: E402
from sky import list_clips, fvd_real_specs, windows_cycle_specs, load_specs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--i3d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    det = F_.load_detector(a.i3d, a.device)
    tr = list_clips(os.path.join(a.root, "sky_train"), 16)
    te = list_clips(os.path.join(a.root, "sky_test"), 16)
    t0 = time.time()
    real = load_specs(fvd_real_specs(tr, a.n, 16, seed=0, mirror=True))
    test = load_specs(windows_cycle_specs(te, a.n, 16, seed=1))
    static = [v[:1].expand_as(v).contiguous() for v in test]
    f_tr = F_.features(det, real, a.device); f_te = F_.features(det, test, a.device); f_st = F_.features(det, static, a.device)
    m_tr = np.array([F_.motion(v) for v in real]); m_te = np.array([F_.motion(v) for v in test])
    np.savez(a.out, f_train=f_tr, f_test=f_te, f_test_static=f_st, motion_train=m_tr, motion_test=m_te)
    refs = dict(n_train_real=len(real), n_test=len(test), train_clips=len(tr), test_clips=len(te),
                fvd_train_vs_test=F_.fvd(f_tr, f_te), fvd_train_vs_static_test=F_.fvd(f_tr, f_st),
                motion_train=float(m_tr.mean()), motion_test=float(m_te.mean()), seconds=time.time() - t0)
    json.dump(refs, open(os.path.splitext(a.out)[0] + "_refs.json", "w"), indent=1)
    print(json.dumps(refs), flush=True)


if __name__ == "__main__":
    main()
