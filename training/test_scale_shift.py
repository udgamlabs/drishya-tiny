"""Check that scale_shift() gives bit-identical outputs to the textbook adaLN form x * (1 + scale) + shift.

  python training/test_scale_shift.py                     # random weights, CPU
  python training/test_scale_shift.py --ckpt last.pt      # also the full model with trained weights (EMA), CPU

The model is run twice on the same inputs: once as written, once with scale_shift swapped for the reference form.
Pass = torch.equal on every output (no tolerance).
"""
import argparse, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmodel  # noqa: E402
from vmodel import VideoDiT, VideoDiTConfig  # noqa: E402


def reference(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def run(model, inputs, fn):
    saved = vmodel.scale_shift
    vmodel.scale_shift = fn
    try:
        with torch.no_grad():
            return model(*inputs)
    finally:
        vmodel.scale_shift = saved


def check(model, B=2, seed=0):
    c = model.cfg
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, c.in_ch, c.frames, c.grid, c.grid, generator=g)
    t = torch.rand(B, generator=g)
    txt = torch.randn(B, 77, c.txt_dim, generator=g)
    mask = torch.ones(B, 77, dtype=torch.bool); mask[:, 40:] = False
    cond = torch.tensor([True, False][:B])
    for dt in (torch.float32, torch.bfloat16):
        with torch.autocast("cpu", dtype=dt, enabled=dt != torch.float32):
            a = run(model, (x, t, txt, mask, cond), vmodel.scale_shift)
            b = run(model, (x, t, txt, mask, cond), reference)
        assert torch.equal(a, b), f"outputs differ ({dt})"
    # the helper itself, on awkward values
    h = torch.randn(3, 5, 8, generator=g) * 1e3
    s, sc = torch.randn(3, 8, generator=g), torch.randn(3, 8, generator=g) * 1e-4
    assert torch.equal(vmodel.scale_shift(h, s, sc), reference(h, s, sc))
    return a.abs().mean().item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=""); a = ap.parse_args()
    torch.manual_seed(0)
    m = VideoDiT(VideoDiTConfig(width=96, depth=2, heads=4, frames=5)).eval()
    for p in m.parameters():  # non-zero adaLN weights, so the modulation actually matters
        torch.nn.init.normal_(p, std=0.05)
    print("random weights: identical, mean |v| =", round(check(m), 6))
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
        m = VideoDiT(VideoDiTConfig(**ck["cfg"])); m.load_state_dict(ck["ema"]); m.eval()
        print("checkpoint EMA weights (step %s): identical, mean |v| = %.6f" % (ck.get("step"), check(m)))
