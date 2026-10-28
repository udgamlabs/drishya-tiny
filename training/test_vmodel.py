"""Offline tests for vmodel.py (random tensors, CPU). Run: .venv/bin/python train/test_vmodel.py"""
import torch
from vmodel import VideoDiT, VideoDiTConfig, flow_inputs, euler_sample, sincos_3d

torch.manual_seed(0)
cfg = VideoDiTConfig(width=96, depth=2, heads=4)
m = VideoDiT(cfg)
x = torch.randn(3, 48, 9, 16, 16)
assert torch.equal(m.unpatchify(m.patchify(x)), x), "patchify round trip"
assert m.patchify(x).shape == (3, 576, 192)
assert sincos_3d(96, 9, 8, 8).shape == (576, 96)
txt, mask = torch.randn(3, 7, 768), torch.ones(3, 7, dtype=torch.bool)
t = torch.rand(3); n = torch.randn_like(x); cond = torch.tensor([True, False, True])
xt, tgt, lm = flow_inputs(x, t, n, cond)
assert torch.equal(xt[0, :, 0], x[0, :, 0]) and torch.equal(xt[2, :, 0], x[2, :, 0]), "clean first frame when cond"
assert not torch.equal(xt[1, :, 0], x[1, :, 0]), "noisy first frame without cond"
assert lm.shape == (3, 1, 9, 1, 1) and lm[0, 0, 0].item() == 0 and lm[1, 0, 0].item() == 1 and lm[0, 0, 1].item() == 1
v = m(xt, t, txt, mask, cond)
assert v.shape == x.shape and v.abs().max() == 0, "zero-init output"
# one step of training must change the output, and the cond embedding must matter after training
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
for _ in range(3):
    loss = (((m(xt, t, txt, mask, cond) - tgt) ** 2) * lm).mean(); opt.zero_grad(); loss.backward(); opt.step()
v1 = m(xt, t, txt, mask, cond); v2 = m(xt, t, txt, mask, None)
assert v1.abs().max() > 0 and not torch.allclose(v1[0], v2[0]), "cond embedding has an effect"
first = x[:, :, :1]
s = euler_sample(m.eval(), txt, mask, (48, 9, 16, 16), steps=3, first=first)
assert s.shape == x.shape and torch.equal(s[:, :, :1], first), "i2v keeps the given first frame"
m.cfg.tread_ratio = 0.5; m.train(); assert m(xt, t, txt, mask, cond).shape == x.shape, "TREAD path"
print("all vmodel tests passed; params(width 384, depth 8) =",
      sum(p.numel() for p in VideoDiT(VideoDiTConfig()).parameters()) / 1e6, "M")
