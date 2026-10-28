"""Offline tests for the Stage 2 arms (CPU, random tensors). Run: .venv/bin/python train/test_arms.py"""
import torch
from vmodel import VideoDiT, VideoDiTConfig, flow_inputs, meanflow_sample

torch.manual_seed(0)
# 1. images as 1-frame videos: shapes round-trip, same weights
m = VideoDiT(VideoDiTConfig(width=96, depth=2, heads=4))
for p in m.parameters():
    torch.nn.init.normal_(p, std=0.02)
txt, mask = torch.zeros(2, 1, 768), torch.ones(2, 1, dtype=torch.bool)
xi = torch.randn(2, 48, 1, 16, 16)
assert m(xi, torch.rand(2), txt, mask).shape == xi.shape, "T=1 forward"
# 2. manual attention == SDPA
ms = VideoDiT(VideoDiTConfig(width=96, depth=2, heads=4, attn="sdpa")); mm = VideoDiT(VideoDiTConfig(width=96, depth=2, heads=4, attn="manual"))
ms.load_state_dict(m.state_dict()); mm.load_state_dict(m.state_dict())
x = torch.randn(2, 48, 5, 16, 16); t = torch.rand(2)
assert torch.allclose(ms(x, t, txt, mask), mm(x, t, txt, mask), atol=1e-5), "manual attention matches SDPA"
# 3. MeanFlow: r = t gives the plain flow output at init (h_in output layer is zero-init); JVP matches finite differences
mf = VideoDiT(VideoDiTConfig(width=96, depth=2, heads=4, meanflow=True, attn="manual"))
sd = {k: v for k, v in m.state_dict().items()}; mf.load_state_dict(sd, strict=False)
assert torch.allclose(mf(x, t, txt, mask, r=t), m(x, t, txt, mask), atol=1e-5), "meanflow(r=t) == flow at init"
for p in mf.h_in.parameters():
    torch.nn.init.normal_(p, std=0.05)
r = t * 0.3; v = torch.randn_like(x)
# finite-difference check in float64 (the timestep embedding scales t by 1000, so float32 FD is too coarse)
mf64 = mf.double(); x64, r64, t64, v64 = x.double(), r.double(), t.double(), v.double()
fn64 = lambda z_, r_, t_: mf64(z_, t_, txt.double(), mask, None, r=r_)
_, d64 = torch.func.jvp(fn64, (x64, r64, t64), (v64, torch.zeros_like(r64), torch.ones_like(t64)))
eps = 1e-6
fd = (fn64(x64 + eps * v64, r64, t64 + eps) - fn64(x64 - eps * v64, r64, t64 - eps)) / (2 * eps)
dudt = d64
mf = mf.float()
fn = lambda z_, r_, t_: mf(z_, t_, txt, mask, None, r=r_)
u, _ = torch.func.jvp(fn, (x, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
rel = ((dudt - fd).norm() / fd.norm()).item()
assert rel < 1e-2, f"JVP vs finite difference rel err {rel}"
# 4. backward through the JVP-built loss
loss = ((u - (v - (t - r).view(-1, 1, 1, 1, 1) * dudt.float()).detach()) ** 2).mean(); loss.backward()
miss = [n for n, p in mf.named_parameters() if p.grad is None and n != "cond_emb"]  # cond_emb is unused without image-to-video
assert not miss, f"no gradient for {miss}"
# 5. few-step sampler shapes and image-to-video first frame
s = meanflow_sample(mf.eval(), txt, mask, (48, 5, 16, 16), steps=2, first=x[:, :, :1])
assert s.shape == x.shape and torch.equal(s[:, :, :1], x[:, :, :1])
print(f"all arm tests passed (JVP rel err vs finite difference {rel:.2e})")
# 6. split MeanFlow loss == full-batch JVP formulation (same random draws)
import types
from train_arm import meanflow_loss
mf2 = VideoDiT(VideoDiTConfig(width=96, depth=2, heads=4, meanflow=True)); mf2.load_state_dict(mf.state_dict())
for p in mf2.parameters():
    p.grad = None
a = types.SimpleNamespace(mf_ratio=0.5, mf_p=1.0)
x0 = torch.randn(6, 48, 5, 16, 16); tx, mk = torch.zeros(6, 1, 768), torch.ones(6, 1, dtype=torch.bool)
cnd = torch.tensor([True, False, True, False, False, True])
torch.manual_seed(5); l_split, e_split = meanflow_loss(mf2, x0, tx, mk, cnd, a, "cpu")
torch.manual_seed(5)  # reference: same draws, JVP on the whole batch
B = 6; t = torch.sigmoid(torch.randn(B)); r = torch.sigmoid(torch.randn(B)); t, r = torch.maximum(t, r), torch.minimum(t, r)
k = max(1, round(a.mf_ratio * B)); diff = torch.zeros(B, dtype=torch.bool); diff[torch.randperm(B)[:k]] = True
r = torch.where(diff, r, t); noise = torch.randn_like(x0)
z, v, lm = flow_inputs(x0, t, noise, cnd)
for mod in mf2.modules():
    if hasattr(mod, "impl"): mod.impl = "manual"
u, dudt = torch.func.jvp(lambda z_, r_, t_: mf2(z_, t_, tx, mk, cnd, r=r_), (z, r, t), (v * lm, torch.zeros_like(r), torch.ones_like(t)))
tgt = (v - (t - r).view(-1, 1, 1, 1, 1) * dudt).detach()
err = (((u - tgt) ** 2) * lm).flatten(1).sum(1) / (lm.flatten(1).sum(1) * 48 * 256)
l_ref = ((1.0 / (err.detach() + 1e-3)) * err).mean()
assert torch.allclose(l_split, l_ref, rtol=1e-4), (l_split.item(), l_ref.item())
print(f"split MeanFlow loss matches the full-batch JVP loss ({l_split.item():.5f} vs {l_ref.item():.5f}, {int(diff.sum())}/6 samples with r != t)")
