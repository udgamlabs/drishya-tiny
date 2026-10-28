"""Export the Drishya DiT to ONNX with classifier-free guidance folded in.

Inputs : x (1,48,5,16,16) f32, t (1,) f32, txt (2,77,768) f32 [cond, uncond], mask (2,77) bool, cfg (1,) f32
         [+ cond (1,) f32 with `cond` as the 2nd argument: 0 = text-to-video, 1 = image-to-video]
Output : v (1,48,5,16,16) f32 = vu + cfg*(vc-vu)
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch
from safetensors.torch import load_file
from drishya.model import DrishyaConfig, DrishyaDiT


class Guided(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x, t, txt, mask, cfg):
        v = self.m(torch.cat([x, x]), t.expand(2), txt, mask, None)
        vc, vu = v[:1], v[1:]
        return vu + cfg * (vc - vu)


class GuidedCond(torch.nn.Module):
    """Guided, plus image-to-video: cond (1,) f32 is 0 (text-to-video) or 1 (the first latent frame is a clean photo
    latent); it is applied to both CFG branches, as pipeline.py passes cond = ones(2). cond = 0 adds exactly zero."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x, t, txt, mask, cfg, cond):
        v = self.m(torch.cat([x, x]), t.expand(2), txt, mask, cond.expand(2))
        vc, vu = v[:1], v[1:]
        return vu + cfg * (vc - vu)


class PlainRMSNorm(torch.nn.Module):
    """RMSNorm written with basic ops so the ONNX export (opset 17) has no RMSNormalization node."""
    def __init__(self, ref):
        super().__init__(); self.weight = ref.weight; self.eps = ref.eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def load_dit():
    cfg = DrishyaConfig(**json.load(open(os.path.join(WEIGHTS, "config.json")))["model"])
    m = DrishyaDiT(cfg); m.load_state_dict(load_file(os.path.join(WEIGHTS, "model.safetensors")))
    for blk in m.blocks:
        for a in (blk.attn, blk.xattn):
            a.q_norm = PlainRMSNorm(a.q_norm); a.k_norm = PlainRMSNorm(a.k_norm)
    return m.eval().requires_grad_(False)


if __name__ == "__main__":
    out = sys.argv[1]
    with_cond = len(sys.argv) > 2 and sys.argv[2] == "cond"
    dynamo = False
    g = (GuidedCond if with_cond else Guided)(load_dit())
    x = torch.randn(1, 48, 5, 16, 16); t = torch.tensor([0.7]); txt = torch.randn(2, 77, 768)
    mask = torch.zeros(2, 77, dtype=torch.bool); mask[0, :20] = True; mask[1, :1] = True
    cfg = torch.tensor([5.0])
    t0 = time.time()
    args, names = ((x, t, txt, mask, cfg, torch.tensor([1.0])), ["x", "t", "txt", "mask", "cfg", "cond"]) if with_cond \
        else ((x, t, txt, mask, cfg), ["x", "t", "txt", "mask", "cfg"])
    torch.onnx.export(g, args, out, input_names=names, output_names=["v"],
                      opset_version=17, dynamo=dynamo, do_constant_folding=True)
    print("exported", out, time.time() - t0)
