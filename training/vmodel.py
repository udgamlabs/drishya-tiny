"""Drishya Tiny generator (VideoDiT): flow-matching text/image-to-video transformer on Wan2.2-VAE latents.

Extends the Chitra Tiny image DiT (Udgam Labs) to video:
- input latents (B, 48, T, h, w); a 33-frame 256px clip is (48, 9, 16, 16). Patch 1x2x2 gives 9 x 8 x 8 = 576 tokens;
- fixed 3D sin-cos position embedding (time, height, width), full softmax self-attention over all tokens;
- image-to-video by frame replacement: the clean first latent frame replaces the noisy one, a learned embedding marks
  its tokens, and its loss is masked out (STIV / SANA-Video recipe). Without the condition the model does text-to-video;
- each block: adaLN-zero self-attention (QK-norm) -> cross-attention to Flan-T5 tokens -> adaLN MLP (as in Chitra);
- optional TREAD token routing during training (as in Chitra).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VideoDiTConfig:
    width: int = 384
    depth: int = 8
    heads: int = 6
    patch: int = 2           # spatial patch; temporal patch is 1
    in_ch: int = 48          # Wan2.2 VAE
    frames: int = 9          # latent frames (33 video frames)
    grid: int = 16           # latent side (256px)
    txt_dim: int = 768       # Flan-T5-Base
    mlp_ratio: float = 4.0
    tread_ratio: float = 0.0
    tread_start: int = 2
    tread_end: int = -2
    meanflow: bool = False   # Stage 2: second time input (t - r) for MeanFlow average-velocity training
    attn: str = "sdpa"       # "sdpa" | "manual" (plain softmax attention; forward-mode AD friendly)


def sincos_1d(n, d):
    f = torch.exp(-math.log(10000.0) * torch.arange(d // 2, dtype=torch.float32) / (d // 2))
    a = torch.arange(n, dtype=torch.float32)[:, None] * f
    return torch.cat([a.sin(), a.cos()], dim=1)


def sincos_3d(d, T, H, W):
    """(T*H*W, d): a quarter of the channels encode time, the rest split between height and width."""
    dt = (d // 4) // 2 * 2
    dh = ((d - dt) // 2) // 2 * 2
    dw = d - dt - dh
    t = sincos_1d(T, dt)[:, None, None].expand(T, H, W, dt)
    h = sincos_1d(H, dh)[None, :, None].expand(T, H, W, dh)
    w = sincos_1d(W, dw)[None, None, :].expand(T, H, W, dw)
    return torch.cat([t, h, w], dim=-1).reshape(T * H * W, d)


def timestep_embedding(t, dim=256):
    half = dim // 2
    dt = torch.float64 if t.dtype == torch.float64 else torch.float32  # float64 only for gradient checks
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=dt) / half)
    a = t.to(dt)[:, None] * 1000.0 * f[None]
    return torch.cat([a.cos(), a.sin()], dim=-1)


def scale_shift(h, shift, scale):
    """adaLN modulation: per-sample scale and shift, broadcast over tokens."""
    gain = 1 + scale[:, None]
    return h * gain + shift[:, None]


class Attention(nn.Module):
    def __init__(self, d, h, cross=False, kv_dim=None, impl="sdpa"):
        super().__init__()
        self.h, self.cross, self.impl = h, cross, impl
        if cross:
            self.q = nn.Linear(d, d)
            self.kv = nn.Linear(kv_dim or d, 2 * d)
        else:
            self.qkv = nn.Linear(d, 3 * d)
        self.q_norm = nn.RMSNorm(d // h, eps=1e-6)
        self.k_norm = nn.RMSNorm(d // h, eps=1e-6)
        self.proj = nn.Linear(d, d)

    def forward(self, x, ctx=None, mask=None):
        B, N, D = x.shape
        if self.cross:
            q = self.q(x); k, v = self.kv(ctx).chunk(2, dim=-1)
        else:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
        sh = lambda t: t.view(B, -1, self.h, D // self.h).transpose(1, 2)
        if self.impl == "manual":  # forward-mode-AD-safe RMSNorm (Apple's fused rms_norm has no JVP); same maths
            rn = lambda x_, n_: x_ * torch.rsqrt(x_.pow(2).mean(-1, keepdim=True) + n_.eps) * n_.weight
            q, k, v = rn(sh(q), self.q_norm), rn(sh(k), self.k_norm), sh(v)
        else:
            q, k, v = self.q_norm(sh(q)), self.k_norm(sh(k)), sh(v)
        attn_mask = None if mask is None else mask[:, None, None, :]
        if self.impl == "manual":  # same maths as SDPA, but every op supports forward-mode AD (MeanFlow's JVP)
            a = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
            if attn_mask is not None:
                a = a.masked_fill(~attn_mask, float("-inf"))
            o = a.softmax(-1) @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, d, h, mlp_ratio, impl="sdpa"):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, h, impl=impl)
        self.n2 = nn.LayerNorm(d, elementwise_affine=True, eps=1e-6)
        self.xattn = Attention(d, h, cross=True, kv_dim=d, impl=impl)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        hid = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hid), nn.GELU(approximate="tanh"), nn.Linear(hid, d))
        self.ada = nn.Linear(d, 6 * d)

    def forward(self, x, c, cmask, temb):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(temb)).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(scale_shift(self.n1(x), sh1, sc1))
        x = x + self.xattn(self.n2(x), c, cmask)
        return x + g2.unsqueeze(1) * self.mlp(scale_shift(self.n3(x), sh2, sc2))


class VideoDiT(nn.Module):
    def __init__(self, cfg: VideoDiTConfig):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.width, cfg.patch
        self.ns = cfg.grid // p
        self.tok_per_frame = self.ns * self.ns
        self.patch_in = nn.Linear(cfg.in_ch * p * p, d)
        self.register_buffer("pos", sincos_3d(d, cfg.frames, self.ns, self.ns)[None], persistent=False)
        self.cond_emb = nn.Parameter(torch.zeros(1, 1, d))  # marks the clean first frame (image-to-video)
        self.txt_in = nn.Sequential(nn.Linear(cfg.txt_dim, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
        self.t_in = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.pool_in = nn.Linear(d, d)
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp_ratio, cfg.attn) for _ in range(cfg.depth))
        if cfg.meanflow:  # embedding of the interval h = t - r (zero-init output, so it starts as a flow model)
            self.h_in = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, cfg.in_ch * p * p)
        self.tread_end = cfg.tread_end if cfg.tread_end >= 0 else cfg.depth + cfg.tread_end
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.t_in[0].weight, std=0.02); nn.init.normal_(self.t_in[2].weight, std=0.02)
        for b in self.blocks:
            nn.init.zeros_(b.ada.weight); nn.init.zeros_(b.ada.bias)
            nn.init.zeros_(b.xattn.proj.weight)
        nn.init.zeros_(self.ada_out.weight); nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        if self.cfg.meanflow:
            nn.init.normal_(self.h_in[0].weight, std=0.02); nn.init.zeros_(self.h_in[2].weight)

    def patchify(self, x):
        B, C, T, H, W = x.shape; p = self.cfg.patch
        x = x.view(B, C, T, H // p, p, W // p, p).permute(0, 2, 3, 5, 1, 4, 6)
        return x.reshape(B, T * (H // p) * (W // p), C * p * p)

    def unpatchify(self, x, T=None):
        B = x.shape[0]; p, C, n = self.cfg.patch, self.cfg.in_ch, self.ns
        T = T or self.cfg.frames
        x = x.view(B, T, n, n, C, p, p).permute(0, 4, 1, 2, 5, 3, 6)
        return x.reshape(B, C, T, n * p, n * p)

    def forward(self, x, t, txt, txt_mask, cond=None, r=None):
        """x: (B,C,T,h,w) model input (frame 0 already replaced by the clean latent where cond is True); T may be
        smaller than cfg.frames (T=1: images, for joint image+video training); t: (B,) in [0,1]; txt: (B,M,768);
        txt_mask: (B,M) bool; cond: (B,) bool or None; r: (B,) MeanFlow interval start (None = r = t, plain flow).
        Returns the velocity (or MeanFlow average velocity u(z, r, t))."""
        T = x.shape[2]
        h = self.patch_in(self.patchify(x)) + self.pos[:, : T * self.tok_per_frame]
        if cond is not None:
            first = torch.zeros(1, h.shape[1], 1, device=h.device, dtype=h.dtype)
            first[:, : self.tok_per_frame] = 1.0
            h = h + cond.to(h.dtype).view(-1, 1, 1) * first * self.cond_emb
        c = self.txt_in(txt)
        m = txt_mask.to(c.dtype).unsqueeze(-1)
        pooled = (c * m).sum(1) / m.sum(1).clamp(min=1.0)
        temb = self.t_in(timestep_embedding(t)) + self.pool_in(pooled)
        if self.cfg.meanflow:
            temb = temb + self.h_in(timestep_embedding(t - (t if r is None else r)))
        B, N, D = h.shape
        keep_idx, full = None, None
        use_tread = self.training and self.cfg.tread_ratio > 0
        for i, blk in enumerate(self.blocks):
            if use_tread and i == self.cfg.tread_start:
                n_keep = max(1, int(round(N * (1 - self.cfg.tread_ratio))))
                keep_idx = torch.rand(B, N, device=h.device).argsort(dim=1)[:, :n_keep]
                full = h
                h = torch.gather(h, 1, keep_idx[..., None].expand(-1, -1, D))
            if use_tread and i == self.tread_end and full is not None:
                h = full.scatter(1, keep_idx[..., None].expand(-1, -1, D), h)
                keep_idx, full = None, None
            h = blk(h, c, txt_mask, temb)
        if full is not None:
            h = full.scatter(1, keep_idx[..., None].expand(-1, -1, D), h)
        sh, sc = self.ada_out(F.silu(temb)).chunk(2, dim=-1)
        return self.unpatchify(self.out(scale_shift(self.norm_out(h), sh, sc)), T)


    def param_groups(self, lr_adam, lr_muon, wd, optim="muon"):
        """Same split as Chitra Tiny: Muon for 2D weights inside blocks (grouped by shape), AdamW for the rest."""
        if optim == "adamw":
            return [dict(kind="adamw", params=list(self.parameters()), lr=lr_adam, betas=(0.9, 0.95), eps=1e-8,
                         weight_decay=wd)]
        muon, adam = {}, []
        for name, p in self.named_parameters():
            if name.startswith("blocks.") and p.ndim == 2:
                muon.setdefault(tuple(p.shape), []).append(p)
            else:
                adam.append(p)
        groups = [dict(kind="adamw", params=adam, lr=lr_adam, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)]
        for shape, ps in sorted(muon.items()):
            groups.append(dict(kind="muon", params=ps, lr=lr_muon, momentum=0.95, ns_steps=5, beta2=0.95,
                               weight_decay=wd))
        return groups


SIZES = {"tiny": (384, 8, 6), "s130": (640, 14, 10), "s210": (768, 16, 12)}  # (width, depth, heads), as in Chitra


def build(size, **kw):
    w, d, h = SIZES[size]
    return VideoDiT(VideoDiTConfig(width=w, depth=d, heads=h, **kw))


def flow_inputs(x0, t, noise, cond):
    """Rectified flow (t=0 data, t=1 noise). Returns model input, velocity target and a loss mask (B,1,T,1,1)."""
    tt = t.view(-1, 1, 1, 1, 1)
    xt = (1 - tt) * x0 + tt * noise
    target = noise - x0
    mask = torch.ones(x0.shape[0], 1, x0.shape[2], 1, 1, device=x0.device)
    if cond is not None:
        c = cond.view(-1, 1, 1, 1, 1).to(x0.dtype)
        xt = xt.clone()
        xt[:, :, :1] = c * x0[:, :, :1] + (1 - c) * xt[:, :, :1]
        mask[:, :, :1] = 1 - c
    return xt, target, mask


@torch.no_grad()
def euler_sample(model, txt, mask, shape, steps=30, first=None, generator=None, shift=1.0):
    """Plain Euler from t=1 to t=0 (no guidance). first: (B,C,1,h,w) clean first-frame latent for image-to-video."""
    B = txt.shape[0]
    x = torch.randn((B, *shape), device=txt.device, generator=generator)
    cond = None
    if first is not None:
        cond = torch.ones(B, dtype=torch.bool, device=txt.device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=txt.device)
    if shift != 1.0:
        ts = shift * ts / (1 + (shift - 1) * ts)
    for i in range(steps):
        if first is not None:
            x[:, :, :1] = first
        v = model(x, ts[i].expand(B), txt, mask, cond)
        x = x + (ts[i + 1] - ts[i]) * v
    if first is not None:
        x[:, :, :1] = first
    return x


@torch.no_grad()
def meanflow_sample(model, txt, mask, shape, steps=1, first=None, generator=None):
    """MeanFlow few-step sampler: z_{t'} = z_t - (t - t') * u(z_t, r=t', t). steps=1 is one-step generation."""
    B = txt.shape[0]
    x = torch.randn((B, *shape), device=txt.device, generator=generator)
    cond = None if first is None else torch.ones(B, dtype=torch.bool, device=txt.device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=txt.device)
    for i in range(steps):
        if first is not None:
            x[:, :, :1] = first
        u = model(x, ts[i].expand(B), txt, mask, cond, r=ts[i + 1].expand(B))
        x = x - (ts[i] - ts[i + 1]) * u
    if first is not None:
        x[:, :, :1] = first
    return x
