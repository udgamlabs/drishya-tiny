"""Drishya Tiny generator: a 212M flow-matching video transformer on Wan2.2-VAE latents (inference copy).

Input latents are (B, 48, T, h, w); a 17-frame 256px clip is (48, 5, 16, 16). A 1x2x2 patch gives 5 x 8 x 8 = 320
tokens with a fixed 3D sin-cos position embedding. Each block: adaLN-zero self-attention (QK-norm), cross-attention to
Flan-T5-Base tokens, adaLN MLP. Image-to-video replaces the first latent frame with the clean one and marks its tokens
with a learned embedding. Training code: training/vmodel.py in the repo.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DrishyaConfig:
    width: int = 768
    depth: int = 16
    heads: int = 12
    patch: int = 2           # spatial patch; temporal patch is 1
    in_ch: int = 48          # Wan2.2 VAE latent channels
    frames: int = 5          # latent frames (17 video frames)
    grid: int = 16           # latent side (256px)
    txt_dim: int = 768       # Flan-T5-Base
    mlp_ratio: float = 4.0


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
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * 1000.0 * f[None]
    return torch.cat([a.cos(), a.sin()], dim=-1)


def scale_shift(h, shift, scale):
    """Adaptive norm conditioning: per-sample gain (1 + scale) and bias, broadcast over tokens."""
    gain = 1 + scale[:, None]
    return h * gain + shift[:, None]


class Attention(nn.Module):
    def __init__(self, d, h, cross=False):
        super().__init__()
        self.h, self.cross = h, cross
        if cross:
            self.q = nn.Linear(d, d)
            self.kv = nn.Linear(d, 2 * d)
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
        heads = lambda t: t.view(B, -1, self.h, D // self.h).transpose(1, 2)
        q, k, v = self.q_norm(heads(q)), self.k_norm(heads(k)), heads(v)
        attn_mask = None if mask is None else mask[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, d, h, mlp_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, h)
        self.n2 = nn.LayerNorm(d, elementwise_affine=True, eps=1e-6)
        self.xattn = Attention(d, h, cross=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        hid = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hid), nn.GELU(approximate="tanh"), nn.Linear(hid, d))
        self.ada = nn.Linear(d, 6 * d)

    def forward(self, x, c, cmask, temb):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(temb)).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(scale_shift(self.n1(x), sh1, sc1))
        x = x + self.xattn(self.n2(x), c, cmask)
        return x + g2.unsqueeze(1) * self.mlp(scale_shift(self.n3(x), sh2, sc2))


class DrishyaDiT(nn.Module):
    def __init__(self, cfg: DrishyaConfig):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.width, cfg.patch
        self.ns = cfg.grid // p
        self.tok_per_frame = self.ns * self.ns
        self.patch_in = nn.Linear(cfg.in_ch * p * p, d)
        self.register_buffer("pos", sincos_3d(d, cfg.frames, self.ns, self.ns)[None], persistent=False)
        self.cond_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.txt_in = nn.Sequential(nn.Linear(cfg.txt_dim, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
        self.t_in = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.pool_in = nn.Linear(d, d)
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, cfg.in_ch * p * p)

    def patchify(self, x):
        B, C, T, H, W = x.shape; p = self.cfg.patch
        x = x.view(B, C, T, H // p, p, W // p, p).permute(0, 2, 3, 5, 1, 4, 6)
        return x.reshape(B, T * (H // p) * (W // p), C * p * p)

    def unpatchify(self, x, T):
        B = x.shape[0]; p, C, n = self.cfg.patch, self.cfg.in_ch, self.ns
        x = x.view(B, T, n, n, C, p, p).permute(0, 4, 1, 2, 5, 3, 6)
        return x.reshape(B, C, T, n * p, n * p)

    def forward(self, x, t, txt, txt_mask, cond=None):
        """x: (B,C,T,h,w); t: (B,) in [0,1] (1 = noise); txt: (B,M,768); txt_mask: (B,M) bool; cond: (B,) bool marks
        a clean first frame (image-to-video) or None. Returns the flow velocity."""
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
        for blk in self.blocks:
            h = blk(h, c, txt_mask, temb)
        sh, sc = self.ada_out(F.silu(temb)).chunk(2, dim=-1)
        return self.unpatchify(self.out(scale_shift(self.norm_out(h), sh, sc)), T)
