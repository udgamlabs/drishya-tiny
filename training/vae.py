"""Frozen Wan2.2 VAE wrapper: uint8 clips <-> normalised latents (mean/std from the VAE config, std ~1.01 in P0)."""
import torch

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class VAE:
    def __init__(self, device="cuda", dtype=torch.float32):
        from diffusers import AutoencoderKLWan
        self.m = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=dtype).to(device).eval()
        self.m.requires_grad_(False)
        self.dev, self.dt = device, dtype
        self.mean = torch.tensor(self.m.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(self.m.config.latents_std, device=device).view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def encode(self, x_u8):
        """x_u8: (B,T,3,H,W) uint8 -> (B,48,T',h,w) float32 normalised latents."""
        x = x_u8.to(self.dev).permute(0, 2, 1, 3, 4).to(self.dt) / 127.5 - 1.0
        z = self.m.encode(x).latent_dist.mode().float()
        return (z - self.mean) / self.std

    @torch.no_grad()
    def decode(self, z):
        """z: (B,48,T',h,w) normalised -> (B,T,3,H,W) uint8."""
        y = self.m.decode((z.float() * self.std + self.mean).to(self.dt)).sample.float()
        return ((y.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 1, 3, 4).cpu()
