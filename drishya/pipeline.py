"""Drishya Tiny pipeline: text (and optionally a first frame) -> a 2-second 256x256 clip at 8 fps.

The 212M generator is ours (trained from scratch). Two frozen parts are downloaded from their owners at first use:
the Wan2.2 VAE (Wan-AI/Wan2.2-TI2V-5B-Diffusers, Apache-2.0) and the Flan-T5-Base encoder (google/flan-t5-base,
Apache-2.0).
"""
import json
import os

import torch

from .model import DrishyaConfig, DrishyaDiT

REPO = "udgamlabs/Drishya-Tiny"
VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
T5_REPO = "google/flan-t5-base"
MARK = "AI-generated · Drishya Tiny"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Drishya:
    def __init__(self, model, text_tok, text_enc, vae, device):
        self.model, self.tok, self.enc, self.vae, self.device = model, text_tok, text_enc, vae, device
        self.mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        self._null = None

    @property
    def null(self):
        """Encoding of the empty prompt (classifier-free guidance), made on first use: on ZeroGPU the GPU is only
        available inside the generate call, not while the app starts."""
        if self._null is None:
            self._null = self.encode_text([""])
        return self._null

    @classmethod
    def from_pretrained(cls, path=REPO, device=None):
        """path: a local folder or a Hugging Face repo holding config.json and model.safetensors."""
        from safetensors.torch import load_file
        from diffusers import AutoencoderKLWan
        from transformers import AutoTokenizer, T5EncoderModel
        device = device or pick_device()
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, allow_patterns=["config.json", "model.safetensors"])
        cfg = DrishyaConfig(**json.load(open(os.path.join(path, "config.json")))["model"])
        model = DrishyaDiT(cfg)
        model.load_state_dict(load_file(os.path.join(path, "model.safetensors")))
        model = model.to(device).eval().requires_grad_(False)
        tok = AutoTokenizer.from_pretrained(T5_REPO)
        enc = T5EncoderModel.from_pretrained(T5_REPO, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        enc = enc.to(device).eval().requires_grad_(False)
        vdt = torch.float32 if device == "cpu" else torch.bfloat16
        vae = AutoencoderKLWan.from_pretrained(VAE_REPO, subfolder="vae", torch_dtype=vdt).to(device).eval()
        vae.requires_grad_(False)
        return cls(model, tok, enc, vae, device)

    @torch.no_grad()
    def encode_text(self, texts):
        e = self.tok(texts, max_length=77, truncation=True, padding="max_length", return_tensors="pt")
        h = self.enc(input_ids=e.input_ids.to(self.device), attention_mask=e.attention_mask.to(self.device)).last_hidden_state
        return h.float(), e.attention_mask.to(self.device).bool()

    @torch.no_grad()
    def encode_image(self, image):
        """A PIL image -> the clean first-frame latent (1,48,1,16,16), centre-cropped to 256x256."""
        from PIL import Image
        img = image.convert("RGB"); w, h = img.size; s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((256, 256), Image.BICUBIC)
        x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(256, 256, 3).permute(2, 0, 1)
        x = x[None, :, None].to(self.device, self.vae.dtype) / 127.5 - 1.0  # (1,3,1,H,W)
        z = self.vae.encode(x).latent_dist.mode().float()
        return (z - self.mean) / self.std

    @torch.no_grad()
    def sample_latents(self, prompt, first=None, steps=50, cfg=5.0, shift=5.0, seed=0):
        """Rectified-flow Euler sampling with classifier-free guidance, from t=1 (noise) to t=0."""
        m = self.model.cfg
        c, cm = self.encode_text([prompt]); nc, nm = self.null
        g = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn((1, m.in_ch, m.frames, m.grid, m.grid), device=self.device, generator=g)
        cond = None if first is None else torch.ones(2, dtype=torch.bool, device=self.device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=self.device)
        ts = shift * ts / (1 + (shift - 1) * ts)
        cc, mm = torch.cat([c, nc]), torch.cat([cm, nm])
        for i in range(steps):
            if first is not None:
                x[:, :, :1] = first
            v = self.model(torch.cat([x, x]), ts[i].expand(2), cc, mm, cond)
            vc, vu = v.chunk(2)
            x = x + (ts[i + 1] - ts[i]) * (vu + cfg * (vc - vu))
        if first is not None:
            x[:, :, :1] = first
        return x

    @torch.no_grad()
    def decode(self, z):
        """Normalised latents -> (T,256,256,3) uint8 frames."""
        y = self.vae.decode((z.float() * self.std + self.mean).to(self.vae.dtype)).sample.float()
        return ((y.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[0].permute(1, 2, 3, 0).cpu()

    def __call__(self, prompt, image=None, steps=50, cfg=5.0, shift=5.0, seed=0):
        first = None if image is None else self.encode_image(image)
        return self.decode(self.sample_latents(prompt, first, steps, cfg, shift, seed))


def add_mark(frames, text=MARK):
    """Burn a small visible 'AI-generated' label into the bottom-left corner of every frame."""
    from PIL import Image, ImageDraw, ImageFont
    out = []
    for fr in frames:
        im = Image.fromarray(fr.numpy()); d = ImageDraw.Draw(im, "RGBA")
        font = ImageFont.load_default(size=11)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
        w, h = x1 - x0, y1 - y0
        d.rectangle((4, im.height - h - 12, 4 + w + 8, im.height - 4), fill=(0, 0, 0, 150))
        d.text((8 - x0, im.height - h - 8 - y0), text, font=font, fill=(255, 255, 255, 230))
        out.append(torch.from_numpy(__import__("numpy").asarray(im)))
    return torch.stack(out)


def save_mp4(frames, path, prompt="", fps=8, mark=True, seed=None):
    """Write an H.264 mp4 with the visible mark and provenance metadata (generator, model, prompt)."""
    import av
    if mark:
        frames = add_mark(frames)
    with av.open(path, "w") as c:
        c.metadata["title"] = "AI-generated video"
        c.metadata["comment"] = json.dumps({"ai_generated": True, "generator": "Drishya Tiny by Udgam Labs",
                                            "model": REPO, "prompt": prompt, "seed": seed})
        s = c.add_stream("libx264", rate=fps); s.height, s.width = frames.shape[1], frames.shape[2]
        s.pix_fmt = "yuv420p"; s.options = {"crf": "18"}
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr.numpy(), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path
