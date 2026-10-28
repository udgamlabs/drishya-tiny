"""The browser pipeline re-done in Python with onnxruntime, to check parity with PyTorch and with the page.

Noise: mulberry32(seed) uniform -> Box-Muller pairs (z0 = r cos, z1 = r sin), u1 = 1 - uniform, fp64 maths, cast to f32.
The page (drishya-web.js) uses the same generator, so the same seed gives the same starting noise.
"""
import math
import numpy as np

M32 = 0xFFFFFFFF
SHAPE = (1, 48, 5, 16, 16)
N = 48 * 5 * 16 * 16


def mulberry32(seed):
    a = seed & M32
    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = (((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0
    return nxt


def noise(seed):
    r = mulberry32(seed); out = np.empty(N, dtype=np.float32)
    for i in range(0, N, 2):
        u1 = 1.0 - r(); u2 = r()
        rad = math.sqrt(-2.0 * math.log(u1))
        out[i] = rad * math.cos(2 * math.pi * u2)
        out[i + 1] = rad * math.sin(2 * math.pi * u2)
    return out.reshape(SHAPE)


def timesteps(steps, shift=5.0):
    ts = [1.0 - i / steps for i in range(steps + 1)]
    return [shift * t / (1 + (shift - 1) * t) for t in ts]


def tokenize(tok, text):
    e = tok([text], max_length=77, truncation=True, padding="max_length", return_tensors="np")
    return e.input_ids.astype(np.int64), e.attention_mask.astype(np.int64)


def encode(t5, tok, text):
    ids, am = tokenize(tok, text)
    h = t5.run(None, {"input_ids": ids, "attention_mask": am})[0].astype(np.float32)
    return h, am.astype(bool)


def sample(dit, c, cm, nc, nm, x0, steps=25, cfg=5.0, shift=5.0, on_step=None, first=None):
    """first: (1,48,1,16,16) photo latent for image-to-video (needs the DiT file with the `cond` input), else None."""
    txt = np.concatenate([c, nc]).astype(np.float32); mask = np.concatenate([cm, nm])
    ts = timesteps(steps, shift); x = x0.copy()
    names = {i.name for i in dit.get_inputs()}
    extra = {"cond": np.array([0.0 if first is None else 1.0], np.float32)} if "cond" in names else {}
    for i in range(steps):
        if first is not None:
            x[:, :, :1] = first
        v = dit.run(None, {"x": x, "t": np.array([ts[i]], np.float32), "txt": txt, "mask": mask,
                           "cfg": np.array([cfg], np.float32), **extra})[0]
        x = (x.astype(np.float64) + (ts[i + 1] - ts[i]) * v.astype(np.float64)).astype(np.float32)
        if on_step: on_step(i)
    if first is not None:
        x[:, :, :1] = first
    return x


def prep_image(path):
    """pipeline.py encode_image preprocessing: centre square crop, bicubic resize to 256 -> (256,256,3) uint8."""
    from PIL import Image
    img = Image.open(path).convert("RGB"); w, h = img.size; s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((256, 256), Image.BICUBIC)
    return np.asarray(img)


def encode_image(enc, rgb):
    return enc.run(None, {"image": rgb.astype(np.float32)})[0]


def decode(tae, z):
    return np.clip(np.round(tae.run(None, {"z": z.astype(np.float32)})[0]), 0, 255).astype(np.uint8)
