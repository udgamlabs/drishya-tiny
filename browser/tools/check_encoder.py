"""Image-to-video first-frame latent: Wan2.2 VAE encoder (the Python package) vs the TAEW2.2 encoder.

Compares the latents, then samples the same photo + prompt + seed with each and decodes both with TAEW2.2.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, torch
from PIL import Image
from drishya.pipeline import Drishya

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get("STEPS", 25))
IMG = os.environ.get("PD12M_IMG", "pd12m_raw/img")  # test photos: PD12M (public domain / CC0)
CASES = [("51", "A large white cloud drifting slowly over a grassy field with trees, summer day, gentle wind."),
         ("39", "An old stone building in a green field, clouds drifting across a blue sky, the grass moving in the wind."),
         ("41", "A view over a park with a lake surrounded by trees, clouds moving slowly across the sky.")]


def prep(path):
    img = Image.open(path).convert("RGB"); w, h = img.size; s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((256, 256), Image.BICUBIC)
    return img


def psnr_u8(a, b):
    return float(10 * np.log10(255 ** 2 / ((a.astype(np.float64) - b) ** 2).mean()))


pipe = Drishya.from_pretrained(WEIGHTS, device="mps")
mod = load_taehv_module(); tae = load_tae(mod).to("mps")
res = []
for name, prompt in CASES:
    img = prep(f"{IMG}/{name}.jpg"); img.save(f"{OUT}/photo{name}.png")
    photo = np.asarray(img)
    z_real = pipe.encode_image(Image.open(f"{IMG}/{name}.jpg"))  # (1,48,1,16,16) normalised
    x = torch.from_numpy(photo.copy()).permute(2, 0, 1)[None, None].float().div(255).to("mps")  # NTCHW [0,1]
    with torch.no_grad():
        z_tae = tae.encode_video(x, parallel=True, show_progress_bar=False).permute(0, 2, 1, 3, 4)  # -> (1,48,1,16,16)
    a, b = z_tae.float().cpu(), z_real.float().cpu()
    lat = {"rel_rmse": float(((a - b) ** 2).mean().sqrt() / (b ** 2).mean().sqrt()),
           "cos": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)), "shape": list(a.shape)}
    clips = {}
    for k, first in (("real", z_real), ("tae", z_tae.to(z_real.dtype))):
        z = pipe.sample_latents(prompt, first=first, steps=STEPS, seed=0)
        with torch.no_grad():
            fr = (tae.decode_video(z.float().permute(0, 2, 1, 3, 4), parallel=True, show_progress_bar=False)[0]
                  .permute(0, 2, 3, 1).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        clips[k] = fr; np.save(f"{OUT}/clip_{k}_{name}.npy", fr)
        if k == "real":
            full = pipe.decode(z).numpy(); np.save(f"{OUT}/clip_fullvae_{name}.npy", full)
    r = {"photo": name, "latent": lat,
         "clip_psnr_tae_vs_real_enc": psnr_u8(clips["tae"], clips["real"]),
         "frame0_psnr_vs_photo": {"real_enc": psnr_u8(clips["real"][0], photo), "tae_enc": psnr_u8(clips["tae"][0], photo),
                                  "real_enc_fullvae": psnr_u8(full[0], photo)},
         "last_frame_psnr_tae_vs_real": psnr_u8(clips["tae"][16], clips["real"][16])}
    res.append(r); print(json.dumps(r), flush=True)
    rows = [np.concatenate([photo] + [full[i] for i in (0, 8, 16)], 1),
            np.concatenate([photo] + [clips["real"][i] for i in (0, 8, 16)], 1),
            np.concatenate([photo] + [clips["tae"][i] for i in (0, 8, 16)], 1)]
    Image.fromarray(np.concatenate(rows, 0)).save(f"{OUT}/enc_cmp_{name}.png")
json.dump(res, open(f"{OUT}/encoder_check.json", "w"), indent=1)
