"""Stage 3 checkpoint (.pt, EMA weights) -> model.safetensors + config.json for udgamlabs/Drishya-Tiny."""
import argparse, hashlib, json, os

import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
keep = ("width", "depth", "heads", "patch", "in_ch", "frames", "grid", "txt_dim", "mlp_ratio")
assert ck["cfg"]["tread_ratio"] == 0 and not ck["cfg"]["meanflow"]
os.makedirs(a.out, exist_ok=True)
sd = {k: v.contiguous() for k, v in ck["ema"].items()}
save_file(sd, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt", "weights": "ema", "step": str(ck["step"])})
src = hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest()
cfg = {"model": {k: ck["cfg"][k] for k in keep}, "weights": "EMA (decay 0.9999)", "step": ck["step"],
       "parameters": sum(v.numel() for v in sd.values()), "source_sha256": src,
       "sampling": {"steps": 50, "cfg": 5.0, "shift": 5.0, "fps": 8, "frames": 17, "size": 256},
       "frozen_parts": {"vae": "Wan-AI/Wan2.2-TI2V-5B-Diffusers (subfolder vae)", "text_encoder": "google/flan-t5-base"}}
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
print(json.dumps({k: cfg[k] for k in ("step", "parameters", "source_sha256")}))
