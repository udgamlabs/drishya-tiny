"""Sample latents with the PyTorch pipeline, decode with the real Wan2.2 VAE and with TAEW2.2; compare."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch
from drishya.pipeline import Drishya
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
pipe = Drishya.from_pretrained(WEIGHTS, device="mps")
print("loaded", time.time() - t0, flush=True)
tae = load_tae().to("mps")
prompts = ["Ocean waves rolling onto a sandy beach on a sunny day, blue sky with white clouds, the camera is static.",
           "Snow-covered mountain peaks under a blue sky, clouds drifting slowly past the summit."]
res = []
from PIL import Image
for k, p in enumerate(prompts):
    t1 = time.time()
    z = pipe.sample_latents(p, steps=25, seed=k)
    print("sampled", time.time() - t1, flush=True)
    torch.save(z.cpu(), f"{OUT}/z{k}.pt")
    ref = pipe.decode(z)  # (T,H,W,3) uint8
    zz = z.float().permute(0, 2, 1, 3, 4)  # NTCHW
    out = {}
    for name, inp in [("norm", zz), ("raw", (z.float() * pipe.std + pipe.mean).permute(0, 2, 1, 3, 4))]:
        y = tae.decode_video(inp, parallel=True, show_progress_bar=False)  # N,T,3,H,W in [0,1]
        fr = (y[0].clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu()
        out[name] = fr
        print(p[:20], name, tuple(fr.shape), "psnr vs VAE", psnr(fr, ref) if fr.shape == ref.shape else "shape mismatch", flush=True)
    res.append({"prompt": p, "ref_shape": list(ref.shape), "psnr_norm": psnr(out["norm"], ref), "psnr_raw": psnr(out["raw"], ref)})
    idx = [0, 8, 16]
    rows = [torch.cat([ref[i] for i in idx], 1), torch.cat([out["norm"][i] for i in idx], 1)]
    Image.fromarray(torch.cat(rows, 0).numpy()).save(f"{OUT}/cmp{k}.png")
    torch.save(ref, f"{OUT}/ref{k}.pt")
json.dump(res, open(f"{OUT}/decoder_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
