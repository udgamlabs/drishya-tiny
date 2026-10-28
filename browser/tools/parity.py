"""Parity: PyTorch (T5 fp32 + DiT + TAEW2.2) vs the ONNX pipeline the page runs, same PRNG noise.

    python tools/parity.py <onnx_dir> <xenova_dir> <out_dir> [t5 variants] [dit files]   (TAE=<decoder file> env)
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, torch, onnxruntime as ort
from onnx_pipeline import noise, timesteps, encode, sample, decode
from transformers import AutoTokenizer, T5EncoderModel
from export_dit import load_dit, Guided
from PIL import Image

ONNX, XEN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
variants = sys.argv[4].split(",") if len(sys.argv) > 4 else ["encoder_model.onnx", "encoder_model_quantized.onnx", "encoder_model_fp16.onnx"]
dits = sys.argv[5].split(",") if len(sys.argv) > 5 else ["dit_fp32.onnx", "dit_fp16.onnx"]
STEPS = int(os.environ.get("STEPS", 25))
os.makedirs(OUT, exist_ok=True)
prompts = ["Ocean waves rolling onto a sandy beach on a sunny day, blue sky with white clouds, the camera is static.",
           "Sunset time lapse at the beach with moving clouds and orange colors in the sky.",
           "Snow-covered mountain peaks under a blue sky, clouds drifting slowly past the summit."]
SEED = 42
tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
t5 = T5EncoderModel.from_pretrained("google/flan-t5-base", dtype=torch.float32).eval()
def t5_torch(text):
    e = tok([text], max_length=77, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad(): h = t5(input_ids=e.input_ids, attention_mask=e.attention_mask).last_hidden_state
    return h.numpy(), e.attention_mask.bool().numpy()
res = {"seed": SEED, "steps": STEPS, "text": {}, "runs": []}
# 1) text encoder variants vs PyTorch
xs = {v: ort.InferenceSession(os.path.join(XEN, "onnx", v), providers=["CPUExecutionProvider"]) for v in variants}
emb = {}
for p in prompts + [""]:
    ref, m = t5_torch(p); emb[("torch", p)] = (ref, m)
    for v, s in xs.items():
        h, m2 = encode(s, tok, p); emb[(v, p)] = (h, m2)
        mm = m[0]
        d = np.abs(h - ref)[0][mm]; cos = (h[0][mm] * ref[0][mm]).sum(-1) / (np.linalg.norm(h[0][mm], axis=-1) * np.linalg.norm(ref[0][mm], axis=-1))
        res["text"].setdefault(v, []).append({"prompt": p[:30], "max_abs": float(d.max()), "mean_abs": float(d.mean()), "min_cos": float(cos.min())})
print(json.dumps(res["text"], indent=0), flush=True)
# 2) PyTorch reference sample (CPU DiT) + TAEW2.2 (torch) + full VAE
dit = Guided(load_dit())
mod = load_taehv_module(); tae_t = load_tae(mod)
tae_o = ort.InferenceSession(os.path.join(ONNX, os.environ.get("TAE", "taew2_2_decoder.onnx")), providers=["CPUExecutionProvider"])
for k, p in enumerate(prompts):
    x0 = noise(SEED + k)
    c, cm = emb[("torch", p)]; nc, nm = emb[("torch", "")]
    ts = timesteps(STEPS); x = torch.from_numpy(x0.copy())
    txt = torch.from_numpy(np.concatenate([c, nc])); mask = torch.from_numpy(np.concatenate([cm, nm]))
    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            v = dit(x, torch.tensor([ts[i]], dtype=torch.float32), txt, mask, torch.tensor([5.0]))
            x = (x.double() + (ts[i + 1] - ts[i]) * v.double()).float()
    z_ref = x.numpy(); np.save(f"{OUT}/z_torch_{k}.npy", z_ref)
    with torch.no_grad():
        fr_ref = (tae_t.decode_video(x.permute(0, 2, 1, 3, 4), parallel=True, show_progress_bar=False)[0].permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    np.save(f"{OUT}/frames_torch_{k}.npy", fr_ref)
    print("torch ref", k, time.time() - t0, flush=True)
    rows = [np.concatenate([fr_ref[i] for i in (0, 8, 16)], 1)]
    for dname in dits:
        ds = ort.InferenceSession(os.path.join(ONNX, dname), providers=["CPUExecutionProvider"])
        for v in variants:
            c2, cm2 = emb[(v, p)]; nc2, nm2 = emb[(v, "")]
            t0 = time.time()
            z = sample(ds, c2, cm2, nc2, nm2, x0, steps=STEPS)
            fr = decode(tae_o, z)
            rel = float(np.sqrt(((z - z_ref) ** 2).mean() / (z_ref ** 2).mean()))
            ps = float(10 * np.log10(255 ** 2 / ((fr.astype(np.float64) - fr_ref) ** 2).mean()))
            r = {"prompt": k, "dit": dname, "t5": v, "latent_rel_rmse": rel, "psnr_vs_torch_tae": ps, "sec": time.time() - t0}
            res["runs"].append(r); print(r, flush=True)
            np.save(f"{OUT}/frames_{dname}_{v}_{k}.npy", fr)
            rows.append(np.concatenate([fr[i] for i in (0, 8, 16)], 1))
        del ds
    Image.fromarray(np.concatenate(rows, 0)).save(f"{OUT}/parity_{k}.png")
json.dump(res, open(f"{OUT}/parity.json", "w"), indent=1)
