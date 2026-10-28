"""Image-to-video parity: PyTorch (T5 fp32, TAEW2.2 encoder + decoder, DiT with cond) vs the ONNX files the page runs.

    python tools/parity_i2v.py <onnx_dir: t5, dit_cond, tae_enc, tae_dec> <out_dir>
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, torch, onnxruntime as ort
from onnx_pipeline import noise, timesteps, encode, sample, decode, prep_image, encode_image
from transformers import AutoTokenizer, T5EncoderModel
from export_dit import load_dit, GuidedCond
from PIL import Image

D, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get("STEPS", 25)); SEED = 42
IMG = os.environ.get("PD12M_IMG", "pd12m_raw/img")  # test photos: PD12M (public domain / CC0)
CASES = [("51", "A large white cloud drifting slowly over a grassy field with trees, summer day, gentle wind."),
         ("39", "An old stone building in a green field, clouds drifting across a blue sky, the grass moving in the wind."),
         ("41", "A view over a park with a lake surrounded by trees, clouds moving slowly across the sky.")]
tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
t5 = T5EncoderModel.from_pretrained("google/flan-t5-base", dtype=torch.float32).eval()
dit = GuidedCond(load_dit()); mod = load_taehv_module(); tae = load_tae(mod)
S = lambda f: ort.InferenceSession(os.path.join(D, f), providers=["CPUExecutionProvider"])
o_t5, o_dit, o_enc, o_dec = S("flan_t5_base_encoder.onnx"), S("drishya_tiny_dit_cond.onnx"), S("taew2_2_encoder.onnx"), S("taew2_2_decoder.onnx")


def t5_torch(text):
    e = tok([text], max_length=77, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad(): return t5(input_ids=e.input_ids, attention_mask=e.attention_mask).last_hidden_state.numpy(), e.attention_mask.bool().numpy()


res = []
nc, nm = t5_torch("")
for k, (name, prompt) in enumerate(CASES):
    rgb = prep_image(f"{IMG}/{name}.jpg"); rgb.tofile(f"{OUT}/photo{name}_256.rgb")
    x0 = noise(SEED + k)
    # PyTorch reference
    with torch.no_grad():
        first_t = tae.encode_video(torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None, None].float() / 255, parallel=True,
                                   show_progress_bar=False).permute(0, 2, 1, 3, 4)
    c, cm = t5_torch(prompt); ts = timesteps(STEPS)
    x = torch.from_numpy(x0.copy()); txt = torch.from_numpy(np.concatenate([c, nc])); mask = torch.from_numpy(np.concatenate([cm, nm]))
    with torch.no_grad():
        for i in range(STEPS):
            x[:, :, :1] = first_t
            v = dit(x, torch.tensor([ts[i]], dtype=torch.float32), txt, mask, torch.tensor([5.0]), torch.tensor([1.0]))
            x = (x.double() + (ts[i + 1] - ts[i]) * v.double()).float()
        x[:, :, :1] = first_t
        fr_t = (tae.decode_video(x.permute(0, 2, 1, 3, 4), parallel=True, show_progress_bar=False)[0].permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    # ONNX (what the page runs)
    t0 = time.time()
    first_o = encode_image(o_enc, rgb)
    c2, cm2 = encode(o_t5, tok, prompt); nc2, nm2 = encode(o_t5, tok, "")
    z = sample(o_dit, c2, cm2, nc2, nm2, x0, steps=STEPS, first=first_o)
    fr_o = decode(o_dec, z); fr_o.tofile(f"{OUT}/frames_onnx_{name}.rgb")
    ps = lambda a, b: float(10 * np.log10(255 ** 2 / ((a.astype(np.float64) - b) ** 2).mean()))
    r = {"photo": name, "seed": SEED + k, "first_latent_max_abs": float(np.abs(first_o - first_t.numpy()).max()),
         "latent_rel_rmse": float(np.sqrt(((z - x.numpy()) ** 2).mean() / (x.numpy() ** 2).mean())),
         "clip_psnr_onnx_vs_torch": ps(fr_o, fr_t), "frame0_psnr_vs_photo": ps(fr_o[0], rgb), "onnx_sec": time.time() - t0}
    res.append(r); print(json.dumps(r), flush=True)
    Image.fromarray(np.concatenate([np.concatenate([rgb] + [fr_t[i] for i in (0, 8, 16)], 1),
                                    np.concatenate([rgb] + [fr_o[i] for i in (0, 8, 16)], 1)], 0)).save(f"{OUT}/i2v_{name}.png")
json.dump(res, open(f"{OUT}/parity_i2v.json", "w"), indent=1)
