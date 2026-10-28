"""Stage 5 CPU/Mac tests (small model size "tiny" so they stay light; the real shape was checked on the GPU before the run).
  .venv/bin/python train/test_s5.py --work DIR --vg_lat <lat_2040_00.npy> --img_dir <encode_images.py output>
1. short_caption on real caption styles.
2. Plumbing: train_s5.py end to end (video + image steps, short captions, 128 tokens, milestones, every eval, mini videos).
3. Learnability: a tiny model trained from random on 16 images + 16 videos must reproduce its training images from their
   captions: PSNR with the own caption must beat PSNR with a shuffled caption by >= 1.5 dB, and a grid is saved to look at.
"""
import argparse, glob, json, os, shutil, subprocess, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from train_s5 import short_caption  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True); ap.add_argument("--vg_lat", required=True); ap.add_argument("--img_dir", required=True)
ap.add_argument("--learn_steps", type=int, default=800); ap.add_argument("--device", default="mps")
ap.add_argument("--only", default="")
a = ap.parse_args()
W = a.work; os.makedirs(W, exist_ok=True)

# 1 -------------------------------------------------------------------------------------------------------------------
cases = {
    "The video shows a basketball game in progress. The players wear red.": "a basketball game in progress",
    "In the video, a man is standing in front of a mirror and taking a selfie with a camera.": "a man is standing in front of a mirror and taking a selfie with a camera",
    "The video features a close-up of a large animal skull, which appears to be that of a cow.": "a close-up of a large animal skull, which appears to be that of a cow",
    "A child's hands holding a pencil over a lined notebook. More text.": "A child's hands holding a pencil over a lined notebook",
    "The video captures a dark and eerie scene of a small village at night.": "a dark and eerie scene of a small village at night",
    "Sunset.": "Sunset",
    "The video shows.": "The video shows",
}
for c, want in cases.items():
    got = short_caption(c)
    assert got == want, (c, got)
print("1 short_caption: ok", len(cases), "cases")


def fixture(name, n_vid, n_img):
    d = os.path.join(W, name); shutil.rmtree(d, ignore_errors=True)
    for sub in ("tr", "va", "img"):
        os.makedirs(os.path.join(d, sub))
    lat = np.load(a.vg_lat, mmap_mode="r"); caps = json.load(open(a.vg_lat.replace("lat_", "cap_").replace(".npy", ".json")))
    np.save(os.path.join(d, "tr", "lat_0000_00.npy"), np.array(lat[:n_vid])); json.dump(caps[:n_vid], open(os.path.join(d, "tr", "cap_0000_00.json"), "w"))
    np.save(os.path.join(d, "va", "lat_9000_00.npy"), np.array(lat[400:408])); json.dump(caps[400:408], open(os.path.join(d, "va", "cap_9000_00.json"), "w"))
    np.save(os.path.join(d, "va", "real.npy"), np.random.default_rng(0).integers(0, 255, (8, 16, 3, 256, 256), dtype=np.uint8))
    il = np.concatenate([np.load(p) for p in sorted(glob.glob(os.path.join(a.img_dir, "lat_*.npy")))])[:n_img]
    ic = sum([json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.img_dir, "cap_*.json")))], [])[:n_img]
    np.save(os.path.join(d, "img", "lat_00000.npy"), il); json.dump(ic, open(os.path.join(d, "img", "cap_00000.json"), "w"))
    from vmodel import build
    torch.manual_seed(0); net = build("tiny", in_ch=48, frames=5, grid=16)
    torch.save({"model": net.state_dict(), "ema": net.state_dict(), "step": 0}, os.path.join(d, "init.pt"))
    vp = os.path.join(d, "vidprom.json"); json.dump({"prompts": ["a dog running on a beach", "a red car in the rain", "a chef cooking pasta", "city lights at night"]}, open(vp, "w"))
    return d


def run(d, extra):
    i3d = os.path.join(HERE, "..", "data", "i3d_torchscript.pt")
    cmd = [sys.executable, os.path.join(HERE, "train_s5.py"), "--lat_train", f"{d}/tr", "--lat_val", f"{d}/va", "--real_frames", f"{d}/va/real.npy",
           "--i3d", i3d, "--init_from", f"{d}/init.pt", "--img_dir", f"{d}/img", "--size", "tiny", "--device", a.device, "--no_compile",
           "--vidprom", f"{d}/vidprom.json", "--mini", os.path.join(HERE, "..", "runs", "stage5", "prompts", "mini_semantic.json"),
           "--out", f"{d}/out", "--final_n", "8", "--i2v_n", "4", "--vidprom_n", "4", "--gen_seeds", "1", "--gen_batch", "4", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(f"{d}/run.log", "w").write(r.stdout + r.stderr)
    assert r.returncode == 0, r.stderr[-3000:]
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]


# 2 -------------------------------------------------------------------------------------------------------------------
if a.only in ("", "plumb"):
    d = fixture("plumb", 32, 51)
    recs = run(d, ["--steps", "40", "--batch", "4", "--img_batch", "8", "--p_img_step", "0.5", "--p_short", "0.5", "--text_len", "128",
                   "--milestones", "20", "--mini_limit", "1", "--sample_steps", "2", "--warmup", "5"])
    fin = json.load(open(f"{d}/out/final.json"))
    steps = [r for r in recs if "loss" in r]
    assert any(r["img_step"] for r in steps) or fin["img_steps"] > 0, "no image step"
    for k in ("t2v_heldout", "i2v_heldout", "vidprom", "val_loss", "mini_videos"):
        assert k in fin, k
    assert sorted(os.listdir(f"{d}/out/milestones")) == ["step_000020.pt", "step_000040.pt"], os.listdir(f"{d}/out/milestones")
    ms = torch.load(f"{d}/out/milestones/step_000040.pt", map_location="cpu", weights_only=False)
    assert ms["step"] == 40 and "ema" in ms and "opt" not in ms
    nm = len(os.listdir(f"{d}/out/mini_videos")); assert nm == fin["mini_videos"]["prompts"] >= 5, nm
    print("2 plumbing: ok", {"img_steps": fin["img_steps"], "of": fin["steps"], "mini_videos": nm, "vidprom_gap": round(fin["vidprom"]["clip_gap"], 2),
                             "fvd": round(fin["t2v_heldout"]["fvd"], 1)})

# 3 -------------------------------------------------------------------------------------------------------------------
if a.only in ("", "learn"):
    d = fixture("learn", 16, 16)
    run(d, ["--steps", str(a.learn_steps), "--batch", "4", "--img_batch", "8", "--p_img_step", "0.5", "--p_short", "0", "--text_len", "128",
            "--lr", "1e-3", "--warmup", "50", "--ema", "0.99", "--cfg_drop", "0.1", "--mini_limit", "1", "--sample_steps", "4"])
    from vmodel import VideoDiT, VideoDiTConfig
    from vae import VAE
    from train_s5 import Text
    from train_t2v import cfg_sample
    ck = torch.load(f"{d}/out/milestones/step_{a.learn_steps:06d}.pt", map_location="cpu", weights_only=False)
    m = VideoDiT(VideoDiTConfig(**ck["cfg"])); m.load_state_dict(ck["ema"]); m = m.to(a.device).eval()
    text = Text(a.device, 128); nc, nm_ = text.null; vae = VAE(a.device, torch.bfloat16)
    il = torch.from_numpy(np.load(f"{d}/img/lat_00000.npy").astype(np.float32)); ic = [r["caption"] for r in json.load(open(f"{d}/img/cap_00000.json"))]
    with torch.no_grad():
        real = vae.decode(il.to(a.device))[:, 0].float()
        def sample(caps):
            c, cm = text(caps); g = torch.Generator(device=a.device).manual_seed(0)
            z = cfg_sample(m, c, cm, nc, nm_, (48, 1, 16, 16), 30, 3.0, 3.0, None, g)
            return vae.decode(z.float())[:, 0].float()
        own = sample(ic); perm = list(range(1, 16)) + [0]; shuf = sample([ic[i] for i in perm])
    psnr = lambda x, y: (10 * torch.log10(255 ** 2 / ((x - y) ** 2).mean(dim=(1, 2, 3)))).mean().item()
    p_own, p_shuf = psnr(own, real), psnr(shuf, real)
    from PIL import Image
    grid = torch.cat([torch.cat(list(real[:8]), 2), torch.cat(list(own[:8]), 2), torch.cat(list(shuf[:8]), 2)], 1)
    Image.fromarray(grid.permute(1, 2, 0).clamp(0, 255).byte().numpy()).save(os.path.join(W, "learnability_grid.png"))
    json.dump({"psnr_own": p_own, "psnr_shuffled": p_shuf, "steps": a.learn_steps}, open(os.path.join(W, "learnability.json"), "w"), indent=1)
    print(f"3 learnability: PSNR own caption {p_own:.2f} dB vs shuffled {p_shuf:.2f} dB (need +1.5)")
    assert p_own >= p_shuf + 1.5, "learnability gate failed"
    print("3 learnability: ok (grid: rows = training image, sample from its caption, sample from another caption)")
