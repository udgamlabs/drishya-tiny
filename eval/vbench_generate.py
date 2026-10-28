"""Drishya Tiny Stage 4: sampling sweep + VBench video generation (EMA weights; the training environment, CUDA GPU).

  python eval/vbench_generate.py sweep  --ckpt last.pt --val <held-out dir> --i3d i3d.pt --out <dir>          -> sweep.json, chosen.json
  python eval/vbench_generate.py t2v    --ckpt last.pt --vbench <VBench repo> --out <dir> [--long] [--limit N] -> <dir>/<prompt>-<i>.mp4
  python eval/vbench_generate.py i2v    --ckpt last.pt --vbench <VBench repo> --images <crop/1-1> --out <dir> [--limit N]

sweep (pre-registered before the run): 512 held-out VidGen clips, FVD-512 vs the same 512 real clips + CLIP gap.
  Grid: CFG {3,5,7} x shift {3,5} at 50 steps; baseline = CFG 5, shift 3 (the Stage 3 setting).
  Pick: highest CLIP gap among grid settings whose FVD-512 <= baseline FVD-512 + 20 (noise ~ +-20).
  Then 100 steps at the pick: adopted only if its CLIP gap is higher and its FVD-512 <= baseline + 20.
t2v: the VBench standard suite (prompts/all_dimension.txt, 946 prompts x 5, indices 0-4) plus indices 5-24 for the 75
  temporal_flickering prompts. --long uses prompts/augmented_prompts/gpt_enhanced_prompts/all_dimension_longer.txt (the
  official GPT-enhanced prompts, line-aligned) as the model input while the file keeps the ORIGINAL prompt as its name.
i2v: VBench-I2V standard suite (vbench2_beta_i2v/vbench2_i2v_full_info.json, 1,118 entries x 5) with the 1:1 crops,
  resized to 256 (PIL bicubic); the image is the first frame.
Videos: 17 frames at 8 fps, 256x256, H.264 crf 10 (as in the VBench examples). Seeds: one generator per batch seeded with
  base_seed + batch index (logged in seeds.json), so every video has its own noise and the run is reproducible.
"""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
from vmodel import VideoDiT, VideoDiTConfig  # noqa: E402
from vae import VAE  # noqa: E402
from train_t2v import Text, Clip, cfg_sample  # noqa: E402
import fvd as F_  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["sweep", "t2v", "i2v"])
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--val", default=""); ap.add_argument("--i3d", default="")
ap.add_argument("--vbench", default=""); ap.add_argument("--images", default="")
ap.add_argument("--long", action="store_true"); ap.add_argument("--limit", type=int, default=0, help="TEST: first N prompts/entries")
ap.add_argument("--smoke", type=int, default=0, help="SMOKE: first N prompts of EACH dimension file, 2 videos each")
ap.add_argument("--n", type=int, default=512); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--cfg", type=float, default=None); ap.add_argument("--shift", type=float, default=None); ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--chosen", default="", help="chosen.json from the sweep (sampling settings for t2v/i2v)")
ap.add_argument("--seed", type=int, default=20261021); ap.add_argument("--device", default="cuda")
ap.add_argument("--mini", action="store_true", help="PREFLIGHT: sweep = baseline + one grid setting, no 100-step check")
a = ap.parse_args()
dev = a.device
os.makedirs(a.out, exist_ok=True)
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfgm = VideoDiTConfig(**ck["cfg"])
model = VideoDiT(cfgm); model.load_state_dict(ck["ema"]); model = model.to(dev).eval().requires_grad_(False)
SHAPE = (cfgm.in_ch, cfgm.frames, cfgm.grid, cfgm.grid)
text = Text(dev); nc, nm = text.null
vae = VAE(dev, torch.bfloat16)


def settings():
    s = {"cfg": 5.0, "shift": 3.0, "steps": 50}
    if a.chosen:
        s.update({k: v for k, v in json.load(open(a.chosen)).items() if k in s})
    for k in s:
        if getattr(a, k) is not None:
            s[k] = getattr(a, k)
    return s


@torch.no_grad()
def generate(prompts, firsts=None, st=None, seed=0):
    """prompts: list[str] (<= batch); firsts: optional (B,C,1,H,W) latents. -> list of (17,3,256,256) uint8 cpu tensors."""
    g = torch.Generator(device=dev).manual_seed(seed)
    c, cm = text(prompts)
    with torch.autocast(**amp):
        z = cfg_sample(model, c, cm, nc, nm, SHAPE, st["steps"], st["cfg"], st["shift"], firsts, g)
    out = []
    for j in range(0, len(prompts), 8):
        out += [v.cpu() for v in vae.decode(z[j:j + 8].float())]
    return out


def save_mp4(frames, path, fps=8):
    import av
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=fps, options={"crf": "10"}); s.width, s.height = frames.shape[-1], frames.shape[-2]; s.pix_fmt = "yuv420p"
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr.permute(1, 2, 0).numpy(), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def run_jobs(jobs, st, tag):
    """jobs: list of (prompt_for_model, file_name, first_latent_or_None). Skips files that already exist (resumable)."""
    jobs = [j for j in jobs if not os.path.exists(os.path.join(a.out, j[1]))]
    t0, seeds = time.time(), {}
    for b in range(0, len(jobs), a.batch):
        chunk = jobs[b:b + a.batch]; sd = a.seed + b // a.batch
        firsts = torch.cat([j[2] for j in chunk]).to(dev) if chunk[0][2] is not None else None
        vids = generate([j[0] for j in chunk], firsts, st, sd)
        for j, v in zip(chunk, vids):
            save_mp4(v, os.path.join(a.out, j[1])); seeds[j[1]] = [sd, chunk.index(j)]
        if (b // a.batch) % 20 == 0:
            print(json.dumps({"gen": tag, "done": b + len(chunk), "of": len(jobs), "per_s": round((b + len(chunk)) / (time.time() - t0), 2)}), flush=True)
    old = json.load(open(os.path.join(a.out, "seeds.json"))) if os.path.exists(os.path.join(a.out, "seeds.json")) else {}
    old.update(seeds); json.dump({"settings": st, "base_seed": a.seed, "videos": old}, open(os.path.join(a.out, "seeds.json"), "w"))
    print(json.dumps({"gen": tag, "videos": len(jobs), "minutes": round((time.time() - t0) / 60, 1), "settings": st}), flush=True)


if a.mode == "sweep":
    import glob
    class Args: clip_model = "ViT-B-32"; clip_pretrained = "laion2b_s34b_b79k"
    clip = Clip(Args, dev); det = F_.load_detector(a.i3d, dev)
    lat, caps = [], []
    for p in sorted(glob.glob(os.path.join(a.val, "lat_*.npy"))):
        lat.append(np.load(p, mmap_mode="r")); caps += [r["caption"] for r in json.load(open(p.replace("lat_", "cap_").replace(".npy", ".json")))]
    caps = caps[: a.n]
    real = torch.from_numpy(np.load(os.path.join(a.val, "val_real.npy"), mmap_mode="r")[: a.n].copy())
    f_real = F_.features(det, list(real), dev)
    own, shuf, gap_real = clip.gap(list(real), caps)
    res = {"n": a.n, "real_clip_gap": gap_real, "real_motion": float(np.mean([F_.motion(v) for v in real])), "runs": []}

    def score(st):
        t0 = time.time(); vids = []
        for i in range(0, a.n, a.batch):
            vids += [v[:16] for v in generate(caps[i:i + a.batch], None, st, a.seed + i // a.batch)]
        f = F_.features(det, vids, dev); own, _, gap = clip.gap(vids, caps)
        r = dict(st, fvd512=F_.fvd(f_real, f), clip_gap=gap, clip_own=own,
                 motion_ratio=float(np.mean([F_.motion(v) for v in vids])) / res["real_motion"], seconds=round(time.time() - t0, 1))
        print(json.dumps(r), flush=True); res["runs"].append(r); json.dump(res, open(os.path.join(a.out, "sweep.json"), "w"), indent=1)
        return r

    base = score({"cfg": 5.0, "shift": 3.0, "steps": 50})
    G = [(3.0, 5.0)] if a.mini else [(c, s) for c in (3.0, 5.0, 7.0) for s in (3.0, 5.0) if (c, s) != (5.0, 3.0)]
    grid = [base] + [score({"cfg": c, "shift": s, "steps": 50}) for c, s in G]
    ok = [r for r in grid if r["fvd512"] <= base["fvd512"] + 20]
    pick = max(ok, key=lambda r: r["clip_gap"])
    long = None if a.mini else score({"cfg": pick["cfg"], "shift": pick["shift"], "steps": 100})
    chosen = long if (long and long["clip_gap"] > pick["clip_gap"] and long["fvd512"] <= base["fvd512"] + 20) else pick
    res["chosen"] = {k: chosen[k] for k in ("cfg", "shift", "steps", "fvd512", "clip_gap", "motion_ratio")}
    json.dump(res, open(os.path.join(a.out, "sweep.json"), "w"), indent=1)
    json.dump(res["chosen"], open(os.path.join(a.out, "chosen.json"), "w"), indent=1)
    print(json.dumps({"CHOSEN": res["chosen"]}), flush=True)

elif a.mode == "t2v":
    st = settings()
    P = os.path.join(a.vbench, "prompts")
    orig = [l.strip() for l in open(os.path.join(P, "all_dimension.txt")) if l.strip()]
    model_in = orig
    if a.long:
        model_in = [l.strip() for l in open(os.path.join(P, "augmented_prompts/gpt_enhanced_prompts/all_dimension_longer.txt")) if l.strip()]
        assert len(model_in) == len(orig), (len(model_in), len(orig))
    flick = set(l.strip() for l in open(os.path.join(P, "prompts_per_dimension/temporal_flickering.txt")) if l.strip())
    idx = range(min(a.limit, len(orig))) if a.limit else range(len(orig))
    if a.smoke:  # the first N prompts of every dimension's list, so every VBench dimension has something to score
        import glob
        want = set()
        for f in sorted(glob.glob(os.path.join(P, "prompts_per_dimension", "*.txt"))):
            want |= set([l.strip() for l in open(f) if l.strip()][: a.smoke])
        idx = [k for k in range(len(orig)) if orig[k] in want]
    jobs = []
    for k in idx:
        n = 2 if a.smoke else (25 if (orig[k] in flick and not a.long) else 5)
        jobs += [(model_in[k], f"{orig[k]}-{i}.mp4", None) for i in range(n)]
    run_jobs(jobs, st, "t2v_long" if a.long else "t2v")

else:  # i2v
    from PIL import Image
    st = settings()
    info = json.load(open(os.path.join(a.vbench, "vbench2_beta_i2v/vbench2_i2v_full_info.json")))
    if a.limit or a.smoke:
        a.limit = a.limit or a.smoke
        seen, keep = {}, []
        for e in info:  # first N entries of each dimension group
            key = tuple(e["dimension"]); seen[key] = seen.get(key, 0) + 1
            if seen[key] <= a.limit:
                keep.append(e)
        info = keep
    cache, jobs = {}, []
    for e in info:
        iname = e["image_name"]  # (not "nm": that is the null-text mask used by cfg_sample)
        if iname not in cache:
            im = Image.open(os.path.join(a.images, iname)).convert("RGB").resize((256, 256), Image.BICUBIC)
            x = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None, None]  # (1,1,3,256,256) uint8
            with torch.no_grad():
                cache[iname] = vae.encode(x.to(dev))[:, :, :1].cpu()
        jobs += [(e["prompt_en"], f"{e['prompt_en']}-{i}.mp4", cache[iname]) for i in range(2 if a.smoke else 5)]
    run_jobs(jobs, st, "i2v")

if torch.cuda.is_available():
    print(json.dumps({"peak_mem_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}), flush=True)
