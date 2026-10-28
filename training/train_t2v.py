"""Drishya Tiny Stage 3 trainer: open-domain text-to-video + image-to-video on VidGen-1M Wan2.2 latents.

Recipe = Stage 1 (A0, unchanged by Stage 2): VideoDiT s210, rectified flow, MuonAdamW, EMA 0.9999, batch 64, 25% image-to-video
steps (frame replacement), plus captions: Flan-T5-Base (frozen, on the fly, 77 tokens), 10% caption dropout so
sampling can use classifier-free guidance (CFG).

Evaluation (pre-registered; EMA weights; 50 Euler steps, shift 3, CFG 5):
  - text-to-video FVD-2048 on held-out VidGen clips' captions vs those clips (16 frames at 8 fps, 256px); motion ratio;
  - text following: CLIP (ViT-B/32) score of each video with its own caption minus with a shuffled caption ("gap"),
    compared with the same gap for the real clips;
  - image-to-video FVD (real first frame + caption);
  - still-frame baseline (real held-out clips frozen on frame 0); held-out flow loss.
Gate: G1 T2V FVD < still-frame FVD and motion 0.5-2; G2 CLIP gap >= 30% of the real gap (and > 0);
      G3 held-out loss at the end < at 25%; G4 I2V FVD < still-frame FVD.
"""
import argparse, copy, glob, json, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import fvd as F_  # noqa: E402
from vmodel import build, flow_inputs  # noqa: E402
from train_video import lr_factor, sync, save_video_grid, push  # noqa: E402


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat_train", required=True); ap.add_argument("--lat_val", required=True)
    ap.add_argument("--real_frames", required=True, help="(N,16,3,256,256) uint8 held-out clips (encode_vidgen.py)")
    ap.add_argument("--i3d", required=True)
    ap.add_argument("--size", default="s210"); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=600); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr_muon", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--decay_frac", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.9999); ap.add_argument("--p_cond", type=float, default=0.25)
    ap.add_argument("--cfg_drop", type=float, default=0.1); ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=3.0); ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--eval_fracs", default="0.25,0.5,0.75"); ap.add_argument("--mid_n", type=int, default=512)
    ap.add_argument("--final_n", type=int, default=2048); ap.add_argument("--i2v_n", type=int, default=512)
    ap.add_argument("--vae_dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--clip_model", default="ViT-B-32"); ap.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--guard_clips_per_s", type=float, default=0); ap.add_argument("--ckpt_minutes", type=float, default=30)
    ap.add_argument("--out", required=True); ap.add_argument("--hf_repo", default=""); ap.add_argument("--hf_prefix", default="stage3")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init_from", default="", help="warm start from a checkpoint: load model+EMA weights, fresh optimiser and schedule")
    ap.add_argument("--no_compile", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--det_device", default="")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class Shards:
    """Memory-mapped latent shards + captions; random batches without loading everything into RAM."""
    def __init__(self, d):
        self.lat, self.cap, self.idx = [], [], []
        for p in sorted(glob.glob(os.path.join(d, "lat_*.npy"))):
            a = np.load(p, mmap_mode="r")
            c = json.load(open(os.path.join(os.path.dirname(p), os.path.basename(p).replace("lat_", "cap_").replace(".npy", ".json"))))
            assert a.shape[0] == len(c), p
            s = len(self.lat); self.lat.append(a); self.cap.append(c); self.idx += [(s, i) for i in range(a.shape[0])]
        self.n = len(self.idx); self.shape = self.lat[0].shape[1:]

    def get(self, ids):
        x = np.stack([self.lat[self.idx[i][0]][self.idx[i][1]] for i in ids])
        return torch.from_numpy(x.astype(np.float32)), [self.cap[self.idx[i][0]][self.idx[i][1]]["caption"] for i in ids]


class Text:
    def __init__(self, dev):
        from transformers import AutoTokenizer, T5EncoderModel
        self.tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.enc = T5EncoderModel.from_pretrained("google/flan-t5-base", dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval().requires_grad_(False)
        self.dev = dev
        self.null = self(["", ])

    @torch.no_grad()
    def __call__(self, texts):
        e = self.tok(texts, max_length=77, truncation=True, padding="max_length", return_tensors="pt")
        h = self.enc(input_ids=e.input_ids.to(self.dev), attention_mask=e.attention_mask.to(self.dev)).last_hidden_state
        return h.float(), e.attention_mask.to(self.dev).bool()


class Clip:
    def __init__(self, a, dev):
        import open_clip
        self.m, _, self.pre = open_clip.create_model_and_transforms(a.clip_model, pretrained=a.clip_pretrained, device=dev)
        self.tok = open_clip.get_tokenizer(a.clip_model); self.m.eval(); self.dev = dev
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def gap(self, vids, caps, seed=0):
        """Mean CLIP score (x100) of each video with its own caption minus with a shuffled caption."""
        img, own = [], []
        for i in range(0, len(vids), 32):
            fr = torch.stack([v[[0, len(v) // 2, len(v) - 1]] for v in vids[i:i + 32]]).to(self.dev)  # (b,3,3,H,W)
            b = fr.shape[0]; x = F.interpolate(fr.flatten(0, 1).float() / 255, size=224, mode="bicubic", antialias=True)
            e = F.normalize(self.m.encode_image((x - self.mean) / self.std).float(), dim=-1).view(b, 3, -1).mean(1)
            img.append(F.normalize(e, dim=-1))
        img = torch.cat(img)
        t = F.normalize(torch.cat([self.m.encode_text(self.tok(caps[i:i + 64]).to(self.dev)).float() for i in range(0, len(caps), 64)]), dim=-1)
        perm = list(range(len(caps))); random.Random(seed).shuffle(perm)
        s_own = (img * t).sum(-1).mean().item() * 100; s_shuf = (img * t[perm]).sum(-1).mean().item() * 100
        return s_own, s_shuf, s_own - s_shuf


@torch.no_grad()
def cfg_sample(model, c, cm, nc, nm, shape, steps, scale, shift, first=None, generator=None):
    B = c.shape[0]; dev = c.device
    x = torch.randn((B, *shape), device=dev, generator=generator)
    cond = None if first is None else torch.ones(2 * B, dtype=torch.bool, device=dev)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    ts = shift * ts / (1 + (shift - 1) * ts)
    cc, mm = torch.cat([c, nc.expand(B, -1, -1)]), torch.cat([cm, nm.expand(B, -1)])
    for i in range(steps):
        if first is not None:
            x[:, :, :1] = first
        v = model(torch.cat([x, x]), ts[i].expand(2 * B), cc, mm, cond)
        vc, vu = v.chunk(2); v = vu + scale * (vc - vu)
        x = x + (ts[i + 1] - ts[i]) * v
    if first is not None:
        x[:, :, :1] = first
    return x


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True); dev = a.device; det_dev = a.det_device or dev
    torch.manual_seed(a.seed); random.seed(a.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    tr, va = Shards(a.lat_train), Shards(a.lat_val)
    C, T, H, W = tr.shape
    net = build(a.size, in_ch=C, frames=T, grid=H).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    if dev == "cuda":
        from optim import MuonAdamW
        opt = MuonAdamW(net.param_groups(a.lr, a.lr_muon, 0.0, "muon"))
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    step, elapsed_prev, done_evals, hist = 0, 0.0, [], []
    if a.resume and os.path.exists(os.path.join(a.out, "last.pt")):
        ck = torch.load(os.path.join(a.out, "last.pt"), map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step, elapsed_prev, done_evals, hist = ck["step"], ck["elapsed"], ck["done_evals"], ck["hist"]
    if a.init_from and not (a.resume and os.path.exists(os.path.join(a.out, "last.pt"))):
        ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        print(json.dumps({"init_from": a.init_from, "from_step": ck.get("step")}), flush=True)
    fwd = net if (a.no_compile or dev != "cuda") else torch.compile(net)
    text = Text(dev)
    from vae import VAE
    vae = VAE(dev, torch.float32 if a.vae_dtype == "fp32" else torch.bfloat16)
    det = F_.load_detector(a.i3d, det_dev)
    clip = Clip(a, dev)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def emit(r):
        print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()

    # --- references on the held-out set (computed once, cached) ---
    rp = os.path.join(a.out, "refs.json")
    real = torch.from_numpy(np.load(a.real_frames))[: a.final_n]
    n_ref = real.shape[0]
    vids_val = list(range(n_ref)); val_lat, val_caps = va.get(vids_val)
    if os.path.exists(rp):
        refs = json.load(open(rp)); f_real = np.load(os.path.join(a.out, "f_real.npy"))
    else:
        f_real = F_.features(det, list(real), det_dev); np.save(os.path.join(a.out, "f_real.npy"), f_real)
        f_still = F_.features(det, [v[:1].expand_as(v).contiguous() for v in real], det_dev)
        own, shuf, gap = clip.gap(list(real), val_caps)
        refs = {"n": n_ref, "fvd_still": F_.fvd(f_real, f_still), "motion_real": float(np.mean([F_.motion(v) for v in real])),
                "clip_real_own": own, "clip_real_shuf": shuf, "clip_real_gap": gap}
        json.dump(refs, open(rp, "w"), indent=1)
    emit({"params_M": round(n_params / 1e6, 2), "train_windows": tr.n, "val_windows": va.n, "latent": [C, T, H, W], "refs": refs})
    nc, nm = text.null

    @torch.no_grad()
    def score(n, tag, i2v=False, seed=0):
        t0 = time.time(); g = torch.Generator(device=dev).manual_seed(seed); vids = []
        for i in range(0, n, 32):
            b = min(32, n - i); c, cm = text(val_caps[i:i + b])
            first = val_lat[i:i + b, :, :1].to(dev) if i2v else None
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                z = cfg_sample(ema, c, cm, nc, nm, (C, T, H, W), a.sample_steps, a.cfg, a.shift, first, g)
            for j in range(0, b, 4):
                vids += list(vae.decode(z[j:j + 4].float())[:, :16])
        f = F_.features(det, vids, det_dev)
        own, shuf, gap = clip.gap(vids, val_caps[:n])
        r = {"tag": tag, "n": n, "fvd": F_.fvd(f_real[:n] if n < n_ref else f_real, f), "motion_ratio": float(np.mean([F_.motion(v) for v in vids])) / refs["motion_real"],
             "clip_own": own, "clip_gap": gap, "seconds": round(time.time() - t0, 1)}
        return r, vids

    @torch.no_grad()
    def val_loss():
        ema.eval(); tot, k = 0.0, 0; g = torch.Generator(device=dev).manual_seed(1234)
        for i in range(0, min(512, len(val_caps)), 64):
            x0 = val_lat[i:i + 64].to(dev); c, cm = text(val_caps[i:i + 64])
            for tv in (0.2, 0.4, 0.6, 0.8):
                t = torch.full((x0.shape[0],), tv, device=dev); xt, tgt, _ = flow_inputs(x0, t, torch.randn(x0.shape, device=dev, generator=g), None)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    p = ema(xt, t, c, cm)
                tot += F.mse_loss(p.float(), tgt, reduction="sum").item() / x0[0].numel(); k += x0.shape[0]
        return tot / k

    def ckpt(tag, weights=False):
        torch.save({"model": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "elapsed": elapsed,
                    "done_evals": done_evals, "hist": hist, "args": vars(a), "cfg": vars(net.cfg)}, os.path.join(a.out, "last.pt.tmp"))
        os.replace(os.path.join(a.out, "last.pt.tmp"), os.path.join(a.out, "last.pt"))
        push(a, [os.path.join(a.out, "log.jsonl")] + ([os.path.join(a.out, "last.pt")] if weights else []), f"{tag} step {step}", background=True)

    fracs = [float(x) for x in a.eval_fracs.split(",") if x]
    warm, t_start, elapsed, last_ck, guard_done = 20, None, elapsed_prev, time.time(), False
    while True:
        if t_start is None and step >= warm and a.steps == 0:
            sync(dev); t_start = time.time() - elapsed_prev
        if t_start is not None:
            elapsed = time.time() - t_start
        frac = (step / a.steps) if a.steps else elapsed / (a.minutes * 60)
        if frac >= 1.0:
            break
        for fr in fracs:
            if frac >= fr and fr not in done_evals:
                sync(dev); p0 = time.time()
                r, _ = score(a.mid_n, f"t2v@{fr}"); r.update(step=step, frac=round(frac, 4), val=val_loss()); emit(r); hist.append(r)
                done_evals.append(fr); ckpt(f"eval@{fr}", weights=True); last_ck = time.time()
                if t_start is not None:
                    t_start += time.time() - p0
        f_ = lr_factor(frac, step, a.warmup, a.decay_frac)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f_
        x0, caps = tr.get(np.random.randint(0, tr.n, a.batch)); x0 = x0.to(dev)
        c, cm = text(caps)
        drop = torch.rand(a.batch, device=dev) < a.cfg_drop
        c = torch.where(drop[:, None, None], nc.expand_as(c), c); cm = torch.where(drop[:, None], nm.expand_as(cm), cm)
        t = torch.sigmoid(torch.randn(a.batch, device=dev)); cond = torch.rand(a.batch, device=dev) < a.p_cond
        xt, tgt, lm = flow_inputs(x0, t, torch.randn_like(x0), cond)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            pred = fwd(xt, t, c, cm, cond)
        loss = (((pred.float() - tgt) ** 2) * lm).sum() / (lm.sum() * C * H * W)
        lv = loss.item() if step % 50 == 0 else None
        if lv is not None and (not np.isfinite(lv) or lv == 0.0):
            emit({"STOP": "invalid loss", "step": step}); ckpt("invalid"); sys.exit(2)
        loss.backward(); gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - d)
        step += 1
        if lv is not None:
            cps = round((step - warm) * a.batch / max(elapsed - elapsed_prev, 1e-9), 1) if t_start and step > warm else None
            emit({"step": step, "frac": round(frac, 4), "loss": round(lv, 5), "gnorm": round(gn.item(), 3), "lr_f": round(f_, 4),
                  "min": round(elapsed / 60, 2), "clips_per_s": cps, "mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1) if dev == "cuda" else None})
            if a.guard_clips_per_s and not guard_done and cps and step >= warm + 300 and not a.resume:
                guard_done = True
                if cps < a.guard_clips_per_s:
                    emit({"STOP": "SLOW_HOST", "clips_per_s": cps}); sys.exit(3)
        if time.time() - last_ck > a.ckpt_minutes * 60:
            ckpt("periodic", weights=True); last_ck = time.time()
    ckpt("end", weights=False)
    fin, vids = score(a.final_n, "t2v_final"); fin.update(step=step, minutes=round(elapsed / 60, 2), val=val_loss(), clips_seen=step * a.batch)
    save_video_grid(vids, os.path.join(a.out, "samples_t2v.png"), os.path.join(a.out, "samples_t2v.mp4"))
    i2v, vi = score(min(a.i2v_n, va.n), "i2v", i2v=True)
    save_video_grid(vi, os.path.join(a.out, "samples_i2v.png"), os.path.join(a.out, "samples_i2v.mp4"))
    v25 = next((h["val"] for h in hist if h["tag"] == "t2v@0.25"), None)
    checks = {"G1_fvd_below_still_and_motion": fin["fvd"] < refs["fvd_still"] and 0.5 <= fin["motion_ratio"] <= 2.0,
              "G2_text_following": fin["clip_gap"] > 0 and fin["clip_gap"] >= 0.3 * refs["clip_real_gap"],
              "G3_val_loss_improved": v25 is not None and fin["val"] < v25,
              "G4_i2v_below_still": i2v["fvd"] < refs["fvd_still"]}
    out = {"final": fin, "i2v": i2v, "hist": hist, "refs": refs, "checks": checks, "gate_pass": all(checks.values()),
           "params_M": round(n_params / 1e6, 2), "train_windows": tr.n}
    json.dump(out, open(os.path.join(a.out, "final.json"), "w"), indent=1)
    emit({"final": fin, "i2v": i2v, "checks": checks, "gate_pass": out["gate_pass"]})
    ckpt("final", weights=True)
    push(a, [os.path.join(a.out, f) for f in ("final.json", "log.jsonl", "refs.json", "samples_t2v.png", "samples_t2v.mp4",
                                              "samples_i2v.png", "samples_i2v.mp4", "last.pt")], f"final step {step}")


if __name__ == "__main__":
    main()
