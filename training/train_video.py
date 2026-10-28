"""Drishya Tiny Stage 1 trainer: unconditional SkyTimelapse VideoDiT on pre-encoded Wan2.2 latents, with built-in
evaluation (FVD + motion) and the pre-registered gate checks.

Built from the Chitra Tiny trainer: MuonAdamW, EMA, wall-clock budget timed after warm-up, optional checkpoint
pushes to a Hugging Face repo (--hf_repo), resume. Text conditioning is a constant zero token (unconditional benchmark); a share of
steps (--p_cond) trains image-to-video by frame replacement so that path is exercised at scale.

Stop rules in code: invalid loss (NaN or exactly 0) -> exit 2; host slower than --guard_clips_per_s after 300 timed
steps -> exit 3 (SLOW_HOST); wall-clock budget --minutes (training only, evals excluded).
"""
import argparse, copy, glob, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import fvd as F_  # noqa: E402
from vmodel import build, flow_inputs, euler_sample  # noqa: E402


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat_train", required=True)
    ap.add_argument("--lat_val", required=True)
    ap.add_argument("--realstats", required=True, help="fvd_stats.py output (.npz; refs next to it)")
    ap.add_argument("--i3d", required=True)
    ap.add_argument("--size", default="s210")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--steps", type=int, default=0, help="if > 0, overrides --minutes (tests)")
    ap.add_argument("--optim", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_muon", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--decay_frac", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--p_cond", type=float, default=0.25)
    ap.add_argument("--eval_fracs", default="0.1,0.25,0.5,0.75")
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--final_n", type=int, default=2048)
    ap.add_argument("--i2v_n", type=int, default=512)
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--extra_shift", type=float, default=3.0, help="secondary final eval; 0 = skip")
    ap.add_argument("--det_device", default="", help="I3D device (default: --device)")
    ap.add_argument("--vae_dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--guard_clips_per_s", type=float, default=0, help="host-speed stop rule; 0 = off")
    ap.add_argument("--fvd_max", type=float, default=200.0, help="pre-registered P1 threshold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf_repo", default="")
    ap.add_argument("--hf_prefix", default="stage1")
    ap.add_argument("--ckpt_minutes", type=float, default=30)
    ap.add_argument("--resume", default="")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--wandb", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def load_lat(d, device, limit=0):
    arr = np.concatenate([np.load(p) for p in sorted(glob.glob(os.path.join(d, "lat_*.npy")))])
    if limit:
        arr = arr[:limit]
    return torch.from_numpy(arr).to(device)  # float16


def lr_factor(frac, step, warmup, decay_frac):
    w = min(1.0, (step + 1) / max(1, warmup))
    d = 1.0 if frac < 1 - decay_frac else max(0.0, (1 - frac) / decay_frac)
    return w * d


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


class Evaluator:
    def __init__(self, a, shape):
        from vae import VAE
        self.a, self.shape, self.dev = a, shape, a.device
        self.vae = VAE(a.device, torch.float32 if a.vae_dtype == "fp32" else torch.bfloat16)
        self.det_dev = a.det_device or a.device
        self.det = F_.load_detector(a.i3d, self.det_dev)
        z = np.load(a.realstats)
        self.f_train, self.f_test = z["f_train"], z["f_test"]
        self.m_train = float(z["motion_train"].mean())
        self.refs = json.load(open(os.path.splitext(a.realstats)[0] + "_refs.json"))
        self.txt = torch.zeros(1, 1, 768, device=self.dev); self.mask = torch.ones(1, 1, dtype=torch.bool, device=self.dev)

    @torch.no_grad()
    def videos(self, model, n, shift, first=None, seed=0, bs=32):
        g = torch.Generator(device=self.dev).manual_seed(seed)
        out = []
        for i in range(0, n, bs):
            b = min(bs, n - i)
            fr = None if first is None else first[i:i + b].float()
            with torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                z = euler_sample(model, self.txt.expand(b, -1, -1), self.mask.expand(b, -1), self.shape,
                                 steps=self.a.sample_steps, first=fr, generator=g, shift=shift)
            for j in range(0, b, 4):
                out += list(self.vae.decode(z[j:j + 4].float())[:, :16])
        return out

    def score(self, model, n, shift, tag, first=None, ref="train"):
        t0 = time.time()
        vids = self.videos(model, n, shift, first)
        f = F_.features(self.det, vids, self.det_dev)
        mot = float(np.mean([F_.motion(v) for v in vids]))
        r = dict(tag=tag, n=n, shift=shift, fvd=F_.fvd(self.f_train if ref == "train" else self.f_test, f), ref=ref,
                 motion=mot, motion_ratio=mot / self.m_train, seconds=round(time.time() - t0, 1))
        return r, vids


@torch.no_grad()
def val_loss(model, val, dev):
    """Flow loss at fixed t and fixed noise on held-out test latents (unconditional)."""
    model.eval(); tot, n = 0.0, 0
    g = torch.Generator(device=dev).manual_seed(1234)
    txt = torch.zeros(1, 1, 768, device=dev); m = torch.ones(1, 1, dtype=torch.bool, device=dev)
    for i in range(0, val.shape[0], 64):
        x0 = val[i:i + 64].float(); B = x0.shape[0]
        for tv in (0.2, 0.4, 0.6, 0.8):
            noise = torch.randn(x0.shape, device=dev, generator=g)
            t = torch.full((B,), tv, device=dev)
            xt, tgt, _ = flow_inputs(x0, t, noise, None)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                pred = model(xt, t, txt.expand(B, -1, -1), m.expand(B, -1))
            tot += F.mse_loss(pred.float(), tgt, reduction="sum").item() / x0[0].numel(); n += B
    model.train()
    return tot / n


def save_video_grid(vids, path_png, path_mp4):
    import av
    from PIL import Image
    v = torch.stack(vids[:16])  # (n<=16,T,3,H,W); pad to a multiple of 4 with black clips
    if v.shape[0] % 4:
        v = torch.cat([v, torch.zeros(4 - v.shape[0] % 4, *v.shape[1:], dtype=v.dtype)])
    rows = [torch.cat([v[r * 4 + c] for c in range(4)], dim=-1) for r in range(v.shape[0] // 4)]
    grid = torch.cat(rows, dim=-2)  # (T,3,4H,4W)
    Image.fromarray(torch.cat([grid[0], grid[7], grid[15]], dim=-2).permute(1, 2, 0).numpy()).save(path_png)
    with av.open(path_mp4, "w") as c:
        s = c.add_stream("libx264", rate=8); s.width, s.height = grid.shape[-1], grid.shape[-2]; s.pix_fmt = "yuv420p"
        for fr in grid:
            for p in s.encode(av.VideoFrame.from_ndarray(fr.permute(1, 2, 0).numpy(), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def push(a, files, msg, background=False):
    """Upload to the Hugging Face repo in --hf_repo (skipped if empty). Background pushes (checkpoints during training)
    run in a thread so the clock is not charged; the final push is synchronous."""
    if not a.hf_repo:
        return
    if background:
        import threading
        threading.Thread(target=push, args=(a, files, msg), daemon=True).start()
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for f in files:
            if os.path.exists(f):
                api.upload_file(path_or_fileobj=f, path_in_repo=f"{a.hf_prefix}/{os.path.basename(f)}", repo_id=a.hf_repo,
                                commit_message=msg)
    except Exception as e:  # logged loudly; check the upload before relying on it
        print("PUSH FAILED", repr(e)[:200], flush=True)


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    torch.manual_seed(a.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    lat = load_lat(a.lat_train, dev)
    val = load_lat(a.lat_val, dev, limit=512)
    N, C, T, H, W = lat.shape
    net = build(a.size, in_ch=C, frames=T, grid=H).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    if dev == "cuda":
        from optim import MuonAdamW
        opt = MuonAdamW(net.param_groups(a.lr, a.lr_muon, a.wd, a.optim))
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    step, elapsed_prev, evals_done, history = 0, 0.0, [], []
    if a.resume and os.path.exists(os.path.join(a.resume, "last.pt")):
        ck = torch.load(os.path.join(a.resume, "last.pt"), map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step, elapsed_prev, evals_done, history = ck["step"], ck["elapsed"], ck["evals_done"], ck["history"]
        print(json.dumps({"resumed": step, "elapsed_min": round(elapsed_prev / 60, 2)}), flush=True)
    torch.manual_seed(a.seed * 1000 + step)
    fwd = net if (a.no_compile or dev != "cuda") else torch.compile(net)
    ev = Evaluator(a, (C, T, H, W))
    txt = torch.zeros(a.batch, 1, 768, device=dev); m = torch.ones(a.batch, 1, dtype=torch.bool, device=dev)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    print(json.dumps({"params_M": round(n_params / 1e6, 2), "train_windows": N, "val_windows": val.shape[0],
                      "latent": [C, T, H, W], "tokens": T * (H // 2) * (W // 2), "refs": ev.refs}), flush=True)

    def emit(rec):
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()

    def ckpt(tag, upload_weights=False):
        tmp = os.path.join(a.out, "last.pt.tmp")
        torch.save({"model": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "elapsed": elapsed, "evals_done": evals_done, "history": history, "args": vars(a),
                    "cfg": vars(net.cfg)}, tmp)
        os.replace(tmp, os.path.join(a.out, "last.pt"))
        files = [os.path.join(a.out, "log.jsonl")] + ([os.path.join(a.out, "last.pt")] if upload_weights else [])
        push(a, files, f"ckpt {tag} step {step}", background=True)

    fracs = [float(x) for x in a.eval_fracs.split(",") if x]
    warm, t_start, last_ckpt, elapsed, guard_done = 20, None, time.time(), elapsed_prev, False
    while True:
        if t_start is None and step >= warm and a.steps == 0:
            sync(dev); t_start = time.time() - elapsed_prev
        if t_start is not None:
            elapsed = time.time() - t_start
        frac = (step / a.steps) if a.steps else (elapsed / (a.minutes * 60))
        if frac >= 1.0:
            break
        for fr in fracs:  # evals pause the clock
            if frac >= fr and fr not in evals_done:
                sync(dev); p0 = time.time()
                r, _ = ev.score(ema, a.eval_n, a.shift, f"eval@{fr}")
                r.update(step=step, frac=round(frac, 4), val_ema=val_loss(ema, val, dev), val_raw=val_loss(net, val, dev))
                emit(r); history.append(r); evals_done.append(fr)
                ckpt(f"eval@{fr}")
                if t_start is not None:
                    t_start += time.time() - p0
                last_ckpt = time.time()
        f = lr_factor(frac, step, a.warmup, a.decay_frac)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        idx = torch.randint(0, N, (a.batch,), device=dev)
        x0 = lat[idx].float()
        t = torch.sigmoid(torch.randn(a.batch, device=dev))
        cond = torch.rand(a.batch, device=dev) < a.p_cond
        xt, tgt, lm = flow_inputs(x0, t, torch.randn_like(x0), cond)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            pred = fwd(xt, t, txt, m, cond)
        loss = (((pred.float() - tgt) ** 2) * lm).sum() / (lm.sum() * C * H * W)
        lv = loss.item() if step % 50 == 0 else None
        if lv is not None and (not np.isfinite(lv) or lv == 0.0):
            emit({"STOP": "invalid loss", "step": step, "loss": lv}); ckpt("invalid"); sys.exit(2)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - d)
        step += 1
        if lv is not None:
            rec = {"step": step, "frac": round(frac, 4), "loss": round(lv, 5), "gnorm": round(gn.item(), 3), "lr_f": round(f, 4),
                   "min": round(elapsed / 60, 2),
                   "clips_per_s": round((step - warm) * a.batch / max(elapsed - elapsed_prev, 1e-9), 1) if t_start and step > warm else None,
                   "mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1) if dev == "cuda" else None}
            emit(rec)
            if (a.guard_clips_per_s and not guard_done and t_start is not None and step >= warm + 300 and not a.resume):
                guard_done = True
                if rec["clips_per_s"] < a.guard_clips_per_s:
                    emit({"STOP": "SLOW_HOST", "clips_per_s": rec["clips_per_s"], "need": a.guard_clips_per_s}); sys.exit(3)
        if time.time() - last_ckpt > a.ckpt_minutes * 60:
            ckpt("periodic", upload_weights=True); last_ckpt = time.time()

    # final evaluation (pre-registered): EMA, 2,048 unconditional samples, shift a.shift, vs the train real set
    ckpt("end")
    if a.final_n == 0:  # resume tests
        emit({"end_without_final": step}); return
    final, vids = ev.score(ema, a.final_n, a.shift, "final")
    final.update(step=step, minutes=round(elapsed / 60, 2), val_ema=val_loss(ema, val, dev), val_raw=val_loss(net, val, dev),
                 params_M=round(n_params / 1e6, 2), train_windows=N, clips_seen=step * a.batch)
    save_video_grid(vids, os.path.join(a.out, "samples_final.png"), os.path.join(a.out, "samples_final.mp4"))
    extra = []
    if a.extra_shift:
        r, _ = ev.score(ema, a.final_n, a.extra_shift, "final_shift"); extra.append(r)
    if a.i2v_n:  # image-to-video: real test first frames -> generated rest, scored against the test real set
        from vae import VAE  # noqa: F401  (ev.vae already loaded)
        firsts = val[: a.i2v_n, :, :1]
        r, _ = ev.score(ema, min(a.i2v_n, firsts.shape[0]), a.shift, "i2v_test", first=firsts, ref="test"); extra.append(r)
    v10 = next((h["val_ema"] for h in history if h["tag"] == "eval@0.1"), None)
    # P2 only counts if the still-frame baseline is clearly worse than real video (>= 1.5x the real-vs-real FVD);
    # Sky moves slowly, so otherwise it cannot discriminate and P3 (motion ratio) alone judges motion. Pre-registered.
    p2_disc = ev.refs["fvd_train_vs_static_test"] >= 1.5 * ev.refs["fvd_train_vs_test"]
    checks = dict(
        P1_fvd_below_max=final["fvd"] < a.fvd_max,
        P2_beats_static_baseline=(final["fvd"] < ev.refs["fvd_train_vs_static_test"]) if p2_disc else "not discriminative",
        P3_motion_ratio_0p5_to_2=0.5 <= final["motion_ratio"] <= 2.0,
        P4_val_loss_improved=(v10 is not None and final["val_ema"] < v10),
    )
    gate = all(v for v in checks.values() if not isinstance(v, str))
    out = dict(final=final, extra=extra, history=history, refs=ev.refs, checks=checks, gate_pass=gate,
               context_external={"Latte-XL_674M_sky_fvd": 59.82, "StyleGAN-V": 79.52, "DIGAN": 83.11, "VideoGPT": 222.7,
                                 "MoCoGAN": 206.6, "source": "Latte paper table (EXTERNAL; our harness not yet checked against it)"})
    json.dump(out, open(os.path.join(a.out, "final.json"), "w"), indent=1)
    emit({"final": final, "extra": extra, "checks": checks, "gate_pass": out["gate_pass"]})
    push(a, [os.path.join(a.out, f) for f in ("final.json", "log.jsonl", "samples_final.png", "samples_final.mp4", "last.pt")],
         f"final step {step}")


if __name__ == "__main__":
    main()
