"""Drishya Tiny Stage 2 bake-off trainer: one arm per call, equal wall-clock budget on the same GPU.

Arms (all start from the Stage 1 recipe, which is A0):
  base     A0: Stage 1 recipe (rectified flow, Muon, EMA, batch 64, 25% image-to-video steps)
  joint    A1: + joint image+video training: every step also trains on --img_batch single frames (T=1 latents),
               as in Latte's ablation (Sky FVD 59.8 -> 42.7 there)
  meanflow A2: MeanFlow (Geng et al., NeurIPS 2025): learn the average velocity u(z, r, t) with a JVP target, so
               generation works in 1-4 steps; trained from scratch, no teacher
  tread    A3: + TREAD token routing (50% of tokens skip blocks 2..13)

Evaluation (pre-registered, same for every arm): EMA weights; FVD-2048 at 50 Euler steps, shift 3.0 (the better Stage 1
shift) vs the Stage 1 real set; motion ratio; held-out flow loss at fixed t and noise. MeanFlow is also scored at 1, 2
and 4 steps. A0 is also scored with a second sampling seed (eval noise) and at 4 Euler steps (few-step baseline).
"""
import argparse, copy, glob, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import fvd as F_  # noqa: E402
from vmodel import build, flow_inputs, euler_sample, meanflow_sample  # noqa: E402
from train_video import load_lat, lr_factor, sync, val_loss, save_video_grid, push, Evaluator  # noqa: E402


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "joint", "meanflow", "tread"])
    ap.add_argument("--lat_train", required=True); ap.add_argument("--lat_val", required=True)
    ap.add_argument("--lat_img", default="", help="joint arm: single-frame latents (N,48,1,16,16)")
    ap.add_argument("--realstats", required=True); ap.add_argument("--i3d", required=True)
    ap.add_argument("--size", default="s210"); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--img_batch", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=60); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr_muon", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--decay_frac", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.9999); ap.add_argument("--p_cond", type=float, default=0.25)
    ap.add_argument("--mf_ratio", type=float, default=0.25, help="MeanFlow: share of samples with r != t")
    ap.add_argument("--mf_p", type=float, default=1.0, help="MeanFlow adaptive loss weight power")
    ap.add_argument("--tread_ratio", type=float, default=0.5)
    ap.add_argument("--mid_n", type=int, default=512); ap.add_argument("--final_n", type=int, default=2048)
    ap.add_argument("--shift", type=float, default=3.0); ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--mf_steps", default="1,2,4")
    ap.add_argument("--vae_dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--guard_clips_per_s", type=float, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--hf_repo", default=""); ap.add_argument("--hf_prefix", default="stage2")
    ap.add_argument("--no_compile", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--det_device", default=""); ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class ArmEval(Evaluator):
    """Evaluator with a choice of sampler: Euler (flow) or MeanFlow k-step. park()/unpark() move the VAE and I3D off
    and back onto the GPU, so they take no GPU memory during training."""
    def park(self):
        self._move("cpu")

    def _move(self, d):
        if self.dev != "cuda":
            return
        self.vae.m.to(d); self.vae.mean = self.vae.mean.to(d); self.vae.std = self.vae.std.to(d); self.vae.dev = d
        self.det = self.det.to(d); self.det_dev = d
        if d == "cpu":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def videos(self, model, n, shift, first=None, seed=0, bs=32, mf_steps=0):
        g = torch.Generator(device=self.dev).manual_seed(seed)
        out = []
        for i in range(0, n, bs):
            b = min(bs, n - i)
            fr = None if first is None else first[i:i + b].float()
            with torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                txt, m = self.txt.expand(b, -1, -1), self.mask.expand(b, -1)
                if mf_steps:
                    z = meanflow_sample(model, txt, m, self.shape, steps=mf_steps, first=fr, generator=g)
                else:
                    z = euler_sample(model, txt, m, self.shape, steps=self.a.sample_steps, first=fr, generator=g, shift=shift)
            for j in range(0, b, 4):
                out += list(self.vae.decode(z[j:j + 4].float())[:, :16])
        return out

    def score(self, model, n, shift, tag, first=None, ref="train", seed=0, mf_steps=0, euler_steps=None):
        t0 = time.time()
        self._move(self.dev)
        keep = self.a.sample_steps
        if euler_steps:
            self.a.sample_steps = euler_steps
        vids = self.videos(model, n, shift, first, seed=seed, mf_steps=mf_steps)
        self.a.sample_steps = keep
        f = F_.features(self.det, vids, self.det_dev)
        mot = float(np.mean([F_.motion(v) for v in vids]))
        self.park()
        r = dict(tag=tag, n=n, shift=shift, seed=seed, mf_steps=mf_steps, euler_steps=euler_steps or (0 if mf_steps else keep),
                 fvd=F_.fvd(self.f_train if ref == "train" else self.f_test, f), ref=ref, motion=mot,
                 motion_ratio=mot / self.m_train, seconds=round(time.time() - t0, 1))
        return r, vids


def set_attn(net, impl):
    for mod in net.modules():
        if hasattr(mod, "impl"):
            mod.impl = impl


def meanflow_loss(net, x0, txt, m, cond, a, dev, fwd=None):
    """MeanFlow objective: u_target = v - (t - r) * du/dt, du/dt via a JVP with tangent (v, 0, 1). Adaptive weight.
    Samples with r == t (75%) have u_target = v exactly (the JVP term is multiplied by t - r = 0), so only the r != t
    samples run the JVP (plain-attention, eager); the rest use the normal compiled forward. Same maths, ~4x less JVP memory."""
    fwd = net if fwd is None else fwd
    B = x0.shape[0]
    t = torch.sigmoid(torch.randn(B, device=dev)); r = torch.sigmoid(torch.randn(B, device=dev))
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    k = max(1, round(a.mf_ratio * B))  # a FIXED count of r != t samples, so the compiled part keeps one batch shape
    diff = torch.zeros(B, dtype=torch.bool, device=dev); diff[torch.randperm(B, device=dev)[:k]] = True
    r = torch.where(diff, r, t)
    noise = torch.randn_like(x0)
    z, v, lm = flow_inputs(x0, t, noise, cond)
    norm = lm.flatten(1).sum(1) * x0.shape[1] * x0.shape[3] * x0.shape[4]
    err = torch.zeros(B, device=dev)
    same = ~diff
    if same.any():
        i = same.nonzero().squeeze(1)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            u0 = fwd(z[i], t[i], txt[i], m[i], cond[i], r=t[i])
        err = err.index_put((i,), (((u0.float() - v[i]) ** 2) * lm[i]).flatten(1).sum(1) / norm[i])
    if diff.any():
        j = diff.nonzero().squeeze(1)
        set_attn(net, "manual")
        fn = lambda z_, r_, t_: net(z_, t_, txt[j], m[j], cond[j], r=r_)
        v_tan = v[j] * lm[j]  # the clean first frame (image-to-video steps) does not move with t
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            u1, dudt = torch.func.jvp(fn, (z[j], r[j], t[j]), (v_tan, torch.zeros_like(r[j]), torch.ones_like(t[j])))
        set_attn(net, "sdpa")
        tgt = (v[j] - (t[j] - r[j]).view(-1, 1, 1, 1, 1) * dudt.float()).detach()
        err = err.index_put((j,), (((u1.float() - tgt) ** 2) * lm[j]).flatten(1).sum(1) / norm[j])
    w = 1.0 / (err.detach() + 1e-3) ** a.mf_p
    return (w * err).mean(), err.mean()


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    torch.manual_seed(a.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    # training latents stay in pinned CPU memory (4.5 GB); each batch (~8 MB) is copied over. Frees GPU memory for
    # MeanFlow's JVP (Stage 2 dry run: OOM at 30.8 of 31.4 GB with all latents on the GPU). Same for every arm.
    lat = load_lat(a.lat_train, "cpu"); val = load_lat(a.lat_val, dev, limit=512)
    img = load_lat(a.lat_img, "cpu") if a.arm == "joint" else None
    # (no pin_memory: page-locking 4.5 GB failed with "CUDA error: invalid argument" on one Stage 2 host)
    N, C, T, H, W = lat.shape
    kw = dict(in_ch=C, frames=T, grid=H)
    if a.arm == "meanflow":
        kw.update(meanflow=True)  # SDPA normally; meanflow_loss switches to plain attention only for its JVP
    if a.arm == "tread":
        kw.update(tread_ratio=a.tread_ratio)
    net = build(a.size, **kw).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    if dev == "cuda":
        from optim import MuonAdamW
        opt = MuonAdamW(net.param_groups(a.lr, a.lr_muon, 0.0, "muon"))
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    torch.manual_seed(a.seed * 1000)
    fwd = net if (a.no_compile or dev != "cuda") else torch.compile(net)  # MeanFlow's JVP part always runs eagerly
    ev = ArmEval(a, (C, T, H, W)); ev.park()  # scoring networks live on the GPU only while scoring
    txt = torch.zeros(a.batch, 1, 768, device=dev); m = torch.ones(a.batch, 1, dtype=torch.bool, device=dev)
    txt_i = torch.zeros(a.img_batch, 1, 768, device=dev); m_i = torch.ones(a.img_batch, 1, dtype=torch.bool, device=dev)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def emit(rec):
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()

    emit({"arm": a.arm, "params_M": round(n_params / 1e6, 2), "train_windows": N, "img_frames": 0 if img is None else img.shape[0],
          "tokens": T * (H // 2) * (W // 2), "refs": ev.refs})
    warm, t_start, elapsed, step, mid_done, history = 20, None, 0.0, 0, False, []
    while True:
        if t_start is None and step >= warm and a.steps == 0:
            sync(dev); t_start = time.time()
        if t_start is not None:
            elapsed = time.time() - t_start
        frac = (step / a.steps) if a.steps else (elapsed / (a.minutes * 60))
        if frac >= 1.0:
            break
        if frac >= 0.5 and not mid_done:  # mid-run check pauses the clock
            sync(dev); p0 = time.time()
            kwm = dict(mf_steps=1) if a.arm == "meanflow" else {}
            r, _ = ev.score(ema, a.mid_n, a.shift, "mid@0.5", **kwm)
            r.update(step=step, val_ema=val_loss(ema, val, dev)); emit(r); history.append(r); mid_done = True
            if t_start is not None:
                t_start += time.time() - p0
        f = lr_factor(frac, step, a.warmup, a.decay_frac)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        x0 = lat[torch.randint(0, N, (a.batch,))].to(dev).float()
        cond = torch.rand(a.batch, device=dev) < a.p_cond
        if a.arm == "meanflow":
            loss, raw = meanflow_loss(net, x0, txt, m, cond, a, dev, fwd=fwd)
        else:
            t = torch.sigmoid(torch.randn(a.batch, device=dev))
            xt, tgt, lm = flow_inputs(x0, t, torch.randn_like(x0), cond)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                pred = fwd(xt, t, txt, m, cond)
            loss = (((pred.float() - tgt) ** 2) * lm).sum() / (lm.sum() * C * H * W); raw = loss
            if a.arm == "joint":
                xi = img[torch.randint(0, img.shape[0], (a.img_batch,))].to(dev).float()
                ti = torch.sigmoid(torch.randn(a.img_batch, device=dev))
                xti, tgti, _ = flow_inputs(xi, ti, torch.randn_like(xi), None)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    pi = fwd(xti, ti, txt_i, m_i)
                loss = loss + F.mse_loss(pi.float(), tgti)
        lv = raw.item() if step % 50 == 0 else None
        if lv is not None and (not np.isfinite(lv) or lv == 0.0):
            emit({"STOP": "invalid loss", "step": step, "loss": lv}); sys.exit(2)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - d)
        step += 1
        if lv is not None:
            cps = round((step - warm) * a.batch / max(elapsed, 1e-9), 1) if t_start and step > warm else None
            emit({"step": step, "frac": round(frac, 4), "loss": round(lv, 5), "gnorm": round(gn.item(), 3), "lr_f": round(f, 4),
                  "min": round(elapsed / 60, 2), "clips_per_s": cps,
                  "mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1) if dev == "cuda" else None})
            if a.guard_clips_per_s and t_start is not None and step == warm + 301 and cps is not None and cps < a.guard_clips_per_s:
                emit({"STOP": "SLOW_HOST", "clips_per_s": cps, "need": a.guard_clips_per_s}); sys.exit(3)
    # save the arm model first, then the pre-registered final evaluation
    torch.save({"model": net.state_dict(), "ema": ema.state_dict(), "step": step, "args": vars(a), "cfg": vars(net.cfg)},
               os.path.join(a.out, "arm.pt"))
    res = dict(arm=a.arm, step=step, clips_seen=step * a.batch, minutes=round(elapsed / 60, 2), params_M=round(n_params / 1e6, 2),
               val_ema=val_loss(ema, val, dev), history=history, refs=ev.refs, scores=[])
    if a.arm == "meanflow":
        ks = [int(k) for k in a.mf_steps.split(",") if k]
        quick = []
        for k in ks:
            r, _ = ev.score(ema, a.mid_n, a.shift, f"mf{k}_512", mf_steps=k); res["scores"].append(r); emit(r); quick.append((r["fvd"], k))
        best_k = min(quick)[1]
        r, vids = ev.score(ema, a.final_n, a.shift, f"final_mf{best_k}", mf_steps=best_k); r["primary"] = True
        res["scores"].append(r); emit(r)
        r2, _ = ev.score(ema, a.mid_n, a.shift, "euler50_512"); res["scores"].append(r2); emit(r2)
    else:
        r, vids = ev.score(ema, a.final_n, a.shift, "final_euler50"); r["primary"] = True; res["scores"].append(r); emit(r)
        if a.arm == "base":
            r2, _ = ev.score(ema, a.final_n, a.shift, "final_euler50_seed1", seed=1); res["scores"].append(r2); emit(r2)
            r3, _ = ev.score(ema, a.mid_n, a.shift, "euler4_512", euler_steps=4); res["scores"].append(r3); emit(r3)
    save_video_grid(vids, os.path.join(a.out, "samples.png"), os.path.join(a.out, "samples.mp4"))
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    emit({"result": {k: v for k, v in res.items() if k not in ("history", "refs")}})
    push(a, [os.path.join(a.out, f) for f in ("result.json", "log.jsonl", "samples.png", "samples.mp4", "arm.pt")], f"{a.arm} done")


if __name__ == "__main__":
    main()
