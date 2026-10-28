"""Drishya Tiny Stage 5a arm trainer: warm start from the Stage 3 final model, then the same recipe with up to three changes:
  --text_len 128      caption limit (Stage 3: 77; 84.5% of VidGen captions were cut);
  --p_short 0.2       share of samples whose caption is replaced by its short form (first sentence, "The video shows" removed);
  --img_dir DIR       1-frame image latents (lat_*.npy (N,48,1,16,16) + cap_*.json) mixed in as whole image steps:
  --p_img_step 0.3    share of steps that are image steps, --img_batch 256 images per image step (~80% of a video step's tokens).
Arms run the same --steps, so A0 vs A1 vs A2 differ only in data.

Evaluation at the end (EMA weights, 50 Euler steps, --cfg/--shift = the Stage 4 choice 5/5):
  - FVD-512 + CLIP gap on the held-out VidGen clips (the Stage 4 sweep harness), I2V FVD on 256, held-out flow loss;
  - VidProM: 200 real user prompts, CLIP gap (own prompt minus a shuffled prompt);
  - mini-semantic videos: the six weak VBench semantic dimensions' prompts x --gen_seeds, saved as <prompt>-<i>.mp4 for the
    VBench scoring job (scores are computed there, in VBench's own env).
Checkpoints: weights-only (model + EMA) milestones/step_XXXXXX.pt at --milestones, never overwritten. No mid-arm resume:
an interrupted arm reruns. The kit(...) calls and KIT_* variables are progress hooks for our internal job runner; they
do nothing unless KIT_RUN is set.
"""
import argparse, copy, json, os, random, re, subprocess, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fvd as F_  # noqa: E402
from vmodel import build, flow_inputs  # noqa: E402
from train_video import lr_factor, sync, save_video_grid  # noqa: E402
from train_t2v import Shards, Clip, cfg_sample  # noqa: E402

E = os.environ
LEAD = re.compile(r"^(in|throughout) (the|this) (video|clip|footage|scene),?\s+|^(the|this) (video|clip|footage|scene|image) "
                  r"(shows|features|depicts|captures|displays|presents|showcases|is of|contains)\s+(that\s+)?", re.I)


def short_caption(c):
    """First sentence, without the captioner's lead-in ("The video shows ..."), no final full stop. Never empty."""
    s = re.split(r"(?<=[.!?])\s+", c.strip(), maxsplit=1)[0].strip()
    t = LEAD.sub("", s).strip().rstrip(".").strip()
    return t or c


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat_train", required=True); ap.add_argument("--lat_val", required=True)
    ap.add_argument("--real_frames", required=True); ap.add_argument("--i3d", required=True)
    ap.add_argument("--init_from", required=True, help="Stage 3 final last.pt (model + EMA; fresh optimiser and schedule)")
    ap.add_argument("--size", default="s210"); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr_muon", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--decay_frac", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.9995); ap.add_argument("--p_cond", type=float, default=0.25)
    ap.add_argument("--cfg_drop", type=float, default=0.1); ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0); ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--text_len", type=int, default=77); ap.add_argument("--p_short", type=float, default=0.0)
    ap.add_argument("--img_dir", default=""); ap.add_argument("--p_img_step", type=float, default=0.0); ap.add_argument("--img_batch", type=int, default=256)
    ap.add_argument("--final_n", type=int, default=512); ap.add_argument("--i2v_n", type=int, default=256)
    ap.add_argument("--vidprom", required=True); ap.add_argument("--vidprom_n", type=int, default=200)
    ap.add_argument("--mini", required=True, help="mini_semantic.json"); ap.add_argument("--mini_limit", type=int, default=0, help="TEST: first N prompts per dimension")
    ap.add_argument("--gen_seeds", type=int, default=2); ap.add_argument("--gen_batch", type=int, default=32)
    ap.add_argument("--milestones", default=""); ap.add_argument("--kit_prefix", default="")
    ap.add_argument("--deadline_ts", type=float, default=0, help="speed guard: at step --guard_step, stop (exit 3) if training + eval cannot end by then")
    ap.add_argument("--guard_step", type=int, default=500); ap.add_argument("--eval_minutes", type=float, default=18)
    ap.add_argument("--clip_model", default="ViT-B-32"); ap.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_compile", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class Text:
    def __init__(self, dev, max_len):
        from transformers import AutoTokenizer, T5EncoderModel
        self.tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.enc = T5EncoderModel.from_pretrained("google/flan-t5-base", dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval().requires_grad_(False)
        self.dev, self.max_len = dev, max_len
        self.null = self(["", ])

    @torch.no_grad()
    def __call__(self, texts):
        e = self.tok(texts, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        h = self.enc(input_ids=e.input_ids.to(self.dev), attention_mask=e.attention_mask.to(self.dev)).last_hidden_state
        return h.float(), e.attention_mask.to(self.dev).bool()


def kit(*x):
    if E.get("KIT_RUN"):
        subprocess.run(["kit", *map(str, x)], check=False)


def save_mp4(frames, path, fps=8):
    import av
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=fps, options={"crf": "10"}); s.width, s.height = frames.shape[-1], frames.shape[-2]; s.pix_fmt = "yuv420p"
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr.permute(1, 2, 0).numpy(), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(os.path.join(a.out, "milestones"), exist_ok=True); dev = a.device
    torch.manual_seed(a.seed); random.seed(a.seed); rng = np.random.default_rng(a.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    max_steps = int(E.get("KIT_MAX_STEPS") or 0) if E.get("KIT_PREFLIGHT") == "1" else 0
    tr, va = Shards(a.lat_train), Shards(a.lat_val)
    im = Shards(a.img_dir) if a.img_dir and a.p_img_step > 0 else None
    C, T, H, W = tr.shape
    if im is not None:
        assert im.shape == (C, 1, H, W), im.shape
    net = build(a.size, in_ch=C, frames=T, grid=H).to(dev)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    if dev == "cuda":
        from optim import MuonAdamW
        opt = MuonAdamW(net.param_groups(a.lr, a.lr_muon, 0.0, "muon"))
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    step, n_img_steps = 0, 0
    ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); init_step = ck.get("step"); del ck
    fwd = net if (a.no_compile or dev != "cuda") else torch.compile(net)
    text = Text(dev, a.text_len)
    from vae import VAE
    vae = VAE(dev, torch.bfloat16)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def emit(r):
        print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()

    emit({"arm_args": vars(a), "init_step": init_step, "train_windows": tr.n, "images": im.n if im else 0, "latent": [C, T, H, W],
          "params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2), "preflight_max_steps": max_steps})
    nc, nm = text.null
    val_lat, val_caps = va.get(list(range(min(a.final_n, va.n))))

    def caps_for(caps):
        return [short_caption(c) if random.random() < a.p_short else c for c in caps]

    def milestone():
        name = f"step_{step:06d}.pt"; p = os.path.join(a.out, "milestones", name)
        torch.save({"model": net.state_dict(), "ema": ema.state_dict(), "step": step, "img_steps": n_img_steps, "args": vars(a),
                    "cfg": vars(net.cfg)}, p + ".tmp")
        os.replace(p + ".tmp", p)
        if a.kit_prefix:
            kit("save", p, "--final", "--as", f"{a.kit_prefix}/milestones/{name}")
        emit({"milestone": name, "step": step})

    total = a.steps if not max_steps else min(a.steps, max_steps)
    miles = sorted(set(int(x) for x in a.milestones.split(",") if x and int(x) <= total) | {total})
    t0, t_steps = time.time(), None
    while step < total:
        frac = step / a.steps
        f_ = lr_factor(frac, step, a.warmup, a.decay_frac)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f_
        is_img = im is not None and rng.random() < a.p_img_step
        if is_img:
            x0, caps = im.get(rng.integers(0, im.n, a.img_batch)); B = a.img_batch; n_img_steps += 1
        else:
            x0, caps = tr.get(rng.integers(0, tr.n, a.batch)); B = a.batch
        x0 = x0.to(dev)
        c, cm = text(caps_for(caps))
        drop = torch.rand(B, device=dev) < a.cfg_drop
        c = torch.where(drop[:, None, None], nc.expand_as(c), c); cm = torch.where(drop[:, None], nm.expand_as(cm), cm)
        t = torch.sigmoid(torch.randn(B, device=dev))
        cond = torch.zeros(B, dtype=torch.bool, device=dev) if is_img else torch.rand(B, device=dev) < a.p_cond
        xt, tgt, lm = flow_inputs(x0, t, torch.randn_like(x0), cond)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            pred = fwd(xt, t, c, cm, cond)
        loss = (((pred.float() - tgt) ** 2) * lm).sum() / (lm.sum() * C * H * W)
        lv = loss.item() if step % 50 == 0 else None
        if lv is not None and (not np.isfinite(lv) or lv == 0.0):
            emit({"STOP": "invalid loss", "step": step}); sys.exit(2)
        loss.backward(); gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - d)
        step += 1
        if step == 30:
            sync(dev); t_steps = (time.time(), step)
        if lv is not None:
            sps = round((step - t_steps[1]) / (time.time() - t_steps[0]), 3) if t_steps and step > t_steps[1] else None
            r = {"step": step, "loss": round(lv, 5), "img_step": is_img, "gnorm": round(gn.item(), 3), "lr_f": round(f_, 4),
                 "min": round((time.time() - t0) / 60, 2), "steps_per_s": sps, "img_frac": round(n_img_steps / step, 3),
                 "mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1) if dev == "cuda" else None}
            emit(r); kit("beat", step, "--loss", round(lv, 5))
            if a.deadline_ts and t_steps and step >= a.guard_step and not getattr(a, "_guarded", False):
                a._guarded = True; eta = time.time() + (total - step) / max(sps, 1e-6) + a.eval_minutes * 60
                emit({"speed_guard": {"steps_per_s": sps, "eta_ts": round(eta), "deadline_ts": round(a.deadline_ts), "ok": eta <= a.deadline_ts}})
                if eta > a.deadline_ts:
                    emit({"STOP": "SLOW_HOST", "steps_per_s": sps}); sys.exit(3)
        if step in miles:
            milestone()
    if dev == "cuda":
        kit("metric", "train_peak_mem_gb", round(torch.cuda.max_memory_reserved() / 1e9, 2))

    # ---------------- evaluation (EMA) ----------------
    kit("phase", "eval", "--stall-min", 40)
    det = F_.load_detector(a.i3d, dev)

    class CA: clip_model = a.clip_model; clip_pretrained = a.clip_pretrained  # noqa: E701
    clip = Clip(CA, dev)

    @torch.no_grad()
    def gen(prompts, seed, first=None):
        g = torch.Generator(device=dev).manual_seed(seed); c, cm = text(prompts)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            z = cfg_sample(ema, c, cm, nc, nm, (C, T, H, W), a.sample_steps, a.cfg, a.shift, first, g)
        out = []
        for j in range(0, len(prompts), 8):
            out += [v.cpu() for v in vae.decode(z[j:j + 8].float())]
        return out

    real = torch.from_numpy(np.load(a.real_frames, mmap_mode="r")[: len(val_caps)].copy())
    f_real = F_.features(det, list(real), dev)
    res = {"arm": os.path.basename(a.out.rstrip("/")), "steps": step, "img_steps": n_img_steps, "settings": {"cfg": a.cfg, "shift": a.shift, "steps": a.sample_steps}}
    _, _, res["real_clip_gap"] = clip.gap(list(real), val_caps)

    def heldout(i2v):
        vids, n = [], (min(a.i2v_n, len(val_caps)) if i2v else len(val_caps))
        for i in range(0, n, a.gen_batch):
            b = min(a.gen_batch, n - i)
            vids += [v[:16] for v in gen(val_caps[i:i + b], 20261022 + i // a.gen_batch, val_lat[i:i + b, :, :1].to(dev) if i2v else None)]
            kit("beat", step + 1 + i // a.gen_batch)
        f = F_.features(det, vids, dev)
        r = {"n": n, "fvd": F_.fvd(f_real[:n], f), "motion_ratio": float(np.mean([F_.motion(v) for v in vids])) / float(np.mean([F_.motion(v) for v in real[:n]]))}
        if not i2v:
            r["clip_own"], _, r["clip_gap"] = clip.gap(vids, val_caps[:n])
        return r, vids

    res["t2v_heldout"], vids = heldout(False); save_video_grid(vids, os.path.join(a.out, "samples_t2v.png"), os.path.join(a.out, "samples_t2v.mp4"))
    res["i2v_heldout"], _ = heldout(True); emit({"heldout": res["t2v_heldout"], "i2v": res["i2v_heldout"]})

    vp = json.load(open(a.vidprom))["prompts"][: a.vidprom_n]
    vids = []
    for i in range(0, len(vp), a.gen_batch):
        vids += [v[:16] for v in gen(vp[i:i + a.gen_batch], 31000 + i // a.gen_batch)]
        kit("beat", step + 1000 + i // a.gen_batch)
    own, shuf, gap = clip.gap(vids, vp)
    res["vidprom"] = {"n": len(vp), "clip_own": own, "clip_shuf": shuf, "clip_gap": gap}
    save_video_grid(vids, os.path.join(a.out, "samples_vidprom.png"), os.path.join(a.out, "samples_vidprom.mp4"))
    emit({"vidprom": res["vidprom"]})

    @torch.no_grad()
    def val_loss():
        tot, k = 0.0, 0; g = torch.Generator(device=dev).manual_seed(1234)
        for i in range(0, len(val_caps), 64):
            x0 = val_lat[i:i + 64].to(dev); c, cm = text(val_caps[i:i + 64])
            for tv in (0.2, 0.4, 0.6, 0.8):
                tt = torch.full((x0.shape[0],), tv, device=dev); xt, tgt, _ = flow_inputs(x0, tt, torch.randn(x0.shape, device=dev, generator=g), None)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    p = ema(xt, tt, c, cm)
                tot += F.mse_loss(p.float(), tgt, reduction="sum").item() / x0[0].numel(); k += x0.shape[0]
        return tot / k
    res["val_loss"] = val_loss()

    mini = json.load(open(a.mini)); md = os.path.join(a.out, "mini_videos"); os.makedirs(md, exist_ok=True)
    prompts = sorted(set(p for v in mini.values() for p in (v[: a.mini_limit] if a.mini_limit else v)))
    jobs = [(p, f"{p}-{i}.mp4") for p in prompts for i in range(a.gen_seeds) if not os.path.exists(os.path.join(md, f"{p}-{i}.mp4"))]
    tg = time.time()
    for b in range(0, len(jobs), a.gen_batch):
        ch = jobs[b:b + a.gen_batch]
        for (p, fn), v in zip(ch, gen([j[0] for j in ch], 40000 + b // a.gen_batch)):
            save_mp4(v, os.path.join(md, fn))
        kit("beat", step + 2000 + b // a.gen_batch)
    res["mini_videos"] = {"prompts": len(prompts), "videos": len(os.listdir(md)), "minutes": round((time.time() - tg) / 60, 1)}
    if dev == "cuda":
        res["peak_mem_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2); kit("metric", "peak_mem_gb", res["peak_mem_gb"])
    json.dump(res, open(os.path.join(a.out, "final.json"), "w"), indent=1)
    emit({"final": res})


if __name__ == "__main__":
    main()
