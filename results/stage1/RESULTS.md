# Stage 1: method check on SkyTimelapse (2026-10-18). Gate passed.

All numbers are measured by us. One RTX 5090 (Runpod Secure). Unconditional video generation on SkyTimelapse, the
standard FVD benchmark, to check that the recipe (VideoDiT on frozen Wan2.2 VAE latents, rectified flow, MuonAdamW)
learns before spending more. Same 211.9M generator as the release model. Code: `training/train_video.py`.

Files here: `final.json` (the trainer's own gate output), `log.jsonl` (the training log).

## Pass test (written before the run, checked in code)

| Check | Rule | Result |
|---|---|---|
| P1 | FVD-2048 < 200 | **167.7** (pass) |
| P2 | FVD < real test frames frozen still (215.1; discriminative, since it is 3.0x the real-vs-real 72.2) | **167.7** (pass) |
| P3 | 0.5 <= motion ratio <= 2.0 | **0.89** (pass) |
| P4 | Held-out loss at the end < at 10% | 0.3565 < 0.3608 (pass) |

Also reported (not part of the gate):
- FVD-2048 at sampling shift 3.0: **160.7**.
- Image-to-video (real first frame) FVD-512 against the held-out test set: **139.0**, motion 1.23.
- References in our harness: real test vs real train 72.2; still frames 215.1.
- Context only, not a claim (our harness was not matched to theirs at the time): Latte-XL (674M) 59.82, StyleGAN-V 79.52,
  DIGAN 83.11, MoCoGAN 206.6, VideoGPT 222.7 (Latte paper table). Our FVD code was later shown to match StyleGAN-V's
  official FVD code on identical frames (`../stage4/fvd_audit_result.json`).

## Training

- 120 minutes, 43,449 steps x batch 64 = **2.78M clip-samples** (~76 passes over 36,481 windows).
- Speed 386 clips/s, peak memory 25.5 GB.

| Check | FVD-512 | Motion | Held-out loss |
|---|---|---|---|
| 10% | 331.3 | 0.98 | 0.3608 |
| 25% | 249.2 | 0.99 | 0.3555 |
| 50% | 204.9 | 1.04 | **0.3535** (best) |
| 75% | 248.6 | 0.96 | 0.3542 |
| final (EMA after LR decay), FVD-2048 | **167.7** | 0.89 | 0.3565 |

**Reading:** the held-out loss bottomed at about 50% of the run and the 75% check got worse. The model saw each window
about 76 times, so it was probably **data-limited** (overfitting to 1,965 training clips) rather than compute-limited.
That is why Stage 3 moved to a much larger open-domain set.

## What went wrong on the way (cost $7.14; every row in `../COST.md`)

1. Two Community hosts failed before the run (one CUDA error, one never reachable) and one run stopped at a data-count
   check because of our own threshold bug. About $0.30 in total.
2. Community had no stock for the main run, so it ran on Secure at $1.19/h instead of $0.69/h.
3. The real-video statistics step took 50 minutes instead of about 10: thread oversubscription when loading frames
   (16 Python threads x PyTorch intra-op threads on 120 cores). Fix: `torch.set_num_threads(1)` inside the loader threads.
4. Evaluation decode ran at 0.92 s per clip instead of 0.42: on this host bf16 failed our pre-registered VAE precision
   rule, so it decoded in fp32.
