# Stage 3: the release model (2026-10-20). All four pre-registered checks pass.

All numbers are measured by us. Run 2026-10-20, 02:41-18:21 IST, on one NVIDIA L40S 48 GB (Shadeform, Massed Compute).
900 training minutes, 180,080 steps at batch 64, 11.53M clips seen (~24 passes over 475,157 two-second windows from
VidGen-1M zips 0-599). Code: `training/train_t2v.py`. The released weights are this run's final EMA weights.

Files here: `final.json` (the trainer's own gate output), `refs.json` (reference numbers on the real held-out clips),
`log.jsonl` (the training log, every 50 steps, plus the evaluations).

## Final scores (held-out VidGen zips 2040-2047, never trained on)

EMA weights, 50 Euler steps, shift 3, CFG 5.

| Check | Rule | Result | |
|---|---|---|---|
| G1 | Text-to-video FVD-2048 < still-frame FVD (660.4), motion 0.5-2.0 | **FVD 235.7**, motion **0.73** | pass |
| G2 | CLIP gap >= 30% of the real clips' gap (25.73 -> 7.72), > 0 | **20.15** (78% of real) | pass |
| G3 | Held-out loss at the end < at 25% | **0.454** < 0.471 | pass |
| G4 | Image-to-video FVD < still-frame FVD | **FVD-512 240.3** (motion 0.89, CLIP gap 23.7) | pass |

- References (2,048 real held-out clips): still-frame FVD 660.4; CLIP score with the own caption 37.43 vs a shuffled
  caption 11.70 (gap 25.73); real motion 0.063.
- CLIP is OpenCLIP ViT-B/32 `laion2b_s34b_b79k`, not OpenAI CLIP.
- FVD-512 is biased upward against FVD-2048; never compare the two.

## During training (512 held-out clips each)

| | 25% | 50% | 75% | final (2,048 clips) |
|---|---|---|---|---|
| FVD | 446.8 | 431.8 | 453.8 | 235.7 |
| Motion ratio | 0.76 | 0.82 | 0.85 | 0.73 |
| CLIP gap | 16.24 | 18.27 | 19.27 | 20.15 |
| Held-out loss | 0.4706 | 0.4629 | 0.4587 | 0.4541 |

## What the samples look like (our impression, not measured)

- **Text-to-video:** recognisable scene layouts and colours that match the captions (ocean horizon, person at a table,
  animal on grass, red car, road through trees, food on a plate, swimmer in a pool). Motion is small. Faces and hands are
  smeared and fine detail is poor.
- **Image-to-video:** the first frame is the given image. Later frames keep the scene but drift: faces and text melt over
  the 2 seconds and blur grows with time.

## What went wrong on the way

Stage 3 cost $24.71 in total (rows in `../COST.md`). The release model's own path was about $21: the data step ($7.40)
and the main training run ($13.76). The other $3.55 went on failed attempts:
- Data step, five attempts before the one that finished: three machines never became reachable, one could not reach
  PyPI, and one ran at only about 5 windows/s and was stopped for an encoder fix.
- Dry run: two starts never came up; the third passed (train, resume, every score, samples, uploads).
- Main run: three Runpod starts never came up (no machine reachable, or a container that never started), so the run moved
  to an L40S on Shadeform after a $0.28 speed probe there.
