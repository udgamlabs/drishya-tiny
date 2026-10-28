# Stage 2: three techniques against the plain recipe, at equal compute (2026-10-19). No technique beat the plain recipe; MeanFlow diverged.

All numbers are measured by us. Four arms ran one after another, 60 GPU-minutes of training each, on one RTX 5090
(Runpod Secure). Scores are FVD-2048 (shift 3, 50 Euler steps, EMA weights, bf16 decode) against Stage 1's real-video
set on SkyTimelapse. Code: `training/train_arm.py`, verdict rules in `training/stage2_summary.py`.

Files here: `summary.json` (the verdict script's output), `<arm>/result.json` per arm, and `meanflow/log.jsonl` (the
training log of the arm that diverged).

## Verdict (rules written before the run)

Noise floor: the control (A0) scored 160.7 and 148.1 with two sampling seeds, so sigma = 12.6 and the margin is
**+-25.1 FVD**.

| Arm | Speed | Clips seen | FVD-2048 | Motion | Held-out loss | Verdict |
|---|---|---|---|---|---|---|
| **A0** Stage 1 recipe | 383/s | 1.38M | **160.7** (seed 1: 148.1) | 0.88 | **0.3551** | control |
| A1 + joint image+video | 320/s | 1.15M | 182.4 | 0.90 | 0.3621 | TIE (22 worse, inside the margin) |
| A3 + TREAD (50% routing) | **532/s** | **1.92M** | 171.2 | 0.95 | 0.3602 | TIE (11 worse, inside the margin) |
| A2 MeanFlow | 125/s | 0.45M | 4,420 at 4 steps (1 step: 6,771; 2 steps: 11,241) | 3.9 (noise) | 0.3811 | no few-step win (diverged) |

- Context: A0 sampled with only 4 Euler steps gives FVD-512 294.8 with motion 0.62.
- **Recipe for Stage 3 = A0**, unchanged.

## What we learned

1. **A0 at 60 minutes matched Stage 1 at 120 minutes** (160.7 both, shift 3). The model was data-limited on
   SkyTimelapse (1,965 clips): more compute on the same data no longer bought quality.
2. **TREAD gave 39% more throughput but no quality gain** at equal time (171 vs 161, within noise). At about 320 tokens
   per clip, skipping tokens saves compute without helping. It may matter for longer or higher-resolution clips.
3. **Joint image+video training did not help** (182 vs 161, within noise). The gain reported in Latte's ablation did
   not reproduce at our scale and budget; the image batch cost 17% of the video throughput.
4. **Our MeanFlow implementation diverged. This is not a verdict on the method.**
   - The logged average-velocity error climbed from about 1.6 to 300-1,300 (`meanflow/log.jsonl`).
   - The adaptive loss weight 1/(err + c) then silenced those samples, so the few-step path never learned; 1-4-step
     samples are noise (motion 3-5).
   - The plain-flow part (r = t, Euler 50) learned normally for its smaller sample count: FVD-512 289, motion 0.97,
     after 0.45M clips.
   - Likely causes, untested: (a) the time embedding scales t by 1,000, so du/dt and the target -(t - r) du/dt are about
     1,000x too large; (b) the JVP ran under bf16 autocast; (c) the plain-attention JVP path cost 3x throughput, so the arm
     saw 3x fewer clips.
5. **Sampling noise is large:** two seeds of the same model differ by 12.6 FVD at n = 2,048. Comparisons need two or
   more seeds, or a margin like the one used here.
6. **Infrastructure:** Community RTX 5090 hosts failed repeatedly (the same broken machine was assigned twice); Secure
   was reliable. Page-locking 4.5 GB of memory (`pin_memory`) failed on one host but not another, so the trainer does not
   pin large tensors. The first MeanFlow memory probe left out the full data and the evaluation networks and
   under-counted by about 6 GB; the dry run caught it (out of memory at 30.8 of 31.4 GB).

Stage 2 cost $7.20, failed starts included (rows in `../COST.md`).
