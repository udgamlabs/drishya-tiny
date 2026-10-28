# Cost: every machine, failed attempts included

**All-in: ~$57.** $57.43 of GPU time across 55 machine rows, every failed attempt included, plus $0.03 of non-GPU
research on day 0: **$57.46 in total**. The release model's own path was **~$21**: the Stage 3 data step ($7.40, one
RTX 5090, ~6.2 h, Runpod) and the main training run ($13.76, one L40S, 15 h 40 min of machine time including the
evaluations, Shadeform).

| Stage | GPU cost | What it bought |
|---|---|---|
| Day-0 checks (Mac, a Colab probe, research) | $0.03 | feasibility |
| Stage 1 (SkyTimelapse method check) | $7.14 | the recipe works |
| Stage 2 (three techniques vs the plain recipe, equal compute) | $7.20 | keep the plain recipe |
| Stage 3 (data + training of the release model) | $24.71 | **the release model** |
| Stage 4 (sampling sweep + VBench) | $8.48 | the evaluation |
| Stage 5a (three controlled arms + scoring) | $9.90 | the ablation |
| **Total** | **$57.46** | |

**Not included:** the pretraining of the borrowed, frozen Wan2.2 VAE and Flan-T5-Base (not ours); people's time;
electricity for the Mac used for checks; storage.

**How the rows were measured:** Stages 1-3 and the first four Stage 4 rows are provider bills, or rate x time where
marked in our log as approximate. The later rows (our job runner, Stages 4-5a) are rate x minutes. "Waste" is money
that produced no kept result (failed starts, idle or hung time, runs stopped by our own bug), as we logged it per row.
Dates are the start date of each machine. Prices are per GPU-hour at the time.

| # | Stage | Date | What | Machine | Rate | Minutes | Cost | Outcome | Waste |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 2026-10-18 | Dry run 1: bad host (CUDA error) | Runpod Community, 1x RTX 5090 | $0.69/h | 7 | $0.08 | failed | $0.08 |
| 2 | 1 | 2026-10-18 | Dry run 2: machine never reachable | Runpod Community, 1x RTX 5090 | $0.69/h | 6 | $0.07 | failed | $0.07 |
| 3 | 1 | 2026-10-18 | Dry run 3 | Runpod Community, 1x RTX 5090 | $0.69/h | 14 | $0.16 | passed | $0.00 |
| 4 | 1 | 2026-10-18 | Main run 1: stopped at the data-count check (our threshold bug) | Runpod Community, 1x RTX 5090 | $0.69/h | 11 | $0.14 | failed | $0.14 |
| 5 | 1 | 2026-10-18 | **Main run: gate passed** (FVD-2048 167.7) | Runpod Secure, 1x RTX 5090 | $1.19/h | 338 | $6.69 | done | $0.00 |
| 6 | 2 | 2026-10-18 | Dry run: three arms passed; MeanFlow ran out of memory | Runpod Community, 1x RTX 5090 | $0.69/h | 14 | $0.16 | partly passed; fixed | $0.00 |
| 7 | 2 | 2026-10-18 | Dry run 2: bad host (CUDA unusable) | Runpod Community, 1x RTX 5090 | $0.69/h | 6 | $0.06 | failed | $0.06 |
| 8 | 2 | 2026-10-18 | Dry run 3: the same broken host again | Runpod Community, 1x RTX 5090 | $0.69/h | 3 | $0.05 | failed | $0.05 |
| 9 | 2 | 2026-10-18 | Dry run 4: all four arms at the real shape | Runpod Secure, 1x RTX 5090 | $1.19/h | 23 | $0.46 | passed | $0.00 |
| 10 | 2 | 2026-10-18 | Main run 1: every arm failed at start (pin_memory error on this host) | Runpod Secure, 1x RTX 5090 | $1.19/h | 17 | $0.34 | failed | $0.34 |
| 11 | 2 | 2026-10-18/19 | **Main run: four arms scored** (no technique beat the control) | Runpod Secure, 1x RTX 5090 | $1.19/h | 309 | $6.13 | done | $0.00 |
| 12 | 3 | 2026-10-19 | Data step attempt 1: machine never reachable | Runpod Secure, 1x RTX 5090 | $1.19/h | 6 | $0.10 | failed | $0.10 |
| 13 | 3 | 2026-10-19 | Data step attempt 2: machine never reachable | Runpod Secure, 1x RTX 5090 | $1.19/h | 6 | $0.10 | failed | $0.10 |
| 14 | 3 | 2026-10-19 | Data step attempt 3: machine could not reach PyPI | Runpod Secure, 1x RTX 5090 | $1.19/h | 6 | $0.12 | failed | $0.12 |
| 15 | 3 | 2026-10-19 | Data step attempt 4: only ~5 windows/s; stopped for an encoder fix | Runpod Secure, 1x RTX 5090 | $1.19/h | 38 | $0.80 | stopped | $0.80 |
| 16 | 3 | 2026-10-19 | Data step attempt 5: machine never reachable | Runpod Secure, 1x RTX 5090 | $1.19/h | 10 | $0.20 | failed | $0.20 |
| 17 | 3 | 2026-10-19 | **Data step: 600 VidGen zips encoded**, plus the held-out set | Runpod Secure, 1x RTX 5090 | $1.19/h | 371 | $7.40 | done | $0.00 |
| 18 | 3 | 2026-10-19 | Dry run attempts 1-2: machines never reachable (two machines) | Runpod Secure, 1x RTX 5090 | $1.19/h | 22 | $0.40 | failed | $0.40 |
| 19 | 3 | 2026-10-19 | Dry run: train, resume, every score, samples, uploads | Runpod Secure, 1x RTX 5090 | $1.19/h | 27 | $0.55 | passed | $0.00 |
| 20 | 3 | 2026-10-19 | Main run attempt 1: machine never reachable | Runpod Secure, 1x RTX 5090 | $1.19/h | 10 | $0.20 | failed | $0.20 |
| 21 | 3 | 2026-10-19/20 | Main run attempt 2: machine never reachable | Runpod Secure, 1x RTX 5090 | $1.19/h | 20 | $0.40 | failed | $0.40 |
| 22 | 3 | 2026-10-20 | Main run attempt 3: container never started | Runpod Secure, 1x RTX 5090 | $1.19/h | 20 | $0.40 | failed | $0.40 |
| 23 | 3 | 2026-10-20 | Speed probe on the L40S | Shadeform (Massed Compute), 1x L40S 48 GB | $0.88/h | 22 | $0.28 | done | $0.00 |
| 24 | 3 | 2026-10-20 | **Main run: the release model** (900 training minutes; gate passed) | Shadeform (Massed Compute), 1x L40S 48 GB | $0.88/h | 940 | $13.76 | done | $0.00 |
| 25 | 4 | 2026-10-20 | Attempt 1: SSH too slow, the job never started | Shadeform (Massed Compute), 1x L40S 48 GB | $0.88/h | 20 | $0.31 | failed | $0.31 |
| 26 | 4 | 2026-10-20 | Attempt 2: provider error, machine never active | Shadeform (Massed Compute), 1x L40S 48 GB | $0.88/h | 23 | $0.00 | failed | $0.00 |
| 27 | 4 | 2026-10-20 | Attempt 3: job killed when the start-up script exited | Shadeform (Massed Compute), 1x L40S 48 GB | $0.88/h | 27 | $0.29 | failed | $0.29 |
| 28 | 4 | 2026-10-21 | Attempt 4: smoke test failed (MKL/libgomp clash; missing dreamsim) | Shadeform (Massed Compute), 1x L40S 48 GB | $0.88/h | 20 | $0.22 | failed | $0.22 |
| 29 | 4 | 2026-10-21 | VBench scoring (try 1) | Runpod Community, 1x RTX A6000 | $0.33/h | 46 | $0.25 | environment build failed | $0.25 |
| 30 | 4 | 2026-10-21 | VBench scoring (try 2) | Shadeform (Massed Compute), 1x RTX A6000 | $0.57/h | 13 | $0.08 | environment build failed | $0.08 |
| 31 | 4 | 2026-10-21 | VBench video generation (try 1) | Shadeform (Massed Compute), 1x RTX A6000 | $0.57/h | 13 | $0.08 | done | $0.01 |
| 32 | 4 | 2026-10-21 | VBench scoring (try 3) | Shadeform (Massed Compute), 1x RTX A6000 | $0.57/h | 10 | $0.07 | job failed | $0.07 |
| 33 | 4 | 2026-10-21 | VBench video generation preflight (try 1) | Shadeform (Massed Compute), 1x RTX A6000 | $0.57/h | 12 | $0.08 | done | $0.01 |
| 34 | 4 | 2026-10-21 | VBench scoring preflight (try 1) | Shadeform (Massed Compute), 1x RTX A6000 | $0.57/h | 6 | $0.02 | job failed | $0.02 |
| 35 | 4 | 2026-10-21 | VBench video generation preflight (try 2) | Shadeform (Massed Compute), 1x RTX A6000 | $0.57/h | 13 | $0.08 | done | $0.01 |
| 36 | 4 | 2026-10-21 | VBench scoring preflight (try 2) | Runpod Secure, 1x L4 | $0.59/h | 1 | $0.01 | environment build failed | $0.01 |
| 37 | 4 | 2026-10-21 | VBench scoring preflight (try 3) | Runpod Secure, 1x RTX 3090 | $0.50/h | 2 | $0.01 | environment build failed | $0.01 |
| 38 | 4 | 2026-10-21 | VBench scoring preflight (try 4) | Runpod Secure, 1x A40 | $0.59/h | 1 | $0.01 | environment build failed | $0.01 |
| 39 | 4 | 2026-10-21 | VBench scoring preflight (try 5) | Runpod Secure, 1x L4 | $0.59/h | 3 | $0.03 | environment build failed | $0.03 |
| 40 | 4 | 2026-10-21 | VBench scoring (try 4) | Runpod Secure, 1x A40 | $0.59/h | 7 | $0.07 | job failed | $0.07 |
| 41 | 4 | 2026-10-21 | VBench scoring preflight (try 6) | Runpod Secure, 1x RTX 3090 | $0.50/h | 5 | $0.04 | never started | $0.04 |
| 42 | 4 | 2026-10-21 | VBench scoring preflight (try 7) | Runpod Secure, 1x L4 | $0.59/h | 11 | $0.11 | job failed | $0.11 |
| 43 | 4 | 2026-10-21 | VBench scoring (try 5) | Runpod Secure, 1x RTX 3090 | $0.50/h | 9 | $0.08 | done | $0.01 |
| 44 | 4 | 2026-10-21 | VBench scoring preflight (try 8) | Runpod Secure, 1x RTX 3090 | $0.50/h | 10 | $0.08 | job failed | $0.08 |
| 45 | 4 | 2026-10-21 | VBench scoring preflight (try 9) | Runpod Secure, 1x RTX 3090 | $0.50/h | 18 | $0.15 | done | $0.00 |
| 46 | 4 | 2026-10-21 | VBench scoring preflight (try 10) | Runpod Secure, 1x A40 | $0.59/h | 17 | $0.17 | done | $0.01 |
| 47 | 4 | 2026-10-21 | **VBench video generation: every video** (try 2) | Shadeform (Massed Compute), 1x RTX A6000 | $0.57/h | 240 | $2.25 | done | $0.01 |
| 48 | 4 | 2026-10-21 | **VBench scoring: every dimension** (try 6) | Runpod Secure, 1x A40 | $0.59/h | 406 | $3.99 | done | $0.00 |
| 49 | 5a | 2026-10-22 | Photo prep preflight (try 1) | Shadeform (Massed Compute), 1x RTX A6000 | $0.57/h | 10 | $0.06 | done | $0.01 |
| 50 | 5a | 2026-10-22 | Three-arm training preflight (try 1) | Runpod Secure, 1x L40S | $1.09/h | 16 | $0.29 | done | $0.02 |
| 51 | 5a | 2026-10-22 | Arm scoring preflight (try 1) | Runpod Secure, 1x RTX A6000 | $0.59/h | 5 | $0.05 | never started | $0.05 |
| 52 | 5a | 2026-10-22 | Arm scoring preflight (try 2) | Runpod Secure, 1x A40 | $0.59/h | 9 | $0.09 | done | $0.00 |
| 53 | 5a | 2026-10-22 | Photo prep (CC12M via BLIP3o) (try 1) | Shadeform (Massed Compute), 1x RTX A6000 | $0.57/h | 181 | $1.68 | done | $0.00 |
| 54 | 5a | 2026-10-22 | **Three arms, 18,000 steps each** (try 1) | Runpod Secure, 1x L40S | $1.09/h | 366 | $6.65 | done | $0.00 |
| 55 | 5a | 2026-10-22 | Arm scoring (VBench) (try 1) | Runpod Secure, 1x A40 | $0.59/h | 110 | $1.08 | done | $0.01 |

Stage sums are taken from the rows. Each stage's `RESULTS.md` tells the story behind its failed rows.
