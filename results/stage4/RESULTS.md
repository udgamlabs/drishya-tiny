# Stage 4: sampling sweep, VBench and the FVD harness audit (2026-10-20 to 2026-10-22)

All numbers are measured by us. VBench numbers are **our own sampling and scoring with the official VBench code, not a
certified leaderboard entry.** Model: the Stage 3 final EMA weights, 256x256, 17 frames at 8 fps.

Files here:
- `fvd_audit_result.json`: our FVD code against StyleGAN-V's official FVD code (`eval/fvd_audit.py`).
- `vbench_summary.json`: per-dimension raw scores and the official totals (`eval/vbench_summary.py`).
- `vbench/{t2v,i2v,t2v_long}/`: VBench's own per-dimension output (`*_eval_results.json` with the score of every video,
  `*_full_info.json` with the prompt-to-video lists). Video paths are relative; `:` in the file names became `-`.

## FVD harness audit: pass

Our `training/fvd.py` and StyleGAN-V's official FVD code give the same FVD on identical frames: 21.5059 vs 21.5059 and
279.374 vs 279.374, a relative difference of about 1e-10.

## Sampling sweep (512 held-out VidGen clips; rule written before the run)

| CFG | shift | steps | FVD-512 | CLIP gap | motion |
|---|---|---|---|---|---|
| 5 | 3 | 50 (Stage 3 setting) | 426.9 | 20.47 | 0.74 |
| 3 | 3 | 50 | 373.3 | 18.55 | 0.74 |
| 3 | 5 | 50 | 376.0 | 18.61 | 0.73 |
| **5** | **5** | **50 (chosen)** | 432.0 | 20.50 | 0.74 |
| 7 | 3 | 50 | 517.1 | 21.13 | 0.76 |
| 7 | 5 | 50 | 519.1 | 21.22 | 0.76 |
| 5 | 5 | 100 | 437.3 | 20.47 | 0.76 |

Real-clip CLIP gap 25.38. The rule: the highest CLIP gap among settings within the baseline's FVD + 20 (the noise is
about +-20). It picked CFG 5, shift 5, 50 steps. CFG 3 is better by FVD (373), with slightly weaker prompt-following.

## VBench (commit fd18b3d055cb, official normalisation)

| Suite | Total | Quality | Semantic / I2V |
|---|---|---|---|
| **Text-to-video, standard prompts** (946 prompts x 5 videos; temporal flickering on 25 videos per prompt after the static filter) | **66.92** | **75.59** | semantic **32.26** |
| Text-to-video, VBench long prompts (GPT-enhanced) | not computed (flickering not run, by design) | - | semantic 33.03 |
| **Image-to-video, 1:1 crops** (1,118 x 5) | **82.62** | 75.00 | I2V **90.24** |

**Text-to-video per dimension (%):**
- Quality: subject consistency 87.54, background consistency 94.52, temporal flickering 98.78, motion smoothness 96.87,
  dynamic degree 80.56, aesthetic 34.77, imaging 51.49.
- Semantic: object class 24.97, multiple objects 2.91, human action 20.00, colour 84.39, spatial relationship 6.03,
  scene 6.03, appearance style 20.40, temporal style 12.15, overall consistency 14.53.

**Image-to-video per dimension (%):** i2v subject 93.64, i2v background 95.92, camera motion 24.80, subject consistency
85.77, background consistency 93.63, motion smoothness 96.49, dynamic degree 69.35, aesthetic 50.87, imaging 63.43.

## Reading

- **Video quality (75.6) is level with the 256px VideoCrafter-0.9 (74.9) and close to ModelScope (78.1)** on the
  official leaderboard, both built on Stable Diffusion. Their numbers are the VBench team's own scoring; ours are not.
- The motion is real, not a still video gaming the score: dynamic degree 80.6 together with high smoothness and flicker
  scores.
- **Prompt-following is the weak spot:** semantic 32.3 against about 65-67 for those peers, with multiple objects,
  scene and spatial relationship all under 10%. Long prompts barely help (33.0).
- Image-to-video (82.62) only matches VideoCrafter-I2V (82.57); DynamiCrafter-256 (85.25) and ConsistI2V (84.07) are
  ahead.

## Caveats

- These are our own runs of the VBench scorers on our own samples, not a certified listing.
- The I2V input images came from the Hugging Face mirror `dc-ai/vbench_i2v`. On 2026-10-25 we checked all 355 1:1 crops
  against the official `crop.zip` named in VBench's `vbench2_beta_i2v/download_data.sh`: all 355 are byte-identical
  (sha256), 0 differ, 0 missing.
- The long-prompt suite has no total: temporal flickering needs 25 samples per prompt and was not run for long prompts,
  by design.

## What went wrong on the way

Stage 4 cost $8.48 (rows in `../COST.md`), of which the generation run was $2.25 and the VBench scoring run $3.99. The
rest went on attempts that failed:
- Four early attempts on one machine (before we moved to our job runner): slow SSH, a provider error, a background job
  killed when the start-up script exited, and a smoke test that failed as designed (an MKL/libgomp threading clash in
  VBench's subprocesses, fixed with `MKL_THREADING_LAYER=GNU`, and the `dreamsim` package that VBench's I2V background
  scorer needs but does not install).
- Building VBench's environment: detectron2 needs CUDA 12.1 or older, so it had to be compiled against torch 2.1.2
  (cu121); several environment builds and preflights failed before one passed.
