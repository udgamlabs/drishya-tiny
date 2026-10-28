# Stage 5a: can data fixes improve prompt-following? (2026-10-22/23). A controlled finding, not the release model.

All numbers are measured by us unless marked. **None of these arms is the release model.**

- **Arms:** each trained 18,000 steps from the Stage 3 final model with the same schedule and evaluation; they differ
  only in data. Code: `training/train_s5.py`, decision rule in `eval/stage5a/mini_summary.py`.
  - **A0 control:** VidGen only, captions cut at 77 tokens (as Stage 3).
  - **A1 caption fix:** captions up to 128 tokens, plus 20% short captions (the first sentence).
  - **A2 caption fix + photos:** A1 plus 30% image steps (5,405 of 18,000) on CC12M photos with long captions from
    BLIP3o (`training/encode_images.py`): 887,004 photos, after 20,934 were dropped for watermarks.
- **Metric ("mini-semantic"):** the mean of six VBench semantic dimensions (object class, multiple objects, human action,
  colour, spatial relationship, scene) on their 516 prompts x 2 seeds, with a paired bootstrap 95% CI against A0.
  Also held-out FVD-512 and CLIP gap, a CLIP gap on 200 real user prompts from VidProM (`prompts/`), and I2V FVD-256.
- **Cost:** $9.90 (rows in `../COST.md`).

Files here: `summary.json` (the decision script's output), `final_A*.json` (per-arm held-out, VidProM, I2V and
validation results), `imgprep_summary.json` (the photo prep counts).

## Results

| Arm | Mini-semantic | vs A0 (paired bootstrap, 95% CI) | Held-out FVD-512 | Held-out CLIP gap | VidProM CLIP gap | I2V FVD-256 |
|---|---|---|---|---|---|---|
| A0 control | 21.42 | - | 414.4 | 20.35 | 8.67 | 457.7 |
| **A1 caption fix** | 23.71 | **+3.01 [+0.93, +5.26]** | 413.3 | 20.53 | 8.85 | 441.1 |
| A2 caption fix + photos | **25.36** | **+4.34 [+1.99, +6.72]** | **518.7** | **21.97** | **10.47** | 481.4 |

**Per dimension (%):**

| | object class | multiple objects | human action | colour | spatial | scene |
|---|---|---|---|---|---|---|
| A0 | 23.18 | 1.18 | 15.50 | 75.80 | 6.41 | 6.47 |
| A1 | 26.66 | 2.52 | 21.00 | 78.68 | 4.65 | 8.76 |
| A2 | 32.67 | 2.25 | 21.00 | 77.38 | 6.25 | 12.61 |

## The decision rule (written before the run)

- Winner = the highest mini-semantic score among arms that meet all three: at least +3 against A0 with the CI above 0;
  a VidProM CLIP gap not below A0's; held-out FVD-512 no more than A0's + 20.
- **A2 fails** the FVD guard (+104 against A0; Stage 4 measured the noise at about +-20), despite the larger semantic gain.
- **A1 passes, but only just:** +3.01 against a +3.00 threshold.

## Reading

- **Photos help prompt-following, but much less than we expected** (our estimate before the run: +5 to +15). A2 is the
  caption fix plus photos: +4.34 against the control, of which the photos' own share (A2 − A1, mini-semantic) is +1.65.
  Against the control the gain is in object class (+9.5) and scene (+6.1); on top of the caption fix, photos add +6.0
  and +3.9 there, concepts our video data rarely shows. (+3.01 and +4.34 are paired, per-prompt statistics; the plain
  score gaps are +2.29 and +3.94.) Multiple objects and
  spatial relationship did not move: composition is not learned from single-subject photos at this scale and compute.
- **Photos cost held-out FVD** (+104) and some I2V FVD (+24). The samples do not look worse to us, and objects look
  cleaner (an impression from 16 prompts, not measured). One possible cause is a style shift from VidGen clips toward
  stock-photo stills, which FVD on VidGen penalises (an untested guess). By the rule written before the run, it is a
  failure.
- **The caption fix alone is worth about +3** (CI +0.9 to +5.3), mostly human action (+5.5) and object class (+3.5), at
  no quality cost (FVD 413 vs 414).
- **Against the released Stage 3 model:** the Stage 4 VBench run, unpaired, scored 24.06 on these six dimensions with 5
  different seeds; A0 scores 21.42 and A1 23.71. So continued training on VidGen alone may have lowered prompt-following a
  little while improving FVD (414 vs 432). The arms' gains are relative to A0, the fair control, not gains over the
  released model. (Different seeds and seed counts; not a paired comparison.)
- **Scale of the gap:** the best arm (A2, 25.4) is still far below what the 1-1.7B peers reach on these dimensions.
  Data changes at 212M parameters and about 10% of Stage 3's compute move prompt-following by 3-4 points, not 30.
