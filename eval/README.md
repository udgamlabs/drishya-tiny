# Evaluation

Everything here is the code behind the numbers in `results/`. **VBench is not included:** `vbench_score.py` clones the
official repository (https://github.com/Vchitect/VBench) at commit `fd18b3d055cb` when you run it.

| Script | What it does | Results |
|---|---|---|
| `../training/fvd.py` | FVD with the StyleGAN-V / Latte protocol (I3D torchscript detector, 16 frames, population covariance) | Stages 1-3, 5a |
| `fvd_audit.py` | Checks `fvd.py` against StyleGAN-V's official FVD code on identical frames | `results/stage4/fvd_audit_result.json` |
| `vbench_generate.py` | The sampling sweep (FVD-512 + CLIP gap on 512 held-out clips) and the VBench videos: text-to-video (standard and GPT-enhanced long prompts) and image-to-video | Stage 4 |
| `vbench_score.py` | Fetches VBench at the pinned commit and scores the videos, one dimension at a time | `results/stage4/vbench/` |
| `vbench_summary.py` | Quality, semantic, I2V and total scores with VBench's official normalisation | `results/stage4/vbench_summary.json` |
| `stage5a/make_prompts.py` | Builds the Stage 5a prompt sets (VBench's six weak semantic dimensions; 200 VidProM prompts) | `prompts/` |
| `stage5a/mini_summary.py` | The Stage 5a decision: mini-semantic score, paired bootstrap against the control, the pre-registered rule | `results/stage5a/summary.json` |

The Stage 3 gate (FVD-2048, motion ratio, CLIP gap, held-out loss, image-to-video FVD) runs inside the trainer,
`training/train_t2v.py`, at the end of training.

## Checkpoints

`vbench_generate.py --ckpt` takes a training checkpoint (a dict with `cfg` and `ema`, as `training/train_t2v.py` saves
it). For the released safetensors weights, use `drishya.Drishya.from_pretrained("udgamlabs/Drishya-Tiny")` (its `model` is the same network with the same outputs; see `training/test_scale_shift.py`).

## VBench, step by step (one GPU with 24 GB or more; we used 48 GB cards)

1. **Generate** (training environment, `../requirements.txt`). The generator reads VBench's prompt lists from a clone
   at the pinned commit (step 2's `setup` makes the same clone and checks the scoring environment).

   ```bash
   git clone https://github.com/Vchitect/VBench.git work/VBench && git -C work/VBench checkout fd18b3d055cb
   python eval/vbench_generate.py t2v --ckpt last.pt --vbench work/VBench --cfg 5 --shift 5 --steps 50 --out out/videos_t2v
   python eval/vbench_generate.py t2v --ckpt last.pt --vbench work/VBench --cfg 5 --shift 5 --steps 50 --long --out out/videos_t2v_long
   python eval/vbench_generate.py i2v --ckpt last.pt --vbench work/VBench --cfg 5 --shift 5 --steps 50 \
       --images <VBench-I2V crop/1-1 folder> --out out/videos_i2v
   ```

   946 prompts x 5 videos (25 for the temporal-flickering prompts) for text-to-video; 1,118 entries x 5 for
   image-to-video. Videos are 17 frames at 8 fps, 256x256, H.264 at CRF 10. Each batch has its own seed (base seed
   20261021 + batch index), logged in `seeds.json`.

2. **Score** (a separate environment, `requirements-vbench.txt`: Python 3.10, torch 2.1.2 with CUDA 12.1, transformers
   4.33.2, detectron2 at commit `b0fe2f0a5bb6`, `dreamsim` 0.2.1; plus `wget`, `unzip` and `ffmpeg` on the machine).
   detectron2 needs CUDA 12.1 or older to compile; a CUDA 12.1 "devel" container is the easy way.

   ```bash
   python eval/vbench_score.py setup --work work/      # reuses work/VBench, checks the commit and the environment
   python eval/vbench_score.py score --work work/ --suite t2v      --videos out/videos_t2v      --out results/t2v
   python eval/vbench_score.py score --work work/ --suite i2v      --videos out/videos_i2v      --out results/i2v --i2v_images <crop/1-1>
   python eval/vbench_score.py score --work work/ --suite t2v_long --videos out/videos_t2v_long --out results/t2v_long
   python eval/vbench_summary.py --repo work/VBench --t2v results/t2v --t2v_long results/t2v_long --i2v results/i2v --out summary.json
   ```

   Text-to-video scores every standard dimension, then runs VBench's static filter and scores temporal flickering on the
   filtered videos. The long-prompt suite skips temporal flickering by design, so it has no total.

**The VBench-I2V images** (Pexels and Pixabay photos) are not redistributed. Get the official `crop.zip` with VBench's
`vbench2_beta_i2v/download_data.sh` and use its `1-1` folder (355 images). We used the Hugging Face mirror
`dc-ai/vbench_i2v` (`crop/1-1/`) and checked that all 355 files are byte-identical (sha256) to the official ones.

**Two problems we hit, already handled in `vbench_score.py`:** an MKL/libgomp threading clash made every text-to-video
dimension fail in VBench's subprocesses (fixed with `MKL_THREADING_LAYER=GNU`), and the image-to-video background scorer
imports `dreamsim`, which VBench does not install.

These are our own sampling and scoring with the official VBench code, not a certified leaderboard entry.

## FVD

- The detector is the I3D torchscript file used by StyleGAN-V (`i3d_torchscript.pt`, linked from StyleGAN-V's FVD code).
  We don't redistribute it.
- FVD-2048 (2,048 clips each side) is the protocol; FVD-512 and FVD-256 are biased upward and are only compared with
  numbers of the same size.
- `fvd_audit.py` needs a clone of https://github.com/universome/stylegan-v (not included; it has no licence file).
