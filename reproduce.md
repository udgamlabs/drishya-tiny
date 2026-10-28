# Reproduce Drishya Tiny on one GPU

Plain steps for one machine with one CUDA GPU. We ran every stage on a single rented GPU; the commands below are the
settings we used, without our own job runner. Paths under `work/` are examples.

**What you need**
- Python 3.12 and `pip install -r requirements.txt` (the release model was trained with torch 2.13, CUDA 12.9).
- A Hugging Face account for the downloads (VidGen-1M, the Wan2.2 VAE, Flan-T5-Base); the scripts download them.
- The I3D detector for FVD, `i3d_torchscript.pt`, from the link in `training/fvd.py` (StyleGAN-V's copy). The file we
  used has sha256 `bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2`.
- For the release model: a 48 GB GPU (we used one NVIDIA L40S), and disk for the latents: 475,157 windows of
  48 x 5 x 16 x 16 float16 values each, about 58 GB, plus room for one VidGen zip at a time while encoding.

**Data terms.** VidGen-1M's clips come from HD-VILA-100M, which Microsoft provides for research use only. Read
`data/README.md` before you download anything.

## Check the code first (CPU, minutes)

```bash
python training/test_vmodel.py        # shapes, frame replacement, zero-init output, TREAD path
python training/test_arms.py          # Stage 2 arms: attention paths agree, MeanFlow JVP vs finite differences
python training/test_scale_shift.py   # the adaLN helper is bit-identical to the textbook form
```

## Stage 3: the release model

**1. Encode the data with the frozen Wan2.2 VAE.** Each clip gives up to two 2-second windows (17 frames at 8 fps,
256x256, centre crop): one from 0.5 s, one from the middle if the clip is long enough. Held-out set first:

```bash
python training/encode_vidgen.py --zips 2040-2047 --out work/lat_val --work work/zips --windows 1 --frames 17 --fps 8 \
    --batch 8 --workers 32 --group 4 --dtype bf16 --real_frames work/lat_val/val_real.npy
python training/encode_vidgen.py --zips 0-599 --out work/lat_train --work work/zips --windows 2 --frames 17 --fps 8 \
    --batch 8 --workers 32 --group 4 --dtype bf16
```

On one RTX 5090 the training zips took about 6.2 hours. The encoder is resumable: finished zips are marked in
`<out>/done/` and skipped on restart. Use `--workers` up to your CPU count.

**2. Train.** The recipe from Stage 1, unchanged by Stage 2, plus captions:

```bash
python training/train_t2v.py --lat_train work/lat_train --lat_val work/lat_val --real_frames work/lat_val/val_real.npy \
    --i3d i3d_torchscript.pt --size s210 --batch 64 --lr 2e-4 --lr_muon 0.02 --warmup 1000 --decay_frac 0.25 \
    --ema 0.9999 --p_cond 0.25 --cfg_drop 0.1 --cfg 5 --shift 3 --vae_dtype bf16 \
    --minutes 900 --eval_fracs 0.25,0.5,0.75 --mid_n 512 --final_n 2048 --i2v_n 512 --out work/train
```

- `--minutes 900` is a wall-clock training budget (evaluations pause the clock). On our L40S it gave 180,080 steps. On a
  different GPU, use `--steps 180080` instead to match the release run step for step.
- What it does: VideoDiT "s210" (width 768, depth 16, 12 heads, patch 1x2x2; 211.92M parameters) from random
  initialisation; rectified flow; 25% of samples train image-to-video by replacing the first latent frame with the clean
  one; 10% of captions dropped for classifier-free guidance; Flan-T5-Base (frozen) reads the first 77 caption tokens;
  MuonAdamW; EMA 0.9999.
- Checkpoints go to `work/train/last.pt` every 30 minutes (`--resume` continues a run). Add `--hf_repo <your repo>` to
  also upload them.
- At the end it runs the pre-registered gate on the held-out clips (EMA weights, 50 Euler steps, shift 3, CFG 5):
  text-to-video FVD-2048, motion ratio, CLIP gap, held-out loss, image-to-video FVD-512. It writes `final.json`;
  compare with `results/stage3/final.json`. Sampling noise moves FVD by tens of points, so expect close, not identical,
  numbers.

**3. Evaluate with VBench.** See `eval/README.md`: generate the VBench videos (CFG 5, shift 5, 50 steps), score them
with the official VBench code (fetched at commit fd18b3d055cb), and summarise.

**4. Sample.** `python -m drishya "your prompt"` uses the released weights (see the README). To sample your own run, convert its `last.pt` with `python tools/convert.py --ckpt work/train/last.pt --out my-drishya` (EMA weights to `model.safetensors`, model shape to `config.json`), then `Drishya.from_pretrained("my-drishya")`.

## Earlier stages (optional)

**Stage 1, method check on SkyTimelapse** (unconditional; one RTX 5090, 120 training minutes):

```bash
# data: sky_timelapse.zip from https://huggingface.co/datasets/maxin-cn/SkyTimelapse, unzipped to work/sky/sky_timelapse
python training/encode_clips.py --root work/sky/sky_timelapse --split sky_train --frames 17 --stride 32 \
    --out work/sky_lat/lat_train --batch 8 --dtype fp32 --max_minutes 45
python training/encode_clips.py --root work/sky/sky_timelapse --split sky_test --frames 17 --stride 32 \
    --out work/sky_lat/lat_test --max_windows 1024 --batch 8 --dtype fp32
python training/fvd_stats.py --root work/sky/sky_timelapse --i3d i3d_torchscript.pt --out work/sky_lat/realstats.npz
python training/train_video.py --lat_train work/sky_lat/lat_train --lat_val work/sky_lat/lat_test \
    --realstats work/sky_lat/realstats.npz --i3d i3d_torchscript.pt --size s210 --batch 64 --optim muon --lr 2e-4 \
    --lr_muon 0.02 --warmup 1000 --decay_frac 0.25 --ema 0.9999 --p_cond 0.25 --sample_steps 50 --shift 1.0 \
    --extra_shift 3.0 --vae_dtype fp32 --minutes 120 --eval_fracs 0.1,0.25,0.5,0.75 --eval_n 512 --final_n 2048 --i2v_n 512 --out work/stage1
```

`encode_clips.py --bench` first compares fp32 and bf16 VAE encoding; our rule was to use bf16 only if it is at least
1.3x faster and within 0.1 dB PSNR of fp32. On the Stage 1 machine bf16 failed the rule, so Stage 1 used fp32.

**Stage 2, three techniques against the plain recipe at equal compute** (60 training minutes per arm, on Stage 1's latents and real-video set):

```bash
# images for the joint arm: single frames between the video windows
python training/encode_clips.py --root work/sky/sky_timelapse --split sky_train --frames 1 --stride 32 --offset 24 \
    --out work/sky_lat/lat_img --batch 8 --dtype bf16 --max_minutes 12
for ARM in base joint tread meanflow; do
  B=64; [ $ARM = meanflow ] && B=48     # MeanFlow's JVP needs more memory
  python training/train_arm.py --arm $ARM --lat_train work/sky_lat/lat_train --lat_val work/sky_lat/lat_test \
      --lat_img work/sky_lat/lat_img --realstats work/sky_lat/realstats.npz --i3d i3d_torchscript.pt \
      --size s210 --batch $B --minutes 60 --shift 3.0 --vae_dtype bf16 --out work/stage2/$ARM
done
python training/stage2_summary.py --runs work/stage2 --out work/stage2/summary.json
```

**Stage 5a, three controlled arms** (18,000 steps each from the Stage 3 model; not the release model):

```bash
python eval/stage5a/make_prompts.py <VidProM_semantic_unique.csv> work/VBench work/prompts   # mini_semantic.json + vidprom200.json
python training/encode_images.py --out work/images --n_images 1000000 --max_minutes 180          # arm A2 only
COMMON="--lat_train work/lat_train --lat_val work/lat_val --real_frames work/lat_val/val_real.npy --i3d i3d_torchscript.pt
  --init_from work/train/last.pt --steps 18000 --vidprom work/prompts/vidprom200.json --mini work/prompts/mini_semantic.json
  --milestones 9000,18000"
python training/train_s5.py $COMMON --text_len 77  --p_short 0.0 --out work/stage5a/A0
python training/train_s5.py $COMMON --text_len 128 --p_short 0.2 --out work/stage5a/A1
python training/train_s5.py $COMMON --text_len 128 --p_short 0.2 --img_dir work/images --p_img_step 0.3 --img_batch 256 --out work/stage5a/A2
# score each arm's mini_videos/ on the six dimensions, then decide:
python eval/vbench_score.py score --work work/ --suite t2v --videos work/stage5a/A0/mini_videos --out work/stage5a_scores/A0 \
    --dims object_class multiple_objects human_action color spatial_relationship scene     # repeat for A1, A2
python eval/stage5a/mini_summary.py --results work/stage5a_scores --arms A0 A2 A1 --finals work/stage5a --out summary.json
```

The other Stage 5a settings are the script defaults (learning rates 1e-4 / 0.01, warm-up 500, EMA 0.9995, CFG 5,
shift 5). `prompts/vidprom_screened.json` is the published, screened subset of the 200 VidProM prompts; the reported
Stage 5a numbers used all 200.
