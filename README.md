**UDGAM LABS** · Video model · 212M-parameter generator, text-to-video and image-to-video

# Drishya Tiny

Drishya Tiny is a 212M-parameter video generator trained from scratch, using the frozen Wan2.2 VAE and Flan-T5-Base
text encoder. It turns a text prompt, or a text prompt plus a first frame, into a 2-second 256×256 clip.

This repository holds everything behind it: the training code for the release model and for every earlier stage, the data
preparation scripts, the evaluation code, the training logs and results of every stage (failed ones included), and the
cost of every machine we rented. The weights are licensed separately (CC BY-NC 4.0).

[Try it](https://huggingface.co/spaces/udgamlabs/Drishya-Tiny) · [Model card and weights](https://huggingface.co/udgamlabs/Drishya-Tiny) · [udgamlabs.com/models/drishya-tiny](https://udgamlabs.com/models/drishya-tiny) · [How we built it](https://udgamlabs.com/blog/drishya-tiny) · [Technical report](https://udgamlabs.com/research/drishya-tiny) ([PDF](docs/Drishya-Tiny-technical-report.pdf))

Drishya (दृश्य) means "a scene, something seen".

## Get started

**Generate a clip** (CUDA, Apple silicon or CPU; the weights, the Wan2.2 VAE and Flan-T5-Base download on first use):

```bash
pip install git+https://github.com/udgamlabs/drishya-tiny
python -m drishya "waves crashing against dark rocks at sunset, golden light" --out waves.mp4
python -m drishya "a sailboat drifting on the ocean" --image photo.jpg --out boat.mp4   # animate a photo
```

```python
from drishya import Drishya, save_mp4

pipe = Drishya.from_pretrained("udgamlabs/Drishya-Tiny")
frames = pipe("a campfire burning at night, sparks rising", seed=0)  # (17, 256, 256, 3) uint8, 8 fps
save_mp4(frames, "campfire.mp4", "a campfire burning at night, sparks rising")
```

Defaults match our evaluation: 50 Euler steps, guidance 5, shift 5. Every clip `save_mp4` writes carries a small
visible "AI-generated" mark and provenance metadata; pass `mark=False` only for research use, and keep the use
policy in mind. The package is `drishya/` (Apache-2.0).

**Rebuild it:** [reproduce.md](./reproduce.md) has plain one-GPU steps for data preparation, training and evaluation.
Run the CPU tests first:

```bash
pip install -r requirements.txt
python training/test_vmodel.py && python training/test_arms.py && python training/test_scale_shift.py
```

**What is in this repository**

| Path | What it is |
|---|---|
| `training/` | The generator (`vmodel.py`), the trainers for every stage (`train_video.py` Stage 1, `train_arm.py` Stage 2, `train_t2v.py` Stage 3 = the release model, `train_s5.py` Stage 5a), the VAE and video I/O helpers, FVD, the optimiser, and tests |
| `data/` | Data notes (what was used and under which terms) and the clip-ID tools. No clips, images or captions |
| `eval/` | VBench generation and scoring (VBench is fetched from GitHub at commit fd18b3d055cb when you run it), the FVD audit, the Stage 5a decision script |
| `results/` | Per-stage results and training logs, VBench's per-dimension output, and `COST.md` (every machine, every failed attempt) |
| `prompts/` | VidProM user prompts used in Stage 5a's evaluation (CC BY-NC 4.0, screened for personal information) |
| `drishya/` | The inference package (`python -m drishya`), Apache-2.0 |
| `browser/` | The run-in-your-browser version: page, ONNX export and parity tools, and the TAEW2.2 tiny decoder (MIT) it uses |
| `reproduce.md` | One-GPU steps for every stage |

## How it works

### The model

| Part | Parameters | Trained by us? |
|---|---|---|
| **Generator: VideoDiT "s210"** (width 768, depth 16, 12 heads, patch 1×2×2) | **211.92M** | **Yes, from random initialisation** |
| Text encoder: Flan-T5-Base encoder | 109.63M | No, frozen |
| Wan2.2 VAE (from `Wan-AI/Wan2.2-TI2V-5B-Diffusers`) | 704.69M (encoder 149.63M, decoder 555.05M) | No, frozen |

- The whole text-to-video system at inference (generator, T5 encoder, VAE decoder) is about 877M parameters; image-to-video
  adds the VAE encoder, about 1.03B. Only the 212M generator was trained by us.
- **Input and output:** a prompt (cut at 77 Flan-T5 tokens), and optionally a first frame, become 17 frames at 8 fps
  (about 2 seconds) at 256×256. The generator works on Wan2.2 latents of 48 channels × 5 frames × 16 × 16, which become
  320 tokens after patching.
- **Each block:** adaLN-zero self-attention with QK-norm over all 320 tokens, cross-attention to the Flan-T5 tokens, and an
  adaLN MLP. Positions use a fixed 3D sine-cosine embedding (time, height, width).
- **Image-to-video by first-frame replacement:** the clean latent of the given frame replaces the first noisy latent frame,
  a learned embedding marks its tokens, and that frame is kept as given (the STIV / SANA-Video recipe).
- **Sampling:** rectified flow with Euler steps and classifier-free guidance. The VBench numbers below use 50 steps, CFG 5,
  shift 5.

### Training data

- **VidGen-1M** (arXiv 2408.02629), whose clips come from **HD-VILA-100M** (Microsoft). We used zips 0-599 (about 300K
  clips), which gave **475,157 two-second training windows**: a window from the start of each clip and, for long clips, one
  from the middle.
- **Held out:** zips 2040-2047 (3,918 windows), never trained on; the Stage 3 gate scored 2,048 of those clips.
- **Captions:** VidGen-1M's own captions, about 80 words on average; the model reads the first 77 tokens.
- No photos, no synthetic data and no other video set are in the release model.
- **Clips and captions are not redistributed.** See [data/README.md](./data/README.md) for the terms (HD-VILA-100M is for
  research use only) and the clip-ID list.

### The training run (Stage 3, 20 October 2026)

| What | Value |
|---|---|
| Hardware | 1× NVIDIA L40S 48 GB (Shadeform, Massed Compute) |
| Training time | 900 minutes (15 h) |
| Steps | 180,080 at batch 64 |
| Clips seen | 11.53M (about 24 passes over the 475,157 windows) |
| Objective | Rectified flow (flow matching); 25% of samples train image-to-video by first-frame replacement; 10% caption dropout for classifier-free guidance |
| Optimiser | MuonAdamW (AdamW learning rate 2e-4, Muon 0.02), 1,000 warm-up steps, learning rate decay over the final 25%, EMA 0.9999 |

The release weights are the run's final EMA weights (step 180,080). The recipe is Stage 1's, unchanged by Stage 2.

## Results

### The pre-registered gate (Stage 3, 20 October 2026): pass

Held-out VidGen clips (zips 2040-2047). EMA weights, 50 Euler steps, shift 3, CFG 5. All numbers measured by Udgam Labs.

| Check | Result | Reference |
|---|---|---|
| Text-to-video FVD-2048 | **235.7** | real clips frozen on frame 0: 660.4 |
| Motion ratio (against real clips) | **0.73** | rule: 0.5-2.0 |
| CLIP gap (own caption minus a shuffled caption; OpenCLIP ViT-B/32 `laion2b_s34b_b79k`) | **20.15** | real clips: 25.73, so 78% of real |
| Held-out flow loss, end vs 25% of the run | 0.454 vs 0.471 | must fall |
| Image-to-video FVD-512 | **240.3** (motion 0.89, CLIP gap 23.7) | |

Our FVD code (`training/fvd.py`) matches StyleGAN-V's official FVD code to a relative 1e-10 on identical frames
(21.5059 vs 21.5059; 279.374 vs 279.374). FVD-512 is biased upward against FVD-2048; the two are never compared.

### VBench (21-22 October 2026)

**These are our own sampling and scoring with the official VBench code, not a certified leaderboard entry.** VBench at
commit fd18b3d055cb; sampling with CFG 5, shift 5, 50 steps, picked by a sampling sweep with a rule fixed in
advance (CFG 3 gives better FVD, 373 against 432 FVD-512, with slightly weaker prompt-following).

| Suite | Total | Quality | Semantic / I2V |
|---|---|---|---|
| **Text-to-video, standard prompts** (946 prompts × 5 videos) | **66.92** | **75.59** | semantic **32.26** |
| Text-to-video, VBench long prompts (GPT-enhanced) | not computed (temporal flickering not run, by design) | - | semantic 33.03 |
| **Image-to-video, 1:1 crops** (1,118 × 5) | **82.62** | 75.00 | I2V **90.24** |

**Text-to-video per dimension (%):**
- Quality: subject consistency 87.54, background consistency 94.52, temporal flickering 98.78, motion smoothness 96.87,
  dynamic degree 80.56, aesthetic 34.77, imaging 51.49.
- Semantic: object class 24.97, multiple objects 2.91, human action 20.00, colour 84.39, spatial relationship 6.03,
  scene 6.03, appearance style 20.40, temporal style 12.15, overall consistency 14.53.

**Image-to-video per dimension (%):** i2v subject 93.64, i2v background 95.92, camera motion 24.80, subject consistency
85.77, background consistency 93.63, motion smoothness 96.49, dynamic degree 69.35, aesthetic 50.87, imaging 63.43.

The VBench-I2V input images we used are byte-identical (sha256) to the official suite: 355 of 355, checked 25 October 2026.

**Compared with the 256px models on the official VBench leaderboard** (the VBench team's own sampling and scoring, at
256×256 and 16 frames, about 2 s; checked against the leaderboard on 25 October 2026). Every one of them has Stable
Diffusion in its lineage.

| Model | Parameters | Total | Quality | Semantic |
|---|---|---|---|---|
| **Drishya Tiny** (our own scoring) | 211.92M generator, trained from scratch; frozen Wan2.2 VAE and Flan-T5-Base | 66.92 | 75.59 | 32.26 |
| ModelScope T2V | 1.7B system (UNet about 1.35B), initialised from SD 2.1 | 75.75 | 78.05 | 66.54 |
| VideoCrafter-0.9 | not stated; spatial layers from SD | 73.02 | 74.91 | 65.46 |
| InstructVideo | a reward fine-tune of ModelScope | 76.61 | 81.56 | 56.81 |

| Image-to-video model | I2V total |
|---|---|
| DynamiCrafter-256 | 85.25 |
| ConsistI2V (from SD 2.1-base) | 84.07 |
| **Drishya Tiny** (our own scoring) | 82.62 |
| VideoCrafter-I2V | 82.57 |

VBench video quality is 75.6, level with the 256px VideoCrafter-0.9 (74.9) and close to ModelScope (78.1), both built on
Stable Diffusion, with a generator trained from scratch (on the frozen Wan2.2 VAE and Flan-T5-Base) and about a sixth the
size of ModelScope's 1.35B UNet. Prompt-following (semantic 32.3) is about half theirs (65-67), and the total is lower.

## How we got here: every stage, failures included

The full results and logs of each stage are in [`results/`](./results); every machine and its cost is in
[`results/COST.md`](./results/COST.md).

1. **Day-0 checks (Mac and a Colab probe; $0.03).** The Wan2.2 VAE round trip, an overfit test and the end-to-end
   pipeline on a laptop before renting anything.
2. **Stage 1: method check on SkyTimelapse (18 October; $7.14).** The same 211.9M generator, unconditional, on the standard
   FVD benchmark: FVD-2048 **167.7** (real frames frozen still 215.1; real against real 72.2), motion 0.89, 2.78M
   clip-samples in 120 minutes on one RTX 5090. The recipe learned. The held-out loss bottomed at about half the run: the
   model was limited by data, not compute. For context only (not a matched comparison), Latte-XL reports 59.82 and
   StyleGAN-V 79.52 on this benchmark.
   *Went wrong:* two bad rented machines and one run stopped by our own threshold bug before the main run.
3. **Stage 2: three techniques against the plain recipe, at equal compute (19 October; $7.20).** 60 GPU-minutes each. **No technique beat the plain
   recipe** (noise margin ±25.1 FVD): joint image+video training 182.4 and TREAD token routing 171.2 tied the control
   (160.7); TREAD gave 39% more throughput but no quality gain. **Our MeanFlow implementation diverged**: its few-step
   samples are noise (likely causes in `results/stage2/RESULTS.md`; it is not a verdict on the method).
   *Went wrong:* MeanFlow ran out of memory in the first dry run, the same broken machine was assigned twice, and the
   first main run failed on one host's page-locked memory.
4. **Stage 3: the release model (19-20 October; $24.71).** The data step on one RTX 5090 ($7.40) and the training run on
   one L40S ($13.76): **about $21 for the release model's own path**. All four pre-registered checks passed.
   *Went wrong:* failed data-step attempts (machines that never became reachable, one without access to PyPI, an encoder
   too slow to finish) and main-run starts that never came up. With a dry run and a speed probe, the rest of Stage 3 cost
   $3.55.
5. **Stage 4: evaluation (20-22 October; $8.48).** The FVD harness audit, the sampling sweep, and VBench.
   *Went wrong:* the first four attempts failed (slow SSH, a provider error, a job killed by its start-up script, and a
   threading clash plus a missing package inside VBench); VBench's environment (detectron2 needs CUDA 12.1 or older) took
   several builds.
6. **Stage 5a: can data fixes improve prompt-following? (22-23 October; $9.90).** A controlled test with three arms, each
   trained 18,000 more steps from the Stage 3 model, scored on six VBench semantic dimensions (516 prompts × 2 seeds,
   paired bootstrap 95% CI). **None of these arms is the release model.**
   - A caption fix (128 tokens and 20% short captions): **+3.01** (CI +0.93 to +5.26), held-out FVD-512 413 against 414.
   - The caption fix plus 30% CC12M photo steps: **+4.34** against the control (CI +1.99 to +6.72), but held-out FVD-512 **+104** (518.7 against 414.4), so it
     failed our pre-registered quality guard.
   - What the photos add on top of the caption fix: **+1.65**, in object class (+6.0) and scene (+3.9); multiple objects
     and spatial relationship did not improve. (+3.01 and +4.34 are paired, per-prompt statistics; see results/stage5a.) Photos
     teach *what* things are, not composition, at an FVD cost. At this size, data fixes move prompt-following by 3-4
     points, not 30.

## Cost

**About $57 all-in**, every failed attempt included; **about $21** for the release model's own path (the Stage 3 data step
on one RTX 5090 for about 6.2 hours, and the training run on one L40S: 15 h 40 min of machine time including the
evaluations).

| Stage | GPU cost | What it bought |
|---|---|---|
| Day-0 checks (Mac, a Colab probe, research) | $0.03 | feasibility |
| Stage 1 (SkyTimelapse method check) | $7.14 | the recipe works |
| Stage 2 (three techniques vs the plain recipe) | $7.20 | keep the plain recipe |
| Stage 3 (data + training of the release model) | $24.71 | **the release model** |
| Stage 4 (sampling sweep + VBench) | $8.48 | the evaluation |
| Stage 5a (three controlled arms + scoring) | $9.90 | the ablation |
| **Total** | **$57.46** | |

Not included: the pretraining of the borrowed Wan2.2 VAE and Flan-T5-Base (not ours), people's time, Mac electricity and
storage. Every row, with machine type, minutes and price: [results/COST.md](./results/COST.md).

## Where it falls short

- **Prompt-following is about half the peers'** (VBench semantic 32.3 against about 65-67 for the 1-1.7B 256px peers). It
  almost never draws two named objects (multiple objects 2.9%), rarely gets spatial relations (6.0%) or named scenes
  (6.0%), and gets the object class right about a quarter of the time (25.0%).
- **Faces and hands smear. Text melts.**
- **Short and small:** 2 s, 256×256, 8 fps. Motion is small.
- **Image-to-video drifts:** later frames lose detail and blur over the 2 s; faces and text melt.
- **Visual quality is modest:** aesthetic 34.8%, imaging 51.5%.
- **Data fixes alone won't close the gap at this size:** Stage 5a moved prompt-following by 3-4 points.
- **Not certified:** the VBench numbers are our own sampling and scoring.
- **Non-commercial weights:** the training clips come from HD-VILA-100M, whose terms allow research use only.
- It works best on nature and scenery (water, waves, sky, snow, boats, fire); people degrade. This is our impression from
  picked clips, not a measurement.

## Licences

- **Code** (this repository, except `prompts/`): **Apache-2.0** ([LICENSE](./LICENSE)). Copyright 2026 Udgam Labs LLP.
  `training/optim.py` is vendored from nanochat and modded-nanogpt under MIT; their notices are in [NOTICE](./NOTICE)
  and in the file.
- **Weights:** **CC BY-NC 4.0**, non-commercial, because the training clips come from HD-VILA-100M (research use only).
  Credit "Drishya Tiny by Udgam Labs". See the model card.
- **Use policy:** [USE_POLICY.md](./USE_POLICY.md) asks you not to use Drishya Tiny to deceive, impersonate or harm, and
  to label outputs as AI-generated. It is guidance, not an extra licence term.
- **Prompts:** `prompts/` holds VidProM prompts under **CC BY-NC 4.0** ([prompts/LICENSE](./prompts/LICENSE)), not Apache.
- The code is Apache-2.0, so you can retrain the generator on clips you have commercial rights to.

## Credits

- **Wan2.2 VAE** (Wan Team, Alibaba; `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, Apache-2.0): the frozen video autoencoder.
- **Flan-T5-Base** (Google; Chung et al., 2022; Apache-2.0): the frozen text encoder.
- **DiT** (Peebles and Xie, "Scalable Diffusion Models with Transformers", 2023): the adaLN-zero transformer design we
  re-implemented. No DiT code is included.
- **FVD:** Unterthiner et al. (2018), with the evaluation settings and I3D detector of **StyleGAN-V** (Skorokhodov et al.,
  2022); our audit checks our code against StyleGAN-V's.
- **Image-to-video by first-frame replacement:** STIV and SANA-Video.
- **VBench** and VBench++ (Huang et al.; Apache-2.0): the benchmark, fetched at commit fd18b3d055cb. VBench-I2V images
  come from Pexels and Pixabay.
- **Data:** VidGen-1M (Tan et al., 2024) and HD-VILA-100M (Xue et al., CVPR 2022, Microsoft); SkyTimelapse (Xiong et al.,
  CVPR 2018) via the Latte authors' copy; CC12M (Changpinyo et al., 2021) via BLIP3o for one Stage 5a arm; VidProM (Wang
  and Yang, 2024) for evaluation prompts.
- **Optimiser:** MuonAdamW from nanochat (Andrej Karpathy) and modded-nanogpt (Keller Jordan), both MIT.
- **Methods we tested:** rectified flow (Liu et al., 2022; Lipman et al., 2022), TREAD token routing, MeanFlow (Geng et
  al., 2025), joint image-video training as in Latte; OpenCLIP for the CLIP gap.
- **A cheaper peer:** [MiniDance](https://huggingface.co/JuniverseLi/MiniDance) trains a smaller text-to-video
  generator from scratch on the same frozen Wan2.2 VAE for much less money than ours, on nature scenery only (people,
  animals, vehicles and text removed from its data). It publishes no VBench scores.

## Citation

```bibtex
@misc{udgamlabs2026drishyatiny,
  title  = {Drishya Tiny: a 212M-parameter text- and image-to-video generator on the frozen Wan2.2 VAE and Flan-T5-Base},
  author = {Prasad, Jitesh K},
  year   = {2026},
  howpublished = {Udgam Labs},
  url    = {https://github.com/udgamlabs/drishya-tiny}
}
```

---

Drishya Tiny by Udgam Labs · AI research from first principles

**About Udgam Labs.** Udgam Labs is an independent AI research lab. We work from first principles: before we build, we understand why things work, whether we are training a model, fine-tuning one or applying AI to a new problem. Udgam (उद्गम) is Sanskrit for origin. [udgamlabs.com](https://udgamlabs.com)
