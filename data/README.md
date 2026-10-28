# Data

**No video clips, images or captions are redistributed in this repository.** It holds the scripts that prepare the data,
and (to come) the list of clips the release model was trained on.

## The release model's training data: VidGen-1M (clips from HD-VILA-100M)

- **VidGen-1M** (Tan, Yang, Qin and Li, 2024, arXiv 2408.02629): https://huggingface.co/datasets/Fudan-FUXI/VIDGEN-1M.
  We used zips 0-599 for training and held out zips 2040-2047 (never trained on).
- VidGen-1M's clips are cut from **HD-VILA-100M** (Xue et al., "Advancing High-Resolution Video-Language Representation
  with Large-Scale Video Transcriptions", CVPR 2022, Microsoft): https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m.
  HD-VILA-100M is provided under Microsoft's **Research Use of Data Agreement v1.0**, which allows **research use only**.
  The underlying videos are from YouTube; their creators keep their rights.
- **Why the weights are non-commercial:** the agreement treats models trained on the data as "Results" and bars their use
  in any commercial offering. The code here is Apache-2.0, so you can retrain on clips you have the rights to.
- **Captions:** the model was trained on VidGen-1M's own captions, which VidGen-1M's authors made with the VILA model
  (non-commercial terms). We do not redistribute them; get them from VidGen-1M (`VidGen_1M_video_caption.json`).

### Clip-ID list

`clip_ids/clip_ids_train.csv` (475,157 training windows) and `clip_ids/clip_ids_held_out.csv` (3,918 held-out
windows) list every 2-second window the release model saw or was tested on: VidGen-1M's clip name (`vid`, a YouTube ID
plus a scene index), the window start in seconds, and the VidGen-1M zip number. Row i is the trainer's window i. No
captions are included. They were made with `data/make_clip_ids.py` from the encoded shards:

```bash
python data/make_clip_ids.py --shards <train shards dir> --split train    --out clip_ids_train.csv    --expect 475157
python data/make_clip_ids.py --shards <held-out dir>     --split held_out --out clip_ids_held_out.csv --expect 3918
```

**Terms for these lists:** the clip IDs identify clips from HD-VILA-100M (Microsoft) as distributed by VidGen-1M. The
underlying data is for research use only under the HD-VILA-100M Research Use of Data Agreement v1.0, whose full text
is in `clip_ids/HD-VILA-100M-R-UDA.txt` (source: https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/LICENSE).
Anyone who uses these lists to obtain or use the data is bound by that agreement's terms. Please also credit
HD-VILA-100M (Xue et al., CVPR 2022) and VidGen-1M (Tan et al., 2024).

## Scripts

| Script | What it does | Stage |
|---|---|---|
| `training/encode_vidgen.py` | Downloads VidGen-1M zips one at a time, cuts up to two 2-second windows per clip (17 frames at 8 fps, 256x256, centre crop), encodes them with the frozen Wan2.2 VAE, writes latent shards `lat_*.npy` and `cap_*.json` (clip ID, start, caption) | 3 (release) |
| `data/make_clip_ids.py` | Clip-ID list from the `cap_*.json` shards, without captions | 3 |
| `training/encode_clips.py` | SkyTimelapse frame folders to Wan2.2 latent shards | 1, 2 |
| `training/fvd_stats.py` | Real-video FVD statistics and baselines for SkyTimelapse | 1, 2 |
| `data/fetch_sky_subset.py` | A small SkyTimelapse subset by HTTP range reads, for local tests | tests |
| `training/encode_images.py` | CC12M photos (via BLIP3o-Pretrain-Long-Caption) to 1-frame latents | 5a only |

## Data used in the other stages (not in the release model)

- **SkyTimelapse** (Stages 1-2; Xiong et al., "Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative
  Adversarial Networks", CVPR 2018). We used the Latte authors' copy, https://huggingface.co/datasets/maxin-cn/SkyTimelapse.
  The original authors give no licence and the footage comes from YouTube: research use.
- **CC12M** photos (Stage 5a arm A2 only; Changpinyo et al., CVPR 2021; Google as the data source) with long captions by
  Qwen2.5-VL-7B, from https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption. The photos belong to their
  owners.
- **VBench** prompts and the VBench-I2V image suite (evaluation only): fetched from VBench at run time; see `eval/`.
