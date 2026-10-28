# VidProM prompts (CC BY-NC 4.0)

`vidprom_screened.json` holds real user prompts from **VidProM** (Wenhao Wang and Yi Yang, "VidProM: A Million-scale Real
Prompt-Gallery Dataset for Text-to-Video Diffusion Models", NeurIPS 2024; https://huggingface.co/datasets/WenhaoWang/VidProM).

**Licence:** these prompts are licensed by their authors under **CC BY-NC 4.0** (full text: `LICENSE` in this folder;
https://creativecommons.org/licenses/by-nc/4.0/). They are **not** covered by this repository's Apache-2.0 licence.
Non-commercial use only; credit VidProM.

**What we used them for:** evaluation only, never training. In Stage 5a, each arm generated one video per prompt and we
measured a CLIP gap (the CLIP score with the prompt minus with a shuffled prompt). See `results/stage5a/`.

**How they were chosen:** `eval/stage5a/make_prompts.py`, seed 0, from `VidProM_semantic_unique.csv`: English, 4-40 words,
every Detoxify score below 0.05, generator flags such as `-motion 2` removed, and none equal to a VBench prompt.

**Changes we made (personal-information screen):** the evaluation used 200 prompts. Before publishing we removed **22**:
prompts that name real people (actors, singers, directors, artists, a historical figure), first names or initials that
could belong to a private person, on-screen text that may be a personal name, and one prompt containing a local file path
with a user name. 178 remain. No prompt was edited. To rebuild the full 200, run `make_prompts.py` on VidProM.

The prompts were written by members of the public and are reproduced as written, including spelling mistakes. Some
describe frightening or odd scenes; none was screened for taste beyond VidProM's toxicity scores and the filter above.
