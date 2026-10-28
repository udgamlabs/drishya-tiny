# TAEHV (third-party)

- Files: `taehv.py`, `taew2_2.pth`, `LICENSE`, `README.md`, downloaded unchanged on 2026-10-28 from
  https://github.com/madebyollin/taehv (commit `011dfc2112197741c540e0bdd5b7b67bcc930771`).
- Licence: MIT, Copyright (c) 2025 Ollin Boer Bohan (see `LICENSE`).
- Use here: `tools/export_tae.py` and `tools/export_tae_enc.py` load `taehv.py` by path and export the TAEW2.2
  decoder to `model/taew2_2_decoder.onnx` and its encoder (one photo, repeated to 4 frames as `encode_video` pads) to
  `model/taew2_2_encoder.onnx` (weights stored fp16). TAEW2.2 decodes Wan2.2 latents as the diffusion model sees them
  (no latents_mean/std step), which matched Drishya's normalised latents: 28.9 and 31.9 dB PSNR against the full
  Wan2.2 VAE decoder on two clips (the raw-latent alternative gave 23.3 and 24.5 dB).
- Encoder for image-to-video (2026-10-28, `tools/check_encoder.py`, 25 steps, seed 0, three PD12M CC0 photos): the
  TAEW2.2 photo latent vs the Wan2.2 VAE encoder's normalised latent: cosine 0.981-0.983, relative RMSE 0.19. Clips from
  the two latents (same noise, both decoded with TAEW2.2) agree at 26.9-31.2 dB PSNR; frame 0 vs the photo is
  24.6/26.8/26.9 dB with the tiny encoder and 24.9/27.3/27.3 dB with the Wan2.2 encoder.
