# Drishya Tiny in the browser

`index.html` and `drishya-web.js` run Drishya Tiny inside a web page with
[onnxruntime-web](https://onnxruntime.ai/) 1.30.0 (WebGPU, with a WebAssembly fallback). The prompt is tokenised with
[transformers.js](https://github.com/huggingface/transformers.js) 4.3.0 and never leaves the device. The model files
download once and are kept in the browser's Cache Storage.

| File (`model/`) | Size | What it is |
|---|---|---|
| `flan_t5_base_encoder.onnx` | 219 MB | Flan-T5-Base encoder (google/flan-t5-base, Apache-2.0; from the Xenova/flan-t5-base ONNX export) |
| `drishya_tiny_dit_cond.onnx` | 426 MB | The 212M Drishya Tiny generator, classifier-free guidance folded in (batch 2: prompt and empty prompt), with a `cond` input: 0 text-to-video, 1 image-to-video |
| `taew2_2_decoder.onnx` | 20 MB | TAEW2.2, a tiny decoder for Wan2.2 latents ([madebyollin/taehv](https://github.com/madebyollin/taehv), MIT) |
| `taew2_2_encoder.onnx` | 3 MB | TAEW2.2's encoder, for the photo in image-to-video; fetched only when a photo is added |

`drishya_tiny_dit.onnx` (the first export, no `cond` input) is no longer used: with cond 0 the new file gives
bit-identical output.

All three store their weights in fp16 and compute in fp32 (full fp16 compute gave NaNs on WebGPU).
The sampler is the one in `drishya/pipeline.py`: rectified-flow Euler steps from t=1 to 0 with shift 5 and CFG 5;
the page defaults to 25 steps. Image-to-video copies `encode_image` and `sample_latents(first=...)`: the photo is
centre-cropped to a square and resized to 256×256 with Pillow's bicubic filter (re-implemented in JS, byte-identical on
the test photos), encoded to the first-frame latent, written into latent frame 0 before every model call and at the
end, and the generator gets cond = 1. The browser uses TAEW2.2's 3 MB encoder instead of the Wan2.2 VAE encoder. The browser uses TAEW2.2 instead of the 555M Wan2.2 VAE decoder, so clips are softer.

```bash
python3 browser/serve.py 8811   # serves the folder with the headers multi-threaded WebAssembly needs
# then open http://localhost:8811  (uses ./model/ on localhost; ?base=<url> points elsewhere)
```

Page options: `?base=` model folder URL, `?prompt=`, `?steps=`, `?seed=`, `?backend=wasm` (force the CPU path),
`?auto=1` (load and generate at once, for tests), with `&photo=<same-origin image URL>` for image-to-video tests.

Noise is seeded: mulberry32(seed) uniforms turned into Gaussians by Box-Muller. `tools/onnx_pipeline.py` does the same
in Python, so a seed gives the same clip in Python and in the page.

`tools/`: `export_dit.py` (`cond` argument for the image-to-video file), `export_tae.py`, `export_tae_enc.py` (ONNX
exports), `fp16_weights.py` (fp16 storage), `parity.py`, `parity_i2v.py`, `check_decoder.py`, `check_encoder.py`,
`check_dit_onnx.py` (checks against PyTorch), `chrome_check.py` (headless Chrome runs).
