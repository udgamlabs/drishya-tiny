// Drishya Tiny in the browser: Flan-T5-Base encoder -> 212M video transformer (25 Euler steps, CFG 5) -> TAEW2.2 decoder.
// Everything runs on the device with onnxruntime-web (WebGPU, WASM fallback). Model files are fetched once and kept in
// the browser's Cache Storage. The sampler mirrors drishya/pipeline.py `sample_latents`, for text-to-video and for
// image-to-video (a photo as the first frame: TAEW2.2's encoder gives its latent, loaded only when a photo is added).
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { AutoTokenizer, env as tjsEnv } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";

export const ORT_VERSION = "1.30.0";
export const DEFAULT_BASE = "https://huggingface.co/udgamlabs/Drishya-Tiny/resolve/main/onnx/";
export const TOKENIZER = { repo: "Xenova/flan-t5-base", revision: "c6a1b98a4a35e1063d88edaceef4dc52cf3b2784" };
// Weights are stored as fp16 and cast to fp32 inside each graph: half the download, fp32 maths (full fp16 compute gave
// NaNs on WebGPU). bytes are shown before the download starts; the real Content-Length is used once it arrives.
export const FILES = {
  text: { name: "flan_t5_base_encoder.onnx", bytes: 219487367 },
  // the generator with a `cond` input (0 text-to-video, 1 image-to-video); with cond 0 its output is identical to
  // drishya_tiny_dit.onnx (no cond input), which it replaces
  dit: { name: "drishya_tiny_dit_cond.onnx", bytes: 425712830 },
  dec: { name: "taew2_2_decoder.onnx", bytes: 19903284 },
  enc: { name: "taew2_2_encoder.onnx", bytes: 3039523 },  // image-to-video only, fetched on the first photo
};
export const MARK = "AI-generated · Drishya Tiny";
const CACHE = "drishya-tiny-onnx-v2";
const C = 48, T = 5, S = 16, N = C * T * S * S, L = 77, D = 768, FRAMES = 17, H = 256, W = 256, SS = S * S;

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;

// ---- seeded noise: mulberry32 uniforms -> Box-Muller pairs (tools/onnx_pipeline.py does the same in Python) ----
export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function gaussianNoise(seed, n = N) {
  const r = mulberry32(seed >>> 0), out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = 1 - r(), u2 = r(), rad = Math.sqrt(-2 * Math.log(u1));
    out[i] = rad * Math.cos(2 * Math.PI * u2); out[i + 1] = rad * Math.sin(2 * Math.PI * u2);
  }
  return out;
}
export function timesteps(steps, shift = 5) {
  const ts = [];
  for (let i = 0; i <= steps; i++) { const t = 1 - i / steps; ts.push(shift * t / (1 + (shift - 1) * t)); }
  return ts;
}

// ---- downloads with progress, kept in Cache Storage ----
// Files are stored in 64 MB parts: Chrome refused one 425 MB Cache entry ("Unexpected internal error") but takes parts.
// No big Blobs: headless Chrome could not read back a 425 MB Blob (NotReadableError), so bytes go into one Uint8Array.
const PART = 64 * 1024 * 1024;
async function cacheRead(cache, url) {
  const meta = await cache.match(`${url}?parts`);
  if (!meta) return null;
  const { n, size } = await meta.json(), out = new Uint8Array(size);
  let off = 0;
  for (let i = 0; i < n; i++) {
    const r = await cache.match(`${url}?part=${i}`);
    if (!r) return null;
    const part = new Uint8Array(await r.arrayBuffer());
    if (off + part.byteLength > size) return null;
    out.set(part, off); off += part.byteLength;
  }
  return off === size ? out : null;
}
async function cacheWrite(cache, url, bytes) {
  const n = Math.ceil(bytes.byteLength / PART);
  for (let i = 0; i < n; i++) await cache.put(`${url}?part=${i}`, new Response(bytes.slice(i * PART, (i + 1) * PART)));
  await cache.put(`${url}?parts`, new Response(JSON.stringify({ n, size: bytes.byteLength }), { headers: { "content-type": "application/json" } }));
}

async function fetchCached(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* no Cache API (private mode): download every time */ }
  let hit = null;
  try { hit = cache && (await cacheRead(cache, url)); } catch (e) { console.warn("cache read failed", e); }
  if (hit) { onProgress(hit.byteLength, hit.byteLength, true); return hit; }
  const res = await fetch(url, { mode: "cors" });
  if (!res.ok) throw new Error(`${res.status} for ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  let buf = new Uint8Array(total || PART), got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.byteLength > buf.byteLength) { const nb = new Uint8Array(Math.max(buf.byteLength * 2, got + value.byteLength)); nb.set(buf.subarray(0, got)); buf = nb; }
    buf.set(value, got); got += value.byteLength; onProgress(got, total || got, false);
  }
  const bytes = got === buf.byteLength ? buf : buf.slice(0, got);
  if (cache) { try { await cacheWrite(cache, url, bytes); } catch (e) { console.warn("cache put failed", e); } }
  return bytes;
}

// ---- photo -> 256x256 RGB, as pipeline.py encode_image: centre square crop, then Pillow's BICUBIC resize ----
// Pillow's resampling (Resample.c) re-done: separable bicubic (a = -0.5) with the kernel widened by the scale factor
// (antialiasing), 22-bit fixed-point weights, a horizontal pass to 8-bit, then a vertical pass to 8-bit.
const PB = 22;
function bicubic(x) {
  const a = -0.5; x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}
function coeffs(inSize, outSize) {
  const scale = inSize / outSize, fs = Math.max(scale, 1), support = 2 * fs, ss = 1 / fs, out = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const k = new Float64Array(xmax); let ww = 0;
    for (let x = 0; x < xmax; x++) { const w = bicubic((x + xmin - center + 0.5) * ss); k[x] = w; ww += w; }
    const ki = new Int32Array(xmax);
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      ki[x] = v < 0 ? Math.trunc(-0.5 + v * (1 << PB)) : Math.trunc(0.5 + v * (1 << PB));
    }
    out.push({ xmin, k: ki });
  }
  return out;
}
const clip8 = (v) => (v >= 255 * (1 << PB) + (1 << PB) ? 255 : v <= 0 ? 0 : Math.min(255, Math.floor(v / (1 << PB))));
// rgba: Uint8ClampedArray (w*h*4) -> Uint8Array (ow*oh*3)
function pilResize(rgba, w, h, ow, oh) {
  let src = rgba, sw = w, sc = 4;
  if (ow !== w) {
    const cs = coeffs(w, ow), tmp = new Uint8Array(ow * h * 3);
    for (let y = 0; y < h; y++) for (let xx = 0; xx < ow; xx++) {
      const { xmin, k } = cs[xx];
      for (let ch = 0; ch < 3; ch++) {
        let acc = 1 << (PB - 1);
        for (let x = 0; x < k.length; x++) acc += src[(y * sw + xmin + x) * sc + ch] * k[x];
        tmp[(y * ow + xx) * 3 + ch] = clip8(acc);
      }
    }
    src = tmp; sw = ow; sc = 3;
  }
  const out = new Uint8Array(ow * oh * 3);
  if (oh !== h) {
    const cs = coeffs(h, oh);
    for (let yy = 0; yy < oh; yy++) {
      const { xmin: ymin, k } = cs[yy];
      for (let x = 0; x < ow; x++) for (let ch = 0; ch < 3; ch++) {
        let acc = 1 << (PB - 1);
        for (let y = 0; y < k.length; y++) acc += src[((ymin + y) * sw + x) * sc + ch] * k[y];
        out[(yy * ow + x) * 3 + ch] = clip8(acc);
      }
    }
  } else {
    for (let p = 0; p < ow * oh; p++) for (let ch = 0; ch < 3; ch++) out[p * 3 + ch] = src[p * sc + ch];
  }
  return out;
}

// File/Blob/ImageBitmap -> { rgb: Uint8Array(256*256*3) } prepared exactly like the Python package (no colour-profile
// conversion, like Pillow). Runs locally; the photo is never uploaded.
export async function prepareImage(source) {
  const bmp = source instanceof ImageBitmap ? source
    : await createImageBitmap(source, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const w = bmp.width, h = bmp.height, s = Math.min(w, h), x0 = Math.floor((w - s) / 2), y0 = Math.floor((h - s) / 2);
  const cv = document.createElement("canvas"); cv.width = s; cv.height = s;
  const g = cv.getContext("2d", { willReadFrequently: true });
  g.imageSmoothingEnabled = false; g.drawImage(bmp, x0, y0, s, s, 0, 0, s, s);
  const px = g.getImageData(0, 0, s, s).data;
  // Pillow's convert("RGB") drops alpha without compositing; canvas keeps the colour of opaque pixels unchanged
  return { rgb: pilResize(px, s, s, W, H), size: [w, h] };
}

export class DrishyaWeb {
  constructor(base = DEFAULT_BASE, { forceWasm = false } = {}) {
    this.base = base.endsWith("/") ? base : base + "/"; this.stats = {}; this.forceWasm = forceWasm;
  }

  async plan() {
    this.webgpu = !this.forceWasm && !!navigator.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
    this.files = [FILES.text, FILES.dit, FILES.dec];
    this.totalBytes = this.files.reduce((s, f) => s + f.bytes, 0);
    return { webgpu: this.webgpu, totalBytes: this.totalBytes };
  }

  async load(onProgress) {
    if (!this.files) await this.plan();
    const t0 = performance.now();
    const done = new Array(this.files.length).fill(0), tot = this.files.map((f) => f.bytes);
    let fromCache = true;
    const bufs = await Promise.all(this.files.map((f, i) => fetchCached(this.base + f.name, (g, t, cached) => {
      done[i] = g; tot[i] = t; fromCache = fromCache && cached;
      onProgress?.(done.reduce((a, b) => a + b, 0), tot.reduce((a, b) => a + b, 0));
    })));
    this.stats.downloadMs = performance.now() - t0; this.stats.fromCache = fromCache;
    this.stats.bytes = bufs.reduce((s, b) => s + b.byteLength, 0);
    tjsEnv.allowLocalModels = false;
    this.tok = await AutoTokenizer.from_pretrained(TOKENIZER.repo, { revision: TOKENIZER.revision });
    const gpu = this.webgpu ? ["webgpu", "wasm"] : ["wasm"];
    const opts = (eps) => ({ executionProviders: eps, graphOptimizationLevel: "all" });
    const t1 = performance.now();
    this.text = await ort.InferenceSession.create(bufs[0], opts(["wasm"]));
    this.dit = await ort.InferenceSession.create(bufs[1], opts(gpu));
    this.dec = await ort.InferenceSession.create(bufs[2], opts(gpu));
    this.stats.sessionMs = performance.now() - t1;
    this.stats.backend = this.webgpu ? "WebGPU" : "WebAssembly";
    this.null = null;
    return this.stats;
  }

  // the photo encoder (image-to-video), fetched and cached only when a photo is first used
  loadEncoder(onProgress) {
    if (!this.encP) {
      this.encP = (async () => {
        if (this.webgpu === undefined) await this.plan();
        const t0 = performance.now();
        const buf = await fetchCached(this.base + FILES.enc.name, (g, t) => onProgress?.(g, t));
        this.enc = await ort.InferenceSession.create(buf, {
          executionProviders: this.webgpu ? ["webgpu", "wasm"] : ["wasm"], graphOptimizationLevel: "all" });
        this.stats.encoderBytes = buf.byteLength; this.stats.encoderLoadMs = performance.now() - t0;
        return this.enc;
      })().catch((e) => { this.encP = null; throw e; });
    }
    return this.encP;
  }

  // 256x256 RGB (Uint8Array) -> first-frame latent (48*1*16*16 floats, Drishya's normalised space)
  async encodeImage(rgb) {
    await this.loadEncoder();
    const out = await this.enc.run({ image: new ort.Tensor("float32", Float32Array.from(rgb), [H, W, 3]) });
    const f = out.first.data;
    return f instanceof Float32Array ? f : Float32Array.from(f);
  }

  async encode(text) {
    const e = this.tok(text, { padding: "max_length", max_length: L, truncation: true });
    const ids = BigInt64Array.from(e.input_ids.data, (v) => BigInt(v));
    const am = BigInt64Array.from(e.attention_mask.data, (v) => BigInt(v));
    const out = await this.text.run({
      input_ids: new ort.Tensor("int64", ids, [1, L]), attention_mask: new ort.Tensor("int64", am, [1, L]),
    });
    const h = out.last_hidden_state.data;
    return { h: h instanceof Float32Array ? h : Float32Array.from(h), mask: Uint8Array.from(am, (v) => Number(v)), ids: Array.from(ids, Number) };
  }

  // image: optional 256x256 RGB Uint8Array from prepareImage (image-to-video). returns { frames: Uint8Array RGB, timings }
  async generate(prompt, { steps = 25, cfg = 5, shift = 5, seed = 42, onStep, image = null } = {}) {
    const t0 = performance.now();
    let first = null, imageMs = 0;
    if (image) { const i0 = performance.now(); first = await this.encodeImage(image); imageMs = performance.now() - i0; }
    // x[:, :, :1] = first  (layout 1 x 48 x 5 x 16 x 16: the first 256 values of each channel)
    const pin = (x) => { if (first) for (let c = 0; c < C; c++) x.set(first.subarray(c * SS, (c + 1) * SS), c * T * SS); };
    const condT = new ort.Tensor("float32", Float32Array.of(first ? 1 : 0), [1]);
    const c = await this.encode(prompt);
    if (!this.null) this.null = await this.encode("");
    const txt = new Float32Array(2 * L * D); txt.set(c.h, 0); txt.set(this.null.h, L * D);
    const mask = new Uint8Array(2 * L); mask.set(c.mask, 0); mask.set(this.null.mask, L);
    const txtT = new ort.Tensor("float32", txt, [2, L, D]), maskT = new ort.Tensor("bool", mask, [2, L]);
    const cfgT = new ort.Tensor("float32", Float32Array.of(cfg), [1]);
    const tText = performance.now() - t0 - imageMs;
    const x = gaussianNoise(seed), ts = timesteps(steps, shift), stepMs = [];
    for (let i = 0; i < steps; i++) {
      const s0 = performance.now();
      pin(x);
      const out = await this.dit.run({
        x: new ort.Tensor("float32", x, [1, C, T, S, S]), t: new ort.Tensor("float32", Float32Array.of(ts[i]), [1]),
        txt: txtT, mask: maskT, cfg: cfgT, cond: condT,
      });
      const v = out.v.data, dt = ts[i + 1] - ts[i];
      for (let j = 0; j < N; j++) x[j] = x[j] + dt * v[j];
      stepMs.push(performance.now() - s0);
      onStep?.(i + 1, steps, stepMs[i]);
    }
    pin(x);
    const d0 = performance.now();
    const fr = (await this.dec.run({ z: new ort.Tensor("float32", x, [1, C, T, S, S]) })).frames.data;
    const frames = new Uint8Array(fr.length);
    for (let j = 0; j < fr.length; j++) frames[j] = Math.min(255, Math.max(0, Math.round(fr[j])));
    const decodeMs = performance.now() - d0;
    const timings = { textMs: tText, imageMs, stepMs, decodeMs, totalMs: performance.now() - t0, steps, seed, ids: c.ids,
      mode: first ? "image-to-video" : "text-to-video" };
    return { frames, latents: x, timings };
  }
}

// Burn the visible label into each frame (same placement as add_mark in drishya/pipeline.py); returns ImageData[]
export function markedFrames(rgb, text = MARK) {
  const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
  const g = cv.getContext("2d", { willReadFrequently: true }), out = [];
  for (let f = 0; f < FRAMES; f++) {
    const img = g.createImageData(W, H), o = f * H * W * 3;
    for (let p = 0; p < H * W; p++) {
      img.data[4 * p] = rgb[o + 3 * p]; img.data[4 * p + 1] = rgb[o + 3 * p + 1];
      img.data[4 * p + 2] = rgb[o + 3 * p + 2]; img.data[4 * p + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    g.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif"; g.textBaseline = "alphabetic";
    const m = g.measureText(text), tw = Math.ceil(m.width), th = Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent);
    g.fillStyle = "rgba(0,0,0,0.59)"; g.fillRect(4, H - th - 12, tw + 8, th + 8);
    g.fillStyle = "rgba(255,255,255,0.9)"; g.fillText(text, 8, H - 8 - Math.ceil(m.actualBoundingBoxDescent));
    out.push(g.getImageData(0, 0, W, H));
  }
  return out;
}

// Record the clip once to WebM at 8 fps with MediaRecorder; resolves to a Blob
export function recordWebM(frames, fps = 8) {
  return new Promise((resolve, reject) => {
    const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
    const g = cv.getContext("2d"), stream = cv.captureStream(0), track = stream.getVideoTracks()[0];
    const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
    if (!type) return reject(new Error("This browser cannot record WebM."));
    const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 4e6 }), chunks = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => resolve(new Blob(chunks, { type: "video/webm" }));
    rec.start();
    let i = 0;
    const tick = () => {
      if (i === frames.length) { setTimeout(() => rec.stop(), 1000 / fps); return; }
      g.putImageData(frames[i++], 0, 0); track.requestFrame?.(); setTimeout(tick, 1000 / fps);
    };
    tick();
  });
}
