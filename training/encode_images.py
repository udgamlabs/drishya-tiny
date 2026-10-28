"""Stage 5 image data: BLIP3o-Pretrain-Long-Caption, CC12M part only (files webdataset_shard_*.tar)
-> Wan2.2 VAE 1-frame latents + captions, in the same shard format as the VidGen latents.

  python encode_images.py --out DIR --n_images 1000000            (download from HF, shards in numeric order)
  python encode_images.py --out DIR --local_tars a.tar b.tar      (Mac smoke test)

Images: RGB, short side resized to 256 (PIL bicubic), centre crop 256x256. Skipped: a short side under 192 px, an empty caption,
or a caption that mentions a watermark (stock-photo marks are common in CC12M; the captioner names them; counted separately).
Captions: the Qwen2.5-VL-7B long caption (.txt) as is; the trainer derives the short form itself (train_s5.short_caption).
Output: lat_XXXXX.npy (N,48,1,16,16) float16 normalised latents + cap_XXXXX.json [{"caption", "key", "src"}], N = --per_file,
plus encode_summary.json. The run stops early (and keeps what it has) at --max_minutes. The kit(...) calls are progress
hooks for our internal job runner; they do nothing unless KIT_RUN is set.
"""
import argparse, io, json, os, re, subprocess, sys, tarfile, threading, time, queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
E = os.environ
REPO = "BLIP3o/BLIP3o-Pretrain-Long-Caption"


def kit(*x):
    if E.get("KIT_RUN"):
        subprocess.run(["kit", *map(str, x)], check=False)


def prep(b):
    from PIL import Image
    im = Image.open(io.BytesIO(b)); im.draft("RGB", (512, 512)); im = im.convert("RGB")
    w, h = im.size
    if min(w, h) < 192:
        return None
    s = 256 / min(w, h); im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
    w, h = im.size; l, t = (w - 256) // 2, (h - 256) // 2
    return np.asarray(im.crop((l, t, l + 256, t + 256)), dtype=np.uint8)


def samples(path, src):
    """(key, jpg bytes, caption) for every complete pair in a webdataset tar."""
    cur, parts = None, {}
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            k, e = m.name.rsplit(".", 1)
            if k != cur:
                if cur and "jpg" in parts and "txt" in parts:
                    yield cur, parts["jpg"], parts["txt"]
                cur, parts = k, {}
            parts[e.lower().replace("jpeg", "jpg")] = tf.extractfile(m).read()
        if cur and "jpg" in parts and "txt" in parts:
            yield cur, parts["jpg"], parts["txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n_images", type=int, default=1_000_000)
    ap.add_argument("--local_tars", nargs="*", default=[]); ap.add_argument("--dl_dir", default="")
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--per_file", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) - 2)); ap.add_argument("--prefetch", type=int, default=6)
    ap.add_argument("--max_minutes", type=float, default=0); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    from vae import VAE
    vae = VAE(a.device, torch.bfloat16 if a.device == "cuda" else torch.float32)

    if a.local_tars:
        tars = [(p, os.path.basename(p)) for p in a.local_tars]; names = [n for _, n in tars]
    else:
        from huggingface_hub import HfApi, hf_hub_download
        names = sorted([f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if re.fullmatch(r"webdataset_shard_\d+\.tar", f)],
                       key=lambda f: int(re.findall(r"\d+", f)[0]))
        dl = a.dl_dir or os.path.join(a.out, "_dl"); os.makedirs(dl, exist_ok=True)
    q = queue.Queue(maxsize=a.prefetch)

    def producer():  # downloads ahead of the encoder; deletes nothing (pod disk is sized for the ~47 GB of tars)
        for n in names:
            if a.local_tars:
                q.put((dict((m, p) for p, m in tars)[n], n)); continue
            for k in range(4):
                try:
                    q.put((hf_hub_download(REPO, n, repo_type="dataset", local_dir=dl), n)); break
                except Exception as e:  # noqa: BLE001
                    print(json.dumps({"download_retry": n, "err": repr(e)[:160]}), flush=True); time.sleep(10 * (k + 1))
        q.put(None)
    threading.Thread(target=producer, daemon=True).start()

    pool = ThreadPoolExecutor(a.workers)
    lat_buf, cap_buf, fidx, n_done, n_skip, n_wm, n_shards, files = [], [], 0, 0, 0, 0, 0, []
    stop = False

    def flush():
        nonlocal lat_buf, cap_buf, fidx
        if not lat_buf:
            return
        lp, cp = os.path.join(a.out, f"lat_{fidx:05d}.npy"), os.path.join(a.out, f"cap_{fidx:05d}.json")
        np.save(lp, np.concatenate(lat_buf).astype(np.float16)); json.dump(cap_buf, open(cp, "w"))
        assert np.load(lp, mmap_mode="r").shape[0] == len(cap_buf)
        kit("save", lp, "--final", "--as", f"images/{os.path.basename(lp)}"); kit("save", cp, "--final", "--as", f"images/{os.path.basename(cp)}")
        files.append(os.path.basename(lp)); lat_buf, cap_buf, fidx = [], [], fidx + 1

    def encode(batch):
        nonlocal n_done
        x = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2)[:, None]  # (B,1,3,256,256)
        z = vae.encode(x)
        assert z.shape[1:] == (48, 1, 16, 16), z.shape
        lat_buf.append(z.cpu().numpy().astype(np.float16)); cap_buf.extend(b[1] for b in batch); n_done += len(batch)

    while not stop:
        item = q.get()
        if item is None:
            break
        path, name = item; n_shards += 1
        recs = list(samples(path, name))
        ims = list(pool.map(prep, [r[1] for r in recs]))
        batch = []
        for (k, _, cap), im in zip(recs, ims):
            c = cap.decode("utf-8", "replace").strip()
            if im is None or not c:
                n_skip += 1; continue
            if "watermark" in c.lower():
                n_wm += 1; continue
            batch.append((im, {"caption": c, "key": k, "src": name}))
            if len(batch) == a.batch:
                encode(batch); batch = []
                if len(cap_buf) >= a.per_file:
                    flush()
            if n_done + len(batch) >= a.n_images:
                stop = True; break
        if batch:
            encode(batch)
        if len(cap_buf) >= a.per_file:
            flush()
        el = time.time() - t0
        rec = {"shard": name, "shards": n_shards, "images": n_done, "skipped": n_skip, "watermark": n_wm, "per_s": round(n_done / el, 1), "min": round(el / 60, 1)}
        print(json.dumps(rec), flush=True); kit("beat", n_shards); kit("metric", "images", n_done)
        if a.max_minutes and el > a.max_minutes * 60:
            print(json.dumps({"STOP": "max_minutes", "images": n_done}), flush=True); break
    flush()
    summ = {"repo": REPO, "part": "CC12M (webdataset_shard_*)", "images": n_done, "skipped": n_skip, "watermark_skipped": n_wm, "shards_read": n_shards, "files": files,
            "per_s": round(n_done / (time.time() - t0), 1), "minutes": round((time.time() - t0) / 60, 1), "per_file": a.per_file}
    if torch.cuda.is_available():
        summ["peak_mem_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2); kit("metric", "peak_mem_gb", summ["peak_mem_gb"])
    json.dump(summ, open(os.path.join(a.out, "encode_summary.json"), "w"), indent=1); print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
