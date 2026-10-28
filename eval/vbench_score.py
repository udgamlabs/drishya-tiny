"""Score Drishya Tiny videos with the official VBench code, fetched from GitHub at commit fd18b3d055cb at run time.

This is the scoring logic we used for the reported numbers, without our internal job runner: the same VBench commit, the
same per-dimension commands (`python evaluate.py` / `python evaluate_i2v.py` from the VBench source tree, one dimension
per call), the static filter before temporal flickering, and the same dimension lists. VBench is never vendored here.

  # 1. fetch VBench at the pinned commit and check the environment (see eval/README.md for the environment)
  python eval/vbench_score.py setup --work work/

  # 2. score (videos from eval/vbench_generate.py)
  python eval/vbench_score.py score --work work/ --suite t2v      --videos out/videos_t2v      --out results/t2v
  python eval/vbench_score.py score --work work/ --suite i2v      --videos out/videos_i2v      --out results/i2v \
      --i2v_images <folder with the 355 VBench-I2V 1:1 crops>
  python eval/vbench_score.py score --work work/ --suite t2v_long --videos out/videos_t2v_long --out results/t2v_long
  # Stage 5a mini-semantic: --suite t2v --dims object_class multiple_objects human_action color spatial_relationship scene

  # 3. totals with VBench's official normalisation
  python eval/vbench_summary.py --repo work/VBench --t2v results/t2v --t2v_long results/t2v_long --i2v results/i2v --out summary.json

VBench downloads its scorer models (and RAFT, used by dynamic degree and the static filter) into $VBENCH_CACHE_DIR on first
use, with wget. Already-scored dimensions (a *_eval_results.json containing the dimension) are skipped, so a run resumes.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

VBENCH_URL = "https://github.com/Vchitect/VBench.git"
VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"

T2V_QUALITY = ["subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree", "aesthetic_quality",
               "imaging_quality"]
T2V_SEMANTIC = ["object_class", "multiple_objects", "human_action", "color", "spatial_relationship", "scene", "temporal_style",
                "appearance_style", "overall_consistency"]
T2V = T2V_QUALITY + T2V_SEMANTIC  # temporal_flickering is scored separately, after the static filter
I2V = ["i2v_subject", "i2v_background", "camera_motion", "subject_consistency", "background_consistency", "motion_smoothness",
       "dynamic_degree", "aesthetic_quality", "imaging_quality"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def paths(work):
    work = os.path.abspath(work)
    return work, os.path.join(work, "VBench"), os.environ.get("VBENCH_CACHE_DIR") or os.path.join(work, "vbench_cache")


def env_vb(vr, cache):
    env = dict(os.environ)
    env.update(MKL_THREADING_LAYER="GNU",  # conda's MKL clashed with libgomp in VBench's subprocesses (every T2V dimension failed)
               VBENCH_CACHE_DIR=cache, TOKENIZERS_PARALLELISM="false",
               WANDB_MODE="disabled",      # nothing here logs to Weights & Biases
               PYTHONPATH=os.pathsep.join([vr, env.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    return env


def setup(a):
    work, vr, cache = paths(a.work)
    os.makedirs(work, exist_ok=True); os.makedirs(cache, exist_ok=True)
    if not os.path.isdir(os.path.join(vr, ".git")):
        subprocess.run(["git", "clone", "-q", VBENCH_URL, vr], check=True)
    subprocess.run(["git", "-C", vr, "fetch", "-q", "origin", VBENCH_COMMIT], check=False)
    subprocess.run(["git", "-C", vr, "checkout", "-q", VBENCH_COMMIT], check=True)
    head = subprocess.run(["git", "-C", vr, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == VBENCH_COMMIT, head
    for tool in ("wget", "unzip", "ffmpeg"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found: VBench needs it (apt-get install {tool})")
    chk = ("import torch, detectron2; from detectron2 import _C; import cv2, decord, pyiqa, timm, transformers, clip, dreamsim, peft, "
           "fvcore; assert torch.cuda.is_available(); print('VBench environment ok: torch', torch.__version__, 'cuda', "
           "torch.version.cuda, 'transformers', transformers.__version__)")
    subprocess.run([sys.executable, "-c", chk], check=True, env=env_vb(vr, cache))
    log(f"VBench at {head} in {vr}; model cache {cache}")


def has(out, dim):
    for f in glob.glob(os.path.join(out, "*_eval_results.json")):
        if dim in json.load(open(f)):
            return True
    return False


def score_dim(kind, vr, cache, videos, out, dim):
    os.makedirs(out, exist_ok=True)
    if has(out, dim):
        log(f"skip {kind} {dim} (already scored)"); return True
    if kind == "i2v":
        cmd = [sys.executable, "evaluate_i2v.py", "--videos_path", videos, "--dimension", dim, "--ratio", "1-1",
               "--output_path", out + "/"]
    else:
        cmd = [sys.executable, "evaluate.py", "--videos_path", videos, "--dimension", dim, "--output_path", out + "/"]
    t0 = time.time()
    with open(os.path.join(out, f"vbench_{kind}.log"), "a") as lf:
        subprocess.run(cmd, cwd=vr, env=env_vb(vr, cache), stdout=lf, stderr=subprocess.STDOUT)
    ok = has(out, dim)
    log(f"{kind} {dim}: {'scored' if ok else 'NO RESULT (see the log in --out)'} in {time.time() - t0:.0f} s")
    return ok


def static_filter(vr, cache, videos, dst):
    """VBench's static filter (needed before temporal flickering): keeps only videos without large motion."""
    raft = os.path.join(cache, "raft_model", "models", "raft-things.pth")
    if not os.path.exists(raft):
        sys.exit(f"RAFT weights not found at {raft}: score dynamic_degree first (VBench downloads RAFT for it)")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "static_filter.log"), "a") as lf:
        subprocess.run([sys.executable, "static_filter.py", "--videos_path", videos, "--result_path", dst, "--model", raft],
                       cwd=vr, env=env_vb(vr, cache), stdout=lf, stderr=subprocess.STDOUT)
    fv = os.path.join(dst, "filtered_videos")
    return fv if os.path.isdir(fv) else None


def score(a):
    work, vr, cache = paths(a.work)
    assert os.path.isdir(vr), "run `setup` first"
    if a.suite == "i2v":
        crop = os.path.join(vr, "vbench2_beta_i2v", "data", "crop", "1-1")
        if not os.path.isdir(crop) or len(os.listdir(crop)) != 355:
            assert a.i2v_images, "--i2v_images: the folder with VBench-I2V's 355 1:1 crops (see eval/README.md)"
            shutil.rmtree(crop, ignore_errors=True); os.makedirs(os.path.dirname(crop), exist_ok=True)
            shutil.copytree(a.i2v_images, crop)
        assert len(os.listdir(crop)) == 355, len(os.listdir(crop))
        dims = a.dims or I2V
    elif a.suite == "t2v_long":  # GPT-enhanced prompts: semantic first, then quality; no temporal flickering (by design)
        dims = a.dims or (T2V_SEMANTIC + T2V_QUALITY)
    else:
        dims = a.dims or T2V
    videos, out = os.path.abspath(a.videos), os.path.abspath(a.out)
    bad = [d for d in dims if not score_dim("i2v" if a.suite == "i2v" else "t2v", vr, cache, videos, out, d)]
    if a.suite == "t2v" and not a.dims:  # temporal flickering: VBench's protocol scores only the static-filtered videos
        fdir = os.path.join(work, "flicker_filter"); os.makedirs(fdir, exist_ok=True)
        fv = static_filter(vr, cache, videos, fdir)
        if not fv or not score_dim("t2v", vr, cache, fv, out, "temporal_flickering"):
            bad.append("temporal_flickering")
    log("all dimensions scored" if not bad else f"NOT scored: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("--work", required=True)
    s = sub.add_parser("score"); s.add_argument("--work", required=True)
    s.add_argument("--suite", required=True, choices=["t2v", "i2v", "t2v_long"])
    s.add_argument("--videos", required=True); s.add_argument("--out", required=True)
    s.add_argument("--dims", nargs="*", default=None, help="subset of dimensions (default: the whole suite)")
    s.add_argument("--i2v_images", default="")
    a = ap.parse_args()
    setup(a) if a.cmd == "setup" else score(a)
