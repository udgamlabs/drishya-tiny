"""Summarise VBench results with the official normalisation (VBench scripts/constant.py at the pinned commit).

  python eval/vbench_summary.py --repo <VBench repo at fd18b3d055cb> --t2v <dir> [--t2v_long <dir>] [--i2v <dir>] --out summary.json

Per-dimension raw scores always; quality / semantic / total only for groups whose every dimension was scored.
- Text-to-video (as scripts/cal_final_score.py): min-max normalise each dimension, weight, average within the group;
  total = (4 x quality + 1 x semantic) / 5.
- Image-to-video (as scripts/cal_i2v_final_score.py): the same with the I2V dimensions, weights and ranges;
  total = (quality + i2v) / 2.
"""
import argparse, glob, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--t2v", required=True); ap.add_argument("--t2v_long", default="")
ap.add_argument("--i2v", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, "scripts"))
import constant as C  # noqa: E402


def raw(d, space=True):
    r = {}
    for f in glob.glob(os.path.join(d, "*_eval_results.json")):
        for k, v in json.load(open(f)).items():
            r[k.replace("_", " ") if space else k] = float(v[0])
    return r


def t2v_scores(r):
    norm = {k: (r[k] - C.NORMALIZE_DIC[k]["Min"]) / (C.NORMALIZE_DIC[k]["Max"] - C.NORMALIZE_DIC[k]["Min"]) * C.DIM_WEIGHT[k]
            for k in r if k in C.NORMALIZE_DIC}
    out = {"raw": r, "missing": [k for k in C.TASK_INFO if k not in r]}
    for name, lst in (("quality", C.QUALITY_LIST), ("semantic", C.SEMANTIC_LIST)):
        if all(k in norm for k in lst):
            out[name] = sum(norm[k] for k in lst) / sum(C.DIM_WEIGHT[k] for k in lst)
    if "quality" in out and "semantic" in out:
        out["total"] = (out["quality"] * C.QUALITY_WEIGHT + out["semantic"] * C.SEMANTIC_WEIGHT) / (C.QUALITY_WEIGHT + C.SEMANTIC_WEIGHT)
    return out


def i2v_scores(r_us):
    r = {C.I2VKEY[k]: v for k, v in r_us.items() if k in C.I2VKEY}
    norm = {k: (r[k] - C.NORMALIZE_DIC_I2V[k]["Min"]) / (C.NORMALIZE_DIC_I2V[k]["Max"] - C.NORMALIZE_DIC_I2V[k]["Min"]) * C.DIM_WEIGHT_I2V[k]
            for k in r if k in C.NORMALIZE_DIC_I2V}
    out = {"raw": {k.replace("_", " "): v for k, v in r_us.items()}, "missing": [k for k in C.TASK_INFO_I2V if k not in r]}
    for name, lst in (("quality", C.I2V_QUALITY_LIST), ("i2v", C.I2V_LIST)):
        if all(k in norm for k in lst):
            out[name] = sum(norm[k] for k in lst) / sum(C.DIM_WEIGHT_I2V[k] for k in lst)
    if "quality" in out and "i2v" in out:
        out["total"] = (out["quality"] * C.I2V_QUALITY_WEIGHT + out["i2v"] * C.I2V_WEIGHT) / (C.I2V_QUALITY_WEIGHT + C.I2V_WEIGHT)
    return out


S = {"vbench_commit": "fd18b3d055cb", "t2v_original": t2v_scores(raw(a.t2v))}
if a.t2v_long and os.path.isdir(a.t2v_long):
    S["t2v_long_prompts"] = t2v_scores(raw(a.t2v_long))
if a.i2v and os.path.isdir(a.i2v):
    S["i2v"] = i2v_scores(raw(a.i2v, space=False))
json.dump(S, open(a.out, "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "raw"} if isinstance(v, dict) else v for k, v in S.items()}, indent=1))
