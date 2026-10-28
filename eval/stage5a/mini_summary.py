"""Stage 5a decision (pre-registered before the run), from VBench's per-video results and the arms' final.json.
  python mini_summary.py --results R --arms A0 A2 A1 --finals F --out summary.json [--control A0]
R/<arm>/*_eval_results.json: VBench output per dimension ({dim: [aggregate, [{"video_path", "video_results"}, ...]]}).
F/<arm>/final.json: the arm trainer's held-out FVD-512 / CLIP gap and the VidProM CLIP gap.

Mini-semantic score = the mean of the six dimensions' VBench aggregates (x100). Paired bootstrap vs the control: per dimension,
per-prompt means of the per-video results (2 seeds) for both arms on the same prompts; 10,000 resamples of prompts within each
dimension (seed 0); statistic = mean over dimensions of the mean per-prompt difference.
Winner: the arm with the highest mini-semantic score among those with (a) diff >= +3 points and 95% CI low > 0, (b) VidProM CLIP
gap >= the control's, (c) held-out FVD-512 <= the control's + 20. None -> "no winner".
"""
import argparse, glob, json, os, re

import numpy as np

DIMS = ["object_class", "multiple_objects", "human_action", "color", "spatial_relationship", "scene"]


def load(d):
    agg, per = {}, {}
    for f in glob.glob(os.path.join(d, "*_eval_results.json")):
        for dim, v in json.load(open(f)).items():
            if dim not in DIMS:
                continue
            agg[dim] = float(v[0]); pp = {}
            for r in v[1]:
                p = re.sub(r"-\d+\.mp4$", "", os.path.basename(r["video_path"]))
                x = r.get("video_results")
                if isinstance(x, (int, float, bool)):
                    pp.setdefault(p, []).append(float(x))
            per[dim] = {p: float(np.mean(x)) for p, x in pp.items()}
    return agg, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--finals", default=""); ap.add_argument("--control", default="A0"); ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    a = ap.parse_args()
    data = {arm: load(os.path.join(a.results, arm)) for arm in a.arms}
    fin = {}
    for arm in a.arms:
        p = os.path.join(a.finals, arm, "final.json") if a.finals else ""
        fin[arm] = json.load(open(p)) if p and os.path.exists(p) else {}
    out = {"dims": DIMS, "arms": {}, "control": a.control, "rule": __doc__.split("Winner:")[1].strip()}
    for arm, (agg, _) in data.items():
        miss = [d for d in DIMS if d not in agg]
        out["arms"][arm] = {"dims": {d: round(100 * agg[d], 2) for d in DIMS if d in agg}, "missing": miss,
                            "mini_semantic": round(100 * float(np.mean([agg[d] for d in DIMS if d in agg])), 2) if len(miss) < len(DIMS) else None,
                            "fvd512": fin[arm].get("t2v_heldout", {}).get("fvd"), "heldout_clip_gap": fin[arm].get("t2v_heldout", {}).get("clip_gap"),
                            "vidprom_clip_gap": fin[arm].get("vidprom", {}).get("clip_gap"), "i2v_fvd": fin[arm].get("i2v_heldout", {}).get("fvd"),
                            "val_loss": fin[arm].get("val_loss")}
    rng = np.random.default_rng(0)
    c_per = data[a.control][1] if a.control in data else {}
    for arm in a.arms:
        if arm == a.control or a.control not in data:
            continue
        per = data[arm][1]; dims = [d for d in DIMS if d in per and d in c_per]
        diffs = []
        for d in dims:
            ps = sorted(set(per[d]) & set(c_per[d]))
            diffs.append(np.array([per[d][p] - c_per[d][p] for p in ps]))
        if not diffs or any(len(x) == 0 for x in diffs):
            out["arms"][arm]["vs_control"] = None; continue
        point = 100 * float(np.mean([x.mean() for x in diffs]))
        boots = np.zeros(a.n_boot)
        for d in diffs:
            idx = rng.integers(0, len(d), (a.n_boot, len(d)))
            boots += d[idx].mean(1)
        boots = 100 * boots / len(diffs)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        o, c = out["arms"][arm], out["arms"][a.control]
        checks = {"semantic_+3_ci_above_0": bool(point >= 3 and lo > 0),
                  "vidprom_not_below_control": None if o["vidprom_clip_gap"] is None or c["vidprom_clip_gap"] is None else bool(o["vidprom_clip_gap"] >= c["vidprom_clip_gap"]),
                  "fvd_within_control_+20": None if o["fvd512"] is None or c["fvd512"] is None else bool(o["fvd512"] <= c["fvd512"] + 20)}
        o["vs_control"] = {"diff": round(point, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)], "dims_used": dims,
                           "prompts": [len(x) for x in diffs], "checks": checks, "passes": all(v is True for v in checks.values())}
    ok = [arm for arm in a.arms if (out["arms"][arm].get("vs_control") or {}).get("passes")]
    out["winner"] = max(ok, key=lambda x: out["arms"][x]["mini_semantic"]) if ok else None
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"winner": out["winner"], **{k: {"mini": v["mini_semantic"], "vs": v.get("vs_control")} for k, v in out["arms"].items()}}), flush=True)


if __name__ == "__main__":
    main()
