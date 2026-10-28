"""Stage 2 verdict (pre-registered rules) from the per-arm result.json files.

  python stage2_summary.py --runs /workspace/out --out /workspace/out/summary.json

Rules (written before any Stage 2 run):
  - eval noise sigma = |FVD(A0, sampling seed 0) - FVD(A0, seed 1)| (FVD-2048), floored at 2.0;
    margin = max(2 * sigma, 5% of A0's FVD).
  - quality arms (joint, tread): WIN if FVD <= A0 - margin AND 0.5 <= motion ratio <= 2.0; LOSE if FVD >= A0 + margin;
    otherwise TIE (no measurable difference at equal compute).
  - meanflow: FEW-STEP WIN if its FVD-2048 at its best of 1/2/4 steps <= 1.15 x A0's 50-step FVD AND motion in band
    (i.e. >= 12x fewer generation steps for <= 15% quality loss). Context: A0 at 4 Euler steps (FVD-512).
  - The Stage 3 recipe = A0 + the winners (their combination is checked in Stage 3's own dry run).
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {}
    for arm in ("base", "joint", "tread", "meanflow"):
        p = os.path.join(a.runs, arm, "result.json")
        if os.path.exists(p):
            R[arm] = json.load(open(p))
    if "base" not in R:
        json.dump({"error": "no A0 result"}, open(a.out, "w")); print("no A0 result"); return
    prim = lambda r: next(s for s in r["scores"] if s.get("primary"))
    b = prim(R["base"]); b1 = next((s for s in R["base"]["scores"] if s["tag"] == "final_euler50_seed1"), None)
    sigma = max(2.0, abs(b["fvd"] - b1["fvd"])) if b1 else None
    margin = max(2 * sigma, 0.05 * b["fvd"]) if sigma else 0.05 * b["fvd"]
    e4 = next((s["fvd"] for s in R["base"]["scores"] if s["tag"] == "euler4_512"), None)
    out = {"A0_fvd": b["fvd"], "A0_motion": b["motion_ratio"], "A0_fvd_seed1": b1["fvd"] if b1 else None, "sigma": sigma,
           "margin": margin, "A0_euler4_fvd512": e4, "arms": {}}
    for arm, r in R.items():
        if arm == "base":
            continue
        p = prim(r); band = 0.5 <= p["motion_ratio"] <= 2.0
        if arm == "meanflow":
            verdict = "FEW-STEP WIN" if (p["fvd"] <= 1.15 * b["fvd"] and band) else "no few-step win"
        else:
            verdict = "WIN" if (p["fvd"] <= b["fvd"] - margin and band) else "LOSE" if p["fvd"] >= b["fvd"] + margin else "TIE"
        out["arms"][arm] = {"fvd": p["fvd"], "motion": p["motion_ratio"], "steps": p.get("mf_steps") or p.get("euler_steps"),
                            "clips_seen": r["clips_seen"], "val_ema": r["val_ema"], "verdict": verdict}
    out["winners"] = [k for k, v in out["arms"].items() if v["verdict"] in ("WIN", "FEW-STEP WIN")]
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
