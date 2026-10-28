"""Stage 5a eval prompt sets (built once, before the run; eval only, never used to pick training data).
  python eval/stage5a/make_prompts.py <VidProM_semantic_unique.csv> <VBench repo at fd18b3d055cb> <out dir>
mini_semantic.json: VBench's prompt lists for the six weak semantic dimensions (object class, multiple objects, human action,
  colour, spatial relationship, scene), 2 seeds each in the arms' eval.
vidprom200.json: 200 real user prompts from VidProM (CC-BY-NC 4.0, eval only), seed 0: English, 4-40 words, every
  Detoxify score < 0.05, generator flags (-motion 2, --ar 16:9, -gs ...) stripped, none equal to a VBench prompt.
"""
import csv, glob, html, json, os, random, re, sys

src, vb, out = sys.argv[1:4]
DIMS = ["object_class", "multiple_objects", "human_action", "color", "spatial_relationship", "scene"]
P = os.path.join(vb, "prompts", "prompts_per_dimension")
mini = {d: [l.strip() for l in open(os.path.join(P, d + ".txt")) if l.strip()] for d in DIMS}
json.dump(mini, open(os.path.join(out, "mini_semantic.json"), "w"), indent=1)
allvb = set(l.strip().lower() for f in glob.glob(os.path.join(vb, "prompts", "**", "*.txt"), recursive=True) for l in open(f) if l.strip())

FLAG = re.compile(r"\s-{1,2}(motion|gs|ar|camera|fps|seed|neg|neg_prompt|aspect|style)\b.*$", re.I)
keep = []
r = csv.reader(open(src, newline="")); h = next(r)
ix = {k: h.index(k) for k in h}
for row in r:
    if any(float(row[ix[k]] or 0) >= 0.05 for k in ("toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit")):
        continue
    p = FLAG.sub("", html.unescape(row[ix["prompt"]])).strip()
    p = re.sub(r"\s+", " ", p)
    w = p.split()
    if not (4 <= len(w) <= 40) or sum(c.isascii() for c in p) < 0.98 * len(p) or p.lower() in allvb:
        continue
    if re.search(r"https?://|@\w|#\w|\b(ratio|9:16|16:9|4k|8k|resolution)\b", p, re.I):
        continue
    keep.append(p)
keep = sorted(set(keep)); random.Random(0).shuffle(keep)
sel = keep[:200]
json.dump({"source": "WenhaoWang/VidProM VidProM_semantic_unique.csv (CC-BY-NC 4.0), eval only", "filter": __doc__.split("vidprom200.json:")[1].strip(),
           "pool": len(keep), "prompts": sel}, open(os.path.join(out, "vidprom200.json"), "w"), indent=1)
print({d: len(v) for d, v in mini.items()}, "unique mini prompts", len(set(sum(mini.values(), []))), "| vidprom pool", len(keep))
for p in sel[:12]:
    print(" -", p)
