"""Compare the ONNX DiT (fp32 or fp16) with PyTorch on random-but-realistic inputs."""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, onnx, onnxruntime as ort, torch
from export_dit import Guided
from drishya.model import DrishyaConfig, DrishyaDiT
from safetensors.torch import load_file
import json
path = sys.argv[1]
m = onnx.load(path, load_external_data=False)
print("ops", sorted(collections.Counter(n.op_type for n in m.graph.node).items()))
print("opset", [(o.domain, o.version) for o in m.opset_import])
cfg = DrishyaConfig(**json.load(open(os.path.join(WEIGHTS, "config.json")))["model"])
dit = DrishyaDiT(cfg); dit.load_state_dict(load_file(os.path.join(WEIGHTS, "model.safetensors"))); dit.eval()
g = Guided(dit)
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
torch.manual_seed(1)
for t in [1.0, 0.5, 0.05]:
    x = torch.randn(1, 48, 5, 16, 16); txt = torch.randn(2, 77, 768) * 0.2
    mask = torch.zeros(2, 77, dtype=torch.bool); mask[0, :23] = True; mask[1, :1] = True
    tt = torch.tensor([t]); c = torch.tensor([5.0])
    with torch.no_grad(): ref = g(x, tt, txt, mask, c).numpy()
    out = sess.run(None, {"x": x.numpy(), "t": tt.numpy(), "txt": txt.numpy(), "mask": mask.numpy(), "cfg": c.numpy()})[0]
    print(f"t={t} max|d|={np.abs(out-ref).max():.2e} rel={np.abs(out-ref).max()/np.abs(ref).max():.2e} ref_absmax={np.abs(ref).max():.2f}")
