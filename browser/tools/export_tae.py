"""Export the TAEW2.2 decoder to ONNX.

Input : z (1,48,5,16,16) f32 -- Drishya's normalised latents (TAEHV decodes exactly what the diffusion model uses)
Output: frames (17,256,256,3) f32 in [0,255]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch


class Dec(torch.nn.Module):
    """Same maths as TAEHV.decode_video(parallel=True), with the pixel shuffle done on 4-D tensors for ONNX."""
    def __init__(self, tae, apply_fn):
        super().__init__(); self.tae = tae; self.apply_fn = apply_fn

    def forward(self, z):
        y = self.apply_fn(self.tae.decoder, z.permute(0, 2, 1, 3, 4), False)  # N,T',12,128,128
        y = torch.nn.functional.pixel_shuffle(y[0], self.tae.patch_size).clamp(0, 1)  # T',3,256,256
        y = y[self.tae.frames_to_trim:]
        return (y * 255.0).permute(0, 2, 3, 1)


if __name__ == "__main__":
    out = sys.argv[1]
    mod = load_taehv_module(); tae = load_tae(mod)
    d = Dec(tae, mod.apply_model_with_memblocks_parallel).eval()
    z = torch.randn(1, 48, 5, 16, 16)
    with torch.no_grad():
        print("shape", d(z).shape)
    torch.onnx.export(d, (z,), out, input_names=["z"], output_names=["frames"], opset_version=17, dynamo=False)
    import onnxruntime as ort, numpy as np
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    for k in range(2):
        z = torch.randn(1, 48, 5, 16, 16)
        with torch.no_grad(): ref = d(z).numpy()
        o = s.run(None, {"z": z.numpy()})[0]
        ref2 = tae.decode_video(z.permute(0, 2, 1, 3, 4), parallel=True, show_progress_bar=False)[0].permute(0, 2, 3, 1).numpy() * 255
        print("max|d| onnx-vs-wrapper", np.abs(o - ref).max(), "wrapper-vs-decode_video", np.abs(ref - ref2).max(), "shape", o.shape)
