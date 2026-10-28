"""Export the TAEW2.2 encoder for one photo (image-to-video first frame) to ONNX.

Input : image (256,256,3) f32 in [0,255] -- the centre-cropped, bicubic-resized photo
Output: first (1,48,1,16,16) f32 -- the first-frame latent in Drishya's normalised space
Same maths as TAEHV.encode_video(parallel=True) on a 1-frame clip: the frame is repeated to 4 (encode_video's padding),
pixel-unshuffled (on 4-D tensors for ONNX) and run through the encoder.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, torch


class Enc(torch.nn.Module):
    def __init__(self, tae, apply_fn):
        super().__init__(); self.tae = tae; self.apply_fn = apply_fn

    def forward(self, image):
        x = (image / 255.0).permute(2, 0, 1)[None]  # 1,3,H,W
        x = torch.nn.functional.pixel_unshuffle(x, self.tae.patch_size)  # 1,12,128,128
        x = x[:, None].expand(1, self.tae.t_downscale, -1, -1, -1)  # N,T=4,C,H,W
        z = self.apply_fn(self.tae.encoder, x, False)  # 1,1,48,16,16
        return z.permute(0, 2, 1, 3, 4)


if __name__ == "__main__":
    out = sys.argv[1]
    mod = load_taehv_module(); tae = load_tae(mod)
    e = Enc(tae, mod.apply_model_with_memblocks_parallel).eval()
    img = torch.rand(256, 256, 3) * 255
    with torch.no_grad():
        ref = tae.encode_video((img / 255).permute(2, 0, 1)[None, None], parallel=True, show_progress_bar=False).permute(0, 2, 1, 3, 4)
        mine = e(img)
    print("wrapper vs encode_video", float((mine - ref).abs().max()), tuple(mine.shape))
    torch.onnx.export(e, (img,), out, input_names=["image"], output_names=["first"], opset_version=17, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    o = s.run(None, {"image": img.numpy()})[0]
    print("onnx vs torch", float(np.abs(o - mine.numpy()).max()))
