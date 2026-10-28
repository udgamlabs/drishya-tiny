"""Command line: python -m drishya "waves crashing on rocks at sunset" [--image photo.jpg] [--seed 0] [--out clip.mp4]"""
import argparse
import time

from .pipeline import Drishya, save_mp4


def main():
    ap = argparse.ArgumentParser(prog="drishya")
    ap.add_argument("prompt")
    ap.add_argument("--image", default="", help="optional first frame to animate (image-to-video)")
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="udgamlabs/Drishya-Tiny"); ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="drishya.mp4"); ap.add_argument("--no-mark", action="store_true")
    a = ap.parse_args()
    pipe = Drishya.from_pretrained(a.model, a.device)
    image = None
    if a.image:
        from PIL import Image
        image = Image.open(a.image)
    t0 = time.time()
    frames = pipe(a.prompt, image, a.steps, a.cfg, a.shift, a.seed)
    save_mp4(frames, a.out, a.prompt, mark=not a.no_mark, seed=a.seed)
    print(f"{a.out}  ({time.time() - t0:.0f} s on {pipe.device})")


if __name__ == "__main__":
    main()
