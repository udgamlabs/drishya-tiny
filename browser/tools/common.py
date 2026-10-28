"""Shared helpers for the browser build: load Drishya (local weights) and TAEW2.2 (imported by path)."""
import importlib.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
BROWSER = os.path.dirname(HERE)
RELEASE = os.path.dirname(BROWSER)
WEIGHTS = os.path.join(RELEASE, "weights", "Drishya-Tiny")
TAEHV_DIR = os.path.join(BROWSER, "third_party", "taehv")
MODEL_OUT = os.path.join(BROWSER, "model")
sys.path.insert(0, os.path.join(RELEASE, "drishya"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_taehv_module():
    spec = importlib.util.spec_from_file_location("taehv_tp", os.path.join(TAEHV_DIR, "taehv.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def load_tae(mod=None):
    mod = mod or load_taehv_module()
    return mod.TAEHV(checkpoint_path=os.path.join(TAEHV_DIR, "taew2_2.pth")).eval().requires_grad_(False)


def psnr(a, b):
    import torch
    a = a.float(); b = b.float()
    mse = ((a - b) ** 2).mean()
    return float(10 * torch.log10(255.0 ** 2 / mse))
