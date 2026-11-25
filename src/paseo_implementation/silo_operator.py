"""Load SILO's pretrained latent operator H_psi (FFHQ, super-resolution x8).

SILO (Raphaeli, Man, Elad, ICCV 2025) has no licence, so its code is not copied
into this repo: it stays in external/SILO (git-ignored, cloned from
https://github.com/ronraphaeli/SILO). We read only the DegradationModel class
out of silo_utils.py at runtime, without importing the rest of that file
(which needs old diffusers versions).
"""

import ast
from pathlib import Path

import einops
import torch
import torch.nn as nn

SILO_DIR = Path(__file__).resolve().parents[2] / "external" / "SILO"
CHECKPOINT = SILO_DIR / "trained_ckpt" / "CNN" / "sr_8_cnn_ffhq_operator.pt"


def load_silo_class(name):
    """Execute only the class `name` from SILO's silo_utils.py and return it."""
    source = (SILO_DIR / "silo_utils.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {"torch": torch, "nn": nn, "einops": einops}
    exec(ast.get_source_segment(source, node), namespace)
    return namespace[name]


def load_operator(device="cuda"):
    """H_psi(z, sigma_emb): 4x64x64 latent -> 4x64x64 latent of the degraded image."""
    DegradationModel = load_silo_class("DegradationModel")
    model = DegradationModel(4, layer_norm=True, sigma_condition=True)  # as in silo_sample.py
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)  # strict: every layer name must match
    return model.eval().requires_grad_(False).to(device)


if __name__ == "__main__":
    H = load_operator()
    z = torch.randn(1, 4, 64, 64, device="cuda")
    sigma_emb = torch.randn(1, 1280, device="cuda")   # placeholder; the real one comes in step 3b
    out = H(z, sigma_emb)
    print("output shape:", tuple(out.shape))
    print("parameters:", sum(p.numel() for p in H.parameters()))
