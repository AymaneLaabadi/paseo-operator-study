"""Step 3c: the measurement chain H_psi was trained to predict (SILO training/silo_cnn_train.py).

Same order and calls as SILO's training loop:
    y   = A(x)                  A = bicubic downsampling x8 (DPS/SILO Resizer), 512 -> 64
    y_n = y + sigma * eps       eps ~ N(0, I)
    y_z = clamp(E(up(y_n)), -4, 4)      up = torchvision bicubic resize to 512, antialias

sigma is in code units (images in [-1, 1]); sigma = 0.02 is the paper's sigma_y = 0.01.
"""

import importlib.util

import torch
from torchvision.transforms.functional import InterpolationMode, resize

from paseo_implementation.latent_models import encode
from paseo_implementation.silo_operator import SILO_DIR


def load_downsampler(device="cuda"):
    """A: the bicubic x8 Resizer used by DPS and SILO, loaded from external/SILO (not copied)."""
    spec = importlib.util.spec_from_file_location("silo_resizer", SILO_DIR / "util" / "resizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Resizer((1, 3, 512, 512), 1 / 8).to(device)


def add_gaussian_noise(y, sigma, generator=None):
    """y + sigma * eps. Pass a seeded generator to reuse the same eps across sigmas."""
    eps = torch.randn(y.shape, generator=generator, device=y.device)
    return y + sigma * eps


def add_poisson_gaussian_noise(y, sigma, generator=None):
    """Signal-dependent noise with the same average variance as Gaussian noise of std sigma.

    Units: y in code units [-1, 1] -> intensity u = (y + 1) / 2 in [0, 1] (clipped at 0).
        counts ~ Poisson(P u)            P = photons at intensity 1 (per image)
        y_n    = 2 counts / P - 1 + (sigma / sqrt(2)) * eps_read
    E[y_n] = y and the shot-noise variance is 4 u / P (code units). With
    P = 8 mean(u) / sigma^2, shot noise gives sigma^2 / 2 on average and read noise
    the other sigma^2 / 2, so the average total variance is sigma^2.
    """
    u = ((y + 1) / 2).clamp(min=0)
    photons = 8 * u.mean(dim=(1, 2, 3), keepdim=True) / sigma**2
    counts = torch.poisson(photons * u, generator=generator)
    read = torch.randn(y.shape, generator=generator, device=y.device)
    return 2 * counts / photons - 1 + sigma / 2**0.5 * read


def encode_measurement(vae, y):
    """y (B, 3, 64, 64) -> y_z (B, 4, 64, 64): upsample to 512, encode, clamp like SILO."""
    y_up = resize(y, [512, 512], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return encode(vae, y_up).clamp(-4, 4)
