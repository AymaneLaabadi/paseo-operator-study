"""Step 3b: the RV-v5.1 VAE (encoder E, decoder D) and the sigma input of H_psi.

Matches how SILO trained the operator (training/silo_cnn_train.py):
- images in [-1, 1], 512x512; latents = E(x) * scaling_factor (0.18215), 4x64x64;
- sigma (code units, i.e. on the [-1, 1] image scale) is fed to H_psi as the
  diffusion UNet's timestep embedding at t = sigma / 0.1 * 999.
"""

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel

MODEL_ID = "stablediffusionapi/realistic-vision-v51"


def load_vae(device="cuda"):
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae",
                                        torch_dtype=torch.float32, use_safetensors=True)
    return vae.eval().requires_grad_(False).to(device)


def encode(vae, x):
    """E: image (B, 3, 512, 512) in [-1, 1] -> latent (B, 4, 64, 64).

    Uses the mean of the VAE posterior, so encoding is deterministic.
    (SILO's training used a random sample, latent_dist.sample().)
    """
    return vae.encode(x).latent_dist.mean * vae.config.scaling_factor


def decode(vae, z):
    """D: latent (B, 4, 64, 64) -> image (B, 3, 512, 512) in [-1, 1]."""
    return vae.decode(z / vae.config.scaling_factor).sample.clamp(-1, 1)


@torch.no_grad()
def sigma_embeddings(sigmas, device="cuda"):
    """{sigma: (1, 1280) tensor}, the noise-level input that H_psi expects.

    Only the UNet's small time-embedding layers are used; the UNet is freed after.
    """
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet",
                                                torch_dtype=torch.float32, use_safetensors=True)
    t = torch.tensor([s / 0.1 * 999 for s in sigmas], dtype=torch.float64)
    emb = unet.time_embedding(unet.time_proj(t).float())
    return {s: e[None].to(device) for s, e in zip(sigmas, emb)}
