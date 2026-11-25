"""Step 3c: sanity checks of the measurement chain, then H_psi vs the encoded measurement.

1. Noise is added in code units: std(y_n - y) = sigma.
2. VAE scaling: raw latent std, scaled latent std, decode(encode(x)) PSNR.
3. Determinism: encoding the same measurement twice gives the same y_z.
4. For each sigma, with ONE fixed noise draw per image (same eps for all sigmas):
   relative L2 error ||H_psi(z, sigma) - y_z|| / ||y_z||, vs a do-nothing baseline (z itself).
"""

import torch

from paseo_implementation.degradation import add_gaussian_noise, encode_measurement, load_downsampler
from paseo_implementation.ffhq import load_ffhq
from paseo_implementation.latent_models import decode, encode, load_vae, sigma_embeddings
from paseo_implementation.silo_operator import load_operator

SIGMAS = [0.0, 0.02, 0.03, 0.04, 0.08, 0.16, 0.32]   # code units; training range is [0, 0.04]
NUM_IMAGES = 26                                     # all FFHQ samples shipped with SILO
BATCH = 2


def relative_l2(a, b):
    """Per-image ||a - b|| / ||b||."""
    return (a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1)


def rms(t):
    return t.pow(2).mean().sqrt().item()


@torch.no_grad()
def main():
    vae, H, A = load_vae(), load_operator(), load_downsampler()
    emb = sigma_embeddings(SIGMAS)
    x_all = load_ffhq(NUM_IMAGES)

    # Clean latents z = E(x) and measurements y = A(x), in small batches.
    z_all = torch.cat([encode(vae, x_all[i:i + BATCH]) for i in range(0, NUM_IMAGES, BATCH)])
    y_all = A(x_all)

    # ---- 1. Noise scale -------------------------------------------------
    gen = torch.Generator(device="cuda").manual_seed(0)
    noise = add_gaussian_noise(y_all, 0.02, gen) - y_all
    print(f"1. std of added noise at sigma = 0.02: {noise.std():.4f}  (y range [{y_all.min():.2f}, {y_all.max():.2f}])")

    # ---- 2. VAE scaling --------------------------------------------------
    x4 = x_all[:4]
    raw = vae.encode(x4).latent_dist.mean
    mse = ((decode(vae, encode(vae, x4)) - x4) / 2).pow(2).mean(dim=(1, 2, 3))   # [0,1] scale
    print(f"2. scaling_factor {vae.config.scaling_factor} | std raw latent {raw.std():.3f} | "
          f"std scaled latent {z_all.std():.3f} | round-trip PSNR {(-10 * mse.log10()).mean():.1f} dB")

    # ---- 3. Determinism -----------------------------------------------
    first = encode_measurement(vae, y_all[:BATCH])
    second = encode_measurement(vae, y_all[:BATCH])
    print(f"3. encoding the same measurement twice, max difference: {(first - second).abs().max():.1e}")

    # ---- 4. Operator vs encoded measurement, per sigma ---------------------
    # One fixed standard-normal draw per image, reused at every sigma: only sigma changes.
    gen = torch.Generator(device="cuda").manual_seed(0)
    eps = torch.randn(y_all.shape, generator=gen, device="cuda")

    print("\n4. sigma   in train?  RMS(H - y_z)  relL2(H - y_z)   relL2(z - y_z)")
    for sigma in SIGMAS:
        y_n = y_all + sigma * eps
        y_z = torch.cat([encode_measurement(vae, y_n[i:i + BATCH]) for i in range(0, NUM_IMAGES, BATCH)])
        pred = H(z_all, emb[sigma].expand(NUM_IMAGES, -1))
        rel_h, rel_id = relative_l2(pred, y_z), relative_l2(z_all, y_z)
        print(f"   {sigma:5.2f}   {'yes' if sigma <= 0.04 else 'no ':3s}       {rms(pred - y_z):.3f}"
              f"         {rel_h.mean():.3f} ± {rel_h.std():.3f}    {rel_id.mean():.3f} ± {rel_id.std():.3f}")


if __name__ == "__main__":
    main()
