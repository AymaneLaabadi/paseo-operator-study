"""Figure 1 (README header): one PASEO update with the fixed r vs the measured error.

Same images, seeds and settings as step 6 (main stand-in prior eta = 0.3, beta = 1), so every
tile is one of the cases averaged in plot 2. Images: the first two FFHQ samples, not selected.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import InterpolationMode, resize

from paseo_implementation.degradation import encode_measurement, load_downsampler
from paseo_implementation.ffhq import load_ffhq
from paseo_implementation.latent_models import decode, encode, load_vae, sigma_embeddings
from paseo_implementation.one_step_experiment import C, LAM, P, R_FIXED, R_PRIOR_ONLY, RHO, measured_r, psnr
from paseo_implementation.paseo_step import paseo_corrections
from paseo_implementation.plot_one_step import COLOURS
from paseo_implementation.plot_operator_noise import INK, MUTED
from paseo_implementation.silo_operator import load_operator

IMAGES = [0, 1]
SIGMAS_SHOWN = [0.02, 0.32]
ETA, BETA = 0.3, 1.0
OUT = Path(__file__).resolve().parents[2] / "results" / "figure1.png"


def to_numpy(image):
    """(1, 3, H, W) in [-1, 1] -> (H, W, 3) in [0, 1]."""
    return ((image[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def tiles_for_image(i, x, vae, H_net, A, emb, r_measured):
    """List of (title, image, PSNR or None, frame colour or None) for one row."""
    # Same random draws, in the same order, as one_step_experiment.main().
    gen = torch.Generator(device="cuda").manual_seed(i)
    with torch.no_grad():
        z_true = encode(vae, x)
        y = A(x)
    delta, w, eps = (torch.randn(t.shape, generator=gen, device="cuda") for t in (z_true, z_true, y))
    mu = z_true + ETA * delta
    z_start = mu + ETA * w

    def update(sigma, r):
        with torch.no_grad():
            y_z = encode_measurement(vae, y + sigma * eps)
        gen_xi = torch.Generator(device="cuda").manual_seed(10_000 + i)

        def H(z):
            return H_net(z, emb[sigma])

        return paseo_corrections(H, z_start, mu, ETA, y_z, r, LAM, RHO, P, C, BETA, gen_xi)

    def decoded(z):
        with torch.no_grad():
            return to_numpy(decode(vae, z))

    z_prior = update(SIGMAS_SHOWN[0], R_PRIOR_ONLY)          # measurement ignored: same for any sigma
    tiles = [("ground truth $x$", to_numpy(x), None, None),
             ("start $z_0$", decoded(z_start), psnr(vae, z_start, x), None),
             ("prior-only update", decoded(z_prior), psnr(vae, z_prior, x), None)]
    for sigma in SIGMAS_SHOWN:
        y_up = resize(y + sigma * eps, [512, 512], interpolation=InterpolationMode.BICUBIC, antialias=True)
        tiles.append((f"measurement $y$\n$\\sigma$ = {sigma}", to_numpy(y_up), None, None))
        for r_name, r in [("fixed", R_FIXED), ("measured", r_measured[sigma])]:
            z = update(sigma, r)
            title = f"fixed $r$ = 0.02\n$\\sigma$ = {sigma}" if r_name == "fixed" \
                else f"measured $r$ = {r:.2f}\n$\\sigma$ = {sigma}"
            tiles.append((title, decoded(z), psnr(vae, z, x), COLOURS[r_name]))
    return tiles


def main():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    vae, H_net, A = load_vae(), load_operator(), load_downsampler()
    emb = sigma_embeddings(SIGMAS_SHOWN)
    r_measured = measured_r()
    x_all = load_ffhq(max(IMAGES) + 1)
    rows = [tiles_for_image(i, x_all[i:i + 1], vae, H_net, A, emb, r_measured) for i in IMAGES]

    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(15.5, 4.6))
    for r, tiles in enumerate(rows):
        for c, (title, image, value, frame) in enumerate(tiles):
            ax = axes[r, c]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(frame is not None)
                if frame:
                    spine.set_color(frame)
                    spine.set_linewidth(3)
            if r == 0:
                ax.set_title(title, fontsize=9, color=INK)
            if value is not None:
                ax.set_xlabel(f"{value:.1f} dB", fontsize=9, color=INK, labelpad=2)
    fig.suptitle("One PASEO measurement update (SR×8, from-scratch implementation, SILO's learned operator): "
                 "with the paper's fixed $r$ it helps at low noise and hurts at high noise; "
                 "with the operator's measured error it helps at both", fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
