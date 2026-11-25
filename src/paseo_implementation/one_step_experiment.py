"""Step 6: one PASEO measurement update on real latents, with the fixed r vs the measured error.

No diffusion model: a calibrated stand-in prior around the true latent replaces it.
    z_true = E(x)
    mu     = z_true + eta * delta          prior mean (truth is a typical draw from N(mu, eta^2 I))
    z0     = mu + eta * w'                 starting point, as in Algorithm 2 line 10
    y_z    = E(up(A(x) + sigma * eps))     the real encoded measurement
Then P relaxed corrections (paper's SR x8 settings) with H_psi given the true sigma, using
    (i)  r = 0.02, the paper's fixed value
    (ii) r = sqrt(B^2 + V) at that sigma, the operator's error measured in step 5.
Baseline (iii): r = 1e4, i.e. the same update with the measurement ignored. With beta = 1 the
anchor is a second prior sample, and relaxing toward it already averages out some prior noise,
so "helps" = lower latent error / higher PSNR than this prior-only update, not than z0.
Same seeds (delta, w', eps, xi) for every sigma and both r.
"""

from pathlib import Path

import pandas as pd
import torch

from paseo_implementation.degradation import encode_measurement, load_downsampler
from paseo_implementation.ffhq import load_ffhq
from paseo_implementation.latent_models import decode, encode, load_vae, sigma_embeddings
from paseo_implementation.paseo_step import paseo_corrections
from paseo_implementation.silo_operator import load_operator

SIGMAS = [0.02, 0.04, 0.08, 0.16, 0.32]
NUM_IMAGES = 26
# Stand-in prior settings (eta = prior std, beta = arc-anchor weight; beta = 1 is a late step).
# The first is the main setting; the others check that conclusions do not hinge on it.
SETTINGS = [(0.3, 1.0), (0.1, 1.0), (0.5, 1.0), (0.3, 0.5)]
P, C, RHO, LAM, R_FIXED = 4, 2, 0.15, 0.005, 0.02     # Table 6, SR x8
R_PRIOR_ONLY = 1e4      # measurement ignored: the baseline any helpful correction must beat
RESULTS = Path(__file__).resolve().parents[2] / "results"


def measured_r():
    """r(sigma) = sqrt(B^2 + V): the operator's RMS error against one observed y_z (step 5)."""
    df = pd.read_csv(RESULTS / "step5_operator_noise.csv")
    a = df[df.experiment == "A"].groupby("sigma")
    return ((a.bias2.mean() + a["var"].mean()) ** 0.5).to_dict()


def relative_error(z, z_true):
    return ((z - z_true).norm() / z_true.norm()).item()


@torch.no_grad()
def psnr(vae, z, x):
    mse = ((decode(vae, z) - x) / 2).pow(2).mean()        # on the [0, 1] scale
    return (-10 * mse.log10()).item()


def main():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    vae, H_net, A = load_vae(), load_operator(), load_downsampler()
    emb = sigma_embeddings(SIGMAS)
    r_measured = measured_r()
    x_all = load_ffhq(NUM_IMAGES)
    rows = []

    for i in range(NUM_IMAGES):
        x = x_all[i:i + 1]
        gen = torch.Generator(device="cuda").manual_seed(i)
        with torch.no_grad():
            z_true = encode(vae, x)
            y = A(x)
        delta, w, eps = (torch.randn(t.shape, generator=gen, device="cuda") for t in (z_true, z_true, y))
        with torch.no_grad():
            y_z = {sigma: encode_measurement(vae, y + sigma * eps) for sigma in SIGMAS}

        for eta, beta in SETTINGS:
            mu = z_true + eta * delta
            z_start = mu + eta * w
            before = dict(err_before=relative_error(z_start, z_true), psnr_before=psnr(vae, z_start, x))

            for sigma in SIGMAS:
                def H(z):
                    return H_net(z, emb[sigma])

                for r_name, r in [("fixed", R_FIXED), ("measured", r_measured[sigma]),
                                  ("prior only", R_PRIOR_ONLY)]:
                    gen_xi = torch.Generator(device="cuda").manual_seed(10_000 + i)   # same xi everywhere
                    z_after = paseo_corrections(H, z_start, mu, eta, y_z[sigma], r, LAM, RHO, P, C, beta, gen_xi)
                    rows.append(dict(image=i, eta=eta, beta=beta, sigma=sigma, r_name=r_name, r=r, **before,
                                     err_after=relative_error(z_after, z_true),
                                     psnr_after=psnr(vae, z_after, x)))
        print(f"image {i + 1}/{NUM_IMAGES} done")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "step6_one_step.csv", index=False)
    summarize(df)


def gains(df):
    """Per image: gain of each r over the prior-only update (positive = the measurement helped)."""
    keys = ["image", "eta", "beta", "sigma"]
    base = df[df.r_name == "prior only"].set_index(keys)
    out = df[df.r_name != "prior only"].set_index(keys + ["r_name"])
    out["psnr_gain"] = out.psnr_after - base.psnr_after.reindex(out.index.droplevel("r_name")).values
    out["err_reduction"] = base.err_after.reindex(out.index.droplevel("r_name")).values - out.err_after
    return out.reset_index()


def summarize(df):
    g = gains(df)
    n = g.image.nunique()
    print(f"\nGain over the prior-only update, mean ± s.e. over {n} images (positive = measurement helped)")
    print(f"{'eta':>4s} {'beta':>4s} {'sigma':>5s} {'r':>9s} {'PSNR gain dB':>15s} {'helped':>7s} {'latent err red.':>16s}")
    for (eta, beta, sigma, r_name), s in g.groupby(["eta", "beta", "sigma", "r_name"]):
        print(f"{eta:4.1f} {beta:4.1f} {sigma:5.2f} {r_name:>9s} {s.psnr_gain.mean():+7.2f} ± {s.psnr_gain.std() / n**0.5:.2f}"
              f" {100 * (s.psnr_gain > 0).mean():6.0f}% {s.err_reduction.mean():+9.3f}")


if __name__ == "__main__":
    main()
