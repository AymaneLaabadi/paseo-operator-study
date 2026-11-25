"""Step 5: does H_psi get worse under noise shifts? Bias vs noise floor, and Jacobian alignment.

Target chain, for an image latent z = E(x) (VAE mean):
    F(z, sigma, eps) = E(up(A(D(z)) + noise(sigma, eps)))

Bias and noise floor from K draws F_1..F_K with mean Fbar (per image, sums over all n elements):
    V_hat  = 1/(K-1) sum_k ||F_k - Fbar||^2       unbiased variance of the target
    B2_hat = ||H - Fbar||^2 - V_hat / K           unbiased squared bias (Fbar is itself noisy)
Monte-Carlo check: B2_hat is also computed from each half of the draws separately.
Reported per element (divided by n) so they compare with r.

Jacobian: reference J_F v = mean over K_J draws of [F(z + eps v) - F(z - eps v)] / (2 eps),
with the same noise draw on both sides: the derivative of the noise-averaged target.

Experiments (same images, directions and standard-normal draws at every sigma):
  A  Gaussian noise, H given the true sigma
  B  the same Gaussian targets, H given 0.5 sigma or 2 sigma
  C  Poisson-Gaussian noise with the same average variance, H given sigma (values only:
     a Poisson draw is not differentiable in z, so no fixed-noise finite difference)
Output: results/step5_operator_noise.csv, one row per image x experiment x sigma.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.func import jvp

from paseo_implementation.degradation import (add_poisson_gaussian_noise, encode_measurement,
                                              load_downsampler)
from paseo_implementation.ffhq import load_ffhq
from paseo_implementation.latent_models import decode, encode, load_vae, sigma_embeddings
from paseo_implementation.operator_diagnostics import cosine
from paseo_implementation.silo_operator import load_operator

SIGMAS = [0.02, 0.04, 0.08, 0.16, 0.32]   # code units; training range is [0, 0.04]
GIVEN_FACTORS = [0.5, 1.0, 2.0]          # sigma given to H_psi = factor * true sigma
NUM_IMAGES = 26
K = 16                                    # noisy targets per image and sigma
K_J = 4                                   # noise draws averaged in the Jacobian reference
NUM_DIRECTIONS = 4
EPS = 1e-3                                # finite-difference step (stable in step 4)
CHUNK = 4                                 # images per VAE call
RESULTS = Path(__file__).resolve().parents[2] / "results" / "step5_operator_noise.csv"


def encode_all(vae, y):
    return torch.cat([encode_measurement(vae, y[i:i + CHUNK]) for i in range(0, len(y), CHUNK)])


def bias_and_floor(h, F):
    """Per-element squared bias (all draws, first half, second half), variance, total error."""
    n = h.numel()

    def estimate(Fs):
        Fbar = Fs.mean(0, keepdim=True)
        V = (Fs - Fbar).pow(2).sum() / (len(Fs) - 1)
        return ((h - Fbar).pow(2).sum() - V / len(Fs)).item() / n, V.item() / n

    bias2, var = estimate(F)
    half = len(F) // 2
    total = (h - F).pow(2).sum().item() / (n * len(F))     # should be about bias2 + var
    return dict(bias2=bias2, bias2_half1=estimate(F[:half])[0], bias2_half2=estimate(F[half:])[0],
                var=var, total=total)


def jacobian_agreement(H, z, directions, J_ref, J_ref_single):
    """cos / magnitude error vs the K_J-averaged reference, and cos vs a single-draw reference
    (averaging removes the noise-dependent part of the derivative, which raises the cosine)."""
    cos, cos_single, mag = [], [], []
    for v, ref, ref1 in zip(directions, J_ref, J_ref_single):
        Jv = jvp(H, (z,), (v,))[1]
        cos.append(cosine(Jv, ref))
        cos_single.append(cosine(Jv, ref1))
        mag.append(((Jv.norm() - ref.norm()).abs() / ref.norm()).item())
    n = len(directions)
    return dict(cos=sum(cos) / n, cos_single=sum(cos_single) / n, mag=sum(mag) / n)


def main():
    torch.backends.cudnn.allow_tf32 = False      # true fp32, needed for finite differences
    torch.backends.cuda.matmul.allow_tf32 = False
    vae, H_net, A = load_vae(), load_operator(), load_downsampler()
    given = sorted({round(f * s, 4) for s in SIGMAS for f in GIVEN_FACTORS})
    emb = sigma_embeddings(given)
    x_all = load_ffhq(NUM_IMAGES)
    rows = []

    for i in range(NUM_IMAGES):
        gen = torch.Generator(device="cuda").manual_seed(i)
        with torch.no_grad():
            z = encode(vae, x_all[i:i + 1])
            y = A(decode(vae, z))                                         # A(D(z))
            directions = [torch.randn(z.shape, generator=gen, device="cuda")
                          for _ in range(NUM_DIRECTIONS)]
            y_plus = [A(decode(vae, z + s * EPS * v)) for v in directions for s in (1, 3)]
            y_minus = [A(decode(vae, z - s * EPS * v)) for v in directions for s in (1, 3)]
        noise = torch.randn((K, *y.shape[1:]), generator=gen, device="cuda")  # reused at every sigma

        for sigma in SIGMAS:
            with torch.no_grad():
                F = encode_all(vae, y + sigma * noise)
                J_ref, J_ref_single, stability = [], [], []
                for j in range(NUM_DIRECTIONS):
                    n_j = sigma * noise[:K_J]
                    d = (encode_all(vae, y_plus[2 * j] + n_j)
                         - encode_all(vae, y_minus[2 * j] + n_j)) / (2 * EPS)
                    d3 = (encode_all(vae, y_plus[2 * j + 1] + n_j[:1])
                          - encode_all(vae, y_minus[2 * j + 1] + n_j[:1])) / (6 * EPS)
                    J_ref.append(d.mean(0, keepdim=True))
                    J_ref_single.append(d[:1])
                    stability.append(cosine(d[:1], d3))                   # eps vs 3 eps, same draw
                gen_p = torch.Generator(device="cuda").manual_seed(1000 + i)
                F_p = encode_all(vae, add_poisson_gaussian_noise(y.expand(K, -1, -1, -1), sigma, gen_p))

            for factor in GIVEN_FACTORS:
                sigma_given = round(factor * sigma, 4)

                def H(z_):
                    return H_net(z_, emb[sigma_given])

                with torch.no_grad():
                    h = H(z)
                base = dict(image=i, sigma=sigma, sigma_given=sigma_given)
                rows.append(dict(base, experiment="A" if factor == 1 else "B", noise="gaussian",
                                 fd_stability=sum(stability) / len(stability),
                                 **bias_and_floor(h, F),
                                 **jacobian_agreement(H, z, directions, J_ref, J_ref_single)))
                if factor == 1:
                    rows.append(dict(base, experiment="C", noise="poisson-gaussian", **bias_and_floor(h, F_p)))
        print(f"image {i + 1}/{NUM_IMAGES} done")

    RESULTS.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS, index=False)
    summarize(df)


def summarize(df):
    """Per experiment and sigma: RMS bias, RMS floor, Monte-Carlo error, Jacobian agreement."""
    print(f"\n{'exp':3s} {'sigma':>5s} {'given':>6s} {'bias':>6s} {'floor':>6s} {'MC err':>7s} "
          f"{'check':>6s} {'cos':>12s} {'cos1':>6s} {'mag':>6s} {'FDstab':>6s}")
    for (exp, sigma, given), g in df.groupby(["experiment", "sigma", "sigma_given"]):
        n = len(g)
        bias = g.bias2.mean() ** 0.5
        floor = g["var"].mean() ** 0.5
        mc_err = (g.bias2_half1 - g.bias2_half2).abs().mean() / 2 / g.bias2.mean()   # relative, on B^2
        check = g.total.mean() / (g.bias2.mean() + g["var"].mean())               # should be ~1
        cos = f"{g.cos.mean():.3f}±{g.cos.std() / n**0.5:.3f}" if "cos" in g and g.cos.notna().all() else "-"
        cos1 = f"{g.cos_single.mean():.3f}" if "cos_single" in g and g.cos_single.notna().all() else "-"
        mag = f"{g.mag.mean():.3f}" if "mag" in g and g.mag.notna().all() else "-"
        stab = f"{g.fd_stability.mean():.3f}" if g.fd_stability.notna().all() else "-"
        print(f"{exp:3s} {sigma:5.2f} {given:6.3f} {bias:6.3f} {floor:6.3f} {100 * mc_err:6.1f}% "
              f"{check:6.3f} {cos:>12s} {cos1:>6s} {mag:>6s} {stab:>6s}")


if __name__ == "__main__":
    main()
