"""Step 4: PASEO Table 5 (App. E.2) for SILO's SR x8 operator, at the paper's noise level.

Reference: the true chain  T(z) = E(up(A(D(z))))     (no noise, "E o A o D" in Table 5)
Learned:   H(z) = H_psi(z, sigma = 0.02)              (sigma_y = 0.01 on the [0, 1] scale)

Per image (z = E(x), VAE mean) and per random direction v:
  value RMS      RMS(H(z) - T(z))                     compare with r and sqrt(r^2 + lam)
  direction cos  cos(J_H v, J_T v)                    J_T v by central finite differences
  magnitude err  | ||J_H v|| - ||J_T v|| | / ||J_T v||
  adjoint err    |<u, J_H v> - <J_H^T u, v>| / |<u, J_H v>|
  ||J_H||_2      power iteration on J_H^T J_H
  FD stability   cos(J_T v at eps, J_T v at 3 eps): is the reference itself reliable?

TF32 is switched off: on RTX 30xx GPUs PyTorch runs convolutions in TF32 by default
(about 1e-3 relative error), which breaks small-step finite differences and the adjoint check.
"""

import torch
from torch.func import jvp, vjp

from paseo_implementation.degradation import encode_measurement, load_downsampler
from paseo_implementation.ffhq import load_ffhq
from paseo_implementation.latent_models import decode, encode, load_vae, sigma_embeddings
from paseo_implementation.silo_operator import load_operator

SIGMA = 0.02                     # code units, the paper's setting
R, LAM = 0.02, 0.005             # Table 6, SR x8
NUM_IMAGES = 26
NUM_DIRECTIONS = 4               # as in Table 5
EPSILONS = [1e-3, 3e-3, 1e-2, 1e-1]   # finite-difference steps; small ones measure the derivative
POWER_ITERS = 50                       # converged after about 50 (checked up to 200)


def rms(t):
    return t.pow(2).mean().sqrt().item()


def cosine(a, b):
    return ((a * b).sum() / (a.norm() * b.norm())).item()


def spectral_norm(J, Jt, shape, generator, iters=POWER_ITERS):
    """Largest singular value of J by power iteration on J^T J."""
    v = torch.randn(shape, generator=generator, device="cuda")
    for _ in range(iters):
        v = Jt(J(v))
        v = v / v.norm()
    return J(v).norm().item()


def mean_std(values):
    t = torch.tensor(values)
    return f"{t.mean():.3f} ± {t.std():.3f}"


def main():
    torch.backends.cudnn.allow_tf32 = False     # true fp32 convolutions
    torch.backends.cuda.matmul.allow_tf32 = False
    vae, H_net, A = load_vae(), load_operator(), load_downsampler()
    emb = sigma_embeddings([SIGMA])[SIGMA]
    x_all = load_ffhq(NUM_IMAGES)
    gen = torch.Generator(device="cuda").manual_seed(0)

    def H(z):                                   # learned operator at the paper's sigma
        return H_net(z, emb)

    @torch.no_grad()
    def T(z):                                   # true chain, no noise
        return encode_measurement(vae, A(decode(vae, z)))

    value, adjoint, norm, fd_stability = [], [], [], []
    cos = {eps: [] for eps in EPSILONS}
    mag = {eps: [] for eps in EPSILONS}
    saturated = []

    for i in range(NUM_IMAGES):
        with torch.no_grad():
            z = encode(vae, x_all[i:i + 1])
            saturated.append((decode(vae, z).abs() >= 1).float().mean().item())
            value.append(rms(H(z) - T(z)))

        _, vjp_fn = vjp(H, z)

        def J(v):
            return jvp(H, (z,), (v,))[1]

        def Jt(u):
            return vjp_fn(u)[0]

        for _ in range(NUM_DIRECTIONS):
            v = torch.randn(z.shape, generator=gen, device="cuda")
            Jv = J(v)
            Jv_ref = {}
            for eps in EPSILONS:
                Jv_ref[eps] = (T(z + eps * v) - T(z - eps * v)) / (2 * eps)
                cos[eps].append(cosine(Jv, Jv_ref[eps]))
                mag[eps].append(((Jv.norm() - Jv_ref[eps].norm()).abs() / Jv_ref[eps].norm()).item())
            fd_stability.append(cosine(Jv_ref[1e-3], Jv_ref[3e-3]))

            u = torch.randn(z.shape, generator=gen, device="cuda")
            a, b = (u * Jv).sum(), (Jt(u) * v).sum()
            adjoint.append(((a - b).abs() / a.abs()).item())

        norm.append(spectral_norm(J, Jt, z.shape, gen))
        print(f"image {i + 1:2d}/{NUM_IMAGES}: value RMS {value[-1]:.3f}, "
              f"cos(eps=1e-3) {torch.tensor(cos[1e-3][-NUM_DIRECTIONS:]).mean():.3f}, ||J|| {norm[-1]:.1f}")

    r_eff = (R**2 + LAM) ** 0.5
    print(f"\nTable 5, RV-v5.1 SR x8, clean latents ({NUM_IMAGES} images x {NUM_DIRECTIONS} directions)")
    print(f"{'metric':28s} {'ours':>16s}   paper")
    print(f"{'value RMS error':28s} {mean_std(value):>16s}   0.16")
    print(f"{'  / r = 0.02':28s} {torch.tensor(value).mean() / R:16.1f}   (0.16 / 0.02 = 8.0)")
    print(f"{'  / sqrt(r^2 + lam) = 0.073':28s} {torch.tensor(value).mean() / r_eff:16.1f}   (2.2)")
    for eps in EPSILONS:
        print(f"{f'direction cosine, eps={eps:g}':28s} {mean_std(cos[eps]):>16s}   0.59")
    for eps in EPSILONS:
        print(f"{f'magnitude error, eps={eps:g}':28s} {mean_std(mag[eps]):>16s}   0.46")
    print(f"{'FD stability, 1e-3 vs 3e-3':28s} {mean_std(fd_stability):>16s}   (1 = stable)")
    print(f"{'adjoint error':28s} {torch.tensor(adjoint).mean():16.1e}   7e-07")
    print(f"{'||J||_2':28s} {mean_std(norm):>16s}   12.2")
    print(f"{'decoded pixels at +-1':28s} {100 * torch.tensor(saturated).mean():15.2f}%")


if __name__ == "__main__":
    main()
