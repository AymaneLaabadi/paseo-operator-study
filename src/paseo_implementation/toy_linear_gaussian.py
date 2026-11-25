"""
Step 1: PASEO's correction step on a linear-Gaussian toy problem.
Prior:        z ~ N(mu, eta^2 I)
Measurement:  y = H z + n,   n ~ N(0, r^2 I)
The posterior p(z | y) is Gaussian with a closed form, so we can check that
PASEO's "perturb, then solve" step produces exact posterior samples.
"""

import numpy as np
import torch

from paseo_implementation.paseo_step import correction_step

rng = np.random.default_rng(0)

# Problem
n, m = 20, 10           # latent size, measurement size
eta, r = 0.5, 0.1       # prior std, measurement noise std
num_samples = 10_000

H = rng.normal(size=(m, n)) / np.sqrt(n)
mu = rng.normal(size=n)
z_true = mu + eta * rng.normal(size=n)
y = H @ z_true + r * rng.normal(size=m)


# Exact posterior (closed form)
def exact_posterior(H, mu, y, eta, r):
    S = r**2 * np.eye(len(y)) + eta**2 * H @ H.T
    mean = mu + eta**2 * H.T @ np.linalg.solve(S, y - H @ mu)
    cov = eta**2 * np.eye(len(mu)) - eta**4 * H.T @ np.linalg.solve(S, H)
    return mean, cov


# PASEO correction step (perturb, then solve)
def paseo_step(H, mu, y, eta, r, lam, num_samples, perturb_y=True):
    """Return num_samples draws of z* = z_a + eta^2 H^T v."""
    z_a = mu + eta * rng.normal(size=(num_samples, len(mu)))  # perturbed prior
    y_tilde = y + r * rng.normal(size=(num_samples, len(y)))  # perturbed measurement
    if not perturb_y:
        y_tilde = np.broadcast_to(y, y_tilde.shape)

    S = (r**2 + lam) * np.eye(len(y)) + eta**2 * H @ H.T
    b = y_tilde - z_a @ H.T                 # b = y~ - H z_a   (one row per sample)
    v = np.linalg.solve(S, b.T).T           # S v = b
    return z_a + eta**2 * v @ H             # z* = z_a + eta^2 H^T v


def relative_error(estimate, truth):
    return np.linalg.norm(estimate - truth) / np.linalg.norm(truth)


mean, cov = exact_posterior(H, mu, y, eta, r)
print(f"Prior std per coordinate:     {eta:.3f}")
print(f"Posterior std (avg):          {np.sqrt(np.diag(cov)).mean():.3f}\n")

cases = [
    ("PASEO, lambda = 0",            dict(lam=0.0)),
    ("PASEO, lambda = 0.005 (paper)", dict(lam=0.005)),
    ("No measurement perturbation",  dict(lam=0.0, perturb_y=False)),
]

def report(name, samples):
    mean_err = relative_error(samples.mean(axis=0), mean)
    cov_err = relative_error(np.cov(samples.T), cov)
    print(f"{name:32s} {mean_err:9.4f} {cov_err:9.4f}")


print(f"{'case':32s} {'mean err':>9s} {'cov err':>9s}")
# Reference: sampling the exact posterior directly shows the Monte-Carlo noise floor.
report("Exact posterior (noise floor)", rng.multivariate_normal(mean, cov, size=num_samples))
for name, kwargs in cases:
    report(name, paseo_step(H, mu, y, eta, r, num_samples=num_samples, **kwargs))


# Step 2: the same step with CG + JVP/VJP (paseo_step.py)
H_torch = torch.from_numpy(H)
mu_torch = torch.from_numpy(mu)


def H_op(z):
    """The operator. Here a matrix; later the CNN H_psi."""
    return H_torch @ z


def paseo_step_cg(lam, max_cg_iters):
    samples = []
    for _ in range(num_samples):
        z_a = mu_torch + eta * torch.from_numpy(rng.normal(size=n))
        y_tilde = torch.from_numpy(y + r * rng.normal(size=m))
        z_star = correction_step(H_op, mu_torch, z_a, y_tilde, eta, r, lam, max_cg_iters)
        samples.append(z_star.numpy())
    return np.array(samples)


# Check 1: one sample, CG vs the direct solve with the same z_a and y~.
z_a = mu + eta * rng.normal(size=n)
y_tilde = y + r * rng.normal(size=m)
S = r**2 * np.eye(m) + eta**2 * H @ H.T
z_direct = z_a + eta**2 * H.T @ np.linalg.solve(S, y_tilde - H @ z_a)
z_cg = correction_step(H_op, mu_torch, torch.from_numpy(z_a), torch.from_numpy(y_tilde),
                       eta, r, lam=0.0, max_cg_iters=m).numpy()
print(f"\nCG (C = m) vs direct solve, max difference: {np.abs(z_cg - z_direct).max():.1e}\n")

# Check 2: statistics, and what happens with the paper's few CG iterations.
print(f"{'case':32s} {'mean err':>9s} {'cov err':>9s}")
report(f"CG, C = {m} (= m), lambda = 0", paseo_step_cg(lam=0.0, max_cg_iters=m))
report(f"CG, C = {m} (= m), lambda = 0.005", paseo_step_cg(lam=0.005, max_cg_iters=m))
report("CG, C = 2, lambda = 0", paseo_step_cg(lam=0.0, max_cg_iters=2))
report("CG, C = 2, lambda = 0.005 (paper)", paseo_step_cg(lam=0.005, max_cg_iters=2))

# Bias from lambda alone, without Monte-Carlo noise: with a converged solve,
# E[z*] = mu + eta^2 H^T ((r^2 + lam) I + eta^2 H H^T)^-1 (y - H mu).
S_lam = (r**2 + 0.005) * np.eye(m) + eta**2 * H @ H.T
mean_lam = mu + eta**2 * H.T @ np.linalg.solve(S_lam, y - H @ mu)
print(f"\nExact mean error from lambda = 0.005 alone (no sampling): {relative_error(mean_lam, mean):.4f}")
