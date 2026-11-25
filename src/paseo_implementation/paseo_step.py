"""Step 2: PASEO's measurement correction, matrix-free (Algorithm 1, inner solve).

    b  = y~ - h - J (z_a - z0)                      with h = H(z0), J = dH/dz at z0
    solve ((r^2 + lam) I + eta^2 J J^T) v = b      with conjugate gradients
    z* = z_a + eta^2 J^T v

J is never built: it is only used through JVP (J u) and VJP (J^T u).
H can be any differentiable function: a matrix in the toy, the CNN H_psi later.
"""

import torch
from torch.func import jvp, vjp


def dot(a, b):
    return (a * b).sum()


def conjugate_gradient(matvec, b, max_iters, rtol=1e-5):
    """Solve S v = b for symmetric positive definite S, given only p -> S p."""
    v = b * 0
    res = b                                 # residual b - S v  (v = 0 at start)
    p = res                                 # first search direction
    res_sq = dot(res, res)
    for _ in range(max_iters):
        Sp = matvec(p)
        alpha = res_sq / dot(p, Sp)         # exact step length along p
        v = v + alpha * p
        res = res - alpha * Sp
        new_res_sq = dot(res, res)
        if new_res_sq.sqrt() < rtol * b.norm():
            break
        p = res + (new_res_sq / res_sq) * p  # next direction, S-conjugate to the previous ones
        res_sq = new_res_sq
    return v


def correction_step(H, z0, z_a, y_tilde, eta, r, lam, max_cg_iters):
    """One PASEO correction: returns z* = z_a + eta^2 J^T v, with J linearised at z0."""
    h, vjp_fn = vjp(H, z0)

    def J(u):                               # J u    (forward mode)
        return jvp(H, (z0,), (u,))[1]

    def Jt(u):                              # J^T u  (reverse mode = backprop)
        return vjp_fn(u)[0]

    def S(p):                               # S p = (r^2 + lam) p + eta^2 J J^T p
        return (r**2 + lam) * p + eta**2 * J(Jt(p))

    b = y_tilde - h - J(z_a - z0)
    v = conjugate_gradient(S, b, max_cg_iters)
    return z_a + eta**2 * Jt(v)


def paseo_corrections(H, z0, mu, eta, y_z, r, lam, rho, P, max_cg_iters, beta=1.0, generator=None):
    """PASEO's measurement update for one diffusion step (Algorithm 2, lines 11-22).

    The perturbed targets are drawn once; then P relaxed corrections, each linearising H
    at the current estimate, move z0 a fraction rho of the way toward the proposal z*.
    """
    y_tilde = y_z + r * torch.randn(y_z.shape, generator=generator, device=y_z.device)
    mu_tilde = mu + eta * torch.randn(mu.shape, generator=generator, device=mu.device)
    c1 = (1 - beta**2) ** 0.5
    for _ in range(P):
        z_a = mu + c1 * (z0 - mu) + beta * (mu_tilde - mu)          # arc anchor, eq. 3.6
        z_star = correction_step(H, z0, z_a, y_tilde, eta, r, lam, max_cg_iters)
        z0 = z0 + rho * (z_star - z0)                               # relaxation
    return z0
