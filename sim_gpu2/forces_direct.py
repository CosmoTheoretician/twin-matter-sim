"""forces_direct.py -- direct O(N^2) summation of the three-law force.

This is the algorithm of the original 1991-93 runs: every particle sums the
force from the N-1 others.  Conventions (see notes, section 1):

    particle i carries an unsigned weight w_i = |m_i| and a sign s_i = +/-1
    a_i = -G s_i s_j w_j f(r_ij) d_ij ,   d_ij = x_i - x_j

so same-sign pairs attract (Newton), opposite-sign pairs repel (anti-Newton).
The potential returned is the *unsigned* potential of the signed density,

    phi_i = G sum_j s_j w_j phi_unit(r_ij)

such that U = 0.5 sum_i s_i w_i phi_i reproduces the pair sum of
U_ij = -G s_i s_j w_i w_j / r_ij (or the logarithmic analogue).

Boundary conditions: `box=None` is open space; with `box` given the
minimum-image rule is applied, i.e. each particle feels only the nearest
periodic image of every other particle -- Petit's "dashed rectangle" tiling
(a flat torus).  A uniform background contributes no net force under the
minimum-image rule by symmetry, so the non-neutral box (|M_-| = 64 M_+)
is handled by an implicit Jeans swindle (A3).

Softening: 'plummer' (r^2 -> r^2 + eps^2) or 'cutin' (force zeroed and
potential held constant for r < eps, the pages' "cut-in").
Force law: 'r2' (1/r^2, razor-thin sheet in 2-D or genuine 3-D) or 'log'
(1/r force, phi = ln r, true 2-D gravity).

Memory is bounded by chunking over targets (and, internally, over sources).
"""
from __future__ import annotations

import numpy as np

from particles import min_image

# approximate byte budget for the (chunk, jchunk, dim) separation array
_MEM_BUDGET = 96 * 1024 * 1024

_SOFT = ('plummer', 'cutin')
_LAWS = ('r2', 'log')


def pair_kernel(r2: np.ndarray, eps: float, soft: str, force_law: str):
    """Return (f_over_r, phi_unit) for squared separations r2.

    f_over_r is defined so that  acc_i = -G s_i s_j w_j f_over_r d_ij ,
    phi_unit is the potential of a unit positive mass at distance r.

    plummer / r2 :  f = (r^2+eps^2)^-3/2,      phi = -(r^2+eps^2)^-1/2
    plummer / log:  f = (r^2+eps^2)^-1,        phi = 0.5 ln(r^2+eps^2)
    cutin   / r2 :  f = r^-3 (r>eps) else 0,   phi = -1/r (r>eps) else -1/eps
    cutin   / log:  f = r^-2 (r>eps) else 0,   phi = ln r (r>eps) else ln eps

    Division by zero (eps = 0 at r = 0) yields inf; callers must mask the
    self term.  Scalars and arrays of any shape are accepted.
    """
    if soft not in _SOFT:
        raise ValueError(f"soft={soft!r} not in {_SOFT}")
    if force_law not in _LAWS:
        raise ValueError(f"force_law={force_law!r} not in {_LAWS}")
    r2 = np.asarray(r2)
    dt = r2.dtype if r2.dtype.kind == 'f' else np.float64
    e2 = dt.type(eps * eps) if hasattr(dt, 'type') else eps * eps
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if soft == 'plummer':
            q = r2 + e2
            if force_law == 'r2':
                inv = 1.0 / np.sqrt(q)
                return inv * inv * inv, -inv
            return 1.0 / q, 0.5 * np.log(q)
        # cut-in: strictly outside the cutoff radius
        outside = r2 > e2
        r2s = np.where(outside, r2, 1.0)        # safe values inside
        r = np.sqrt(r2s)
        if force_law == 'r2':
            f = np.where(outside, 1.0 / (r2s * r), 0.0)
            inside_phi = -1.0 / eps if eps > 0 else 0.0
            phi = np.where(outside, -1.0 / r, inside_phi)
        else:
            f = np.where(outside, 1.0 / r2s, 0.0)
            inside_phi = np.log(eps) if eps > 0 else 0.0
            phi = np.where(outside, np.log(r), inside_phi)
        return f.astype(dt, copy=False), phi.astype(dt, copy=False)


def direct_forces(pos: np.ndarray, w: np.ndarray, s: np.ndarray, G: float, eps: float,
                  soft: str, force_law: str, box: float | None = None, chunk: int = 1024,
                  need_phi: bool = True):
    """Pairwise sum over all particles.

    Returns (acc, phi): acc (N, dim) *including* the sign s_i of the target,
    phi (N,) the unsigned potential of the signed density (zeros when
    need_phi is False).  Works in the dtype of `pos` (float32 or float64).
    """
    pos = np.asarray(pos)
    if pos.ndim != 2:
        raise ValueError("pos must have shape (N, dim)")
    n, dim = pos.shape
    dt = pos.dtype if pos.dtype.kind == 'f' else np.dtype(np.float64)
    pos = pos.astype(dt, copy=False)
    w = np.asarray(w, dtype=dt)
    s = np.asarray(s, dtype=dt)
    if w.shape != (n,) or s.shape != (n,):
        raise ValueError("w and s must have shape (N,)")
    acc = np.zeros((n, dim), dtype=dt)
    phi = np.zeros(n, dtype=dt)
    if n < 2:
        return acc, phi
    if box is not None and box <= 0:
        raise ValueError("box must be positive or None")
    sw = s * w                                   # signed weights
    chunk = max(1, min(int(chunk), n))
    jmax = max(64, int(_MEM_BUDGET // (chunk * dim * dt.itemsize)))
    jmax = min(jmax, n)
    G = dt.type(G)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        c = i1 - i0
        pi = pos[i0:i1]
        a_i = np.zeros((c, dim), dtype=dt)
        p_i = np.zeros(c, dtype=dt)
        for j0 in range(0, n, jmax):
            j1 = min(n, j0 + jmax)
            d = pi[:, None, :] - pos[None, j0:j1, :]          # (c, jc, dim)
            if box is not None:
                d = min_image(d, box)
            r2 = np.einsum('ijk,ijk->ij', d, d)
            f, pu = pair_kernel(r2, eps, soft, force_law)
            # remove self-interaction (also kills inf/nan from eps = 0)
            lo, hi = max(i0, j0), min(i1, j1)
            if lo < hi:
                k = np.arange(lo, hi)
                f[k - i0, k - j0] = 0.0
                pu[k - i0, k - j0] = 0.0
            coef = f * sw[None, j0:j1]
            a_i -= np.einsum('ij,ijk->ik', coef, d)
            if need_phi:
                p_i += pu @ sw[j0:j1]
        acc[i0:i1] = G * s[i0:i1, None] * a_i
        if need_phi:
            phi[i0:i1] = G * p_i
    return acc, phi


def potential_energy_from_phi(w: np.ndarray, s: np.ndarray, phi: np.ndarray) -> float:
    """U = 0.5 sum_i s_i w_i phi_i (pair sum without self terms)."""
    w = np.asarray(w, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    return 0.5 * float(np.sum(s * w * phi))


def pair_count_estimate(n: int) -> float:
    """Number of distinct pairs, n(n-1)/2, for runtime warnings."""
    n = float(n)
    return 0.5 * n * (n - 1.0)


def estimate_seconds_per_eval(n: int, dim: int = 2, dtype: str = 'float64') -> float:
    """Very rough wall-clock estimate for one force evaluation (numpy, one core):
    ~3 s per 10^4 particles in float64 2-D, scaling as N^2."""
    base = 3.0e-8 * (1.0 if dtype == 'float64' else 0.55) * (1.0 if dim == 2 else 1.4)
    return base * float(n) * float(n)


if __name__ == '__main__':  # minimal self-test
    rng = np.random.default_rng(0)
    N = 300
    pos = rng.random((N, 2))
    w = np.full(N, 1.0 / N)
    s = np.where(np.arange(N) < N // 2, 1.0, -1.0)
    for soft in _SOFT:
        for law in _LAWS:
            for box in (None, 1.0):
                a, p = direct_forces(pos, w, s, 1.0, 0.01, soft, law, box=box, chunk=64)
                a2, p2 = direct_forces(pos, w, s, 1.0, 0.01, soft, law, box=box, chunk=7)
                assert np.allclose(a, a2) and np.allclose(p, p2)
                # Newton's third law for |m| a  ->  sum w_i a_i = 0
                assert np.allclose((w[:, None] * a).sum(0), 0.0, atol=1e-10), (soft, law, box)
    # two-body check: unit masses, same sign, distance 1, open space, Plummer eps=0
    a, p = direct_forces(np.array([[0.0, 0.0], [1.0, 0.0]]), np.ones(2), np.ones(2),
                         1.0, 0.0, 'plummer', 'r2')
    assert np.allclose(a[0], [1.0, 0.0]) and np.allclose(p, [-1.0, -1.0])
    assert np.isclose(potential_energy_from_phi(np.ones(2), np.ones(2), p), -1.0)
    # opposite signs repel
    a, _ = direct_forces(np.array([[0.0, 0.0], [1.0, 0.0]]), np.ones(2), np.array([1.0, -1.0]),
                         1.0, 0.0, 'plummer', 'r2')
    assert a[0, 0] < 0 and a[1, 0] > 0
    print("forces_direct self-test OK")
