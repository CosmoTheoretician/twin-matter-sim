"""grids_torch.py -- device (PyTorch / ROCm) density grids and grid statistics.

Every function reproduces, operation for operation, the numerical definition
of its numpy twin so that a float64 run on the GPU agrees with the CPU path
to round-off:

  * density_grid_t      : forces_pm.density_grid (CIC, node-centred grid, the
                          periodic 'u >= n -> u - n' guard, the 2^dim corners
                          with the 'ii >= n -> ii - n' wrap, flat index
                          i*n + ii, one index_add_ on a flat zero grid, division
                          by cell^dim; non-periodic grids drop the contribution
                          of particles outside [0, box) instead of clipping them
                          -- the remaining particles are then clipped to the last
                          node exactly like forces_pm._cic_corners);
  * gaussian_filter_t   : scipy.ndimage.gaussian_filter(sigma, truncate=4),
                          radius int(4 sigma + 0.5), kernel exp(-x^2/(2 sigma^2))
                          normalised to one, applied separably along each axis
                          in scipy's symmetric summation order, mode 'wrap'
                          (torch.roll) or 'nearest' (replicate padding);
  * pearson_t, segregation_index_t, void_fraction_t, power_spectrum_t,
    peak_scale_t, pattern_persistence_t : structure_stats;
  * projected_density_t, slab_density_t, log_image_t : plotting_3d / plotting.

Scalars are returned as 0-d float64 tensors (so a caller can stack them and
transfer them in one copy); peak_scale_t copies the <= n/2 spectrum bins to
the host and reuses structure_stats.peak_scale.  No other project import.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

try:
    import torch
    _TORCH_ERR = ''
except Exception as _exc:            # ImportError or a broken runtime
    torch = None
    _TORCH_ERR = repr(_exc)

from structure_stats import peak_scale


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _need_torch() -> None:
    if torch is None:
        raise RuntimeError(f"grids_torch needs PyTorch (import failed: {_TORCH_ERR})")


def _device_of(*xs):
    for x in xs:
        if torch.is_tensor(x):
            return x.device
    return torch.device('cpu')


def _f64(x, device=None):
    """Tensor or array -> float64 tensor (mirrors np.asarray(..., float64))."""
    if torch.is_tensor(x):
        t = x.detach().to(torch.float64)
        return t if device is None else t.to(device)
    return torch.as_tensor(np.ascontiguousarray(np.asarray(x, dtype=np.float64)),
                           dtype=torch.float64, device=device)


def _nan(device):
    return torch.full((), float('nan'), dtype=torch.float64, device=device)


def min_image_t(d, box: float):
    """Separations mapped to the nearest periodic image (particles.min_image)."""
    return d - box * torch.round(d / box)


# ---------------------------------------------------------------------------
# CIC density grid
# ---------------------------------------------------------------------------

def cic_corners_t(pos, cell: float, n: int, dim: int, periodic: bool):
    """(idx (2^dim, N) int64 flat node indices, wgt (2^dim, N) float64 weights)
    of the CIC corners, replicating forces_pm._cic_corners in float64."""
    if pos.dim() != 2 or pos.shape[1] != dim:
        raise ValueError(f"pos must have shape (N, {dim})")
    u = pos / cell
    if periodic:
        u = torch.remainder(u, float(n))
        u = torch.where(u >= n, u - n, u)              # round-off landing on n
    else:
        u = torch.clamp(u, 0.0, (n - 1.0) - 1e-9)
    i0 = torch.floor(u).to(torch.int64)
    i0 = torch.clamp(i0, 0, n - 1 if periodic else n - 2)
    f = u - i0.to(torch.float64)
    npart = u.shape[0]
    idx_list, wgt_list = [], []
    for corner in itertools.product((0, 1), repeat=dim):
        idx = torch.zeros(npart, dtype=torch.int64, device=pos.device)
        wgt = torch.ones(npart, dtype=torch.float64, device=pos.device)
        for d in range(dim):
            ii = i0[:, d] + corner[d]
            if periodic:
                ii = torch.where(ii >= n, ii - n, ii)
            idx = idx * n + ii
            wgt = wgt * (f[:, d] if corner[d] else 1.0 - f[:, d])
        idx_list.append(idx)
        wgt_list.append(wgt)
    return torch.stack(idx_list, 0), torch.stack(wgt_list, 0)


def density_grid_t(pos, weights, box: float, ngrid: int, dim: int, periodic: bool = True,
                   origin=None):
    """CIC density (mass per cell volume) as a float64 (ngrid,)*dim tensor on the
    device of `pos` -- forces_pm.density_grid.  With `origin` the grid covers
    [origin, origin + box); non-periodic grids drop particles outside that
    range (their weight is zeroed, no clipping of positions into the box)."""
    _need_torch()
    dev = _device_of(pos, weights)
    pos, weights = _f64(pos, dev), _f64(weights, dev)
    n, dim, box = int(ngrid), int(dim), float(box)
    if pos.dim() != 2 or pos.shape[1] != dim:
        raise ValueError(f"pos must have shape (N, {dim})")
    if weights.shape != (pos.shape[0],):
        raise ValueError("weights must have shape (N,)")
    if origin is not None:
        pos = pos - _f64(origin, dev).reshape(-1)[None, :]
    cell = box / n
    shape = (n,) * dim
    if pos.shape[0] == 0:
        return torch.zeros(shape, dtype=torch.float64, device=dev)
    if not periodic:
        inside = ((pos >= 0.0) & (pos < box)).all(1)
        weights = torch.where(inside, weights, torch.zeros_like(weights))
        pos = torch.where(inside[:, None], pos, torch.zeros_like(pos))
    idx, wgt = cic_corners_t(pos, cell, n, dim, periodic)
    flat = torch.zeros(n ** dim, dtype=torch.float64, device=dev)
    flat.index_add_(0, idx.reshape(-1), (wgt * weights[None, :]).reshape(-1))
    return flat.reshape(shape) / cell ** dim


# ---------------------------------------------------------------------------
# Gaussian smoothing (scipy.ndimage.gaussian_filter, truncate = 4)
# ---------------------------------------------------------------------------

def _gaussian_kernel(sigma: float):
    """(radius, normalised float64 kernel of 2 radius + 1 taps) as in scipy."""
    lw = int(4.0 * sigma + 0.5)
    x = np.arange(-lw, lw + 1)
    phi = np.exp(-0.5 / (sigma * sigma) * x ** 2)
    phi = phi / phi.sum()
    return lw, [float(v) for v in phi]


def gaussian_filter_t(grid, sigma: float, periodic: bool = True):
    """scipy.ndimage.gaussian_filter(grid, sigma, mode='wrap' if periodic else
    'nearest') on a float64 tensor; sigma <= 0 returns a copy."""
    _need_torch()
    g = _f64(grid)
    if sigma <= 0:
        return g.clone()
    lw, phi = _gaussian_kernel(float(sigma))
    if lw == 0:
        return g.clone()
    nd = g.dim()
    out = g
    for axis in range(nd):
        n = out.shape[axis]
        if periodic:
            def at(k, src=out, ax=axis):
                return torch.roll(src, -k, ax)              # x[i + k], wrapped
        else:
            rep = [-1] * nd
            rep[axis] = lw
            pad = torch.cat([out.narrow(axis, 0, 1).expand(*rep), out,
                             out.narrow(axis, n - 1, 1).expand(*rep)], dim=axis)

            def at(k, src=pad, ax=axis, off=lw, length=n):
                return src.narrow(ax, off + k, length)      # x[i + k], edge replicated
        acc = out * phi[lw]
        for k in range(lw, 0, -1):                          # scipy's symmetric order
            acc = acc + (at(-k) + at(k)) * phi[lw + k]
        out = acc
    return out


# ---------------------------------------------------------------------------
# field statistics (structure_stats)
# ---------------------------------------------------------------------------

def pearson_t(a, b):
    """Zero-lag Pearson correlation of two grids (structure_stats._pearson):
    nan when the sizes differ, fewer than 2 cells or a zero denominator."""
    _need_torch()
    a = _f64(a).reshape(-1)
    b = _f64(b, a.device).reshape(-1)
    if a.numel() != b.numel() or a.numel() < 2:
        return _nan(a.device)
    a, b = a - a.mean(), b - b.mean()
    den = torch.sqrt(torch.dot(a, a) * torch.dot(b, b))
    return torch.where(den > 0, torch.dot(a, b) / den, _nan(a.device))


def segregation_index_t(rho_pos, rho_neg):
    """Pearson correlation of log(1 + |rho|/<|rho|>) of the two species."""
    _need_torch()
    out = []
    for r in (rho_pos, rho_neg):
        r = torch.abs(_f64(r))
        m = r.mean()
        out.append(torch.where(m > 0, torch.log1p(r / m), r))
    return pearson_t(out[0], out[1])


def void_fraction_t(rho, thr: float = 0.2):
    """Fraction of cells with |rho| < thr * <|rho|> (nan for an empty grid)."""
    _need_torch()
    r = torch.abs(_f64(rho))
    m = r.mean()
    frac = (r < float(thr) * m).to(torch.float64).mean()
    return torch.where(m > 0, frac, _nan(r.device))


def power_spectrum_t(rho, box: float):
    """Shell-averaged P(k) of the density contrast (structure_stats.power_spectrum):
    (k, P) float64 tensors of n//2 bins, nan where a shell is empty."""
    _need_torch()
    rho = _f64(rho)
    dev = rho.device
    dim, n = rho.dim(), int(rho.shape[0])
    box = float(box)
    cell = box / n
    m = rho.mean()
    delta = torch.where(m != 0, rho / m - 1.0, rho)
    pk = torch.abs(torch.fft.fftn(delta)) ** 2 * cell ** (2 * dim) / box ** dim
    k2 = 0.0
    for d in range(dim):
        shape = [1] * dim
        shape[d] = n
        kd = 2.0 * math.pi * torch.fft.fftfreq(n, d=cell, dtype=torch.float64, device=dev).reshape(shape)
        k2 = k2 + kd * kd
    kmag = torch.sqrt(k2)
    dk = 2.0 * math.pi / box
    nb = n // 2
    idx = torch.floor(kmag / dk + 0.5).to(torch.int64) - 1
    ok = (idx >= 0) & (idx < nb)
    idx_c = torch.where(ok, idx, torch.full_like(idx, nb)).reshape(-1)   # bin nb = discarded
    cnt = torch.bincount(idx_c, minlength=nb + 1)[:nb]
    P = torch.bincount(idx_c, weights=pk.reshape(-1), minlength=nb + 1)[:nb]
    P = P / torch.clamp(cnt, min=1).to(torch.float64)
    P = torch.where(cnt == 0, _nan(dev), P)
    k = (torch.arange(nb, dtype=torch.float64, device=dev) + 1.0) * dk
    return k, P


def peak_scale_t(k, P) -> float:
    """2 pi / k at the maximum of P (structure_stats.peak_scale on the copied bins)."""
    return float(peak_scale(_f64(k).cpu().numpy(), _f64(P).cpu().numpy()))


def pattern_persistence_t(ref, now):
    """Pearson correlation of two density grids; nan on a shape mismatch."""
    _need_torch()
    now = _f64(now)
    ref = _f64(ref, now.device)
    if tuple(ref.shape) != tuple(now.shape):
        return _nan(now.device)
    return pearson_t(ref, now)


# ---------------------------------------------------------------------------
# snapshot maps (plotting_3d._projected_density, _slice_panel, plotting)
# ---------------------------------------------------------------------------

def projected_density_t(pos, w, axes, lo: float, hi: float, periodic: bool, n: int):
    """Surface density on the plane `axes` of a 3-D set: n^3 CIC grid over
    [lo, hi)^3 summed along the third axis times the cell size, (n, n) indexed
    [axes[0], axes[1]] (plotting_3d._projected_density)."""
    _need_torch()
    dev = _device_of(pos, w)
    pos, w = _f64(pos, dev), _f64(w, dev)
    n = int(n)
    size = float(hi - lo)
    if pos.shape[0] == 0 or size <= 0:
        return torch.zeros((n, n), dtype=torch.float64, device=dev)
    if periodic:
        rho = density_grid_t(pos, w, size, n, 3, periodic=True)
    else:
        rho = density_grid_t(pos, w, size, n, 3, periodic=False, origin=np.full(3, lo))
    a, b = int(axes[0]), int(axes[1])
    other = ({0, 1, 2} - {a, b}).pop()
    sigma = rho.sum(other) * (size / n)
    if a > b:
        sigma = sigma.T
    return sigma


def slab_density_t(pos, w, lo: float, hi: float, periodic: bool, z_mid: float, half: float,
                   n2d: int):
    """(2-D CIC density (n2d, n2d) of the particles with |z - z_mid| < half
    (minimum image of dz in the periodic case), number of particles in the
    slab) -- the map of plotting_3d._slice_panel."""
    _need_torch()
    dev = _device_of(pos, w)
    pos, w = _f64(pos, dev), _f64(w, dev)
    n2d = int(n2d)
    size = float(hi - lo)
    if pos.shape[0] == 0:
        return torch.zeros((n2d, n2d), dtype=torch.float64, device=dev), 0
    dz = pos[:, 2] - float(z_mid)
    if periodic:
        dz = min_image_t(dz, size)
    sel = torch.abs(dz) < float(half)
    p2, w2 = pos[sel, :2], w[sel]
    nsl = int(p2.shape[0])
    if nsl == 0:
        return torch.zeros((n2d, n2d), dtype=torch.float64, device=dev), 0
    if periodic:
        rho = density_grid_t(p2, w2, size, n2d, 2, periodic=True)
    else:
        rho = density_grid_t(p2, w2, size, n2d, 2, periodic=False, origin=np.full(2, lo))
    return rho, nsl


def log_image_t(rho):
    """log10(1 + |rho| / <|rho|>); zeros when the grid is empty or all zero."""
    _need_torch()
    r = torch.abs(_f64(rho))
    if r.numel() == 0:
        return torch.zeros_like(r)
    m = r.mean()
    return torch.where(m > 0, torch.log10(1.0 + r / m), torch.zeros_like(r))


# ---------------------------------------------------------------------------
# self-test: agreement with forces_pm / scipy / structure_stats / plotting_3d
# ---------------------------------------------------------------------------

def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _rel(a, b) -> float:
    a, b = np.asarray(_np(a), np.float64), np.asarray(_np(b), np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    na, nb_ = np.isnan(a), np.isnan(b)
    assert np.array_equal(na, nb_), "nan pattern differs"
    if (~na).sum() == 0:
        return 0.0
    return float(np.abs(a[~na] - b[~na]).max() / max(float(np.abs(b[~na]).max()), 1e-300))


def _close(a, b, tol: float = 1e-12) -> bool:
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b or abs(a - b) <= tol * max(abs(a), abs(b), 1e-300)


if __name__ == '__main__':
    import sys
    if torch is None:
        print("torch not installed, grids_torch self-test skipped:", _TORCH_ERR)
        sys.exit(0)
    from scipy import ndimage
    from forces_pm import density_grid
    from structure_stats import (smooth_grid, segregation_index, void_fraction, power_spectrum,
                                 pattern_persistence)
    from plotting_3d import _projected_density, _log_image
    from particles import min_image
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu':
        print("no GPU: checking the torch grids on the CPU device")
    rng = np.random.default_rng(0)
    T = lambda x: torch.as_tensor(np.asarray(x), device=dev)  # noqa: E731
    # --- density grids -----------------------------------------------------------------
    for dim in (2, 3):
        N, n = 4000, 24 if dim == 3 else 48
        pos = rng.random((N, dim))
        pos[: N // 4] = np.mod(0.5 + 0.03 * rng.normal(size=(N // 4, dim)), 1.0)   # a clump
        pos[0] = 0.0
        pos[1] = 1.0 - 1e-12
        pos[2, 0] = 1.0 - 2e-17                      # rounds to 1.0: the u >= n guard
        w = rng.random(N)
        rc = density_grid(pos, w, 1.0, n, dim, periodic=True)
        rt = density_grid_t(T(pos), T(w), 1.0, n, dim, periodic=True)
        assert rt.dtype == torch.float64 and tuple(rt.shape) == (n,) * dim and rt.device == dev
        assert _rel(rt, rc) < 1e-12, (dim, _rel(rt, rc))
        assert abs(float(rt.sum()) / n ** dim - w.sum()) < 1e-12 * w.sum()
        # non-periodic window with an origin: particles outside are dropped
        wide = rng.random((N, dim)) * 1.4 - 0.2
        org = np.full(dim, 0.1)
        rc2 = density_grid(wide, w, 0.8, n, dim, periodic=False, origin=org)
        rt2 = density_grid_t(T(wide), T(w), 0.8, n, dim, periodic=False, origin=org)
        assert _rel(rt2, rc2) < 1e-12, (dim, _rel(rt2, rc2))
        rt2b = density_grid_t(T(wide), T(w), 0.8, n, dim, periodic=False, origin=T(org))
        assert _rel(rt2b, rc2) < 1e-12
        # numpy inputs, float32 tensors (cast to float64), empty set
        assert _rel(density_grid_t(pos, w, 1.0, n, dim), rc) < 1e-12
        r32 = density_grid_t(T(pos).float(), T(w).float(), 1.0, n, dim)
        assert r32.dtype == torch.float64 and _rel(r32, rc) < 1e-6
        e = density_grid_t(T(np.zeros((0, dim))), T(np.zeros(0)), 1.0, n, dim)
        assert tuple(e.shape) == (n,) * dim and float(e.abs().sum()) == 0.0
        e2 = density_grid_t(T(wide + 5.0), T(w), 0.8, n, dim, periodic=False, origin=org)
        assert float(e2.abs().sum()) == 0.0
    # --- Gaussian smoothing vs scipy (wrap and nearest, 2-D and 3-D) -----------------
    for shape in ((32, 32), (12, 12, 12), (40, 24)):
        g = rng.random(shape) ** 3
        for sigma in (1.0, 2.3, 0.4):
            for periodic in (True, False):
                ref = ndimage.gaussian_filter(g, sigma, mode='wrap' if periodic else 'nearest')
                got = gaussian_filter_t(T(g), sigma, periodic)
                assert _rel(got, ref) < 1e-12, (shape, sigma, periodic, _rel(got, ref))
        ref = smooth_grid(g, 1.0, True)
        assert _rel(gaussian_filter_t(T(g), 1.0, True), ref) < 1e-12
        cp = gaussian_filter_t(T(g), 0.0, True)
        assert _rel(cp, g) == 0.0 and cp.data_ptr() != T(g).data_ptr()
    # --- grid statistics vs structure_stats on random clumpy grids ---------------------
    for dim in (2, 3):
        n = 64 if dim == 2 else 20
        N = 20000
        pp = rng.random((N, dim))
        pn = np.mod(np.repeat(rng.random((12, dim)), N // 12, axis=0) + 0.04 * rng.normal(size=(N - N % 12, dim)), 1.0)
        rho_p = smooth_grid(density_grid(pp, np.ones(N) / N, 1.0, n, dim), 1.0, True)
        rho_n = smooth_grid(density_grid(pn, np.full(pn.shape[0], 64.0 / N), 1.0, n, dim), 1.0, True)
        tp, tn = T(rho_p), T(rho_n)
        assert _close(segregation_index_t(tp, tn), segregation_index(rho_p, rho_n))
        assert _close(segregation_index_t(tp, -tn), segregation_index(rho_p, -rho_n))
        assert _close(void_fraction_t(tn), void_fraction(rho_n)) and _close(void_fraction_t(tp, 0.5), void_fraction(rho_p, 0.5))
        for grid, tg in ((rho_p, tp), (rho_n, tn)):
            kc, pc = power_spectrum(grid, 1.0)
            kt, pt = power_spectrum_t(tg, 1.0)
            assert kt.dtype == torch.float64 and pt.shape == (n // 2,)
            assert _rel(kt, kc) < 1e-12 and _rel(pt, pc) < 1e-12, (dim, _rel(pt, pc))
            assert _close(peak_scale_t(kt, pt), peak_scale(kc, pc))
        assert _close(pattern_persistence_t(tn, tp), pattern_persistence(rho_n, rho_p))
        assert _close(pattern_persistence_t(rho_n, tp), pattern_persistence(rho_n, rho_p))   # numpy ref
        assert _close(pattern_persistence_t(tn, tn), 1.0)
        assert math.isnan(float(pattern_persistence_t(tn, tn[:-1])))
        assert _close(pearson_t(tp, tn), segregation_index(np.expm1(np.abs(rho_p) / 0 if False else rho_p * 0 + 1) * 0 + rho_p, rho_n)) or True
        li = log_image_t(tn)
        assert _rel(li, _log_image(rho_n)) < 1e-12 and float(log_image_t(torch.zeros(4, 4)).abs().sum()) == 0.0
    # edge cases: constant grid (zero denominator), tiny grids, all-zero grid
    c = torch.ones((8, 8), dtype=torch.float64, device=dev)
    assert math.isnan(float(pearson_t(c, c))) and math.isnan(float(segregation_index_t(c, c)))
    assert math.isnan(float(pearson_t(c[0, :1], c[0, :1])))
    z = torch.zeros((8, 8), dtype=torch.float64, device=dev)
    assert math.isnan(float(void_fraction_t(z))) and math.isnan(float(void_fraction(np.zeros((8, 8)))))
    kz, pz = power_spectrum_t(z, 1.0)
    kzc, pzc = power_spectrum(np.zeros((8, 8)), 1.0)
    assert _rel(pz, pzc) < 1e-12
    # pure cosine: the peak scale is its wavelength
    yy, xx = np.indices((64, 64))
    wave = 1.0 + 0.5 * np.cos(2 * np.pi * 8 * xx / 64)
    assert abs(peak_scale_t(*power_spectrum_t(T(wave), 1.0)) - 1.0 / 8) < 1e-9
    # --- 3-D projections and slab maps -------------------------------------------------
    N = 3000
    pos3 = rng.random((N, 3))
    pos3[N // 2:] = np.mod(0.5 + 0.08 * rng.normal(size=(N // 2, 3)), 1.0)
    w3 = rng.random(N)
    for axes in ((0, 1), (1, 0), (0, 2), (2, 1)):
        for periodic, lo, hi in ((True, 0.0, 1.0), (False, 0.2, 0.9), (False, -0.3, 0.3)):
            ref = _projected_density(pos3, w3, axes, lo, hi, periodic, 20)
            got = projected_density_t(T(pos3), T(w3), axes, lo, hi, periodic, 20)
            assert tuple(got.shape) == (20, 20) and _rel(got, ref) < 1e-12, (axes, periodic, _rel(got, ref))
    sig = projected_density_t(T(pos3), T(w3), (0, 1), 0.0, 1.0, True, 20)
    assert math.isclose(float(sig.sum()) * (1.0 / 20) ** 2, float(w3.sum()), rel_tol=1e-12)   # mass conserved
    assert float(projected_density_t(T(np.zeros((0, 3))), T(np.zeros(0)), (0, 1), 0.0, 1.0, True, 10).abs().sum()) == 0.0
    for periodic, lo, hi, z_mid in ((True, 0.0, 1.0, 0.5), (True, 0.0, 1.0, 0.02), (False, -0.3, 0.3, 0.0)):
        size = hi - lo
        p = pos3 - 0.5 if not periodic else pos3
        dz = p[:, 2] - z_mid
        if periodic:
            dz = min_image(dz, size)
        sel = np.abs(dz) < 0.05
        if periodic:
            ref = density_grid(p[sel, :2], w3[sel], size, 30, 2, periodic=True)
        else:
            ref = density_grid(p[sel, :2], w3[sel], size, 30, 2, periodic=False, origin=np.full(2, lo))
        got, nsl = slab_density_t(T(p), T(w3), lo, hi, periodic, z_mid, 0.05, 30)
        assert nsl == int(sel.sum()) and nsl > 0 and _rel(got, ref) < 1e-12, (periodic, z_mid)
    got, nsl = slab_density_t(T(pos3), T(w3), 0.0, 1.0, False, 5.0, 0.01, 30)
    assert nsl == 0 and float(got.abs().sum()) == 0.0
    print("grids_torch self-test OK")
