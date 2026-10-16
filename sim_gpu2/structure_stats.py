"""structure_stats.py -- quantitative structure measures for both scenarios.

LSS (pre-registered criteria, notes section 3): per-species clump catalogue
from a CIC density grid (forces_pm.density_grid) smoothed by one cell and
segmented with scipy.ndimage.label above threshold x mean density, with labels
merged across periodic edges; cell size d = mean nearest-neighbour distance of
negative clump centroids; phi/d; segregation index (Pearson correlation of
log-densities, strongly negative = cells); positive-matter void fraction;
power-spectrum peak scale; pattern persistence (correlation of rho_-(t) with a
reference grid).

Galaxy: Fourier amplitudes A_m(R) and phases in radial bins, bar strength
(max A2 inside r_inner), outer spiral strength (mass-weighted A2 in an annulus),
pitch angle from d(phi2/2)/d ln R, pattern speed from the m=2 phase drift,
disk/twin angular momenta (braking test), twin clump count, measured Toomre Q.
All functions are pure numpy/scipy; nothing here tunes parameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from particles import ParticleSet, min_image
from forces_pm import density_grid

_sum_labels = getattr(ndimage, 'sum_labels', ndimage.sum)


@dataclass
class ClumpCatalog:
    n: int
    masses: np.ndarray       # (n,) mass per clump (density x cell volume)
    diameters: np.ndarray    # (n,) equivalent diameter (circle / sphere of the same area / volume)
    centroids: np.ndarray    # (n, dim) density-weighted, periodic-aware
    labels: np.ndarray       # grid of clump ids (0 = background), same shape as the input


# ---------------------------------------------------------------------------
# LSS: grids and clumps
# ---------------------------------------------------------------------------

def smooth_grid(rho: np.ndarray, sigma_cells: float, periodic: bool) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    if sigma_cells <= 0:
        return rho.copy()
    return ndimage.gaussian_filter(rho, sigma_cells, mode='wrap' if periodic else 'nearest')


def _merge_periodic(labels: np.ndarray) -> np.ndarray:
    """Union labels touching across opposite faces; returns relabelled grid."""
    nlab = int(labels.max())
    if nlab == 0:
        return labels
    parent = np.arange(nlab + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ax in range(labels.ndim):
        a = np.take(labels, 0, axis=ax).ravel()
        b = np.take(labels, -1, axis=ax).ravel()
        sel = (a > 0) & (b > 0)
        if not sel.any():
            continue
        for x, y in np.unique(np.stack([a[sel], b[sel]], axis=1), axis=0):
            rx, ry = find(int(x)), find(int(y))
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)
    roots = np.array([find(i) for i in range(nlab + 1)])
    uniq, inv = np.unique(roots, return_inverse=True)
    lut = inv.astype(np.int64)
    lut[0] = 0
    return lut[labels]


def find_clumps(rho: np.ndarray, box: float, threshold_factor: float, min_cells: int,
                periodic: bool) -> ClumpCatalog:
    """Connected regions with |rho| > threshold_factor * mean|rho| and >= min_cells."""
    rho = np.abs(np.asarray(rho, dtype=np.float64))
    dim, n = rho.ndim, rho.shape[0]
    cell = box / n
    vol = cell ** dim
    mean = float(rho.mean())
    empty = ClumpCatalog(0, np.zeros(0), np.zeros(0), np.zeros((0, dim)), np.zeros(rho.shape, np.int64))
    if mean <= 0:
        return empty
    labels, nlab = ndimage.label(rho > threshold_factor * mean)
    if nlab == 0:
        return empty
    if periodic:
        labels = _merge_periodic(labels)
    nlab = int(labels.max())
    idx = np.arange(1, nlab + 1)
    counts = _sum_labels(np.ones_like(rho), labels, idx)
    keep = counts >= max(int(min_cells), 1)
    if not keep.any():
        return empty
    lut = np.zeros(nlab + 1, dtype=np.int64)
    lut[idx[keep]] = np.arange(1, int(keep.sum()) + 1)
    labels = lut[labels]
    nk = int(keep.sum())
    idx = np.arange(1, nk + 1)
    counts = counts[keep]
    masses = _sum_labels(rho, labels, idx) * vol
    size = counts * vol
    diameters = 2.0 * np.sqrt(size / np.pi) if dim == 2 else 2.0 * np.cbrt(3.0 * size / (4.0 * np.pi))
    cent = np.zeros((nk, dim))
    for k in range(dim):
        shape = [1] * dim
        shape[k] = n
        coord = np.arange(n, dtype=np.float64).reshape(shape)
        if periodic:
            th = 2.0 * np.pi * coord / n
            c = _sum_labels(rho * np.cos(th), labels, idx)
            s = _sum_labels(rho * np.sin(th), labels, idx)
            cent[:, k] = np.mod(np.arctan2(s, c), 2.0 * np.pi) / (2.0 * np.pi) * box
        else:
            cent[:, k] = _sum_labels(rho * coord, labels, idx) / np.maximum(masses / vol, 1e-300) * cell
    return ClumpCatalog(nk, np.asarray(masses), np.asarray(diameters), cent, labels)


def nn_distance(points: np.ndarray, box) -> np.ndarray:
    """Nearest-neighbour distance of every point (periodic via cKDTree boxsize)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.full(pts.shape[0], np.nan)
    if box is not None:
        pts = np.mod(pts, box)
        pts[pts >= box] -= box
        tree = cKDTree(pts, boxsize=box)
    else:
        tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    return d[:, 1]


# ---------------------------------------------------------------------------
# LSS: field statistics
# ---------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return float('nan')
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else float('nan')


def segregation_index(rho_pos: np.ndarray, rho_neg_abs: np.ndarray) -> float:
    """Pearson correlation of log(1 + rho/mean) of the two species (cells => < 0)."""
    out = []
    for r in (rho_pos, rho_neg_abs):
        r = np.abs(np.asarray(r, dtype=np.float64))
        m = r.mean()
        out.append(np.log1p(r / m) if m > 0 else r)
    return _pearson(out[0], out[1])


def void_fraction(rho: np.ndarray, thr_factor: float = 0.2) -> float:
    rho = np.abs(np.asarray(rho, dtype=np.float64))
    m = rho.mean()
    return float(np.mean(rho < thr_factor * m)) if m > 0 else float('nan')


def power_spectrum(rho: np.ndarray, box: float):
    """Shell-averaged P(k) of the density contrast; k = 0 is excluded."""
    rho = np.asarray(rho, dtype=np.float64)
    dim, n = rho.ndim, rho.shape[0]
    cell = box / n
    m = rho.mean()
    delta = rho / m - 1.0 if m != 0 else rho
    pk = np.abs(np.fft.fftn(delta)) ** 2 * cell ** (2 * dim) / box ** dim
    k2 = 0.0
    for d in range(dim):
        shape = [1] * dim
        shape[d] = n
        kd = 2.0 * np.pi * np.fft.fftfreq(n, d=cell).reshape(shape)
        k2 = k2 + kd * kd
    kmag = np.sqrt(k2)
    dk = 2.0 * np.pi / box
    nb = n // 2
    idx = np.floor(kmag / dk + 0.5).astype(np.int64) - 1
    ok = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[ok], minlength=nb)
    P = np.bincount(idx[ok], weights=pk[ok], minlength=nb) / np.maximum(cnt, 1)
    P[cnt == 0] = np.nan
    return (np.arange(nb) + 1) * dk, P


def peak_scale(k: np.ndarray, pk: np.ndarray) -> float:
    k, pk = np.asarray(k, dtype=np.float64), np.asarray(pk, dtype=np.float64)
    ok = (k > 0) & np.isfinite(pk)
    if not ok.any():
        return float('nan')
    kk, pp = k[ok], pk[ok]
    return float(2.0 * np.pi / kk[int(np.argmax(pp))])


def pattern_persistence(rho_ref: np.ndarray, rho_now: np.ndarray) -> float:
    """Normalised cross-correlation (zero-lag Pearson) of two density grids."""
    a, b = np.asarray(rho_ref), np.asarray(rho_now)
    if a.shape != b.shape:
        return float('nan')
    return _pearson(a, b)


def lss_structure_row(ps: ParticleSet, t: float, box: float, ngrid: int, dim: int,
                      threshold: float, min_cells: int, ref_grid):
    """One structure.csv record and the current (smoothed) negative grid."""
    grids = {}
    for name, mask in (('pos', ps.pos_mask), ('neg', ps.neg_mask)):
        if mask.any():
            rho = density_grid(np.asarray(ps.pos[mask], np.float64), np.asarray(ps.w[mask], np.float64),
                               box, ngrid, dim, periodic=True)
            grids[name] = smooth_grid(rho, 1.0, True)
        else:
            grids[name] = None
    row = {'t': float(t)}
    nan = float('nan')
    cats = {}
    for name in ('neg', 'pos'):
        g = grids[name]
        cats[name] = find_clumps(g, box, threshold, min_cells, True) if g is not None else None
    cn = cats['neg']
    if cn is not None and cn.n > 0:
        row.update(n_clumps_neg=cn.n, mass_mean=float(cn.masses.mean()), mass_std=float(cn.masses.std()),
                   diam_mean=float(cn.diameters.mean()), diam_std=float(cn.diameters.std()),
                   mass_frac_clumps_neg=float(cn.masses.sum() / max(float(ps.w[ps.neg_mask].sum()), 1e-300)))
        dnn = nn_distance(cn.centroids, box)
        row['d_cell'] = float(np.nanmean(dnn)) if np.isfinite(dnn).any() else nan
        row['phi_over_d'] = row['diam_mean'] / row['d_cell'] if row['d_cell'] > 0 else nan
    else:
        row.update(n_clumps_neg=0 if cn is not None else nan, mass_mean=nan, mass_std=nan, diam_mean=nan,
                   diam_std=nan, mass_frac_clumps_neg=nan, d_cell=nan, phi_over_d=nan)
    cp = cats['pos']
    row['n_clumps_pos'] = cp.n if cp is not None else nan
    gp, gn = grids['pos'], grids['neg']
    row['segregation'] = segregation_index(gp, gn) if (gp is not None and gn is not None) else nan
    row['void_frac_pos'] = void_fraction(gp) if gp is not None else nan
    row['void_frac_neg'] = void_fraction(gn) if gn is not None else nan
    for name in ('pos', 'neg'):
        g = grids[name]
        row[f'lambda_peak_{name}'] = peak_scale(*power_spectrum(g, box)) if g is not None else nan
    cur = gn if gn is not None else (gp if gp is not None else np.zeros((ngrid,) * dim))
    row['persistence'] = pattern_persistence(ref_grid, cur) if ref_grid is not None else nan
    return row, cur


# ---------------------------------------------------------------------------
# Galaxy: Fourier modes
# ---------------------------------------------------------------------------

def _polar(pos, centre):
    d = np.asarray(pos, dtype=np.float64)[:, :2] - np.asarray(centre, dtype=np.float64)[:2]
    return np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0]), d


def radial_mass(pos: np.ndarray, w: np.ndarray, centre: np.ndarray, rbins: np.ndarray) -> np.ndarray:
    R, _, _ = _polar(pos, centre)
    nb = len(rbins) - 1
    idx = np.digitize(R, rbins) - 1
    ok = (idx >= 0) & (idx < nb)
    return np.bincount(idx[ok], weights=np.asarray(w, np.float64)[ok], minlength=nb)


def fourier_modes(pos: np.ndarray, w: np.ndarray, centre: np.ndarray, rbins: np.ndarray, mmax: int = 4):
    """A[m, bin] = |sum w e^{i m phi}| / M_bin (A[0] = 1 where the bin has mass),
    phases[m, bin] = arg(sum w e^{i m phi}); nan in empty bins."""
    R, phi, _ = _polar(pos, centre)
    w = np.asarray(w, dtype=np.float64)
    rbins = np.asarray(rbins, dtype=np.float64)
    nb = rbins.size - 1
    idx = np.digitize(R, rbins) - 1
    ok = (idx >= 0) & (idx < nb)
    A = np.full((mmax + 1, nb), np.nan)
    ph = np.full((mmax + 1, nb), np.nan)
    M = np.bincount(idx[ok], weights=w[ok], minlength=nb)
    good = M > 0
    A[0, good], ph[0, good] = 1.0, 0.0
    for m in range(1, mmax + 1):
        c = np.bincount(idx[ok], weights=w[ok] * np.cos(m * phi[ok]), minlength=nb)
        s = np.bincount(idx[ok], weights=w[ok] * np.sin(m * phi[ok]), minlength=nb)
        A[m, good] = np.hypot(c, s)[good] / M[good]
        ph[m, good] = np.arctan2(s, c)[good]
    return A, ph


def _centres(rbins):
    rbins = np.asarray(rbins, dtype=np.float64)
    return 0.5 * (rbins[1:] + rbins[:-1])


def bar_strength(A2: np.ndarray, rbins: np.ndarray, r_inner: float) -> float:
    Rc = _centres(rbins)
    sel = (Rc < r_inner) & np.isfinite(A2)
    return float(np.max(np.asarray(A2)[sel])) if sel.any() else float('nan')


def spiral_strength(A2: np.ndarray, rbins: np.ndarray, r_inner: float, r_outer: float,
                    weights=None) -> float:
    """Weighted mean A2 for r_inner <= R < r_outer; weights default to annulus
    area (disk_structure_row passes the measured bin masses)."""
    rbins = np.asarray(rbins, dtype=np.float64)
    Rc = _centres(rbins)
    wts = np.asarray(weights, np.float64) if weights is not None else np.pi * (rbins[1:] ** 2 - rbins[:-1] ** 2)
    sel = (Rc >= r_inner) & (Rc < r_outer) & np.isfinite(A2) & (wts > 0)
    if not sel.any():
        return float('nan')
    return float(np.sum(wts[sel] * np.asarray(A2)[sel]) / np.sum(wts[sel]))


def pitch_angle(phase2: np.ndarray, rbins: np.ndarray, r_inner: float, r_outer: float,
                A2=None, a_min: float = 0.05) -> float:
    """Pitch angle (deg, unsigned) of the m=2 pattern: theta(R) = phase2/2 unwrapped,
    least-squares slope b = d theta / d ln R, tan(i) = 1/|b|.  nan when fewer than
    3 bins are usable or the mean A2 in the annulus is below a_min."""
    Rc = _centres(rbins)
    phase2 = np.asarray(phase2, dtype=np.float64)
    sel = (Rc >= r_inner) & (Rc < r_outer) & np.isfinite(phase2) & (Rc > 0)
    if A2 is not None:
        a = np.asarray(A2, dtype=np.float64)
        sel &= np.isfinite(a)
        if sel.any() and float(np.mean(a[sel])) < a_min:
            return float('nan')
    if sel.sum() < 3:
        return float('nan')
    theta = 0.5 * np.unwrap(phase2[sel])
    x = np.log(Rc[sel])
    b = np.polyfit(x, theta, 1)[0]
    return float(np.degrees(np.arctan(1.0 / abs(b)))) if b != 0 else 90.0


def pattern_speed(phase_now: float, phase_prev: float, dt: float, m: int = 2) -> float:
    """Omega_p = d(phase_m)/dt / m with the phase difference wrapped to (-pi, pi].
    Unambiguous only while |Omega_p| m dt < pi (aliased otherwise)."""
    if dt is None or dt <= 0 or not (np.isfinite(phase_now) and np.isfinite(phase_prev)):
        return float('nan')
    dphi = (phase_now - phase_prev + np.pi) % (2.0 * np.pi) - np.pi
    return float(dphi / (m * dt))


# ---------------------------------------------------------------------------
# Galaxy: kinematics
# ---------------------------------------------------------------------------

def measured_toomre_q(ps: ParticleSet, centre: np.ndarray, rbins: np.ndarray, G: float,
                      box=None) -> np.ndarray:
    """Q = sigma_R kappa / (3.36 G Sigma) per radial bin for species 0, with sigma_R
    the measured radial dispersion, kappa from the measured mean v_phi
    (kappa^2 = 4 Omega^2 + R dOmega^2/dR) and Sigma the surface density."""
    m = ps.species == 0
    pos = np.asarray(ps.pos[m], np.float64)[:, :2] - np.asarray(centre, np.float64)[:2]
    if box is not None:
        pos = min_image(pos, box)
    vel = np.asarray(ps.vel[m], np.float64)[:, :2]
    w = np.asarray(ps.w[m], np.float64)
    rbins = np.asarray(rbins, dtype=np.float64)
    nb = rbins.size - 1
    R = np.maximum(np.hypot(pos[:, 0], pos[:, 1]), 1e-12)
    vR = (vel[:, 0] * pos[:, 0] + vel[:, 1] * pos[:, 1]) / R
    vphi = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / R
    idx = np.digitize(R, rbins) - 1
    ok = (idx >= 0) & (idx < nb)
    i, ww = idx[ok], w[ok]
    cnt = np.bincount(i, minlength=nb)
    M = np.bincount(i, weights=ww, minlength=nb)
    Ms = np.maximum(M, 1e-300)
    vphi_m = np.bincount(i, weights=ww * vphi[ok], minlength=nb) / Ms
    vR_m = np.bincount(i, weights=ww * vR[ok], minlength=nb) / Ms
    sigR = np.sqrt(np.maximum(np.bincount(i, weights=ww * (vR[ok] - vR_m[i]) ** 2, minlength=nb) / Ms, 0.0))
    Rc = _centres(rbins)
    Sigma = M / (np.pi * (rbins[1:] ** 2 - rbins[:-1] ** 2))
    om2 = (vphi_m / np.maximum(Rc, 1e-12)) ** 2
    k2 = 4.0 * om2 + (Rc * np.gradient(om2, Rc) if nb > 1 else 0.0)
    Q = np.full(nb, np.nan)
    good = (cnt >= 5) & (Sigma > 0) & (k2 > 0)
    Q[good] = sigR[good] * np.sqrt(k2[good]) / (3.36 * G * Sigma[good])
    return Q


def disk_structure_row(ps: ParticleSet, t: float, centre: np.ndarray, rd: float, rmax: float,
                       box: float, G: float, prev, ring_grid_n: int, threshold: float, min_cells: int):
    """One galaxy structure.csv record and the state for the next call.
    Bar = max A2 inside 2 R_d; spiral = mass-weighted A2 for 2 R_d <= R < rmax;
    escaped = disk particles beyond 1.5 rmax; omega_p from the m=2 phase inside
    2 R_d between successive calls."""
    nan = float('nan')
    centre = np.asarray(centre, dtype=np.float64)
    dim = ps.dim
    use_box = box if (box is not None and box > 0) else None
    rel = np.asarray(ps.pos, np.float64) - centre[None, :dim]
    if use_box is not None:
        rel = min_image(rel, use_box)
    vel = np.asarray(ps.vel, np.float64)
    w = np.asarray(ps.w, np.float64)
    disk, twin = ps.species == 0, ps.species == 1
    zero = np.zeros(2)
    nd = int(disk.sum())
    nb = int(np.clip(nd // 200, 8, 32))
    rbins = np.linspace(0.0, 1.25 * rmax, nb + 1)
    row = {'t': float(t)}
    state = {'t': float(t), 'phase2': nan}
    lz = w * (rel[:, 0] * vel[:, 1] - rel[:, 1] * vel[:, 0])
    row['Lz_disk'] = float(lz[disk].sum()) if nd else nan
    row['Lz_twin_abs'] = float(lz[twin].sum()) if twin.any() else 0.0
    if nd:
        A, ph = fourier_modes(rel[disk], w[disk], zero, rbins)
        Mb = radial_mass(rel[disk], w[disk], zero, rbins)
        row['bar_A2'] = bar_strength(A[2], rbins, 2.0 * rd)
        row['spiral_A2'] = spiral_strength(A[2], rbins, 2.0 * rd, rmax, weights=Mb)
        for m in (1, 3, 4):
            row[f'A{m}'] = spiral_strength(A[m], rbins, 0.0, rmax, weights=Mb)
        row['pitch_deg'] = pitch_angle(ph[2], rbins, 2.0 * rd, rmax, A2=A[2])
        R, phi, _ = _polar(rel[disk], zero)
        inner = R < 2.0 * rd
        if inner.any():
            z = np.sum(w[disk][inner] * np.exp(2j * phi[inner]))
            state['phase2'] = float(np.angle(z))
        row['phase2_deg'] = float(np.degrees(state['phase2'])) if np.isfinite(state['phase2']) else nan
        row['omega_p'] = (pattern_speed(state['phase2'], prev.get('phase2', nan), t - prev.get('t', t))
                          if prev else nan)
        Rs = np.maximum(R, 1e-12)
        vt = (rel[disk, 0] * vel[disk, 1] - rel[disk, 1] * vel[disk, 0]) / Rs
        row['vrot_disk'] = float(np.sum(w[disk] * vt) / w[disk].sum())
        Q = measured_toomre_q(ps, centre, rbins, G, box=use_box)
        row['Q_at_2rd'] = float(Q[int(np.argmin(np.abs(_centres(rbins) - 2.0 * rd)))])
        row['escaped_frac'] = float(np.mean(R > 1.5 * rmax))
        row['r_rms_disk'] = float(np.sqrt(np.sum(w[disk] * R * R) / w[disk].sum()))
    else:
        row.update(bar_A2=nan, spiral_A2=nan, A1=nan, A3=nan, A4=nan, pitch_deg=nan, phase2_deg=nan,
                   omega_p=nan, vrot_disk=nan, Q_at_2rd=nan, escaped_frac=nan, r_rms_disk=nan)
    if twin.any() and use_box is not None and ring_grid_n >= 4:
        rho = density_grid(np.asarray(ps.pos[twin], np.float64), w[twin], use_box, int(ring_grid_n),
                           dim, periodic=True)
        cat = find_clumps(smooth_grid(rho, 1.0, True), use_box, threshold, min_cells, True)
        row['n_clumps_twin'] = cat.n
        row['twin_clump_mass_frac'] = float(cat.masses.sum() / w[twin].sum()) if cat.n else 0.0
        Rt = np.hypot(rel[twin, 0], rel[twin, 1])
        row['r_rms_twin'] = float(np.sqrt(np.sum(w[twin] * Rt * Rt) / w[twin].sum()))
    else:
        row['n_clumps_twin'] = 0 if not twin.any() else nan
        row['twin_clump_mass_frac'] = nan
        row['r_rms_twin'] = nan
    return row, state


if __name__ == '__main__':  # minimal self-test
    rng = np.random.default_rng(0)
    # two Gaussian blobs, one straddling the periodic edge -> 2 clumps, correct centroid
    n = 64
    g = np.zeros((n, n))
    yy, xx = np.indices((n, n))
    for cx, cy in ((10.0, 10.0), (0.5, 40.0)):
        dx = (xx - cx + n / 2) % n - n / 2
        dy = (yy - cy + n / 2) % n - n / 2
        g += np.exp(-(dx * dx + dy * dy) / 8.0)
    cat = find_clumps(g, 1.0, 3.0, 4, True)
    assert cat.n == 2, cat.n
    cx = cat.centroids[np.argmin(np.abs(cat.centroids[:, 1] - 40 / 64.0))]
    assert min(abs(cx[0] - 0.5 / 64), abs(cx[0] - 0.5 / 64 - 1)) < 2.0 / 64, cx
    assert find_clumps(g, 1.0, 3.0, 4, False).n == 3          # edge blob split without merging
    d = nn_distance(cat.centroids, 1.0)
    assert d.shape == (2,) and np.isfinite(d).all()
    # segregation: anti-correlated fields -> negative; persistence of identical grids = 1
    assert segregation_index(g, g.max() - g) < -0.5 and np.isclose(pattern_persistence(g, g), 1.0)
    assert 0.0 <= void_fraction(g) <= 1.0
    # power spectrum of a pure cosine recovers its wavelength
    wave = 1.0 + 0.5 * np.cos(2 * np.pi * 8 * xx / n)
    k, pk = power_spectrum(wave, 1.0)
    assert abs(peak_scale(k, pk) - 1.0 / 8) < 1e-9
    # Fourier modes of a bar: A2 ~ 1 inside, phase = 2 * bar angle
    N = 4000
    R = rng.random(N) * 3.0
    ang = np.where(rng.random(N) < 0.5, 0.3, 0.3 + np.pi) + rng.normal(0, 0.05, N)
    pos = np.column_stack([R * np.cos(ang), R * np.sin(ang)])
    A, ph = fourier_modes(pos, np.ones(N), np.zeros(2), np.linspace(0, 3, 7))
    assert A[2, :].min() > 0.9 and abs(((ph[2, 0] - 0.6) + np.pi) % (2 * np.pi) - np.pi) < 0.1
    assert np.isclose(pattern_speed(0.6, 0.4, 0.1), 1.0) and np.isclose(pattern_speed(-3.0, 3.0, 1.0), 0.5 * (2 * np.pi - 6.0))
    # logarithmic spiral with pitch 20 deg
    theta = np.log(np.linspace(1.0, 4.0, 10)) / np.tan(np.radians(20.0))
    assert abs(pitch_angle(np.angle(np.exp(2j * theta)), np.linspace(0.75, 4.25, 11), 0.0, 5.0) - 20.0) < 0.5
    # full rows on a toy two-species set
    ps = ParticleSet(rng.random((400, 2)), rng.normal(size=(400, 2)), np.ones(400) * 0.01,
                     np.where(np.arange(400) < 200, 1.0, -1.0), (np.arange(400) >= 200).astype(np.int32))
    row, grid = lss_structure_row(ps, 0.0, 1.0, 32, 2, 5.0, 4, None)
    row2, _ = lss_structure_row(ps, 1.0, 1.0, 32, 2, 5.0, 4, grid)
    assert np.isclose(row2['persistence'], 1.0) and 'n_clumps_neg' in row and grid.shape == (32, 32)
    ps2 = ParticleSet(rng.random((400, 2)) * 16, rng.normal(size=(400, 2)), np.ones(400) * 0.01,
                      np.where(np.arange(400) < 200, 1.0, -1.0), (np.arange(400) >= 200).astype(np.int32))
    drow, st = disk_structure_row(ps2, 0.0, np.full(2, 8.0), 1.0, 4.0, 16.0, 1.0, None, 64, 5.0, 4)
    drow2, _ = disk_structure_row(ps2, 0.5, np.full(2, 8.0), 1.0, 4.0, 16.0, 1.0, st, 64, 5.0, 4)
    assert np.isfinite(drow['Lz_disk']) and np.isclose(drow2['omega_p'], 0.0) and 'n_clumps_twin' in drow
    print("structure_stats self-test OK")
