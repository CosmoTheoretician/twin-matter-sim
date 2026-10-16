"""structure_torch.py -- device (PyTorch / ROCm) structure rows of structure_stats.

lss_structure_row_t / disk_structure_row_t have the signatures, the dict keys
(in the same order) and the numerical definitions of
structure_stats.lss_structure_row / disk_structure_row, but take the particles
as a gpu_backend.GPUParticles (or any object with pos/vel/w/s/species tensors)
and keep every per-particle operation on the device:

  * density grids (grids_torch.density_grid_t, CIC) and their one-cell Gaussian
    smoothing, the segregation index, void fractions, power spectra / peak
    scales and the pattern persistence are evaluated on the device; the only
    grid that crosses the bus is the smoothed per-species grid (<= 512^2 or
    128^3 float64) needed by scipy's connected-component labelling in
    structure_stats.find_clumps (clump masses, diameters, centroids);
  * the disk Fourier analysis reduces the particles to per-bin sums (<= 33 bins:
    count, mass, cos/sin moments of m = 1..4, mean v_phi, sigma_R) with
    torch.bincount using the np.digitize convention (bucketize(right=True) - 1);
    those tiny arrays and a dozen scalars are copied once and bar_strength,
    spiral_strength, pitch_angle, pattern_speed and the measured Toomre Q are
    finished with the unchanged structure_stats / numpy code.

Everything is computed in float64 (the CPU code casts to float64 everywhere),
so a float64 particle state reproduces the CPU rows to round-off.  The nan rules
are identical: no negative species -> n_clumps_neg nan, no disk -> nan row, no
twin -> n_clumps_twin = 0.  The returned LSS grid `cur` is a device tensor, so
the persistence-reference logic of scenarios.run_lss works unchanged.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    _TORCH_ERR = ''
except Exception as _exc:            # ImportError or a broken runtime
    torch = None
    _TORCH_ERR = repr(_exc)

from structure_stats import (find_clumps, nn_distance, bar_strength, spiral_strength, pitch_angle,
                             pattern_speed, _centres)
from grids_torch import (density_grid_t, gaussian_filter_t, segregation_index_t, void_fraction_t,
                         power_spectrum_t, peak_scale_t, pattern_persistence_t, min_image_t)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _need_torch() -> None:
    if torch is None:
        raise RuntimeError(f"structure_torch needs PyTorch (import failed: {_TORCH_ERR})")


def _f64(t):
    return t.detach().to(torch.float64)


def _vec(x, device, n: int):
    """Host array or device tensor -> float64 tensor (n,) on `device`."""
    if torch.is_tensor(x):
        v = x.detach().to(device=device, dtype=torch.float64).reshape(-1)
    else:
        v = torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(-1), dtype=torch.float64,
                            device=device)
    return v[:n]


def _host(tensors) -> list:
    """Stack 0-d device scalars and copy them to the host in one transfer."""
    if not tensors:
        return []
    st = torch.stack([x.detach().to(torch.float64).reshape(()) for x in tensors])
    return [float(v) for v in st.cpu().numpy()]


def _div(a, b) -> float:
    """float64 division with numpy semantics (inf / nan instead of an exception)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.float64(a) / np.float64(b))


# ---------------------------------------------------------------------------
# LSS row
# ---------------------------------------------------------------------------

def lss_structure_row_t(gp, t: float, box: float, ngrid: int, dim: int, threshold: float,
                        min_cells: int, ref_grid):
    """One structure.csv record (keys/order of structure_stats.lss_structure_row)
    and the current smoothed negative grid as a float64 device tensor."""
    _need_torch()
    box, ngrid, dim = float(box), int(ngrid), int(dim)
    nan = float('nan')
    pos, w = _f64(gp.pos), _f64(gp.w)
    dev = pos.device
    masks = {'pos': gp.s > 0, 'neg': gp.s < 0}
    counts = torch.stack([masks['pos'].sum(), masks['neg'].sum()]).cpu().numpy()
    grids = {}
    for k, name in enumerate(('pos', 'neg')):
        if int(counts[k]) > 0:
            rho = density_grid_t(pos, w * masks[name].to(torch.float64), box, ngrid, dim, periodic=True)
            grids[name] = gaussian_filter_t(rho, 1.0, True)
        else:
            grids[name] = None
    g_pos, g_neg = grids['pos'], grids['neg']
    nan_t = torch.full((), nan, dtype=torch.float64, device=dev)
    cur = g_neg if g_neg is not None else (
        g_pos if g_pos is not None else torch.zeros((ngrid,) * dim, dtype=torch.float64, device=dev))
    scal = [(w * masks['neg'].to(torch.float64)).sum(),
            segregation_index_t(g_pos, g_neg) if (g_pos is not None and g_neg is not None) else nan_t,
            void_fraction_t(g_pos) if g_pos is not None else nan_t,
            void_fraction_t(g_neg) if g_neg is not None else nan_t,
            pattern_persistence_t(ref_grid, cur) if ref_grid is not None else nan_t]
    w_neg_sum, seg, vf_pos, vf_neg, pers = _host(scal)
    lam = {name: (peak_scale_t(*power_spectrum_t(g, box)) if g is not None else nan)
           for name, g in grids.items()}
    # the only grid transfer: connected-component labelling on the host
    cats = {name: (find_clumps(g.cpu().numpy(), box, threshold, min_cells, True) if g is not None else None)
            for name, g in grids.items()}
    row = {'t': float(t)}
    cn = cats['neg']
    if cn is not None and cn.n > 0:
        row.update(n_clumps_neg=cn.n, mass_mean=float(cn.masses.mean()), mass_std=float(cn.masses.std()),
                   diam_mean=float(cn.diameters.mean()), diam_std=float(cn.diameters.std()),
                   mass_frac_clumps_neg=float(cn.masses.sum() / max(w_neg_sum, 1e-300)))
        dnn = nn_distance(cn.centroids, box)
        row['d_cell'] = float(np.nanmean(dnn)) if np.isfinite(dnn).any() else nan
        row['phi_over_d'] = row['diam_mean'] / row['d_cell'] if row['d_cell'] > 0 else nan
    else:
        row.update(n_clumps_neg=0 if cn is not None else nan, mass_mean=nan, mass_std=nan, diam_mean=nan,
                   diam_std=nan, mass_frac_clumps_neg=nan, d_cell=nan, phi_over_d=nan)
    cp = cats['pos']
    row['n_clumps_pos'] = cp.n if cp is not None else nan
    row['segregation'] = seg
    row['void_frac_pos'] = vf_pos
    row['void_frac_neg'] = vf_neg
    row['lambda_peak_pos'] = lam['pos']
    row['lambda_peak_neg'] = lam['neg']
    row['persistence'] = pers
    return row, cur


# ---------------------------------------------------------------------------
# galaxy row
# ---------------------------------------------------------------------------

def disk_structure_row_t(gp, t: float, centre, rd: float, rmax: float, box, G: float, prev,
                         ring_grid_n: int, threshold: float, min_cells: int):
    """One galaxy structure.csv record (keys/order of structure_stats.disk_structure_row)
    and the state for the next call.  Bar = max A2 inside 2 R_d; spiral = mass-weighted
    A2 for 2 R_d <= R < rmax; escaped = disk particles beyond 1.5 rmax; omega_p from
    the m=2 phase inside 2 R_d between successive calls."""
    _need_torch()
    nan = float('nan')
    rd, rmax, G = float(rd), float(rmax), float(G)
    pos, vel, w = _f64(gp.pos), _f64(gp.vel), _f64(gp.w)
    dev = pos.device
    dim = int(pos.shape[1])
    use_box = float(box) if (box is not None and box > 0) else None
    rel = pos - _vec(centre, dev, dim)[None, :]
    if use_box is not None:
        rel = min_image_t(rel, use_box)
    disk_b, twin_b = gp.species == 0, gp.species == 1
    disk, twin = disk_b.to(torch.float64), twin_b.to(torch.float64)
    counts = torch.stack([disk_b.sum(), twin_b.sum()]).cpu().numpy()
    nd, nt = int(counts[0]), int(counts[1])
    nb = int(np.clip(nd // 200, 8, 32))
    rbins = np.linspace(0.0, 1.25 * rmax, nb + 1)
    row = {'t': float(t)}
    state = {'t': float(t), 'phase2': nan}
    x, y, vx, vy = rel[:, 0], rel[:, 1], vel[:, 0], vel[:, 1]
    lz = w * (x * vy - y * vx)
    scal = [(lz * disk).sum(), (lz * twin).sum()]
    bins = None
    if nd:
        R = torch.hypot(x, y)
        phi = torch.atan2(y, x)
        Rs = torch.clamp(R, min=1e-12)
        vR = (vx * x + vy * y) / Rs
        vphi = (x * vy - y * vx) / Rs
        wd = w * disk
        rbins_t = torch.as_tensor(rbins, dtype=torch.float64, device=dev)
        idx = torch.bucketize(R, rbins_t, right=True) - 1          # == np.digitize(R, rbins) - 1
        ok = (idx >= 0) & (idx < nb) & disk_b
        idx_c = torch.where(ok, idx, torch.full_like(idx, nb))     # bin nb = discarded
        idx_g = torch.clamp(idx_c, max=nb - 1)

        def bc(wt):
            return torch.bincount(idx_c, weights=wt, minlength=nb + 1)[:nb]

        cnt = torch.bincount(idx_c, minlength=nb + 1)[:nb].to(torch.float64)
        M = bc(wd)
        per_bin = [cnt, M]
        for m in (1, 2, 3, 4):
            per_bin.append(bc(wd * torch.cos(m * phi)))
            per_bin.append(bc(wd * torch.sin(m * phi)))
        Ms = torch.clamp(M, min=1e-300)
        vphi_m = bc(wd * vphi) / Ms
        vR_m = bc(wd * vR) / Ms
        sigR = torch.sqrt(torch.clamp(bc(wd * (vR - vR_m[idx_g]) ** 2) / Ms, min=0.0))
        per_bin += [vphi_m, sigR]
        bins = torch.stack(per_bin, 0)
        inner = (disk_b & (R < 2.0 * rd)).to(torch.float64)
        scal += [inner.sum(), (wd * inner * torch.cos(2.0 * phi)).sum(),
                 (wd * inner * torch.sin(2.0 * phi)).sum(), (wd * vphi).sum(), wd.sum(),
                 (disk_b & (R > 1.5 * rmax)).sum(), (wd * R * R).sum()]
    if nt:
        wt = w * twin
        Rt = torch.hypot(x, y)
        scal += [wt.sum(), (wt * Rt * Rt).sum()]
    vals = _host(scal)                                      # one transfer of the scalars
    row['Lz_disk'] = vals[0] if nd else nan
    row['Lz_twin_abs'] = vals[1] if nt else 0.0
    k = 2
    if nd:
        n_inner, zc, zs, vrot_num, wd_sum, n_esc, rr_num = vals[2:9]
        k = 9
        B = bins.cpu().numpy()                              # one transfer of the <= 33-entry bins
        cnt, M = B[0], B[1]
        A = np.full((5, nb), np.nan)
        ph = np.full((5, nb), np.nan)
        good = M > 0
        A[0, good], ph[0, good] = 1.0, 0.0
        for m in (1, 2, 3, 4):
            c, s = B[2 * m], B[2 * m + 1]
            A[m, good] = np.hypot(c, s)[good] / M[good]
            ph[m, good] = np.arctan2(s, c)[good]
        vphi_m, sigR = B[10], B[11]
        row['bar_A2'] = bar_strength(A[2], rbins, 2.0 * rd)
        row['spiral_A2'] = spiral_strength(A[2], rbins, 2.0 * rd, rmax, weights=M)
        for m in (1, 3, 4):
            row[f'A{m}'] = spiral_strength(A[m], rbins, 0.0, rmax, weights=M)
        row['pitch_deg'] = pitch_angle(ph[2], rbins, 2.0 * rd, rmax, A2=A[2])
        if n_inner > 0:
            state['phase2'] = float(np.arctan2(zs, zc))
        row['phase2_deg'] = float(np.degrees(state['phase2'])) if np.isfinite(state['phase2']) else nan
        row['omega_p'] = (pattern_speed(state['phase2'], prev.get('phase2', nan), t - prev.get('t', t))
                          if prev else nan)
        row['vrot_disk'] = _div(vrot_num, wd_sum)
        Rc = _centres(rbins)
        Sigma = M / (np.pi * (rbins[1:] ** 2 - rbins[:-1] ** 2))
        om2 = (vphi_m / np.maximum(Rc, 1e-12)) ** 2
        k2 = 4.0 * om2 + (Rc * np.gradient(om2, Rc) if nb > 1 else 0.0)
        Q = np.full(nb, np.nan)
        goodq = (cnt >= 5) & (Sigma > 0) & (k2 > 0)
        Q[goodq] = sigR[goodq] * np.sqrt(k2[goodq]) / (3.36 * G * Sigma[goodq])
        row['Q_at_2rd'] = float(Q[int(np.argmin(np.abs(Rc - 2.0 * rd)))])
        row['escaped_frac'] = n_esc / nd
        row['r_rms_disk'] = float(np.sqrt(_div(rr_num, wd_sum)))
    else:
        row.update(bar_A2=nan, spiral_A2=nan, A1=nan, A3=nan, A4=nan, pitch_deg=nan, phase2_deg=nan,
                   omega_p=nan, vrot_disk=nan, Q_at_2rd=nan, escaped_frac=nan, r_rms_disk=nan)
    if nt and use_box is not None and int(ring_grid_n) >= 4:
        wt_sum, rt_num = vals[k], vals[k + 1]
        rho = density_grid_t(pos, w * twin, use_box, int(ring_grid_n), dim, periodic=True)
        cat = find_clumps(gaussian_filter_t(rho, 1.0, True).cpu().numpy(), use_box, threshold,
                          min_cells, True)
        row['n_clumps_twin'] = cat.n
        row['twin_clump_mass_frac'] = _div(cat.masses.sum(), wt_sum) if cat.n else 0.0
        row['r_rms_twin'] = float(np.sqrt(_div(rt_num, wt_sum)))
    else:
        row['n_clumps_twin'] = 0 if not nt else nan
        row['twin_clump_mass_frac'] = nan
        row['r_rms_twin'] = nan
    return row, state


# ---------------------------------------------------------------------------
# self-test: agreement with structure_stats on random and clumpy sets
# ---------------------------------------------------------------------------

def _close(a, b, tol: float = 1e-10) -> bool:
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b or abs(a - b) <= tol * max(abs(a), abs(b))


def _check(row_c: dict, row_t: dict, label: str, tol: float = 1e-10) -> None:
    assert list(row_c.keys()) == list(row_t.keys()), (label, list(row_c), list(row_t))
    for key in row_c:
        assert _close(row_c[key], row_t[key], tol), (label, key, row_c[key], row_t[key])
        assert type(row_t[key]) in (int, float), (label, key, type(row_t[key]))


def _lss_set(rng, n: int, dim: int, clumpy: bool):
    """N/2 positive uniform + N/2 negative (clumps, one straddling the box edge)."""
    from particles import ParticleSet
    pos = rng.random((n, dim))
    if clumpy:
        h = n // 2
        k = 8
        cen = rng.random((k, dim))
        cen[0] = 0.0
        pos[h:] = np.mod(cen[np.arange(n - h) % k] + 0.03 * rng.normal(size=(n - h, dim)), 1.0)
    vel = rng.normal(size=(n, dim))
    s = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    w = np.where(s > 0, 1.0 / n, 64.0 / n)
    return ParticleSet(pos, vel, w, s, (s < 0).astype(np.int32))


def _disk_set(rng, n_disk: int, n_twin: int, dim: int, box: float, centre, rd=1.0, rmax=4.0,
              rin=4.5, rout=7.0):
    """Barred exponential disk (species 0) + twin annulus (species 1) around `centre`,
    wrapped into [0, box) so that the ring straddles the periodic edges."""
    from particles import ParticleSet, concat, empty
    c = np.asarray(centre, dtype=np.float64)
    parts = []
    if n_disk:
        R = np.minimum(rd * (-np.log(rng.random(n_disk)) - np.log(rng.random(n_disk))), rmax)
        bar = rng.random(n_disk) < 0.3
        ang = np.where(bar, 0.5 + 0.25 * rng.normal(size=n_disk), 2.0 * np.pi * rng.random(n_disk))
        ang += np.where(bar & (rng.random(n_disk) < 0.5), np.pi, 0.0) + 0.3 * np.log(R + 0.1)
        p = np.zeros((n_disk, dim))
        p[:, 0], p[:, 1] = R * np.cos(ang), R * np.sin(ang)
        vt = 0.8 * R / np.sqrt(R * R + 0.5) + 0.1 * rng.normal(size=n_disk)
        vr = 0.1 * rng.normal(size=n_disk)
        v = np.zeros((n_disk, dim))
        v[:, 0] = vr * np.cos(ang) - vt * np.sin(ang)
        v[:, 1] = vr * np.sin(ang) + vt * np.cos(ang)
        if dim == 3:
            p[:, 2] = 0.1 * rng.normal(size=n_disk)
            v[:, 2] = 0.05 * rng.normal(size=n_disk)
        parts.append(ParticleSet(np.mod(p + c, box), v, np.full(n_disk, 1.0 / n_disk), np.ones(n_disk),
                                 np.zeros(n_disk, np.int32)))
    if n_twin:
        r = np.sqrt(rin ** 2 + rng.random(n_twin) * (rout ** 2 - rin ** 2))
        ang = 2.0 * np.pi * rng.random(n_twin) + 0.4 * rng.normal(size=n_twin) * (rng.random(n_twin) < 0.5)
        p = np.zeros((n_twin, dim))
        p[:, 0], p[:, 1] = r * np.cos(ang), r * np.sin(ang)
        if dim == 3:
            p[:, 2] = 0.3 * rng.normal(size=n_twin)
        parts.append(ParticleSet(np.mod(p + c, box), 0.3 * rng.normal(size=(n_twin, dim)),
                                 np.full(n_twin, 1.0 / n_twin), -np.ones(n_twin), np.ones(n_twin, np.int32)))
    if not parts:
        return empty(dim)
    ps = parts[0]
    for q in parts[1:]:
        ps = concat(ps, q)
    return ps


if __name__ == '__main__':
    import sys
    if torch is None:
        print("torch not installed, structure_torch self-test skipped:", _TORCH_ERR)
        sys.exit(0)
    from particles import ParticleSet, subset
    from gpu_backend import GPUParticles
    from structure_stats import lss_structure_row, disk_structure_row
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu':
        print("no GPU: checking the torch structure rows on the CPU device")
    rng = np.random.default_rng(0)
    up = lambda ps: GPUParticles.from_host(ps, dev, 'float64')  # noqa: E731
    # --- LSS: the structure_stats self-test set ------------------------------------
    ps = ParticleSet(rng.random((400, 2)), rng.normal(size=(400, 2)), np.ones(400) * 0.01,
                     np.where(np.arange(400) < 200, 1.0, -1.0), (np.arange(400) >= 200).astype(np.int32))
    row_c, grid_c = lss_structure_row(ps, 0.0, 1.0, 32, 2, 5.0, 4, None)
    row_t, grid_t = lss_structure_row_t(up(ps), 0.0, 1.0, 32, 2, 5.0, 4, None)
    _check(row_c, row_t, 'lss self-test set')
    assert torch.is_tensor(grid_t) and grid_t.dtype == torch.float64 and tuple(grid_t.shape) == (32, 32)
    assert np.abs(grid_t.cpu().numpy() - grid_c).max() < 1e-12 * np.abs(grid_c).max()
    row2_c, _ = lss_structure_row(ps, 1.0, 1.0, 32, 2, 5.0, 4, grid_c)
    row2_t, _ = lss_structure_row_t(up(ps), 1.0, 1.0, 32, 2, 5.0, 4, grid_t)
    _check(row2_c, row2_t, 'lss self-test set, persistence')
    assert _close(row2_t['persistence'], 1.0, 1e-12)
    row2n_t, _ = lss_structure_row_t(up(ps), 1.0, 1.0, 32, 2, 5.0, 4, grid_c)   # numpy reference grid
    _check(row2_c, row2n_t, 'lss numpy ref grid')
    # --- LSS: clumpy random sets in 2-D and 3-D, single-species controls ---------------
    for dim, ngrid in ((2, 64), (3, 24)):
        for clumpy in (False, True):
            psl = _lss_set(rng, 6000, dim, clumpy)
            gpl = up(psl)
            rc, gc = lss_structure_row(psl, 0.3, 1.0, ngrid, dim, 5.0, 4, None)
            rt, gt = lss_structure_row_t(gpl, 0.3, 1.0, ngrid, dim, 5.0, 4, None)
            _check(rc, rt, f'lss {dim}d clumpy={clumpy}')
            if clumpy:
                assert rt['n_clumps_neg'] >= 2 and np.isfinite(rt['d_cell']) and rt['segregation'] < 0.5
            rc2, _ = lss_structure_row(psl, 0.8, 1.0, ngrid, dim, 5.0, 4, gc)
            rt2, _ = lss_structure_row_t(gpl, 0.8, 1.0, ngrid, dim, 5.0, 4, gt)
            _check(rc2, rt2, f'lss {dim}d clumpy={clumpy} persistence')
            assert _close(rt2['persistence'], 1.0, 1e-12)
            # a reference of another shape -> nan persistence, like the CPU
            rt3, _ = lss_structure_row_t(gpl, 0.8, 1.0, ngrid, dim, 5.0, 4, gt.reshape(-1)[:-1])
            assert math.isnan(rt3['persistence'])
        psl = _lss_set(rng, 3000, dim, True)
        for mask, label in ((psl.s > 0, 'pos only'), (psl.s < 0, 'neg only')):
            sub = subset(psl, mask)
            rc, gc = lss_structure_row(sub, 0.0, 1.0, ngrid, dim, 5.0, 4, None)
            rt, gt = lss_structure_row_t(up(sub), 0.0, 1.0, ngrid, dim, 5.0, 4, None)
            _check(rc, rt, f'lss {dim}d {label}')
            assert math.isnan(rt['segregation']) and tuple(gt.shape) == (ngrid,) * dim
            if label == 'pos only':
                assert math.isnan(rt['n_clumps_neg']) and isinstance(rt['n_clumps_pos'], int)
            else:
                assert math.isnan(rt['n_clumps_pos']) and math.isnan(rt['void_frac_pos'])
    # --- galaxy: the structure_stats self-test set --------------------------------------
    ps2 = ParticleSet(rng.random((400, 2)) * 16, rng.normal(size=(400, 2)), np.ones(400) * 0.01,
                      np.where(np.arange(400) < 200, 1.0, -1.0), (np.arange(400) >= 200).astype(np.int32))
    c8 = np.full(2, 8.0)
    dc, st_c = disk_structure_row(ps2, 0.0, c8, 1.0, 4.0, 16.0, 1.0, None, 64, 5.0, 4)
    dt_, st_t = disk_structure_row_t(up(ps2), 0.0, c8, 1.0, 4.0, 16.0, 1.0, None, 64, 5.0, 4)
    _check(dc, dt_, 'disk self-test set')
    assert set(st_t) == {'t', 'phase2'} and _close(st_c['phase2'], st_t['phase2'])
    dc2, _ = disk_structure_row(ps2, 0.5, c8, 1.0, 4.0, 16.0, 1.0, st_c, 64, 5.0, 4)
    dt2, _ = disk_structure_row_t(up(ps2), 0.5, c8, 1.0, 4.0, 16.0, 1.0, st_t, 64, 5.0, 4)
    _check(dc2, dt2, 'disk self-test set, omega_p')
    assert np.isclose(dt2['omega_p'], 0.0) and np.isfinite(dt2['Lz_disk'])
    # --- galaxy: barred disk + twin ring straddling the box edges, 2-D and 3-D ----------
    for dim in (2, 3):
        box = 16.0
        centre = np.full(dim, 2.0)
        psd = _disk_set(rng, 5000, 3000, dim, box, centre)
        gpd = up(psd)
        prev_c = prev_t = None
        for tt in (0.0, 0.25):
            rc, prev_c = disk_structure_row(psd, tt, centre, 1.0, 4.0, box, 1.0, prev_c, 64, 5.0, 4)
            rt, prev_t = disk_structure_row_t(gpd, tt, centre, 1.0, 4.0, box, 1.0, prev_t, 64, 5.0, 4)
            _check(rc, rt, f'disk {dim}d t={tt}')
            assert _close(prev_c['phase2'], prev_t['phase2'])
        assert rt['bar_A2'] > 0.1 and np.isfinite(rt['Q_at_2rd']) and np.isfinite(rt['pitch_deg'])
        assert isinstance(rt['n_clumps_twin'], int) and rt['n_clumps_twin'] >= 1 and np.isfinite(rt['r_rms_twin'])
        assert np.isfinite(rt['omega_p']) and rt['vrot_disk'] > 0.3 and rt['Lz_twin_abs'] != 0.0
        # tensor centre (as returned by diagnostics_torch.centre_of_mass_t)
        rt_c, _ = disk_structure_row_t(gpd, 0.0, torch.as_tensor(centre, device=dev), 1.0, 4.0, box, 1.0,
                                       None, 64, 5.0, 4)
        rc_c, _ = disk_structure_row(psd, 0.0, centre, 1.0, 4.0, box, 1.0, None, 64, 5.0, 4)
        _check(rc_c, rt_c, f'disk {dim}d tensor centre')
        # isolated (no box): no min-image, no twin clump grid -> nan
        c_abs = np.full(dim, 8.0)
        psi = _disk_set(rng, 3000, 2000, dim, 1e9, c_abs)
        rc_i, _ = disk_structure_row(psi, 0.0, c_abs, 1.0, 4.0, None, 1.0, None, 64, 5.0, 4)
        rt_i, _ = disk_structure_row_t(up(psi), 0.0, c_abs, 1.0, 4.0, None, 1.0, None, 64, 5.0, 4)
        _check(rc_i, rt_i, f'disk {dim}d isolated')
        assert math.isnan(rt_i['n_clumps_twin']) and math.isnan(rt_i['r_rms_twin'])
        # ring grid too small -> nan twin clumps (CPU rule)
        rc_s, _ = disk_structure_row(psd, 0.0, centre, 1.0, 4.0, box, 1.0, None, 2, 5.0, 4)
        rt_s, _ = disk_structure_row_t(gpd, 0.0, centre, 1.0, 4.0, box, 1.0, None, 2, 5.0, 4)
        _check(rc_s, rt_s, f'disk {dim}d tiny ring grid')
        # no twin -> n_clumps_twin = 0, Lz_twin_abs = 0
        psn = _disk_set(rng, 2500, 0, dim, box, centre)
        rc_n, _ = disk_structure_row(psn, 0.0, centre, 1.0, 4.0, box, 1.0, None, 64, 5.0, 4)
        rt_n, _ = disk_structure_row_t(up(psn), 0.0, centre, 1.0, 4.0, box, 1.0, None, 64, 5.0, 4)
        _check(rc_n, rt_n, f'disk {dim}d no twin')
        assert rt_n['n_clumps_twin'] == 0 and rt_n['Lz_twin_abs'] == 0.0 and math.isnan(rt_n['r_rms_twin'])
        # no disk -> nan row, twin statistics still defined
        pst = _disk_set(rng, 0, 2000, dim, box, centre)
        rc_e, _ = disk_structure_row(pst, 0.0, centre, 1.0, 4.0, box, 1.0, None, 64, 5.0, 4)
        rt_e, _ = disk_structure_row_t(up(pst), 0.0, centre, 1.0, 4.0, box, 1.0, None, 64, 5.0, 4)
        _check(rc_e, rt_e, f'disk {dim}d no disk')
        assert math.isnan(rt_e['Lz_disk']) and math.isnan(rt_e['bar_A2']) and np.isfinite(rt_e['r_rms_twin'])
        # float32 particle state: same keys, finite values close to the float64 row
        r32, _ = disk_structure_row_t(GPUParticles.from_host(psd, dev, 'float32'), 0.0, centre, 1.0, 4.0,
                                      box, 1.0, None, 64, 5.0, 4)
        assert list(r32) == list(rc_c) and abs(r32['Lz_disk'] - rc_c['Lz_disk']) < 1e-4 * abs(rc_c['Lz_disk'])
    print("structure_torch self-test OK")
