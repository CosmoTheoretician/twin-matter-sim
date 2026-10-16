"""initial_conditions.py -- initial conditions for both scenarios.

LSS (A4-A7): two uniform populations in a periodic box (Poisson, glass or
lattice positions), Maxwellian velocities with the dispersions derived in
config (sigma_- = vel_ratio * sigma_+), species 0 = ordinary (+), 1 = twin (-).
Single-population controls honour cfg.neg_only / cfg.pos_only.

Galaxy (A11-A12): an exponential disk of ordinary matter (2-D, or 3-D with a
sech^2 vertical profile) plus a twin ring (2-D annulus / 3-D shell) of negative
mass.  The disk rotation is NOT analytic: the circular speed is measured from
the azimuthally averaged radial acceleration of the actual initial force field
(disk self-gravity + live twin ring or rigid control field).  Where the net
radial force points outward v_c is undefined; those particles get no rotation
and the fraction is reported, never patched.  Dispersions follow Toomre Q
(sigma_R = 3.36 G Sigma Q / kappa, sigma_phi = sigma_R kappa / 2 Omega) with the
razor-thin asymmetric-drift correction.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from config import RunConfig, Derived
from particles import (ParticleSet, wrap_periodic, min_image, concat, empty,
                       remove_bulk_velocity)
from integrator import ForceEvaluator

TWIN_SHELL_AXIS_RATIO = 1.0     # 3-D twin shell c/a (pages sketch ellipsoids, no ratio given)
_MIN_PER_BIN = 5                # minimum particles per radial bin for a defined v_c


def _say(cfg, msg: str) -> None:
    if not getattr(cfg, 'quiet', False):
        print("[ic] " + msg, file=sys.stderr)


def _rel(pos: np.ndarray, centre: np.ndarray, box) -> np.ndarray:
    d = np.asarray(pos, dtype=np.float64) - np.asarray(centre, dtype=np.float64)[None, :]
    return min_image(d, box) if box is not None else d


# ---------------------------------------------------------------------------
# position generators (LSS)
# ---------------------------------------------------------------------------

def poisson_positions(n: int, dim: int, box: float, rng) -> np.ndarray:
    return rng.random((int(n), int(dim))) * box


def lattice_positions(n: int, dim: int, box: float, rng, noise: float) -> np.ndarray:
    """Nearest square/cubic lattice (cell-centred) with Gaussian jitter of
    noise*spacing per coordinate.  n is rounded to a perfect power and reported."""
    m = max(1, int(round(n ** (1.0 / dim))))
    n_adj = m ** dim
    if n_adj != n:
        print(f"[ic] lattice: n={n} adjusted to {n_adj} = {m}^{dim}", file=sys.stderr)
    spacing = box / m
    ax = (np.arange(m) + 0.5) * spacing
    grids = np.meshgrid(*([ax] * dim), indexing='ij')
    pos = np.stack([g.reshape(-1) for g in grids], axis=1)
    if noise > 0:
        pos = pos + rng.normal(0.0, noise * spacing, pos.shape)
    return wrap_periodic(pos, box)


def glass_positions(n: int, dim: int, box: float, rng, n_iter: int = 30,
                    noise: float = 0.0) -> np.ndarray:
    """Repulsive relaxation of a Poisson set (cKDTree, periodic), then jitter."""
    from scipy.spatial import cKDTree
    pos = poisson_positions(n, dim, box, rng)
    if n < 2:
        return pos
    spacing = box / n ** (1.0 / dim)
    k = int(min(n - 1, 8 if dim == 2 else 14))
    for it in range(int(n_iter)):
        step = spacing * (0.02 + 0.2 * (1.0 - it / max(n_iter, 1)))
        tree = cKDTree(pos, boxsize=box)
        dist, idx = tree.query(pos, k=k + 1)
        d = min_image(pos[:, None, :] - pos[idx[:, 1:]], box)              # (n,k,dim)
        r = np.maximum(dist[:, 1:], 1e-3 * spacing)[..., None]
        disp = ((d / r) * (spacing / r) ** dim).sum(axis=1)
        norm = np.sqrt((disp * disp).sum(axis=1))
        scale = step / max(float(np.mean(norm)), 1e-30)
        move = disp * np.minimum(scale, 2.0 * step / np.maximum(norm, 1e-30))[:, None]
        pos = wrap_periodic(pos + move, box)
    if noise > 0:
        pos = wrap_periodic(pos + rng.normal(0.0, noise * spacing, pos.shape), box)
    return pos


def maxwellian_velocities(n: int, dim: int, sigma: float, rng) -> np.ndarray:
    """Gaussian per component with dispersion sigma, mean removed per component."""
    n, dim = int(n), int(dim)
    if n == 0:
        return np.zeros((0, dim))
    v = rng.normal(0.0, 1.0, (n, dim)) * float(sigma)
    return v - v.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# LSS
# ---------------------------------------------------------------------------

def _lss_positions(cfg: RunConfig, n: int, dim: int, box: float, rng, offset: float):
    if cfg.ic == 'poisson':
        return poisson_positions(n, dim, box, rng)
    if cfg.ic == 'lattice':
        p = lattice_positions(n, dim, box, rng, cfg.ic_noise)
        if offset:                       # second lattice shifted by half a cell: no coincident points
            spacing = box / p.shape[0] ** (1.0 / dim)
            p = wrap_periodic(p + offset * spacing, box)
        return p
    if cfg.ic == 'glass':
        return glass_positions(n, dim, box, rng, noise=cfg.ic_noise)
    raise ValueError(f"unknown ic {cfg.ic!r}")


def make_lss_ic(cfg: RunConfig, der: Derived, rng) -> ParticleSet:
    """Species 0 (positive, w_pos, sigma_pos) then species 1 (negative, w_neg,
    sigma_neg); net |m|-weighted velocity removed.  Total mass of each population
    is preserved if the lattice rounding changes N."""
    dim, box = der.dim, der.box
    parts = []
    plan = []
    if not cfg.neg_only:
        plan.append((0, cfg.npos, 1.0, der.w_pos * cfg.npos, der.sigma_pos, 0.0))
    if not cfg.pos_only:
        plan.append((1, cfg.nneg, -1.0, der.w_neg * cfg.nneg, der.sigma_neg, 0.5 if plan else 0.0))
    for species, n, sign, mass_total, sigma, offset in plan:
        if n <= 0:
            continue
        pos = _lss_positions(cfg, n, dim, box, rng, offset)
        n_act = pos.shape[0]
        vel = maxwellian_velocities(n_act, dim, sigma, rng)
        w = np.full(n_act, mass_total / n_act)
        parts.append(ParticleSet(pos, vel, w, np.full(n_act, sign), np.full(n_act, species, np.int32)))
    if not parts:
        raise ValueError("LSS initial condition has no particles")
    ps = parts[0]
    for p in parts[1:]:
        ps = concat(ps, p)
    remove_bulk_velocity(ps)
    _say(cfg, f"LSS IC: {ps.npos} positive (sigma={der.sigma_pos:.4g}), {ps.nneg} negative "
              f"(sigma={der.sigma_neg:.4g}), ic={cfg.ic}, noise={cfg.ic_noise:g}, dim={dim}")
    return ps.astype(cfg.dtype)


# ---------------------------------------------------------------------------
# galaxy: disk and ring positions
# ---------------------------------------------------------------------------

def sample_exponential_disk(n: int, rd: float, rmax: float, rng) -> np.ndarray:
    """(n, 2) in-plane offsets from the centre: radii by inverse CDF of the
    truncated exponential disk M(<R) ~ 1 - e^-x (1+x), x = R/rd, random azimuths."""
    n = int(n)
    if n == 0:
        return np.zeros((0, 2))
    xmax = rmax / rd if rmax > 0 else 20.0
    xs = np.linspace(0.0, xmax, 4001)
    F = 1.0 - np.exp(-xs) * (1.0 + xs)
    u = rng.random(n) * F[-1]
    x = np.interp(u, F, xs)
    for _ in range(3):                                    # Newton polish of the table inverse
        fx = 1.0 - np.exp(-x) * (1.0 + x) - u
        dfx = x * np.exp(-x)
        x = np.clip(x - np.where(dfx > 1e-12, fx / np.maximum(dfx, 1e-12), 0.0), 0.0, xmax)
    R = rd * x
    phi = 2.0 * np.pi * rng.random(n)
    return np.column_stack([R * np.cos(phi), R * np.sin(phi)])


def make_disk_positions(cfg: RunConfig, der: Derived, rng, centre: np.ndarray) -> ParticleSet:
    """Species 0 exponential disk around `centre`, velocities zero (placeholder).
    3-D: sech^2(z/z0) with z0 = zscale * rd."""
    n, dim, rd = int(cfg.ndisk), der.dim, cfg.rd
    xy = sample_exponential_disk(n, rd, cfg.rmax, rng)
    pos = np.zeros((n, dim))
    pos[:, :2] = xy
    if dim == 3:
        z0 = cfg.zscale * rd
        u = np.clip(rng.random(n), 1e-6, 1.0 - 1e-6)
        pos[:, 2] = z0 * np.arctanh(2.0 * u - 1.0)
    pos = pos + np.asarray(centre, dtype=np.float64)[None, :]
    if cfg.bc == 'periodic':
        pos = wrap_periodic(pos, der.box)
    return ParticleSet(pos, np.zeros((n, dim)), np.full(n, der.w_pos), np.ones(n),
                       np.zeros(n, np.int32))


def make_twin_ring(cfg: RunConfig, der: Derived, rng, centre: np.ndarray) -> ParticleSet:
    """Species 1 negative-mass annulus (2-D, uniform surface density) or
    spheroidal shell (3-D, uniform density, axis ratio TWIN_SHELL_AXIS_RATIO)
    between twin_rin and twin_rout (R_d units).  Velocities: Maxwellian with
    sigma = twin_sigma * vc_ref plus twin_rot * vc_ref tangential (about z)."""
    n, dim = int(cfg.ntwin), der.dim
    if n <= 0 or cfg.twin_ratio <= 0:
        return empty(dim)
    rin, rout = cfg.twin_rin * cfg.rd, cfg.twin_rout * cfg.rd
    u = rng.random(n)
    if dim == 2:
        r = np.sqrt(rin ** 2 + u * (rout ** 2 - rin ** 2))
        phi = 2.0 * np.pi * rng.random(n)
        off = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    else:
        r = np.cbrt(rin ** 3 + u * (rout ** 3 - rin ** 3))
        cz = 2.0 * rng.random(n) - 1.0
        sz = np.sqrt(1.0 - cz * cz)
        phi = 2.0 * np.pi * rng.random(n)
        off = np.column_stack([r * sz * np.cos(phi), r * sz * np.sin(phi),
                               r * cz * TWIN_SHELL_AXIS_RATIO])
    pos = off + np.asarray(centre, dtype=np.float64)[None, :]
    if cfg.bc == 'periodic':
        pos = wrap_periodic(pos, der.box)
    vel = maxwellian_velocities(n, dim, cfg.twin_sigma * der.vc_ref, rng)
    if cfg.twin_rot != 0.0:
        Rc = np.maximum(np.hypot(off[:, 0], off[:, 1]), 1e-12)
        vt = cfg.twin_rot * der.vc_ref
        vel[:, 0] += -vt * off[:, 1] / Rc
        vel[:, 1] += vt * off[:, 0] / Rc
    return ParticleSet(pos, vel, np.full(n, der.w_neg), -np.ones(n), np.ones(n, np.int32))


# ---------------------------------------------------------------------------
# galaxy: rotation curve measured from the actual field
# ---------------------------------------------------------------------------

def _smooth3(y: np.ndarray) -> np.ndarray:
    if y.size < 3:
        return y.copy()
    out = y.copy()
    out[1:-1] = (y[:-2] + y[1:-1] + y[2:]) / 3.0
    return out


def measure_rotation_curve(ps: ParticleSet, force: ForceEvaluator, centre: np.ndarray,
                           rbins: np.ndarray) -> dict:
    """Azimuthally averaged in-plane radial acceleration on species 0 in the
    radial bins `rbins` (edges).  vc = sqrt(R g) with g = -<a_R>; nan where the
    mean force points outward or the bin is too poor.  Omega = vc/R,
    kappa^2 = 4 Omega^2 + R dOmega^2/dR (3-point smoothed), Sigma = projected
    surface density per annulus."""
    rbins = np.asarray(rbins, dtype=np.float64)
    nb = rbins.size - 1
    acc, _ = force(ps)
    m = ps.species == 0
    box = getattr(force, 'box', None)
    d = _rel(ps.pos[m], centre, box)
    a = np.asarray(acc[m], dtype=np.float64)
    w = np.asarray(ps.w[m], dtype=np.float64)
    R = np.hypot(d[:, 0], d[:, 1])
    aR = (a[:, 0] * d[:, 0] + a[:, 1] * d[:, 1]) / np.maximum(R, 1e-12)
    idx = np.digitize(R, rbins) - 1
    inside = (idx >= 0) & (idx < nb)
    count = np.bincount(idx[inside], minlength=nb).astype(np.float64)
    g = -np.bincount(idx[inside], weights=aR[inside], minlength=nb) / np.maximum(count, 1.0)
    mass = np.bincount(idx[inside], weights=w[inside], minlength=nb)
    area = np.pi * (rbins[1:] ** 2 + 0.0 - rbins[:-1] ** 2)
    Rc = 0.5 * (rbins[1:] + rbins[:-1])
    valid = (count >= _MIN_PER_BIN) & (g > 0)
    vc = np.where(valid, np.sqrt(np.maximum(Rc * g, 0.0)), np.nan)
    omega = vc / np.maximum(Rc, 1e-12)
    kappa = np.full(nb, np.nan)
    if valid.sum() >= 3:
        Rv, om2 = Rc[valid], _smooth3(omega[valid] ** 2)
        k2 = 4.0 * om2 + Rv * np.gradient(om2, Rv)
        kappa[valid] = np.where(k2 > 0, np.sqrt(np.maximum(k2, 0.0)), np.nan)
    elif valid.sum() > 0:
        kappa[valid] = 2.0 * omega[valid]          # solid-body guess
    undefined = ~inside | ~valid[np.clip(idx, 0, nb - 1)]
    frac_undef = float(undefined.mean()) if undefined.size else 0.0
    vmax = float(np.nanmax(vc)) if np.any(valid) else float('nan')
    return {'rbins': rbins, 'R': Rc, 'vc': vc, 'omega': omega, 'kappa': kappa,
            'sigma_surface': mass / area, 'g_r': g, 'n_in_bin': count, 'valid': valid,
            'vc_max': vmax, 'frac_undefined': frac_undef, 'n_undefined': int(undefined.sum())}


def set_disk_velocities(ps: ParticleSet, force: ForceEvaluator, cfg: RunConfig, der: Derived,
                        rng, centre: np.ndarray) -> dict:
    """In place for species 0.  v_phi = sqrt(vc^2 - sigma_R^2 [sigma_phi^2/sigma_R^2
    - 1 - d ln(Sigma sigma_R^2)/d ln R]) (asymmetric drift, razor thin), sigma_R
    from Toomre Q, sigma_phi = sigma_R kappa/(2 Omega), 3-D sigma_z^2 = pi G Sigma z0.
    Particles in bins with undefined vc get no rotation (logged in the returned dict)."""
    rmax = cfg.rmax if cfg.rmax > 0 else 20.0 * cfg.rd
    nb = int(np.clip(cfg.ndisk // 150, 10, 48))
    rbins = np.linspace(0.0, 1.02 * rmax, nb + 1)
    rc = measure_rotation_curve(ps, force, centre, rbins)
    m = ps.species == 0
    n = int(m.sum())
    G = der.G
    Rc, vc, om, kap, Sig, valid = (rc['R'], rc['vc'], rc['omega'], rc['kappa'],
                                   rc['sigma_surface'], rc['valid'])
    sigR_b = np.where(np.isfinite(kap) & (kap > 0) & (Sig > 0), 3.36 * G * Sig * cfg.q / np.maximum(kap, 1e-30), np.nan)
    n_clip = 0
    if np.isfinite(rc['vc_max']):                      # guard against kappa -> 0 at the edge
        n_clip = int(np.sum(sigR_b > rc['vc_max']))
        sigR_b = np.minimum(sigR_b, rc['vc_max'])
    sigphi_b = sigR_b * kap / np.maximum(2.0 * om, 1e-30)
    good = valid & np.isfinite(sigR_b) & np.isfinite(sigphi_b)
    vphi_b = np.full(nb, np.nan)
    if good.sum() >= 2:
        lnR = np.log(np.maximum(Rc[good], 1e-12))
        dln = np.gradient(np.log(Sig[good] * sigR_b[good] ** 2), lnR)
    else:
        dln = -Rc[good] / cfg.rd
    if good.any():
        drift = sigR_b[good] ** 2 * (sigphi_b[good] ** 2 / sigR_b[good] ** 2 - 1.0 - dln)
        vphi_b[good] = np.sqrt(np.maximum(vc[good] ** 2 - drift, 0.0))
    d = _rel(ps.pos[m], centre, getattr(force, 'box', None))
    R = np.hypot(d[:, 0], d[:, 1])
    cphi, sphi = d[:, 0] / np.maximum(R, 1e-12), d[:, 1] / np.maximum(R, 1e-12)
    idx = np.clip(np.digitize(R, rbins) - 1, 0, nb - 1)
    undefined = ~valid[idx] | (R > rbins[-1])
    if good.any():
        sigR_p = np.interp(R, Rc[good], sigR_b[good])
        sigphi_p = np.interp(R, Rc[good], sigphi_b[good])
        vphi_p = np.where(undefined, 0.0, np.interp(R, Rc[good], vphi_b[good]))
        Sig_p = np.interp(R, Rc[good], Sig[good])
    else:                                              # no rotation anywhere: hot, non-rotating blob
        sigR_p = sigphi_p = np.full(n, 0.3 * der.vc_ref)
        vphi_p = np.zeros(n)
        Sig_p = np.interp(R, Rc, Sig)
    vR = rng.normal(0.0, 1.0, n) * sigR_p
    vphi = vphi_p + rng.normal(0.0, 1.0, n) * sigphi_p
    vel = np.zeros((n, ps.dim))
    vel[:, 0] = vR * cphi - vphi * sphi
    vel[:, 1] = vR * sphi + vphi * cphi
    if ps.dim == 3:
        sigz = np.sqrt(np.maximum(np.pi * G * Sig_p * cfg.zscale * cfg.rd, 0.0))
        vel[:, 2] = rng.normal(0.0, 1.0, n) * sigz
    ps.vel[m] = vel.astype(ps.vel.dtype)
    remove_bulk_velocity(ps, m)
    i2 = int(np.argmin(np.abs(Rc - 2.0 * cfg.rd)))
    vc2 = float(vc[i2]) if np.isfinite(vc[i2]) else float('nan')
    rc.update({'Q': cfg.q, 'sigma_R': sigR_b, 'sigma_phi': sigphi_b, 'vphi_mean': vphi_b,
               'frac_undefined': float(undefined.mean()) if n else 0.0,
               'n_undefined': int(undefined.sum()), 'n_sigma_clipped': n_clip,
               'vc_at_2rd': vc2, 't_rot_at_2rd': (4.0 * np.pi * cfg.rd / vc2) if vc2 > 0 else float('nan'),
               'sigma_R_at_2rd': float(sigR_b[i2]) if np.isfinite(sigR_b[i2]) else float('nan')})
    return rc


def make_galaxy_ic(cfg: RunConfig, der: Derived, rng, force_builder):
    """Disk (+ live twin ring when control='live'); force_builder(ps)->ForceEvaluator
    is used to measure the real initial field (including rigid control fields)
    before the disk velocities are set.  Returns (ps, rotation-curve info)."""
    dim = der.dim
    centre = np.full(dim, 0.5 * der.box)
    disk = make_disk_positions(cfg, der, rng, centre)
    if cfg.control == 'live':
        ring = make_twin_ring(cfg, der, rng, centre)
        ps = concat(disk, ring) if ring.n else disk
    else:
        ps = disk
    ps = ps.astype(cfg.dtype)
    force = force_builder(ps)
    rc = set_disk_velocities(ps, force, cfg, der, rng, centre)
    rc['centre'] = centre
    rc['n_disk'] = int(np.sum(ps.species == 0))
    rc['n_twin'] = int(np.sum(ps.species == 1))
    _say(cfg, f"galaxy IC: {rc['n_disk']} disk + {rc['n_twin']} twin; vc(2R_d) = {rc['vc_at_2rd']:.4g} "
              f"(analytic ref {der.vc_ref:.4g}), vc_max = {rc['vc_max']:.4g}, sigma_R(2R_d) = "
              f"{rc['sigma_R_at_2rd']:.4g}; undefined v_c for {100 * rc['frac_undefined']:.1f}% of the disk"
              + (f"; {rc['n_sigma_clipped']} bins with sigma_R clipped at vc_max" if rc['n_sigma_clipped'] else ""))
    return ps, rc


if __name__ == '__main__':  # minimal self-test
    from config import resolve_defaults, derive
    from external_potentials import NoField
    rng = np.random.default_rng(1)
    for ic in ('poisson', 'lattice', 'glass'):
        for dim in (2, 3):
            cfg = resolve_defaults(RunConfig('lss', ic=ic, npos=200, nneg=200, dim=dim, quiet=True))
            der = derive(cfg)
            ps = make_lss_ic(cfg, der, rng)
            assert ps.dim == dim and np.all((ps.pos >= 0) & (ps.pos < 1.0))
            assert abs(ps.w[ps.pos_mask].sum() - 1.0) < 1e-9 and abs(ps.w[ps.neg_mask].sum() - 64.0) < 1e-6
            assert np.abs((ps.w[:, None] * ps.vel).sum(0)).max() < 1e-9
            assert list(ps.species[:ps.npos]) == [0] * ps.npos
    cfgn = resolve_defaults(RunConfig('lss', neg_only=True, nneg=100, quiet=True))
    assert make_lss_ic(cfgn, derive(cfgn), rng).npos == 0
    xy = sample_exponential_disk(20000, 1.0, 4.0, rng)
    r = np.hypot(xy[:, 0], xy[:, 1])
    assert r.max() <= 4.0 and abs(r.mean() - 1.72) < 0.1, r.mean()   # truncated-disk mean radius
    cfg = resolve_defaults(RunConfig('galaxy', ndisk=800, ntwin=800, method='direct', quiet=True))
    der = derive(cfg)
    ps, rc = make_galaxy_ic(cfg, der, rng, lambda p: ForceEvaluator(cfg, der, None, NoField()))
    assert ps.n == 1600 and np.all(np.isfinite(ps.vel))
    d = ps.pos[ps.species == 0] - rc['centre']
    Lz = np.sum(d[:, 0] * ps.vel[ps.species == 0, 1] - d[:, 1] * ps.vel[ps.species == 0, 0])
    assert Lz > 0 and np.isfinite(rc['vc_max']) and 0.0 <= rc['frac_undefined'] <= 1.0
    cfg3 = resolve_defaults(RunConfig('galaxy', ndisk=400, ntwin=200, dim=3, method='direct', quiet=True))
    der3 = derive(cfg3)
    ps3, rc3 = make_galaxy_ic(cfg3, der3, rng, lambda p: ForceEvaluator(cfg3, der3, None, NoField()))
    assert ps3.dim == 3 and np.all(np.isfinite(ps3.vel))
    print("initial_conditions self-test OK")
