"""diagnostics_torch.py -- device (PyTorch / ROCm) counterpart of diagnostics_core.

Every function here reproduces, term for term, the numerical definition of its
numpy twin in diagnostics_core (energies, momenta, angular momenta, species
statistics, the merged diagnostics row) and of particles.centre_of_mass, but
operates on device-resident tensors (gpu_backend.GPUParticles or any object
with pos/vel/w/s tensors) in float64 -- the CPU code casts to float64 with
np.asarray(..., float64) everywhere, so a float64 GPU run reproduces the CPU
numbers to round-off and a float32 run differs only through the float32
particle state.  Conventions (diagnostics_core, notes section 1):

  * KE = 0.5 a^2 sum w |v|^2,  U = 0.5 sum s w phi (+ 0.5 sum s w phi_ext) - E_self,
  * P_abs = sum |m| v (conserved), P_signed = sum m v (not conserved),
  * L_abs / L_signed / L_pos / L_neg: scalar L_z in 2-D, norm of the vector in 3-D,
  * sigma_*: |m|-weighted rms-over-components dispersion, nan for < 2 particles.

The species sums are evaluated with multiplicative masks (w * [s > 0]) instead
of boolean indexing, so no host synchronisation happens inside a row; all the
scalars of a row are stacked into one float64 tensor and transferred with a
single .cpu() call.  The dict keys and their order are exactly those of
diagnostics_core.diagnostics_row.  Only numpy and torch are imported.
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

_AXES = ('x', 'y', 'z')
_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _need_torch() -> None:
    if torch is None:
        raise RuntimeError(f"diagnostics_torch needs PyTorch (import failed: {_TORCH_ERR})")


def _f64(t, like=None):
    """Tensor -> float64 view/copy (mirrors np.asarray(..., float64)); arrays are
    uploaded to the device of `like` (or the CPU)."""
    if torch.is_tensor(t):
        return t.detach().to(torch.float64)
    dev = like.device if like is not None else None
    return torch.as_tensor(np.ascontiguousarray(np.asarray(t, dtype=np.float64)),
                           dtype=torch.float64, device=dev)


def _const(value: float, device):
    return torch.full((), float(value), dtype=torch.float64, device=device)


def _nan(device):
    return _const(float('nan'), device)


def _as_vec(x, device):
    """Small host or device vector (centre, P0) -> float64 tensor on `device`."""
    if torch.is_tensor(x):
        return x.detach().to(device=device, dtype=torch.float64).reshape(-1)
    return torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(-1), dtype=torch.float64,
                           device=device)


def min_image_t(d, box: float):
    """Separation vectors mapped to the nearest periodic image (particles.min_image)."""
    return d - box * torch.round(d / box)


def _wrap_t(pos, box: float):
    """Coordinates into [0, box) with the guards of particles.wrap_periodic."""
    out = torch.remainder(pos, box)
    out = torch.where(out >= box, out - box, out)
    out = torch.where(out < 0, out + box, out)
    return out


def _rel_pos_t(gp, centre, box):
    pos = _f64(gp.pos)
    if centre is None:
        return pos
    dim = pos.shape[1]
    d = pos - _as_vec(centre, pos.device)[None, :dim]
    return min_image_t(d, box) if box is not None else d


class _Collector:
    """Ordered (key, value) pairs; tensor values are stacked and copied to the
    host in one transfer, host scalars pass through.  kind: 'f' float, 'i' int."""

    def __init__(self, device):
        self.device = device
        self.entries = []                        # (key, kind, tensor or host value)

    def add(self, key: str, value, kind: str = 'f') -> None:
        if torch.is_tensor(value):
            value = value.detach().to(torch.float64).reshape(())
        self.entries.append((key, kind, value))

    def update(self, items) -> None:
        for key, value, kind in items:
            self.add(key, value, kind)

    def finish(self) -> dict:
        tens = [v for _, _, v in self.entries if torch.is_tensor(v)]
        host = torch.stack(tens).cpu().numpy() if tens else np.zeros(0)
        out, j = {}, 0
        for key, kind, v in self.entries:
            if torch.is_tensor(v):
                val = float(host[j])
                j += 1
            else:
                val = v
            out[key] = int(round(val)) if kind == 'i' else float(val)
        return out


# ---------------------------------------------------------------------------
# energies (tensor-returning cores and float-returning public versions)
# ---------------------------------------------------------------------------

def _ke(gp, a: float):
    v, w = _f64(gp.vel), _f64(gp.w)
    return 0.5 * float(a) ** 2 * (w * (v * v).sum(1)).sum()


def _pe(gp, phi, self_energy: float, phi_ext=None):
    if gp.pos.shape[0] == 0:
        return _const(-float(self_energy), gp.pos.device)
    sw = _f64(gp.s) * _f64(gp.w)
    U = 0.5 * (sw * _f64(phi, like=gp.pos)).sum()
    if phi_ext is not None:
        U = U + 0.5 * (sw * _f64(phi_ext, like=gp.pos)).sum()
    return U - float(self_energy)


def kinetic_energy_t(gp, a: float = 1.0) -> float:
    """0.5 * a^2 * sum w |v|^2 (diagnostics_core.kinetic_energy)."""
    _need_torch()
    if gp.pos.shape[0] == 0:
        return 0.0
    return float(_ke(gp, a).item())


def potential_energy_t(gp, phi, self_energy: float = 0.0, phi_ext=None) -> float:
    """0.5 sum s w phi (+ 0.5 sum s w phi_ext) - self_energy (diagnostics_core.potential_energy)."""
    _need_torch()
    return float(_pe(gp, phi, self_energy, phi_ext).item())


# ---------------------------------------------------------------------------
# momenta
# ---------------------------------------------------------------------------

def _ang(r, v, weights):
    """Weighted total angular momentum: scalar L_z in 2-D, (3,) vector in 3-D."""
    if r.shape[1] == 2:
        return (weights * (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0])).sum()
    if r.shape[0] == 0:
        return torch.zeros(3, dtype=torch.float64, device=r.device)
    return (weights[:, None] * torch.cross(r, v, dim=1)).sum(0)


def _norm(x):
    return torch.sqrt((x * x).sum())


def _ang_scalar(L):
    return L if L.dim() == 0 else _norm(L)


def _momenta_items(gp, centre, box):
    """[(key, tensor, kind)] in the order of diagnostics_core.momenta, plus the
    raw p_abs / p_sgn vectors (for the drift measures of the row)."""
    dim = gp.pos.shape[1]
    w = _f64(gp.w)
    sw = _f64(gp.s) * w
    v = _f64(gp.vel)
    p_abs = (w[:, None] * v).sum(0)
    p_sgn = (sw[:, None] * v).sum(0)
    items = [(f'p_abs_{_AXES[k]}', p_abs[k], 'f') for k in range(dim)]
    items += [(f'p_signed_{_AXES[k]}', p_sgn[k], 'f') for k in range(dim)]
    items += [('p_abs_norm', _norm(p_abs), 'f'), ('p_signed_norm', _norm(p_sgn), 'f')]
    r = _rel_pos_t(gp, centre, box)
    pm = (gp.s > 0).to(torch.float64)
    nm = (gp.s < 0).to(torch.float64)
    L_pos = _ang(r, v, w * pm)
    L_neg = _ang(r, v, w * nm)
    L_abs, L_sgn = L_pos + L_neg, L_pos - L_neg
    items += [('L_abs', _ang_scalar(L_abs), 'f'), ('L_signed', _ang_scalar(L_sgn), 'f'),
              ('L_pos', _ang_scalar(L_pos), 'f'), ('L_neg', _ang_scalar(L_neg), 'f')]
    if dim == 3:
        items += [('L_pos_z', L_pos[2], 'f'), ('L_neg_z', L_neg[2], 'f')]
    return items, p_abs, p_sgn


def momenta_t(gp, centre=None, box=None) -> dict:
    """p_abs_x.., p_signed_x.., p_abs_norm, p_signed_norm, L_abs, L_signed, L_pos,
    L_neg (+ L_pos_z / L_neg_z in 3-D); one host transfer."""
    _need_torch()
    items, _, _ = _momenta_items(gp, centre, box)
    col = _Collector(gp.pos.device)
    col.update(items)
    return col.finish()


# ---------------------------------------------------------------------------
# per-species statistics
# ---------------------------------------------------------------------------

def _dispersion(v, w, m):
    """|m|-weighted 1-D dispersion of the particles selected by the bool mask m
    (rms over components of the variance); nan for < 2 particles or zero weight."""
    wm = w * m.to(torch.float64)
    n = m.sum()
    wsum = wm.sum()
    mean = (wm[:, None] * v).sum(0) / wsum
    dv = v - mean[None, :]
    var = (wm[:, None] * dv * dv).sum(0) / wsum
    return torch.where((n >= 2) & (wsum > 0), torch.sqrt(var.mean()), _nan(v.device))


def _species_items(gp, centre, r_escape, box):
    r = _rel_pos_t(gp, centre, box)
    v, w = _f64(gp.vel), _f64(gp.w)
    dev = r.device
    nan = _nan(dev)
    rr = (r * r).sum(1)
    items = []
    for name, m in (('pos', gp.s > 0), ('neg', gp.s < 0)):
        n = m.sum()
        wm = w * m.to(torch.float64)
        wsum = wm.sum()
        ok = (n > 0) & (wsum > 0)
        items.append((f'n_{name}', n.to(torch.float64), 'i'))
        items.append((f'sigma_{name}', _dispersion(v, w, m), 'f'))
        items.append((f'r_rms_{name}', torch.where(ok, torch.sqrt((wm * rr).sum() / wsum), nan), 'f'))
        if r_escape is not None:
            cnt = ((torch.sqrt(rr) > float(r_escape)) & m).sum().to(torch.float64)
            esc = torch.where(ok, cnt / n.to(torch.float64), nan)
        else:
            esc = nan
        items.append((f'escaped_frac_{name}', esc, 'f'))
    pm = gp.s > 0
    wp = w * pm.to(torch.float64)
    wsum = wp.sum()
    R = torch.clamp(torch.hypot(r[:, 0], r[:, 1]), min=1e-30)
    vt = (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]) / R
    vrot = torch.where((pm.sum() > 0) & (wsum > 0), (wp * vt).sum() / wsum, nan)
    items.append(('vrot_mean_pos', vrot, 'f'))
    return items


def species_stats_t(gp, centre=None, r_escape=None, box=None) -> dict:
    """n_pos/n_neg, sigma_*, r_rms_*, escaped_frac_* (nan when r_escape is None),
    vrot_mean_pos -- the keys and definitions of diagnostics_core.species_stats."""
    _need_torch()
    col = _Collector(gp.pos.device)
    col.update(_species_items(gp, centre, r_escape, box))
    return col.finish()


# ---------------------------------------------------------------------------
# merged row
# ---------------------------------------------------------------------------

def diagnostics_row_t(gp, phi, t: float, step: int, dt: float, a: float, self_energy: float,
                      E0, P0, centre=None, r_escape=None, wall: float = 0.0, box=None,
                      phi_ext=None) -> dict:
    """One diagnostics record with exactly the keys/order of
    diagnostics_core.diagnostics_row; all device scalars leave the device in a
    single transfer.  dP_abs_rel = |P_abs - P0| / sum |m||v|, dP_signed_rel =
    |P_signed| / sum |m||v|, virial = 2 KE / |U|."""
    _need_torch()
    dev = gp.pos.device
    n, dim = int(gp.pos.shape[0]), int(gp.pos.shape[1])
    nan = _nan(dev)
    KE = _ke(gp, a) if n else _const(0.0, dev)
    PE = _pe(gp, phi, self_energy, phi_ext)
    E = KE + PE
    col = _Collector(dev)
    col.add('t', float(t))
    col.add('step', int(step), 'i')
    col.add('dt', float(dt))
    col.add('a', float(a))
    col.add('n', n, 'i')
    col.add('KE', KE)
    col.add('PE', PE)
    col.add('E', E)
    if E0 is not None and np.isfinite(float(E0)) and float(E0) != 0:
        col.add('dE_rel', (E - float(E0)) / abs(float(E0)))
    else:
        col.add('dE_rel', nan)
    items, p_abs, p_sgn = _momenta_items(gp, centre, box)
    col.update(items)
    w, v = _f64(gp.w), _f64(gp.vel)
    scale = (w * torch.sqrt((v * v).sum(1))).sum() if n else _const(0.0, dev)
    if P0 is not None:
        P0a = _as_vec(P0, dev)[:dim]
        col.add('dP_abs_rel', torch.where(scale > 0, _norm(p_abs - P0a) / scale, nan))
    else:
        col.add('dP_abs_rel', nan)
    col.add('dP_signed_rel', torch.where(scale > 0, _norm(p_sgn) / scale, nan))
    col.add('virial', torch.where(PE != 0, 2.0 * KE / torch.abs(PE), nan))
    col.update(_species_items(gp, centre, r_escape, box))
    col.add('wall', float(wall))
    return col.finish()


# ---------------------------------------------------------------------------
# centre of |m|-weighted mass (particles.centre_of_mass on the device)
# ---------------------------------------------------------------------------

def centre_of_mass_t(gp, mask=None, box=None):
    """|m|-weighted centre of the (optionally masked) particles as a float64
    tensor (dim,) on the device.  With `box` the circular mean per axis is used
    (angle 2 pi x / box, weighted mean of cos/sin, back to [0, box)); an axis
    whose weighted vector sum vanishes (< 1e-24) falls back to the plain mean.
    Empty selection -> nan; zero total weight -> plain unweighted mean."""
    _need_torch()
    pos, w = _f64(gp.pos), _f64(gp.w)
    dim = pos.shape[1]
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(np.asarray(mask), device=pos.device)
        mask = mask.to(device=pos.device)
        pos, w = pos[mask], w[mask]
    if pos.shape[0] == 0:
        return torch.full((dim,), float('nan'), dtype=torch.float64, device=pos.device)
    wsum = w.sum()
    if float(wsum.item()) <= 0:
        return pos.mean(0)
    plain = (w[:, None] * pos).sum(0) / wsum
    if box is None:
        return plain
    box = float(box)
    theta = 2.0 * math.pi * pos / box
    c = (w[:, None] * torch.cos(theta)).sum(0) / wsum
    sn = (w[:, None] * torch.sin(theta)).sum(0) / wsum
    circ = torch.atan2(sn, c) / _TWO_PI * box
    out = torch.where(c * c + sn * sn < 1e-24, plain, circ)
    return _wrap_t(out, box)


# ---------------------------------------------------------------------------
# self-test: agreement with diagnostics_core / particles on random sets
# ---------------------------------------------------------------------------

def _close(a, b, tol: float = 1e-12) -> bool:
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b or abs(a - b) <= tol * max(abs(a), abs(b), 1e-300)


def _check(row_c: dict, row_t: dict, label: str) -> None:
    assert list(row_c.keys()) == list(row_t.keys()), (label, list(row_c), list(row_t))
    for k in row_c:
        assert _close(row_c[k], row_t[k]), (label, k, row_c[k], row_t[k])


if __name__ == '__main__':
    import sys
    if torch is None:
        print("torch not installed, diagnostics_torch self-test skipped:", _TORCH_ERR)
        sys.exit(0)
    from particles import ParticleSet, centre_of_mass, subset
    from diagnostics_core import (diagnostics_row, momenta, species_stats, kinetic_energy,
                                  potential_energy)
    from gpu_backend import GPUParticles
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu':
        print("no GPU: checking the torch diagnostics on the CPU device")
    rng = np.random.default_rng(0)
    for dim in (2, 3):
        n = 3000
        pos = rng.random((n, dim))
        # negative species: a clump straddling the periodic edge at x = y = (z =) 0
        pos[n // 2:] = np.mod(0.02 + 0.03 * rng.normal(size=(n // 2, dim)), 1.0)
        vel = rng.normal(size=(n, dim))
        s = np.where(np.arange(n) < n // 2, 1.0, -1.0)
        w = np.where(s > 0, 0.01, 0.64)
        ps = ParticleSet(pos, vel, w, s, (s < 0).astype(np.int32))
        gp = GPUParticles.from_host(ps, dev, 'float64')
        phi = rng.normal(size=n)
        phi_ext = 0.1 * rng.normal(size=n)
        phi_t = torch.as_tensor(phi, device=dev)
        phi_ext_t = torch.as_tensor(phi_ext, device=dev)
        centre = np.full(dim, 0.5)
        # --- individual functions ---------------------------------------------
        assert _close(kinetic_energy_t(gp, 1.3), kinetic_energy(ps, 1.3))
        assert _close(potential_energy_t(gp, phi_t, 0.1, phi_ext_t), potential_energy(ps, phi, 0.1, phi_ext))
        assert _close(potential_energy_t(gp, phi_t, 0.0), potential_energy(ps, phi, 0.0))
        _check(momenta(ps, centre, 1.0), momenta_t(gp, centre, 1.0), f'momenta {dim}d')
        _check(momenta(ps), momenta_t(gp), f'momenta origin {dim}d')
        _check(momenta(ps, centre, 1.0), momenta_t(gp, torch.as_tensor(centre, device=dev), 1.0),
               f'momenta tensor centre {dim}d')
        _check(species_stats(ps, centre, 0.3, 1.0), species_stats_t(gp, centre, 0.3, 1.0), f'species {dim}d')
        _check(species_stats(ps), species_stats_t(gp), f'species plain {dim}d')
        # --- full rows -------------------------------------------------------------
        args_c = dict(centre=centre, r_escape=0.3, wall=0.5, box=1.0, phi_ext=phi_ext)
        args_t = dict(centre=centre, r_escape=0.3, wall=0.5, box=1.0, phi_ext=phi_ext_t)
        row_c = diagnostics_row(ps, phi, 0.0, 0, 0.01, 1.0, 0.1, None, None, **args_c)
        row_t = diagnostics_row_t(gp, phi_t, 0.0, 0, 0.01, 1.0, 0.1, None, None, **args_t)
        _check(row_c, row_t, f'row0 {dim}d')
        assert isinstance(row_t['n_pos'], int) and isinstance(row_t['step'], int) and isinstance(row_t['n'], int)
        assert math.isnan(row_t['dE_rel']) and math.isnan(row_t['dP_abs_rel'])
        E0 = 0.8 * row_c['E']
        P0 = 0.5 * np.array([row_c[f'p_abs_{_AXES[k]}'] for k in range(dim)])
        row2_c = diagnostics_row(ps, phi, 0.02, 2, 0.01, 1.5, 0.1, E0, P0, **args_c)
        row2_t = diagnostics_row_t(gp, phi_t, 0.02, 2, 0.01, 1.5, 0.1, E0, P0, **args_t)
        _check(row2_c, row2_t, f'row2 {dim}d')
        assert abs(row2_t['dE_rel']) > 0.1 and row2_t['dP_abs_rel'] > 0 and row2_t['a'] == 1.5
        row3_c = diagnostics_row(ps, phi, 0.0, 0, 0.01, 1.0, 0.0, None, None)
        row3_t = diagnostics_row_t(gp, phi_t, 0.0, 0, 0.01, 1.0, 0.0, None, None)
        _check(row3_c, row3_t, f'row plain {dim}d')
        # --- centre of mass (circular mean, clump straddling the box edge) -----------
        for mask in (None, ps.s < 0, ps.s > 0):
            mask_t = None if mask is None else torch.as_tensor(mask, device=dev)
            for box in (1.0, None):
                c_c = centre_of_mass(ps, mask, box)
                c_t = centre_of_mass_t(gp, mask_t, box).cpu().numpy()
                assert c_t.shape == (dim,) and all(_close(x, y) for x, y in zip(c_c, c_t)), (dim, box, c_c, c_t)
        c_neg = centre_of_mass_t(gp, gp.s < 0, 1.0).cpu().numpy()
        assert np.all(np.minimum(c_neg, 1.0 - c_neg) < 0.03), c_neg      # straddling clump near 0
        assert np.all((c_neg >= 0) & (c_neg < 1.0))
        # --- one species only: zeros / nan for the empty population ------------------
        psp = subset(ps, ps.s > 0)
        gpp = GPUParticles.from_host(psp, dev, 'float64')
        phip = torch.as_tensor(phi[: n // 2], device=dev)
        rp_c = diagnostics_row(psp, phi[: n // 2], 0.0, 0, 0.01, 1.0, 0.0, None, None, centre, 0.3, 0.0, 1.0)
        rp_t = diagnostics_row_t(gpp, phip, 0.0, 0, 0.01, 1.0, 0.0, None, None, centre, 0.3, 0.0, 1.0)
        _check(rp_c, rp_t, f'pos only {dim}d')
        assert rp_t['L_neg'] == 0.0 and rp_t['n_neg'] == 0 and math.isnan(rp_t['sigma_neg'])
        if dim == 3:
            assert rp_t['L_neg_z'] == 0.0 and 'L_pos_z' in rp_t
        empty = centre_of_mass_t(gpp, gpp.s < 0, 1.0).cpu().numpy()
        assert np.all(np.isnan(empty)) and empty.shape == (dim,)
        # --- float32 particle state still yields finite float64 diagnostics ---------
        g32 = GPUParticles.from_host(ps, dev, 'float32')
        r32 = diagnostics_row_t(g32, phi_t.to(torch.float32), 0.0, 0, 0.01, 1.0, 0.1, None, None, **args_t)
        assert list(r32) == list(row_c) and abs(r32['KE'] - row_c['KE']) < 1e-5 * abs(row_c['KE'])
    # --- empty set ------------------------------------------------------------------
    from particles import empty as _empty
    ge = GPUParticles.from_host(_empty(2), dev, 'float64')
    re_t = diagnostics_row_t(ge, torch.zeros(0, dtype=torch.float64, device=dev), 0.0, 0, 0.01, 1.0, 0.0,
                             None, None)
    re_c = diagnostics_row(_empty(2), np.zeros(0), 0.0, 0, 0.01, 1.0, 0.0, None, None)
    _check(re_c, re_t, 'empty')
    print("diagnostics_torch self-test OK")
