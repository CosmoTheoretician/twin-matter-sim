"""diagnostics_core.py -- conservation and bulk diagnostics plus CSV/JSON writers.

Conventions (notes, section 1):
  * KE = 0.5 sum w |a v|^2 (peculiar kinetic energy; a = 1 without expansion),
  * U  = 0.5 sum s w phi - self_energy, with phi the unsigned potential of the
    signed density (forces_direct / forces_pm) -- an external rigid field
    contributes sum s w phi_ext without the 1/2 (optional `phi_ext` argument),
  * P_abs = sum |m| v is the conserved momentum of the three-law scheme;
    P_signed = sum m v is NOT conserved (cross-sign pairs receive parallel
    impulses) -- both are logged, only P_abs is a conservation diagnostic,
  * likewise L_abs (conserved) vs L_signed (not); per-species L_pos / L_neg
    allow the "braking" claim to be tested (disk L can only fall if the twin
    ring's |m|-weighted L rises).
Angular momenta are the z-component in 2-D and the vector norm in 3-D.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import is_dataclass, asdict

import numpy as np

from particles import ParticleSet, min_image

_AXES = ('x', 'y', 'z')


# ---------------------------------------------------------------------------
# energies
# ---------------------------------------------------------------------------

def kinetic_energy(ps: ParticleSet, a: float = 1.0) -> float:
    """0.5 * sum w |v|^2 * a^2 (peculiar kinetic energy; vel is comoving dx/dt)."""
    if ps.n == 0:
        return 0.0
    v = np.asarray(ps.vel, dtype=np.float64)
    w = np.asarray(ps.w, dtype=np.float64)
    return 0.5 * float(a) ** 2 * float(np.sum(w * np.einsum('ij,ij->i', v, v)))


def potential_energy(ps: ParticleSet, phi: np.ndarray, self_energy: float = 0.0,
                     phi_ext=None) -> float:
    """0.5 * sum s w phi - self_energy.  If `phi` already includes an external
    rigid potential, pass it separately as `phi_ext` so it is counted once
    (no back-reaction => no factor 1/2 for that part)."""
    if ps.n == 0:
        return -float(self_energy)
    sw = np.asarray(ps.s, dtype=np.float64) * np.asarray(ps.w, dtype=np.float64)
    U = 0.5 * float(np.sum(sw * np.asarray(phi, dtype=np.float64)))
    if phi_ext is not None:
        U += 0.5 * float(np.sum(sw * np.asarray(phi_ext, dtype=np.float64)))
    return U - float(self_energy)


# ---------------------------------------------------------------------------
# momenta
# ---------------------------------------------------------------------------

def _rel_pos(ps: ParticleSet, centre, box):
    pos = np.asarray(ps.pos, dtype=np.float64)
    if centre is None:
        return pos
    d = pos - np.asarray(centre, dtype=np.float64)[None, :ps.dim]
    return min_image(d, box) if box is not None else d


def _ang(r: np.ndarray, v: np.ndarray, weights: np.ndarray):
    """Weighted total angular momentum: scalar L_z in 2-D, (3,) vector in 3-D."""
    if r.shape[0] == 0:
        return 0.0 if r.shape[1] == 2 else np.zeros(3)
    if r.shape[1] == 2:
        return float(np.sum(weights * (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0])))
    return (weights[:, None] * np.cross(r, v)).sum(axis=0)


def _ang_scalar(L) -> float:
    return float(L) if np.ndim(L) == 0 else float(np.linalg.norm(L))


def momenta(ps: ParticleSet, centre=None, box=None) -> dict:
    """p_abs_x.., p_signed_x.., L_abs, L_signed, L_pos, L_neg (+ norms).
    Angular momenta about `centre` (default origin); min-image if box given."""
    dim = ps.dim
    w = np.asarray(ps.w, dtype=np.float64)
    sw = np.asarray(ps.s, dtype=np.float64) * w
    v = np.asarray(ps.vel, dtype=np.float64)
    out = {}
    p_abs = (w[:, None] * v).sum(axis=0) if ps.n else np.zeros(dim)
    p_sgn = (sw[:, None] * v).sum(axis=0) if ps.n else np.zeros(dim)
    for k in range(dim):
        out[f'p_abs_{_AXES[k]}'] = float(p_abs[k])
    for k in range(dim):
        out[f'p_signed_{_AXES[k]}'] = float(p_sgn[k])
    out['p_abs_norm'] = float(np.linalg.norm(p_abs))
    out['p_signed_norm'] = float(np.linalg.norm(p_sgn))
    r = _rel_pos(ps, centre, box)
    pm, nm = ps.pos_mask, ps.neg_mask
    L_pos = _ang(r[pm], v[pm], w[pm])
    L_neg = _ang(r[nm], v[nm], w[nm])
    L_abs = np.add(L_pos, L_neg)
    L_sgn = np.subtract(L_pos, L_neg)
    out['L_abs'] = _ang_scalar(L_abs)
    out['L_signed'] = _ang_scalar(L_sgn)
    out['L_pos'] = _ang_scalar(L_pos)
    out['L_neg'] = _ang_scalar(L_neg)
    if dim == 3:                      # z-components too (disk plane)
        out['L_pos_z'] = float(L_pos[2]) if np.ndim(L_pos) else 0.0
        out['L_neg_z'] = float(L_neg[2]) if np.ndim(L_neg) else 0.0
    return out


# ---------------------------------------------------------------------------
# per-species statistics
# ---------------------------------------------------------------------------

def _dispersion(v: np.ndarray, w: np.ndarray) -> float:
    """|m|-weighted 1-D dispersion (rms over components of the variance)."""
    if v.shape[0] < 2 or w.sum() <= 0:
        return float('nan')
    mean = (w[:, None] * v).sum(axis=0) / w.sum()
    dv = v - mean[None, :]
    var = (w[:, None] * dv * dv).sum(axis=0) / w.sum()
    return float(np.sqrt(var.mean()))


def species_stats(ps: ParticleSet, centre=None, r_escape=None, box=None) -> dict:
    """sigma_pos/neg (1-D dispersion), vrot_mean_pos (|m|-weighted mean tangential
    velocity about `centre`), r_rms_pos/neg, escaped_frac_pos/neg (r > r_escape;
    nan when r_escape is None), n_pos, n_neg."""
    r = _rel_pos(ps, centre, box)
    v = np.asarray(ps.vel, dtype=np.float64)
    w = np.asarray(ps.w, dtype=np.float64)
    out = {}
    for name, m in (('pos', ps.pos_mask), ('neg', ps.neg_mask)):
        n = int(m.sum())
        out[f'n_{name}'] = n
        out[f'sigma_{name}'] = _dispersion(v[m], w[m])
        if n and w[m].sum() > 0:
            rr = np.einsum('ij,ij->i', r[m], r[m])
            out[f'r_rms_{name}'] = float(np.sqrt(np.sum(w[m] * rr) / w[m].sum()))
            out[f'escaped_frac_{name}'] = (float(np.mean(np.sqrt(rr) > r_escape))
                                           if r_escape is not None else float('nan'))
        else:
            out[f'r_rms_{name}'] = float('nan')
            out[f'escaped_frac_{name}'] = float('nan')
    pm = ps.pos_mask
    if pm.any() and w[pm].sum() > 0:
        rp, vp, wp = r[pm], v[pm], w[pm]
        R = np.maximum(np.hypot(rp[:, 0], rp[:, 1]), 1e-30)
        vt = (rp[:, 0] * vp[:, 1] - rp[:, 1] * vp[:, 0]) / R
        out['vrot_mean_pos'] = float(np.sum(wp * vt) / wp.sum())
    else:
        out['vrot_mean_pos'] = float('nan')
    return out


# ---------------------------------------------------------------------------
# merged row
# ---------------------------------------------------------------------------

def diagnostics_row(ps: ParticleSet, phi: np.ndarray, t: float, step: int, dt: float,
                    a: float, self_energy: float, E0, P0, centre=None, r_escape=None,
                    wall: float = 0.0, box=None, phi_ext=None) -> dict:
    """One diagnostics record: energies, momenta, species stats, dE_rel,
    dP_abs_rel (|P_abs - P0| normalised by sum |m||v|, since P0 ~ 0 after bulk
    velocity removal), dP_signed (drift of the non-conserved signed momentum,
    expected physics), virial = 2 KE / |U|."""
    KE = kinetic_energy(ps, a)
    PE = potential_energy(ps, phi, self_energy, phi_ext=phi_ext)
    E = KE + PE
    row = {'t': float(t), 'step': int(step), 'dt': float(dt), 'a': float(a),
           'n': ps.n, 'KE': KE, 'PE': PE, 'E': E}
    if E0 is not None and np.isfinite(E0) and E0 != 0:
        row['dE_rel'] = (E - float(E0)) / abs(float(E0))
    else:
        row['dE_rel'] = float('nan')
    mom = momenta(ps, centre, box)
    row.update(mom)
    dim = ps.dim
    p_abs = np.array([mom[f'p_abs_{_AXES[k]}'] for k in range(dim)])
    p_sgn = np.array([mom[f'p_signed_{_AXES[k]}'] for k in range(dim)])
    w = np.asarray(ps.w, dtype=np.float64)
    v = np.asarray(ps.vel, dtype=np.float64)
    scale = float(np.sum(w * np.sqrt(np.einsum('ij,ij->i', v, v)))) if ps.n else 0.0
    if P0 is not None and scale > 0:
        P0a = np.asarray(P0, dtype=np.float64).reshape(-1)[:dim]
        row['dP_abs_rel'] = float(np.linalg.norm(p_abs - P0a) / scale)
    else:
        row['dP_abs_rel'] = float('nan')
    row['dP_signed_rel'] = float(np.linalg.norm(p_sgn) / scale) if scale > 0 else float('nan')
    row['virial'] = (2.0 * KE / abs(PE)) if PE != 0 else float('nan')
    row.update(species_stats(ps, centre, r_escape, box))
    row['wall'] = float(wall)
    return row


# ---------------------------------------------------------------------------
# CSV / JSON
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return 'nan'
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return 'nan' if not np.isfinite(v) else repr(float(v))
    return str(v)


class CsvLogger:
    """Append-style CSV writer: header from the first row, later rows are
    reordered to the header and missing columns filled with nan; unknown
    columns in later rows are ignored (recorded in `dropped`)."""

    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._f = open(path, 'w', newline='')
        self._w = csv.writer(self._f)
        self.columns = None
        self.n_rows = 0
        self.dropped = set()

    def log(self, row: dict) -> None:
        if self.columns is None:
            self.columns = list(row.keys())
            self._w.writerow(self.columns)
        else:
            self.dropped.update(set(row) - set(self.columns))
        self._w.writerow([_fmt(row.get(c)) for c in self.columns])
        self.n_rows += 1
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, (np.dtype, type)):
        return str(o)
    return str(o)


def write_json(path: str, obj) -> None:
    """json.dump with numpy scalars/arrays and dataclasses made serialisable."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.write('\n')


def read_csv(path: str) -> dict:
    """Column dict: float arrays where every entry parses as a number
    ('nan' included), otherwise arrays of strings."""
    with open(path, newline='') as f:
        rd = csv.reader(f)
        try:
            header = next(rd)
        except StopIteration:
            return {}
        rows = [r for r in rd if r]
    out = {}
    for j, name in enumerate(header):
        col = [r[j] if j < len(r) else 'nan' for r in rows]
        try:
            out[name] = np.array([float(c) if c != '' else np.nan for c in col], dtype=np.float64)
        except ValueError:
            out[name] = np.array(col, dtype=object)
    return out


if __name__ == '__main__':  # minimal self-test
    import tempfile
    rng = np.random.default_rng(0)
    n = 100
    pos = rng.random((n, 2))
    vel = rng.normal(size=(n, 2))
    s = np.where(np.arange(n) < 50, 1.0, -1.0)
    w = np.where(s > 0, 0.01, 0.64)
    ps = ParticleSet(pos, vel, w, s, (s < 0).astype(np.int32))
    assert np.isclose(kinetic_energy(ps), 0.5 * np.sum(w * (vel ** 2).sum(1)))
    assert np.isclose(kinetic_energy(ps, 2.0), 4 * kinetic_energy(ps))
    phi = rng.normal(size=n)
    assert np.isclose(potential_energy(ps, phi, 0.1), 0.5 * np.sum(s * w * phi) - 0.1)
    m = momenta(ps, centre=np.array([0.5, 0.5]))
    assert np.isclose(m['L_abs'], m['L_pos'] + m['L_neg']) and np.isclose(m['L_signed'], m['L_pos'] - m['L_neg'])
    assert np.isclose(m['p_abs_x'], np.sum(w * vel[:, 0]))
    # 3-D: norms
    ps3 = ParticleSet(rng.random((n, 3)), rng.normal(size=(n, 3)), w, s, (s < 0).astype(np.int32))
    m3 = momenta(ps3)
    assert m3['L_abs'] >= 0 and 'p_abs_z' in m3
    st = species_stats(ps, centre=np.array([0.5, 0.5]), r_escape=0.5)
    assert 0 <= st['escaped_frac_pos'] <= 1 and np.isfinite(st['sigma_neg'])
    P0 = np.array([m['p_abs_x'], m['p_abs_y']])
    row = diagnostics_row(ps, phi, 0.0, 0, 0.01, 1.0, 0.0, None, None)
    row2 = diagnostics_row(ps, phi, 0.01, 1, 0.01, 1.0, 0.0, row['E'], P0, wall=0.3)
    assert row2['dE_rel'] == 0.0 and row2['dP_abs_rel'] == 0.0 and np.isnan(row['dE_rel'])
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'd.csv')
        lg = CsvLogger(p)
        lg.log(row)
        r3 = dict(row2)
        r3.pop('virial')
        r3['extra'] = 1.0
        lg.log(r3)
        lg.close()
        d = read_csv(p)
        assert d['t'].shape == (2,) and np.isnan(d['virial'][1]) and 'extra' not in d
        assert lg.dropped == {'extra'}
        write_json(os.path.join(td, 'x.json'), {'a': np.float32(1.5), 'b': np.arange(3), 'c': float('nan')})
        with open(os.path.join(td, 'x.json')) as f:
            back = json.load(f)
        assert back['a'] == 1.5 and back['b'] == [0, 1, 2]
    print("diagnostics_core self-test OK")
