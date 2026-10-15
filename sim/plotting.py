"""plotting.py -- all figure output (matplotlib Agg backend only).

Species snapshots (2-D: scatter + density panels per species; 3-D: xy/xz/yz
projections per species), conservation time series (E, P_abs vs the
non-conserved P_signed, L_abs vs L_signed, virial), LSS structure time series,
disk mode time series, rotation curve, and sweep/ensemble comparison plots.
Nothing here computes physics; every function reads what the other modules
wrote and draws it.  Missing columns produce an 'n/a' panel, never an error.
"""
from __future__ import annotations

import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from particles import ParticleSet, min_image
from forces_pm import density_grid
from diagnostics_core import read_csv

_COL = {'pos': 'tab:blue', 'neg': 'tab:red'}
_NAME = {'pos': 'ordinary matter (+)', 'neg': 'twin matter (-)'}
_AX = 'xyz'
_DENS_GRID = 200


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _save(fig, path: str, dpi: int = 110) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _layout(n: int, ncols: int = 3, w: float = 4.3, h: float = 3.2):
    ncols = max(1, min(ncols, n))
    nrows = max(1, int(math.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def _marker_size(n: int) -> float:
    return float(np.clip(30000.0 / max(n, 1), 0.15, 6.0))


def _col(d: dict, name: str):
    """Numeric column or None."""
    v = d.get(name)
    if v is None or v.dtype.kind not in 'fiu':
        return None
    return np.asarray(v, dtype=np.float64)


def _series(ax, d: dict, t, names, labels=None, ylabel: str = '', title: str = '',
            logy: bool = False, styles=None):
    """Plot the available columns of `names` against t; 'n/a' if none exist."""
    labels = labels or names
    drawn = 0
    for k, name in enumerate(names):
        y = _col(d, name)
        if y is None or t is None or y.shape != t.shape:
            continue
        yy = np.abs(y) if logy else y
        if logy:
            yy = np.where(yy > 0, yy, np.nan)
        kw = (styles[k] if styles and k < len(styles) else {})
        ax.plot(t, yy, label=labels[k], lw=1.2, **kw)
        drawn += 1
    if drawn == 0:
        ax.text(0.5, 0.5, 'n/a', ha='center', va='center', transform=ax.transAxes, color='gray')
    else:
        if logy:
            ax.set_yscale('log')
        if drawn > 1 or labels[0] != ylabel:
            ax.legend(fontsize=7, frameon=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    return drawn


def _band(ax, t, mean, std, color, label):
    if mean is None or t is None:
        return
    ax.plot(t, mean, color=color, lw=1.2, label=label)
    if std is not None:
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2, lw=0)


def _time_axis(d: dict, t_ref, t_ref_name: str):
    t = _col(d, 't')
    if t is None:
        return None, 't'
    if t_ref is not None and t_ref > 0:
        return t / float(t_ref), f't / {t_ref_name}'
    return t, 't (code units)'


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------

def _species_positions(ps: ParticleSet, name: str, box, centre):
    mask = ps.pos_mask if name == 'pos' else ps.neg_mask
    pos = np.asarray(ps.pos[mask], dtype=np.float64)
    w = np.asarray(ps.w[mask], dtype=np.float64)
    if centre is not None and pos.shape[0]:
        d = pos - np.asarray(centre, dtype=np.float64)[None, :ps.dim]
        if box is not None and box > 0:
            d = min_image(d, box)
        pos = d
    return pos, w


def _density_image(ax, pos, w, dim, lo, hi, periodic, color_name):
    """log10(1 + rho/<rho>) of one species in the window [lo, hi]^2."""
    if pos.shape[0] == 0:
        ax.text(0.5, 0.5, 'no particles', ha='center', va='center', transform=ax.transAxes)
        return
    size = hi - lo
    if periodic:
        rho = density_grid(pos, w, size, _DENS_GRID, dim, periodic=True)
    else:
        rho = density_grid(pos, w, size, _DENS_GRID, dim, periodic=False,
                           origin=np.full(dim, lo))
    if dim == 3:
        rho = rho.sum(axis=2)
    m = float(np.mean(np.abs(rho)))
    img = np.log10(1.0 + np.abs(rho) / m) if m > 0 else rho
    cmap = 'Blues' if color_name == 'pos' else 'Reds'
    im = ax.imshow(img.T, origin='lower', extent=[lo, hi, lo, hi], cmap=cmap,
                   interpolation='nearest', aspect='equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='log10(1 + rho/<rho>)')


def snapshot(ps: ParticleSet, box: float, t: float, t_ref_name: str, path: str, dim: int,
             centre=None, extent=None, title: str = '') -> None:
    """2-D: [scatter +, scatter -] over [density +, density -]; 3-D: species rows x
    (xy, xz, yz) projection columns.  With `centre` the view is relative to it
    (minimum image if box > 0) and spans +-extent (default half the box)."""
    relative = centre is not None
    if relative:
        ext = float(extent) if extent is not None else (0.5 * box if box else 1.0)
        lo, hi = -ext, ext
    else:
        lo, hi = 0.0, float(box)
    periodic = (not relative) and box is not None and box > 0
    data = {name: _species_positions(ps, name, box if box else None, centre) for name in ('pos', 'neg')}
    head = f"{title}  t = {t:.4g} {t_ref_name}".strip()
    if dim == 2:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))
        for j, name in enumerate(('pos', 'neg')):
            pos, w = data[name]
            ax = axes[0, j]
            if pos.shape[0]:
                ax.scatter(pos[:, 0], pos[:, 1], s=_marker_size(pos.shape[0]), c=_COL[name],
                           lw=0, rasterized=True)
            else:
                ax.text(0.5, 0.5, 'no particles', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
            ax.set_title(f"{_NAME[name]}: N = {pos.shape[0]}", fontsize=10)
            _density_image(axes[1, j], pos, w, 2, lo, hi, periodic, name)
            axes[1, j].set_title(f"{_NAME[name]}: density", fontsize=10)
        for ax in axes.ravel():
            ax.set_xlabel('x' if not relative else 'x - x_c')
            ax.set_ylabel('y' if not relative else 'y - y_c')
    else:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
        for i, name in enumerate(('pos', 'neg')):
            pos, _ = data[name]
            for j, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                ax = axes[i, j]
                if pos.shape[0]:
                    ax.scatter(pos[:, a], pos[:, b], s=_marker_size(pos.shape[0]), c=_COL[name],
                               lw=0, rasterized=True, alpha=0.6)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect('equal')
                ax.set_xlabel(_AX[a])
                ax.set_ylabel(_AX[b])
                ax.set_title(f"{_NAME[name]} ({_AX[a]}{_AX[b]}), N = {pos.shape[0]}", fontsize=9)
    fig.suptitle(head, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)


# ---------------------------------------------------------------------------
# conservation
# ---------------------------------------------------------------------------

def plot_conservation(csv_path: str, out_png: str, expansion: bool) -> None:
    """Energy drift, |m|-weighted momentum drift (conserved) next to the signed
    momentum (not conserved: expected physics), angular momenta, virial ratio,
    energy budget and a(t) or dispersions."""
    d = read_csv(csv_path)
    t = _col(d, 't')
    fig, axes = _layout(6, 3)
    _series(axes[0], d, t, ['dE_rel'], ['(E - E0)/|E0|'], 'dE/|E0|', 'energy conservation')
    axes[0].axhline(0, color='k', lw=0.5)
    _series(axes[1], d, t, ['dP_abs_rel', 'dP_signed_rel'],
            ['|dP_abs| / sum|m||v|  (conserved)', '|P_signed| / sum|m||v|  (NOT conserved, expected)'],
            'momentum drift', 'linear momentum', logy=True,
            styles=[{'color': 'tab:green'}, {'color': 'tab:orange', 'ls': '--'}])
    _series(axes[2], d, t, ['L_abs', 'L_signed', 'L_pos', 'L_neg'],
            ['L_abs (conserved)', 'L_signed (not conserved)', 'L_+', 'L_-'], 'L', 'angular momentum',
            styles=[{'color': 'k'}, {'color': 'gray', 'ls': '--'}, {'color': _COL['pos']}, {'color': _COL['neg']}])
    _series(axes[3], d, t, ['virial'], ['2 KE / |U|'], 'virial ratio', 'virial ratio')
    axes[3].axhline(1.0, color='k', lw=0.5, ls=':')
    _series(axes[4], d, t, ['KE', 'PE', 'E'], ['KE', 'U', 'E'], 'energy', 'energy budget')
    if expansion:
        _series(axes[5], d, t, ['a'], ['a(t)'], 'scale factor', 'kinematic expansion (A15)')
    else:
        _series(axes[5], d, t, ['sigma_pos', 'sigma_neg'], ['sigma_+', 'sigma_-'], 'velocity dispersion',
                'species dispersions', styles=[{'color': _COL['pos']}, {'color': _COL['neg']}])
    for ax in axes:
        ax.set_xlabel('t (code units)')
    fig.suptitle(os.path.basename(os.path.dirname(os.path.abspath(csv_path))) + ': conservation', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_png)


# ---------------------------------------------------------------------------
# LSS structure time series
# ---------------------------------------------------------------------------

def plot_lss_structure(csv_path: str, out_png: str, t_ref_name: str, t_ref=None) -> None:
    """Clump counts, clump mass/diameter (mean +- std), cell size and phi/d,
    segregation index, void fractions, power-spectrum peak scales, persistence.
    `t_ref` (optional) rescales the time axis to reference units."""
    d = read_csv(csv_path)
    t, tlab = _time_axis(d, t_ref, t_ref_name)
    fig, axes = _layout(9, 3)
    _series(axes[0], d, t, ['n_clumps_neg', 'n_clumps_pos'], ['twin clumps', 'ordinary clumps'],
            'count', 'clump number', styles=[{'color': _COL['neg']}, {'color': _COL['pos']}])
    ax = axes[1]
    _band(ax, t, _col(d, 'mass_mean'), _col(d, 'mass_std'), _COL['neg'], 'twin clump mass (mean +- std)')
    ax.set_ylabel('mass')
    ax.set_title('clump mass', fontsize=9)
    ax = axes[2]
    _band(ax, t, _col(d, 'diam_mean'), _col(d, 'diam_std'), _COL['neg'], 'twin clump diameter phi (mean +- std)')
    ax.set_ylabel('phi')
    ax.set_title('clump diameter', fontsize=9)
    _series(axes[3], d, t, ['d_cell', 'diam_mean'], ['cell size d (NN distance)', 'clump diameter phi'],
            'length', 'cell size')
    _series(axes[4], d, t, ['phi_over_d'], ['phi / d'], 'phi / d', 'clump-to-cell diameter ratio')
    _series(axes[5], d, t, ['segregation'], ['corr(log rho_+, log rho_-)'], 'segregation index', 'segregation')
    axes[5].axhline(-0.3, color='k', lw=0.6, ls=':', label='criterion -0.3')
    axes[5].axhline(0.0, color='k', lw=0.4)
    _series(axes[6], d, t, ['void_frac_pos', 'void_frac_neg'], ['ordinary', 'twin'], 'void fraction',
            'void fraction (rho < 0.2 <rho>)', styles=[{'color': _COL['pos']}, {'color': _COL['neg']}])
    _series(axes[7], d, t, ['lambda_peak_pos', 'lambda_peak_neg'], ['ordinary', 'twin'], 'lambda_peak',
            'power-spectrum peak scale', styles=[{'color': _COL['pos']}, {'color': _COL['neg']}])
    _series(axes[8], d, t, ['persistence'], ['corr(rho_-(t), rho_-(t_ref))'], 'persistence', 'pattern persistence')
    axes[8].axhline(0.7, color='k', lw=0.6, ls=':')
    for ax in axes:
        ax.set_xlabel(tlab)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle('LSS structure statistics', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_png)


# ---------------------------------------------------------------------------
# galaxy modes
# ---------------------------------------------------------------------------

def plot_disk_modes(csv_path: str, out_png: str, t_ref=None, t_ref_name: str = 'T_rot') -> None:
    """Bar (inner) and spiral (outer) A2, other modes, pitch angle, pattern speed,
    disk L_z against the twin ring's |m|-weighted L_z (braking test), twin clump
    count, Toomre Q, mean rotation speed and escaped fraction."""
    d = read_csv(csv_path)
    t, tlab = _time_axis(d, t_ref, t_ref_name)
    fig, axes = _layout(9, 3)
    _series(axes[0], d, t, ['bar_A2', 'spiral_A2'], ['bar A2 (R < 2 R_d)', 'spiral A2 (R > 2 R_d)'],
            'A2', 'm = 2 amplitudes')
    axes[0].axhline(0.2, color='tab:blue', lw=0.6, ls=':')
    axes[0].axhline(0.1, color='tab:orange', lw=0.6, ls=':')
    _series(axes[1], d, t, ['A1', 'A3', 'A4'], ['A1', 'A3', 'A4'], 'A_m', 'other modes (whole disk)')
    _series(axes[2], d, t, ['pitch_deg'], ['pitch angle'], 'degrees', 'm = 2 pitch angle')
    _series(axes[3], d, t, ['omega_p'], ['Omega_p'], 'pattern speed', 'm = 2 pattern speed')
    ax = axes[4]
    lz_d, lz_t = _col(d, 'Lz_disk'), _col(d, 'Lz_twin_abs')
    if t is not None and lz_d is not None:
        ax.plot(t, lz_d, color=_COL['pos'], label='L_z disk')
        if lz_t is not None:
            ax.plot(t, lz_t, color=_COL['neg'], label='L_z twin (|m|-weighted)')
            ax.plot(t, lz_d + lz_t, color='k', ls='--', lw=0.8, label='sum (conserved)')
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, 'n/a', ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.set_ylabel('L_z')
    ax.set_title('braking: disk vs twin angular momentum', fontsize=9)
    _series(axes[5], d, t, ['n_clumps_twin', 'twin_clump_mass_frac'], ['twin clumps', 'mass fraction in clumps'],
            'count / fraction', 'twin ring fragmentation')
    _series(axes[6], d, t, ['Q_at_2rd'], ['Q(2 R_d)'], 'Toomre Q', 'measured Toomre Q')
    _series(axes[7], d, t, ['vrot_disk'], ['<v_phi> disk'], 'v_rot', 'mean disk rotation')
    _series(axes[8], d, t, ['escaped_frac', 'r_rms_disk', 'r_rms_twin'],
            ['escaped fraction', 'r_rms disk', 'r_rms twin'], '', 'escape / sizes')
    for ax in axes:
        ax.set_xlabel(tlab)
        ax.grid(alpha=0.3)
    fig.suptitle('disk modes and twin ring response', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_png)


# ---------------------------------------------------------------------------
# rotation curve
# ---------------------------------------------------------------------------

def plot_rotation_curve(rc: dict, out_png: str) -> None:
    """Measured circular speed (nan where the net force is outward), mean v_phi
    and dispersions, radial acceleration, surface density, Omega and kappa."""
    R = np.asarray(rc.get('R', []), dtype=np.float64)
    fig, axes = _layout(4, 2)
    ax = axes[0]
    vc = np.asarray(rc.get('vc', np.full_like(R, np.nan)), dtype=np.float64)
    ax.plot(R, vc, 'k-o', ms=3, label='v_c = sqrt(R g_R)')
    for key, lab, col in (('vphi_mean', '<v_phi> (asym. drift)', 'tab:blue'),
                          ('sigma_R', 'sigma_R', 'tab:orange'), ('sigma_phi', 'sigma_phi', 'tab:green')):
        if key in rc:
            ax.plot(R, np.asarray(rc[key], dtype=np.float64), label=lab, color=col, lw=1.0)
    bad = ~np.isfinite(vc)
    if bad.any():
        for r in R[bad]:
            ax.axvspan(r - 0.5 * (R[1] - R[0] if R.size > 1 else 0.1),
                       r + 0.5 * (R[1] - R[0] if R.size > 1 else 0.1), color='red', alpha=0.08, lw=0)
        ax.plot([], [], color='red', alpha=0.3, lw=6, label='v_c undefined (outward force)')
    ax.set_ylabel('velocity')
    ax.set_title(f"rotation curve (undefined for {100 * rc.get('frac_undefined', 0.0):.1f}% of disk, "
                 f"Q = {rc.get('Q', float('nan')):g})", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    g = np.asarray(rc.get('g_r', np.full_like(R, np.nan)), dtype=np.float64)
    ax.plot(R, g, 'k-o', ms=3)
    ax.axhline(0, color='red', lw=0.8, ls=':')
    ax.fill_between(R, np.minimum(g, 0), 0, color='red', alpha=0.2, lw=0)
    ax.set_ylabel('-<a_R>')
    ax.set_title('inward radial acceleration on the disk (red: outward)', fontsize=9)
    ax = axes[2]
    sig = np.asarray(rc.get('sigma_surface', np.full_like(R, np.nan)), dtype=np.float64)
    ax.semilogy(R, np.where(sig > 0, sig, np.nan), 'k-o', ms=3)
    ax.set_ylabel('Sigma(R)')
    ax.set_title('disk surface density', fontsize=9)
    ax = axes[3]
    for key, lab, col in (('omega', 'Omega', 'tab:blue'), ('kappa', 'kappa', 'tab:orange')):
        if key in rc:
            ax.plot(R, np.asarray(rc[key], dtype=np.float64), label=lab, color=col)
    ax.set_ylabel('frequency')
    ax.set_title('Omega and epicyclic frequency', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.set_xlabel('R')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_png)


# ---------------------------------------------------------------------------
# sweeps and ensembles
# ---------------------------------------------------------------------------

def _to_float(v):
    try:
        if isinstance(v, (bool, np.bool_)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _metric(rows, name):
    out = []
    for r in rows:
        f = _to_float(r.get(name))
        out.append(float('nan') if f is None else f)
    return np.array(out, dtype=np.float64)


def plot_sweep(rows: list, param: str, metrics: list, out_png: str, logx: bool = False) -> None:
    """One panel per metric against the swept parameter.  Numeric values are
    sorted on a linear or log axis; categorical values (ic, method, bc, ...)
    are placed at integer positions with labelled ticks.  All values shown."""
    if not rows:
        return
    raw = [r.get(param, r.get('value')) for r in rows]
    xs = [_to_float(v) for v in raw]
    numeric = all(x is not None for x in xs) and not (logx and any(x <= 0 for x in xs))
    if numeric:
        order = np.argsort(np.array(xs))
        x = np.array(xs)[order]
    else:
        order = np.arange(len(rows))
        x = np.arange(len(rows), dtype=np.float64)
    fig, axes = _layout(len(metrics), 3)
    for ax, m in zip(axes, metrics):
        y = _metric(rows, m)[order]
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], 'o-', color='tab:blue', ms=5)
        if (~ok).any():
            ax.plot(x[~ok], np.zeros((~ok).sum()), 'x', color='red', label='nan / not measured')
            ax.legend(fontsize=7, frameon=False)
        if numeric and logx:
            ax.set_xscale('log')
        if not numeric:
            ax.set_xticks(x)
            ax.set_xticklabels([str(raw[i]) for i in order], rotation=30, fontsize=8)
        ax.set_xlabel(param)
        ax.set_ylabel(m)
        ax.set_title(m, fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f'sweep over {param}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_png)


def plot_ensemble(rows: list, metrics: list, out_png: str) -> None:
    """Per-seed values of each metric with the mean (line) and +-1 std (band);
    titles report mean, std and the coefficient of variation used by the
    'reproducible' criterion (seed-to-seed std < 20 % of the mean)."""
    if not rows:
        return
    seeds = [r.get('seed', i) for i, r in enumerate(rows)]
    x = np.arange(len(rows), dtype=np.float64)
    fig, axes = _layout(len(metrics), 3)
    for ax, m in zip(axes, metrics):
        y = _metric(rows, m)
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], 'o', color='tab:blue', ms=5)
        if ok.sum() >= 1:
            mu = float(np.mean(y[ok]))
            sd = float(np.std(y[ok])) if ok.sum() > 1 else 0.0
            ax.axhline(mu, color='k', lw=1.0)
            ax.axhspan(mu - sd, mu + sd, color='k', alpha=0.12, lw=0)
            cv = (sd / abs(mu)) if mu != 0 else float('nan')
            ax.set_title(f"{m}: {mu:.4g} +- {sd:.3g} (std/mean = {cv:.2f})", fontsize=8)
        else:
            ax.set_title(f"{m}: n/a", fontsize=8)
        if (~ok).any():
            ax.plot(x[~ok], np.zeros((~ok).sum()), 'x', color='red')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds], fontsize=8)
        ax.set_xlabel('seed')
        ax.set_ylabel(m)
        ax.grid(alpha=0.3)
    fig.suptitle(f'ensemble of {len(rows)} seeds', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_png)


if __name__ == '__main__':  # minimal self-test
    import tempfile
    from diagnostics_core import CsvLogger
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        n = 600
        s = np.where(np.arange(n) < n // 2, 1.0, -1.0)
        ps2 = ParticleSet(rng.random((n, 2)), rng.normal(size=(n, 2)), np.ones(n), s, (s < 0).astype(np.int32))
        snapshot(ps2, 1.0, 0.5, 't_J,neg', os.path.join(td, 's2.png'), 2, title='lss')
        snapshot(ps2, 1.0, 0.5, 'T_rot', os.path.join(td, 's2w.png'), 2, centre=np.array([0.5, 0.5]), extent=0.3)
        ps3 = ParticleSet(rng.random((n, 3)), rng.normal(size=(n, 3)), np.ones(n), s, (s < 0).astype(np.int32))
        snapshot(ps3, 1.0, 1.0, 't_J,neg', os.path.join(td, 's3.png'), 3)
        lg = CsvLogger(os.path.join(td, 'diag.csv'))
        for k in range(5):
            lg.log({'t': 0.1 * k, 'dE_rel': 1e-4 * k, 'dP_abs_rel': 1e-12, 'dP_signed_rel': 0.01 * k,
                    'L_abs': 1.0, 'L_signed': 0.5 - 0.1 * k, 'L_pos': 0.7, 'L_neg': 0.3, 'virial': 0.9,
                    'KE': 1.0, 'PE': -2.0, 'E': -1.0, 'a': 1.0 + k, 'sigma_pos': 0.1, 'sigma_neg': 0.4})
        lg.close()
        plot_conservation(os.path.join(td, 'diag.csv'), os.path.join(td, 'c.png'), True)
        lg = CsvLogger(os.path.join(td, 'struct.csv'))
        for k in range(5):
            lg.log({'t': 0.5 * k, 'n_clumps_neg': 60 - k, 'n_clumps_pos': 3, 'mass_mean': 1.0, 'mass_std': 0.2,
                    'diam_mean': 0.05, 'diam_std': 0.01, 'd_cell': 0.12, 'phi_over_d': 0.4,
                    'segregation': -0.5, 'void_frac_pos': 0.3, 'void_frac_neg': 0.5, 'lambda_peak_pos': 0.1,
                    'lambda_peak_neg': 0.12, 'persistence': 0.9})
        lg.close()
        plot_lss_structure(os.path.join(td, 'struct.csv'), os.path.join(td, 'l.png'), 't_J,neg', t_ref=0.5)
        lg = CsvLogger(os.path.join(td, 'modes.csv'))
        for k in range(5):
            lg.log({'t': k, 'bar_A2': 0.3, 'spiral_A2': 0.15, 'A1': 0.05, 'A3': 0.02, 'A4': 0.01,
                    'pitch_deg': 20.0, 'omega_p': 0.5, 'Lz_disk': 1.0 - 0.02 * k, 'Lz_twin_abs': 0.02 * k,
                    'n_clumps_twin': k, 'Q_at_2rd': 1.5, 'vrot_disk': 0.6, 'escaped_frac': 0.0})
        lg.close()
        plot_disk_modes(os.path.join(td, 'modes.csv'), os.path.join(td, 'm.png'))
        R = np.linspace(0.1, 4.0, 20)
        rc = {'R': R, 'vc': np.where(R < 3.5, 0.5 * np.sqrt(R), np.nan), 'g_r': np.where(R < 3.5, 0.25, -0.1),
              'sigma_surface': np.exp(-R), 'omega': 0.5 / np.sqrt(R), 'kappa': 0.7 / np.sqrt(R),
              'frac_undefined': 0.05, 'Q': 1.5, 'sigma_R': 0.1 * np.ones(20)}
        plot_rotation_curve(rc, os.path.join(td, 'rc.png'))
        rows = [{'mass_ratio': v, 'n_clumps_neg_final': 10 * v, 'segregation_final': -0.1 * v} for v in (1, 4, 16)]
        plot_sweep(rows, 'mass_ratio', ['n_clumps_neg_final', 'segregation_final'], os.path.join(td, 'sw.png'), logx=True)
        rows = [{'ic': v, 'n_clumps_neg_final': 10} for v in ('poisson', 'glass')]
        plot_sweep(rows, 'ic', ['n_clumps_neg_final', 'missing'], os.path.join(td, 'sw2.png'))
        plot_ensemble([{'seed': k, 'n_clumps_neg_final': 50 + k} for k in range(4)], ['n_clumps_neg_final', 'x'],
                      os.path.join(td, 'en.png'))
        for f in ('s2.png', 's2w.png', 's3.png', 'c.png', 'l.png', 'm.png', 'rc.png', 'sw.png', 'sw2.png', 'en.png'):
            assert os.path.getsize(os.path.join(td, f)) > 1000, f
    print("plotting self-test OK")
