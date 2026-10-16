"""plotting_3d.py -- informative snapshots of 3-D runs (matplotlib Agg only).

A 3-D particle set is hard to read as three scatter projections: with 10^5
points every panel saturates.  snapshot_3d draws, per population (ordinary
matter row, twin matter row):

  columns 1-3 : projected surface density of the whole set on the xy, xz and
                yz planes, shown as log10(1 + Sigma/<Sigma>).  The density is
                a 3-D CIC grid (forces_pm.density_grid, ngrid//2 cells per side)
                summed along the third axis, so the three maps are exactly
                consistent with each other and with the structure statistics;
  column 4    : a thin slice |z - z_mid| < slice_frac * box (z_mid = the centre
                of the view: box/2 in an absolute view, the given centre in a
                relative one) as a 2-D density map of the particles inside the
                slab with a scatter overlay, titled with the slab population.

Absolute views (no centre) of a periodic box use periodic CIC; relative views
(centre given) use minimum-image offsets, a +-extent window and non-periodic
CIC with the window origin, exactly like plotting.snapshot in 2-D.  Nothing
here computes physics; only particles and forces_pm.density_grid are imported.
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

_COL = {'pos': 'tab:blue', 'neg': 'tab:red'}
_CMAP = {'pos': 'Blues', 'neg': 'Reds'}
_NAME = {'pos': 'ordinary matter (+)', 'neg': 'twin matter (-)'}
_AX = 'xyz'
_PROJ = ((0, 1), (0, 2), (1, 2))        # (horizontal, vertical) axis of each projection


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _save(fig, path: str, dpi: int = 110) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _marker_size(n: int) -> float:
    return float(np.clip(30000.0 / max(n, 1), 0.15, 6.0))


def _label(k: int, relative: bool) -> str:
    return f"{_AX[k]} - {_AX[k]}_c" if relative else _AX[k]


def _species_positions(ps: ParticleSet, name: str, box, centre):
    """(positions (N,3) float64, weights (N,)) of one species; offsets from
    `centre` (minimum image when box > 0) in a relative view."""
    mask = ps.pos_mask if name == 'pos' else ps.neg_mask
    pos = np.asarray(ps.pos[mask], dtype=np.float64)
    w = np.asarray(ps.w[mask], dtype=np.float64)
    if centre is not None and pos.shape[0]:
        d = pos - np.asarray(centre, dtype=np.float64)[None, :3]
        if box is not None and box > 0:
            d = min_image(d, box)
        pos = d
    return pos, w


def _log_image(rho: np.ndarray) -> np.ndarray:
    """log10(1 + |rho| / <|rho|>); zeros when the grid is empty."""
    rho = np.abs(np.asarray(rho, dtype=np.float64))
    m = float(rho.mean()) if rho.size else 0.0
    return np.log10(1.0 + rho / m) if m > 0 else np.zeros_like(rho)


def _no_particles(ax, text: str = 'no particles') -> None:
    ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes, color='gray')


def _imshow(ax, img: np.ndarray, lo: float, hi: float, name: str, cb_label: str):
    im = ax.imshow(img.T, origin='lower', extent=[lo, hi, lo, hi], cmap=_CMAP[name],
                   interpolation='nearest', aspect='equal')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cb_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return im


# ---------------------------------------------------------------------------
# projected density
# ---------------------------------------------------------------------------

def _projected_density(pos: np.ndarray, w: np.ndarray, axes, lo: float, hi: float,
                       periodic: bool, n: int) -> np.ndarray:
    """Surface density Sigma (mass per unit area) of the particles projected on the
    plane spanned by `axes` (two of 0, 1, 2): a 3-D CIC grid of n^3 cells over
    [lo, hi)^3 (periodic, or non-periodic with origin lo so particles outside the
    window are dropped) summed along the third axis and multiplied by the cell
    size.  Returned as (n, n) indexed [axes[0], axes[1]].  Mass is conserved:
    Sigma.sum() * cell^2 = total mass inside the window."""
    pos = np.asarray(pos, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = int(n)
    size = float(hi - lo)
    if pos.shape[0] == 0 or size <= 0:
        return np.zeros((n, n))
    if periodic:
        rho = density_grid(pos, w, size, n, 3, periodic=True)
    else:
        rho = density_grid(pos, w, size, n, 3, periodic=False, origin=np.full(3, lo))
    a, b = int(axes[0]), int(axes[1])
    other = ({0, 1, 2} - {a, b}).pop()
    sigma = rho.sum(axis=other) * (size / n)
    if a > b:                                   # remaining axes come out in increasing order
        sigma = sigma.T
    return sigma


def _slice_panel(ax, pos: np.ndarray, w: np.ndarray, lo: float, hi: float, periodic: bool,
                 z_mid: float, half: float, name: str, n2d: int) -> int:
    """Thin slab |z - z_mid| < half drawn as a 2-D density map (xy) of the slab
    particles with a scatter overlay; returns the number of particles in the slab."""
    if pos.shape[0] == 0:
        _no_particles(ax)
        return 0
    size = float(hi - lo)
    dz = pos[:, 2] - z_mid
    if periodic:
        dz = min_image(dz, size)
    sel = np.abs(dz) < half
    nsl = int(sel.sum())
    if nsl == 0:
        _no_particles(ax, 'no particles in the slice')
        return 0
    p2, w2 = pos[sel, :2], w[sel]
    if periodic:
        rho = density_grid(p2, w2, size, n2d, 2, periodic=True)
    else:
        rho = density_grid(p2, w2, size, n2d, 2, periodic=False, origin=np.full(2, lo))
    _imshow(ax, _log_image(rho), lo, hi, name, 'log10(1 + Sigma/<Sigma>) in slice')
    ax.scatter(p2[:, 0], p2[:, 1], s=_marker_size(nsl), c=_COL[name], lw=0, alpha=0.5,
               rasterized=True)
    return nsl


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------

def snapshot_3d(ps: ParticleSet, box: float, t: float, t_ref_name: str, path: str,
                centre=None, extent=None, title: str = '', slice_frac: float = 0.05,
                ngrid: int = 200) -> None:
    """2 rows (ordinary, twin) x 4 columns (xy, xz, yz projected log density, thin
    mid-plane slice).  With `centre` the view is relative to it (minimum image if
    box > 0) and spans +-extent (default half the box); otherwise it is the whole
    box [0, box)^3 with periodic density estimates when box > 0."""
    if ps.dim != 3:
        raise ValueError(f"snapshot_3d needs a 3-D particle set (dim = {ps.dim})")
    relative = centre is not None
    use_box = float(box) if (box is not None and box > 0) else None
    if relative:
        ext = float(extent) if extent is not None else (0.5 * use_box if use_box else 1.0)
        lo, hi = -ext, ext
    else:
        if use_box is None:
            raise ValueError("an absolute 3-D view needs box > 0")
        lo, hi = 0.0, use_box
    periodic = (not relative) and use_box is not None
    n3 = max(8, int(ngrid) // 2)
    n2 = max(8, int(ngrid))
    z_mid = 0.0 if relative else 0.5 * (lo + hi)
    half = float(slice_frac) * (use_box if use_box is not None else (hi - lo))
    zl = _label(2, relative)
    head = f"{title}  t = {t:.4g} {t_ref_name}".strip()
    fig, axes = plt.subplots(2, 4, figsize=(19.0, 9.6))
    for i, name in enumerate(('pos', 'neg')):
        pos, w = _species_positions(ps, name, use_box, centre)
        n_sp = int(pos.shape[0])
        for j, (a, b) in enumerate(_PROJ):
            ax = axes[i, j]
            if n_sp:
                sigma = _projected_density(pos, w, (a, b), lo, hi, periodic, n3)
                _imshow(ax, _log_image(sigma), lo, hi, name, 'log10(1 + Sigma/<Sigma>)')
            else:
                _no_particles(ax)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
            ax.set_xlabel(_label(a, relative))
            ax.set_ylabel(_label(b, relative))
            ax.set_title(f"{_NAME[name]}: {_AX[a]}{_AX[b]} projection, N = {n_sp}", fontsize=9)
        ax = axes[i, 3]
        nsl = _slice_panel(ax, pos, w, lo, hi, periodic, z_mid, half, name, n2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(_label(0, relative))
        ax.set_ylabel(_label(1, relative))
        zc = '0' if relative else f"{z_mid:.3g}"
        ax.set_title(f"{_NAME[name]}: slice |{zl} - {zc}| < {half:.3g}, {nsl} particles", fontsize=9)
    fig.suptitle(head, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)


if __name__ == '__main__':  # minimal self-test
    import tempfile
    rng = np.random.default_rng(0)
    n = 3000
    s = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    pos = rng.random((n, 3))
    pos[n // 2:] = np.mod(0.5 + 0.08 * rng.normal(size=(n // 2, 3)), 1.0)   # a twin clump
    w = np.where(s > 0, 2.0 / n, 128.0 / n)
    ps = ParticleSet(pos, rng.normal(size=(n, 3)), w, s, (s < 0).astype(np.int32))
    # projection conserves mass; the three projections are consistent
    pp, wp = _species_positions(ps, 'neg', 1.0, None)
    sig = _projected_density(pp, wp, (0, 1), 0.0, 1.0, True, 20)
    assert sig.shape == (20, 20) and math.isclose(float(sig.sum()) * (1.0 / 20) ** 2, float(wp.sum()), rel_tol=1e-9)
    sig2 = _projected_density(pp, wp, (1, 0), 0.0, 1.0, True, 20)
    assert np.allclose(sig2, sig.T)
    sig_np = _projected_density(pp, wp, (0, 2), 0.0, 1.0, False, 20)
    assert sig_np.shape == (20, 20) and sig_np.sum() > 0
    assert _projected_density(np.zeros((0, 3)), np.zeros(0), (0, 1), 0.0, 1.0, True, 10).shape == (10, 10)
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, 'abs.png')
        snapshot_3d(ps, 1.0, 0.5, 't_J,neg', p1, title='lss 3-D')
        assert os.path.getsize(p1) > 1000
        p2 = os.path.join(td, 'rel.png')
        snapshot_3d(ps, 1.0, 1.5, 'T_rot(2R_d)', p2, centre=np.array([0.5, 0.5, 0.5]), extent=0.3,
                    title='relative view')
        assert os.path.getsize(p2) > 1000
        # single population (empty twin row) and a tiny grid
        only = ParticleSet(pos[: n // 2], np.zeros((n // 2, 3)), w[: n // 2], s[: n // 2],
                           np.zeros(n // 2, np.int32))
        p3 = os.path.join(td, 'one.png')
        snapshot_3d(only, 1.0, 0.0, 't_J,neg', p3, ngrid=32, slice_frac=0.1)
        assert os.path.getsize(p3) > 1000
    print("plotting_3d self-test OK")
