"""snapshot_torch.py -- snapshots of device-resident particles (matplotlib Agg).

The figures have the layout, colours, names, colour maps, extents and axis
labels of plotting.snapshot (2-D) and plotting_3d.snapshot_3d (3-D), but every
per-particle operation stays on the GPU:

  * density maps are CIC grids built with grids_torch.density_grid_t on the
    device (periodic for absolute views, non-periodic with origin lo for
    relative views, exactly like plotting._density_image), turned into
    log10(1 + rho/<rho>) with grids_torch.log_image_t and copied as one
    200 x 200 (or ngrid x ngrid) array per panel;
  * 3-D projections come from one (ngrid//2)^3 grid per population summed
    along the third axis (plotting_3d._projected_density), the mid-plane slice
    map from grids_torch.slab_density_t;
  * the scatter panels draw a fixed random subsample (ScatterSubsample:
    default 200000 particles per population, --scatter-subsample) whose index
    tensors are built once from numpy's default_rng(seed) and uploaded once;
    only those rows are copied per snapshot, and the panel title states it
    ('N = 6400000 (random subsample of 200000 shown)').

No full particle array ever leaves the device.  Imports: grids_torch,
diagnostics_torch.min_image_t, numpy, torch, matplotlib.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

try:
    import torch
    _TORCH_ERR = ''
except Exception as _exc:            # ImportError or a broken runtime
    torch = None
    _TORCH_ERR = repr(_exc)

from grids_torch import density_grid_t, slab_density_t, log_image_t  # noqa: E402
from diagnostics_torch import min_image_t  # noqa: E402

_COL = {'pos': 'tab:blue', 'neg': 'tab:red'}
_CMAP = {'pos': 'Blues', 'neg': 'Reds'}
_NAME = {'pos': 'ordinary matter (+)', 'neg': 'twin matter (-)'}
_AX = 'xyz'
_PROJ = ((0, 1), (0, 2), (1, 2))        # (horizontal, vertical) axis of each projection
_DENS_GRID = 200
DEFAULT_SUBSAMPLE = 200000


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _need_torch() -> None:
    if torch is None:
        raise RuntimeError(f"snapshot_torch needs PyTorch (import failed: {_TORCH_ERR})")


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


def _no_particles(ax, text: str = 'no particles') -> None:
    ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes, color='gray')


def _imshow(ax, img: np.ndarray, lo: float, hi: float, name: str, cb_label: str, small: bool = False):
    im = ax.imshow(img.T, origin='lower', extent=[lo, hi, lo, hi], cmap=_CMAP[name],
                   interpolation='nearest', aspect='equal')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if small:
        cb.set_label(cb_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        cb.set_label(cb_label)
    return im


def _centre_vec(centre, device, dim: int):
    """Host array or device tensor -> float64 tensor (dim,) on `device`."""
    if torch.is_tensor(centre):
        return centre.detach().to(device=device, dtype=torch.float64).reshape(-1)[:dim]
    return torch.as_tensor(np.asarray(centre, dtype=np.float64).reshape(-1)[:dim],
                           dtype=torch.float64, device=device)


def _relative_t(pos, centre, box):
    """plotting._species_positions transform on a device tensor: offsets from
    `centre` (minimum image when box > 0) in a relative view, unchanged otherwise."""
    if centre is None or pos.shape[0] == 0:
        return pos
    d = pos - _centre_vec(centre, pos.device, pos.shape[1])[None, :]
    if box is not None and box > 0:
        d = min_image_t(d, float(box))
    return d


def _mask(gp, name: str):
    return gp.s > 0 if name == 'pos' else gp.s < 0


def _window(box, centre, extent):
    """(relative, use_box, lo, hi, periodic) as in plotting / plotting_3d."""
    relative = centre is not None
    use_box = float(box) if (box is not None and box > 0) else None
    if relative:
        ext = float(extent) if extent is not None else (0.5 * use_box if use_box else 1.0)
        lo, hi = -ext, ext
    else:
        if use_box is None:
            raise ValueError("an absolute view needs box > 0")
        lo, hi = 0.0, use_box
    periodic = (not relative) and use_box is not None
    return relative, use_box, lo, hi, periodic


# ---------------------------------------------------------------------------
# device-side maps (public: reused by diagtest)
# ---------------------------------------------------------------------------

def species_view_t(gp, name: str, centre=None, box=None):
    """(positions (n, dim) float64, weights (n,) float64) of one species on the
    device, relative to `centre` (minimum image when box > 0) when given."""
    _need_torch()
    m = _mask(gp, name)
    pos = gp.pos.detach()[m].to(torch.float64)
    w = gp.w.detach()[m].to(torch.float64)
    return _relative_t(pos, centre, box), w


def density_map_t(pos, w, dim: int, lo: float, hi: float, periodic: bool, ngrid: int = _DENS_GRID):
    """log10(1 + rho/<rho>) map (ngrid, ngrid) of plotting._density_image on the
    device (3-D grids are summed along z); zeros when the grid is empty."""
    _need_torch()
    size = float(hi - lo)
    n, dim = int(ngrid), int(dim)
    if periodic:
        rho = density_grid_t(pos, w, size, n, dim, periodic=True)
    else:
        rho = density_grid_t(pos, w, size, n, dim, periodic=False, origin=np.full(dim, lo))
    if dim == 3:
        rho = rho.sum(2)
    return log_image_t(rho)


def projection_maps_t(pos, w, lo: float, hi: float, periodic: bool, n: int) -> dict:
    """{(a, b): Sigma (n, n)} surface densities on the xy, xz and yz planes from
    ONE n^3 CIC grid over [lo, hi)^3 (plotting_3d._projected_density for each
    plane: sum along the third axis times the cell, transposed when a > b)."""
    _need_torch()
    size = float(hi - lo)
    n = int(n)
    if pos.shape[0] == 0 or size <= 0:
        z = torch.zeros((n, n), dtype=torch.float64, device=pos.device)
        return {ab: z for ab in _PROJ}
    if periodic:
        rho = density_grid_t(pos, w, size, n, 3, periodic=True)
    else:
        rho = density_grid_t(pos, w, size, n, 3, periodic=False, origin=np.full(3, lo))
    cell = size / n
    out = {}
    for a, b in _PROJ:
        other = ({0, 1, 2} - {a, b}).pop()
        sigma = rho.sum(other) * cell
        if a > b:
            sigma = sigma.T
        out[(a, b)] = sigma
    return out


# ---------------------------------------------------------------------------
# fixed random scatter subsample
# ---------------------------------------------------------------------------

class ScatterSubsample:
    """Fixed random subsample of each population for the scatter panels.

    The member indices of s > 0 and s < 0 are found on the device; numpy's
    default_rng(seed) permutation of those indices is taken (pos first, then
    neg, same stream) and the first min(n_per_species, N_species) of each are
    uploaded once as int64 tensors (`idx['pos']`, `idx['neg']`).  Only the
    subsample rows are ever copied to the host (positions())."""

    def __init__(self, gp, n_per_species: int = DEFAULT_SUBSAMPLE, seed: int = 0):
        _need_torch()
        self.n_per_species = max(int(n_per_species), 0)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        dev = gp.pos.device
        self.idx, self.n_species = {}, {}
        for name in ('pos', 'neg'):
            members = torch.nonzero(_mask(gp, name), as_tuple=False).reshape(-1)   # sorted, device
            n_sp = int(members.numel())
            k = min(self.n_per_species, n_sp)
            perm = rng.permutation(n_sp)[:k]            # == rng.permutation(member indices)[:k]
            sel = torch.as_tensor(np.ascontiguousarray(perm), dtype=torch.int64, device=dev)
            self.idx[name] = members.index_select(0, sel).to(torch.int64)
            self.n_species[name] = n_sp
            del members

    @property
    def n_pos(self) -> int:
        return int(self.idx['pos'].numel())

    @property
    def n_neg(self) -> int:
        return int(self.idx['neg'].numel())

    def n_sub(self, name: str) -> int:
        return int(self.idx[name].numel())

    def positions(self, gp, name: str, centre=None, box=None) -> np.ndarray:
        """(k, dim) float64 host array of the subsample positions of one species,
        relative to `centre` (minimum image when box > 0) when given."""
        idx = self.idx[name]
        dim = int(gp.pos.shape[1])
        if idx.numel() == 0:
            return np.zeros((0, dim))
        pos = gp.pos.detach().index_select(0, idx.to(gp.pos.device)).to(torch.float64)
        return _relative_t(pos, centre, box).cpu().numpy()

    def describe(self, name: str) -> str:
        n_sp, k = self.n_species[name], self.n_sub(name)
        if n_sp == 0:
            return 'no particles'
        if k == 0:
            return 'no scatter subsample drawn'
        if k >= n_sp:
            return 'all shown'
        return f"random subsample of {k} shown"

    def __repr__(self) -> str:
        return (f"ScatterSubsample(pos {self.n_pos}/{self.n_species['pos']}, "
                f"neg {self.n_neg}/{self.n_species['neg']}, seed={self.seed})")


# ---------------------------------------------------------------------------
# 2-D snapshot
# ---------------------------------------------------------------------------

def _snapshot_2d(gp, box, t, t_ref_name, path, centre, extent, title, sub, ngrid) -> None:
    relative, use_box, lo, hi, periodic = _window(box, centre, extent)
    head = f"{title}  t = {t:.4g} {t_ref_name}".strip()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))
    for j, name in enumerate(('pos', 'neg')):
        pos_t, w_t = species_view_t(gp, name, centre, use_box)
        n_sp = int(pos_t.shape[0])
        ax = axes[0, j]
        if n_sp and sub.n_sub(name):
            xy = sub.positions(gp, name, centre, use_box)
            ax.scatter(xy[:, 0], xy[:, 1], s=_marker_size(xy.shape[0]), c=_COL[name], lw=0,
                       rasterized=True)
        elif n_sp == 0:
            _no_particles(ax)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_title(f"{_NAME[name]}: N = {n_sp} ({sub.describe(name)})", fontsize=10)
        ax = axes[1, j]
        if n_sp:
            img = density_map_t(pos_t, w_t, 2, lo, hi, periodic, ngrid).cpu().numpy()
            _imshow(ax, img, lo, hi, name, 'log10(1 + rho/<rho>)')
        else:
            _no_particles(ax)
        ax.set_title(f"{_NAME[name]}: density", fontsize=10)
        del pos_t, w_t
    for ax in axes.ravel():
        ax.set_xlabel(_label(0, relative))
        ax.set_ylabel(_label(1, relative))
    fig.suptitle(head, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)


# ---------------------------------------------------------------------------
# 3-D snapshot
# ---------------------------------------------------------------------------

def _slice_panel(ax, gp, name, sub, pos_t, w_t, centre, use_box, lo, hi, periodic, z_mid, half, n2):
    """Slab map + subsample scatter; returns (particles in the slab, subsample drawn)."""
    n_sp = int(pos_t.shape[0])
    if n_sp == 0:
        _no_particles(ax)
        return 0, 0
    rho, nsl = slab_density_t(pos_t, w_t, lo, hi, periodic, z_mid, half, n2)
    if nsl == 0:
        _no_particles(ax, 'no particles in the slice')
        return 0, 0
    _imshow(ax, log_image_t(rho).cpu().numpy(), lo, hi, name, 'log10(1 + Sigma/<Sigma>) in slice',
            small=True)
    n_in = 0
    if sub.n_sub(name):
        xyz = sub.positions(gp, name, centre, use_box)
        dz = xyz[:, 2] - z_mid
        if periodic:
            size = hi - lo
            dz = dz - size * np.round(dz / size)
        sel = np.abs(dz) < half
        n_in = int(sel.sum())
        if n_in:
            ax.scatter(xyz[sel, 0], xyz[sel, 1], s=_marker_size(n_in), c=_COL[name], lw=0, alpha=0.5,
                       rasterized=True)
    return nsl, n_in


def _snapshot_3d(gp, box, t, t_ref_name, path, centre, extent, title, sub, ngrid, slice_frac) -> None:
    relative, use_box, lo, hi, periodic = _window(box, centre, extent)
    n3 = max(8, int(ngrid) // 2)
    n2 = max(8, int(ngrid))
    z_mid = 0.0 if relative else 0.5 * (lo + hi)
    half = float(slice_frac) * (use_box if use_box is not None else (hi - lo))
    zl = _label(2, relative)
    head = f"{title}  t = {t:.4g} {t_ref_name}".strip()
    fig, axes = plt.subplots(2, 4, figsize=(19.0, 9.6))
    for i, name in enumerate(('pos', 'neg')):
        pos_t, w_t = species_view_t(gp, name, centre, use_box)
        n_sp = int(pos_t.shape[0])
        maps = projection_maps_t(pos_t, w_t, lo, hi, periodic, n3) if n_sp else None
        for j, (a, b) in enumerate(_PROJ):
            ax = axes[i, j]
            if n_sp:
                _imshow(ax, log_image_t(maps[(a, b)]).cpu().numpy(), lo, hi, name,
                        'log10(1 + Sigma/<Sigma>)', small=True)
            else:
                _no_particles(ax)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
            ax.set_xlabel(_label(a, relative))
            ax.set_ylabel(_label(b, relative))
            ax.set_title(f"{_NAME[name]}: {_AX[a]}{_AX[b]} projection, N = {n_sp}", fontsize=9)
        del maps
        ax = axes[i, 3]
        nsl, n_in = _slice_panel(ax, gp, name, sub, pos_t, w_t, centre, use_box, lo, hi, periodic,
                                 z_mid, half, n2)
        del pos_t, w_t
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(_label(0, relative))
        ax.set_ylabel(_label(1, relative))
        zc = '0' if relative else f"{z_mid:.3g}"
        k = sub.n_sub(name)
        if n_sp == 0:
            note = 'no particles'
        elif k == 0:
            note = 'no scatter subsample drawn'
        elif k >= n_sp:
            note = f"{n_in} shown (all particles in the subsample)"
        else:
            note = f"{n_in} of the {k}-particle random subsample shown"
        ax.set_title(f"{_NAME[name]}: slice |{zl} - {zc}| < {half:.3g}, {nsl} particles; {note}",
                     fontsize=8)
    fig.suptitle(head, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def snapshot_t(gp, box: float, t: float, t_ref_name: str, path: str, dim: int, centre=None,
               extent=None, title: str = '', subsample=None, ngrid: int = _DENS_GRID,
               slice_frac: float = 0.05) -> None:
    """Snapshot of device-resident particles with the layout of plotting.snapshot:
    2-D: [scatter +, scatter -] over [density +, density -]; 3-D: 2 x 4 panels
    (xy, xz, yz projected log density, thin mid-plane slice) per population.
    With `centre` (host array or device tensor) the view is relative to it
    (minimum image if box > 0) and spans +-extent (default half the box).
    `subsample` is a ScatterSubsample (built with DEFAULT_SUBSAMPLE and seed 0
    when None); only its rows and the density maps are copied to the host."""
    _need_torch()
    dim = int(dim)
    if int(gp.pos.shape[1]) != dim:
        raise ValueError(f"dim = {dim} does not match the particle set (dim = {gp.pos.shape[1]})")
    if subsample is None:
        subsample = ScatterSubsample(gp, DEFAULT_SUBSAMPLE, 0)
    if dim == 3:
        _snapshot_3d(gp, box, t, t_ref_name, path, centre, extent, title, subsample, ngrid, slice_frac)
    elif dim == 2:
        _snapshot_2d(gp, box, t, t_ref_name, path, centre, extent, title, subsample, ngrid)
    else:
        raise ValueError("snapshot_t supports dim = 2 or 3")


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-300))


if __name__ == '__main__':
    import sys
    import tempfile
    if torch is None:
        print("torch not installed, snapshot_torch self-test skipped:", _TORCH_ERR)
        sys.exit(0)
    from particles import ParticleSet, min_image, subset
    from gpu_backend import GPUParticles
    from forces_pm import density_grid
    from plotting_3d import _projected_density, _log_image
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu':
        print("no GPU: checking the torch snapshots on the CPU device")
    rng = np.random.default_rng(0)

    def forbid_host_copy(*a, **k):
        raise AssertionError("full particle array copied to the host")

    with tempfile.TemporaryDirectory() as td:
        n = 3000
        for dim in (2, 3):
            pos = rng.random((n, dim))
            pos[n // 2:] = np.mod(0.5 + 0.08 * rng.normal(size=(n // 2, dim)), 1.0)   # a twin clump
            s = np.where(np.arange(n) < n // 2, 1.0, -1.0)
            w = np.where(s > 0, 2.0 / n, 128.0 / n)
            ps = ParticleSet(pos, rng.normal(size=(n, dim)), w, s, (s < 0).astype(np.int32))
            gp = GPUParticles.from_host(ps, dev, 'float64')
            gp.to_host = forbid_host_copy
            # --- subsample: sizes, determinism, the documented numpy recipe ----------
            sub = ScatterSubsample(gp, 500, 7)
            sub2 = ScatterSubsample(gp, 500, 7)
            assert sub.n_pos == 500 and sub.n_neg == 500 and sub.n_species == {'pos': n // 2, 'neg': n // 2}
            assert torch.equal(sub.idx['pos'], sub2.idx['pos']) and torch.equal(sub.idx['neg'], sub2.idx['neg'])
            assert sub.idx['pos'].dtype == torch.int64 and sub.idx['pos'].device == dev
            ip, ineg = sub.idx['pos'].cpu().numpy(), sub.idx['neg'].cpu().numpy()
            assert len(set(ip.tolist())) == 500 and np.all(s[ip] > 0) and np.all(s[ineg] < 0)
            r = np.random.default_rng(7)
            assert np.array_equal(ip, r.permutation(np.flatnonzero(s > 0))[:500])
            assert np.array_equal(ineg, r.permutation(np.flatnonzero(s < 0))[:500])
            assert not torch.equal(sub.idx['pos'], ScatterSubsample(gp, 500, 8).idx['pos'])
            big = ScatterSubsample(gp, 10 ** 6, 1)
            assert big.n_pos == n // 2 and big.describe('pos') == 'all shown'
            none = ScatterSubsample(gp, 0, 1)
            assert none.n_pos == 0 and none.positions(gp, 'pos').shape == (0, dim)
            assert 'random subsample of 500' in sub.describe('neg')
            xy = sub.positions(gp, 'pos')
            assert xy.shape == (500, dim) and xy.dtype == np.float64 and np.array_equal(xy, pos[ip])
            c = np.full(dim, 0.5)
            xyr = sub.positions(gp, 'neg', c, 1.0)
            assert _rel(xyr, min_image(pos[ineg] - c, 1.0)) < 1e-15
            xyt = sub.positions(gp, 'neg', torch.as_tensor(c, device=dev), 1.0)
            assert np.array_equal(xyt, xyr)
            # --- density maps equal to plotting's numpy maps -------------------------
            for name in ('pos', 'neg'):
                m = s > 0 if name == 'pos' else s < 0
                pos_np, w_np = pos[m], w[m]
                pos_t, w_t = species_view_t(gp, name, None, 1.0)
                assert pos_t.dtype == torch.float64 and int(pos_t.shape[0]) == int(m.sum())
                ref = density_grid(pos_np, w_np, 1.0, 40, dim, periodic=True)
                if dim == 3:
                    ref = ref.sum(axis=2)
                got = density_map_t(pos_t, w_t, dim, 0.0, 1.0, True, 40).cpu().numpy()
                assert got.shape == (40, 40) and _rel(got, _log_image(ref)) < 1e-12, (dim, name)
                d_np = min_image(pos_np - c, 1.0)
                ref2 = density_grid(d_np, w_np, 0.6, 40, dim, periodic=False, origin=np.full(dim, -0.3))
                if dim == 3:
                    ref2 = ref2.sum(axis=2)
                pos_r, w_r = species_view_t(gp, name, c, 1.0)
                got2 = density_map_t(pos_r, w_r, dim, -0.3, 0.3, False, 40).cpu().numpy()
                assert _rel(got2, _log_image(ref2)) < 1e-12, (dim, name)
                if dim == 3:
                    maps = projection_maps_t(pos_t, w_t, 0.0, 1.0, True, 20)
                    for ab in _PROJ:
                        refp = _projected_density(pos_np, w_np, ab, 0.0, 1.0, True, 20)
                        assert _rel(maps[ab].cpu().numpy(), refp) < 1e-12, (name, ab)
                    maps_r = projection_maps_t(pos_r, w_r, -0.3, 0.3, False, 20)
                    for ab in _PROJ:
                        refp = _projected_density(d_np, w_np, ab, -0.3, 0.3, False, 20)
                        assert _rel(maps_r[ab].cpu().numpy(), refp) < 1e-12, (name, ab)
                    e = projection_maps_t(pos_t[:0], w_t[:0], 0.0, 1.0, True, 10)
                    assert tuple(e[(0, 1)].shape) == (10, 10)
            # --- figures written -------------------------------------------------------
            p1 = os.path.join(td, f'abs{dim}.png')
            snapshot_t(gp, 1.0, 0.5, 't_J,neg', p1, dim, title='lss', subsample=sub, ngrid=64)
            p2 = os.path.join(td, f'rel{dim}.png')
            snapshot_t(gp, 1.0, 1.5, 'T_rot(2R_d)', p2, dim, centre=torch.as_tensor(c, device=dev),
                       extent=0.3, title='relative view', subsample=sub, ngrid=64)
            p3 = os.path.join(td, f'all{dim}.png')
            snapshot_t(gp, 1.0, 0.0, 't_J,neg', p3, dim, centre=c, subsample=big, ngrid=32)
            p4 = os.path.join(td, f'none{dim}.png')
            snapshot_t(gp, 1.0, 0.0, 't_J,neg', p4, dim, subsample=none, ngrid=32, slice_frac=0.1)
            p5 = os.path.join(td, f'default{dim}.png')
            snapshot_t(gp, 1.0, 0.0, 't_J,neg', p5, dim, ngrid=32)                 # subsample built inside
            only = GPUParticles.from_host(subset(ps, s > 0), dev, 'float64')     # empty twin row
            only.to_host = forbid_host_copy
            p6 = os.path.join(td, f'one{dim}.png')
            snapshot_t(only, 1.0, 0.0, 't_J,neg', p6, dim, subsample=ScatterSubsample(only, 100, 3), ngrid=32)
            g32 = GPUParticles.from_host(ps, dev, 'float32')
            g32.to_host = forbid_host_copy
            p7 = os.path.join(td, f'f32_{dim}.png')
            snapshot_t(g32, 1.0, 0.0, 't_J,neg', p7, dim, subsample=ScatterSubsample(g32, 200, 1), ngrid=32)
            for p in (p1, p2, p3, p4, p5, p6, p7):
                assert os.path.getsize(p) > 1000, p
            try:
                snapshot_t(gp, 1.0, 0.0, 't', os.path.join(td, 'bad.png'), 5 - dim)
                raise AssertionError("dimension mismatch must be refused")
            except ValueError:
                pass
        print(repr(sub))
    print("snapshot_torch self-test OK")
