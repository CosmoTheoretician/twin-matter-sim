"""forces_pm.py -- FFT particle-mesh solver for the three-law (signed-mass) force.

Scheme (see notes, section 1 and 3):
  * the *signed* density rho_s = sum_j s_j w_j W(x - x_j) is deposited on a grid
    with the cloud-in-cell (CIC) kernel (node-centred grid, node i at x = i*cell);
  * one Poisson solve gives the unsigned potential Phi of that signed density,
        periodic : Phi_k = K(k) rho_k, with K = -2 pi G/|k| (2-D sheet, 1/r^2 law),
                   -2 pi G/k^2 (2-D logarithmic law), -4 pi G/k^2 (3-D);
                   the k = 0 mode is removed (Jeans swindle, A3); optional
                   Gaussian smoothing exp(-k^2 eps^2/2) as extra softening (A8);
        isolated : Hockney-Eastwood zero-padded doubled grid convolved with the
                   real-space Green function of the same law, Plummer-softened
                   with max(eps, cell) (no background subtraction, full monopole);
  * the acceleration of a unit positive mass is the spectral gradient -grad Phi,
    gathered back to the particles with the *same* CIC weights, so there is no
    self-force and |m|-weighted momentum is conserved to round-off;
  * particle i receives a_i = s_i * (-grad Phi)(x_i), i.e. same signs attract,
    opposite signs repel, exactly as in forces_direct.
Phi has the same meaning as in forces_direct: U = 0.5 sum_i s_i w_i Phi_i minus
the self-energy returned by self_energy() (A17).
"""
from __future__ import annotations

import itertools

import numpy as np

_LAWS = ('r2', 'log')
_BCS = ('periodic', 'isolated')


# ---------------------------------------------------------------------------
# CIC helpers (shared by the solver and the standalone density_grid)
# ---------------------------------------------------------------------------

def _cic_corners(pos: np.ndarray, cell: float, n: int, dim: int, periodic: bool) -> list:
    """List of (flat node index, weight) for the 2^dim CIC corners of every particle.

    Grid nodes sit at x = i*cell, i = 0..n-1.  Periodic: coordinates wrapped;
    isolated: coordinates clipped into [0, (n-1)*cell) so all nodes exist."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != dim:
        raise ValueError(f"pos must have shape (N, {dim})")
    u = pos / cell
    if periodic:
        u = np.mod(u, n)
        u[u >= n] -= n                      # guard against round-off landing on n
    else:
        u = np.clip(u, 0.0, (n - 1.0) - 1e-9)
    i0 = np.floor(u).astype(np.int64)
    f = u - i0
    npart = u.shape[0]
    out = []
    for corner in itertools.product((0, 1), repeat=dim):
        idx = np.zeros(npart, dtype=np.int64)
        wgt = np.ones(npart, dtype=np.float64)
        for d in range(dim):
            ii = i0[:, d] + corner[d]
            if periodic:
                ii = np.where(ii >= n, ii - n, ii)
            idx = idx * n + ii
            wgt = wgt * (f[:, d] if corner[d] else 1.0 - f[:, d])
        out.append((idx, wgt))
    return out


def _deposit(corners: list, weights: np.ndarray, n: int, dim: int) -> np.ndarray:
    """Mass per node (not yet divided by the cell volume)."""
    ncell = n ** dim
    acc = np.zeros(ncell, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    for idx, wgt in corners:
        acc += np.bincount(idx, weights=wgt * weights, minlength=ncell)
    return acc.reshape((n,) * dim)


def _gather(corners: list, grid: np.ndarray) -> np.ndarray:
    flat = np.asarray(grid, dtype=np.float64).reshape(-1)
    out = np.zeros(corners[0][0].shape[0], dtype=np.float64)
    for idx, wgt in corners:
        out += wgt * flat[idx]
    return out


# ---------------------------------------------------------------------------
# Green function for isolated boundaries
# ---------------------------------------------------------------------------

def green_function_real(dim: int, n: int, cell: float, G: float, force_law: str,
                        eps: float) -> np.ndarray:
    """Real-space Green function on the doubled (2n)^dim grid, periodically mirrored
    so that circular convolution with the zero-padded density reproduces the open-
    space potential of a unit mass at the origin:
        r2 law : g = -G / sqrt(r^2 + eps^2)      (2-D sheet or 3-D)
        log law: g = 0.5 G ln(r^2 + eps^2)        (true 2-D gravity)
    eps <= 0 falls back to one cell to keep g(0) finite."""
    if force_law not in _LAWS:
        raise ValueError(f"force_law={force_law!r} not in {_LAWS}")
    n2 = 2 * n
    i = np.arange(n2)
    r1 = cell * np.minimum(i, n2 - i).astype(np.float64)
    r2 = np.zeros((n2,) * dim, dtype=np.float64)
    for d in range(dim):
        shape = [1] * dim
        shape[d] = n2
        r2 = r2 + (r1 ** 2).reshape(shape)
    e2 = (eps if eps > 0 else cell) ** 2
    if force_law == 'r2':
        return -G / np.sqrt(r2 + e2)
    return 0.5 * G * np.log(r2 + e2)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class PMSolver:
    def __init__(self, dim: int, grid: int, box: float, G: float, force_law: str,
                 bc: str = 'periodic', eps: float = 0.0, dtype=np.float64):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if force_law not in _LAWS:
            raise ValueError(f"force_law={force_law!r} not in {_LAWS}")
        if force_law == 'log' and dim != 2:
            raise ValueError("force_law='log' is only defined in 2-D")
        if bc not in _BCS:
            raise ValueError(f"bc={bc!r} not in {_BCS}")
        if grid < 4 or box <= 0:
            raise ValueError("grid must be >= 4 and box > 0")
        self.dim, self.n, self.box, self.G = int(dim), int(grid), float(box), float(G)
        self.force_law, self.bc, self.eps = force_law, bc, float(max(eps, 0.0))
        self.dtype = np.dtype(dtype)
        self.periodic = bc == 'periodic'
        self._cell = self.box / self.n
        self.nw = self.n if self.periodic else 2 * self.n      # working (FFT) grid
        self._kd, k2 = self._k_arrays(self.nw)
        if self.periodic:
            with np.errstate(divide='ignore'):
                if dim == 2 and force_law == 'r2':
                    kern = -2.0 * np.pi * G / np.sqrt(k2)
                elif dim == 2:
                    kern = -2.0 * np.pi * G / k2
                else:
                    kern = -4.0 * np.pi * G / k2
            kern[(0,) * dim] = 0.0                              # Jeans swindle (A3)
            if self.eps > 0:
                kern *= np.exp(-0.5 * k2 * self.eps ** 2)
            self._kernel = kern
            self.eps_eff = max(self.eps, self._cell)
        else:
            self.eps_eff = max(self.eps, self._cell)
            g = green_function_real(dim, self.n, self._cell, G, force_law, self.eps_eff)
            self._kernel = np.fft.rfftn(g) * self._cell ** dim
        self._phi_self = self._measure_self_potential()

    # ---- geometry -----------------------------------------------------------
    @property
    def cell(self) -> float:
        return self._cell

    def _k_arrays(self, nw: int):
        """Broadcastable wave-number arrays (Nyquist zeroed for the derivative) and k^2."""
        kd, k2 = [], 0.0
        for d in range(self.dim):
            if d == self.dim - 1:
                k = 2.0 * np.pi * np.fft.rfftfreq(nw, d=self._cell)
            else:
                k = 2.0 * np.pi * np.fft.fftfreq(nw, d=self._cell)
            shape = [1] * self.dim
            shape[d] = k.size
            k = k.reshape(shape)
            k2 = k2 + k * k
            kz = k.copy()
            if nw % 2 == 0:                                      # zero the Nyquist mode
                nyq = nw // 2 if d < self.dim - 1 else k.size - 1
                np.put_along_axis(kz, np.full(shape, nyq, dtype=np.int64)[..., :1] if False else
                                  np.array([nyq]).reshape([1] * self.dim), 0.0, axis=d)
            kd.append(kz)
        return kd, k2

    # ---- internal FFT pipeline -----------------------------------------------
    def _phi_k(self, rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=np.float64)
        if rho.shape != (self.n,) * self.dim:
            raise ValueError(f"rho must have shape {(self.n,) * self.dim}")
        if not self.periodic:
            work = np.zeros((self.nw,) * self.dim, dtype=np.float64)
            work[(slice(0, self.n),) * self.dim] = rho
            rho = work
        return self._kernel * np.fft.rfftn(rho)

    def _to_real(self, fk: np.ndarray) -> np.ndarray:
        out = np.fft.irfftn(fk, s=(self.nw,) * self.dim)
        if not self.periodic:
            out = out[(slice(0, self.n),) * self.dim]
        return out

    def _acc_from_k(self, fk: np.ndarray) -> list:
        return [self._to_real(-1j * k * fk) for k in self._kd]

    def _measure_self_potential(self) -> float:
        pos = np.full((1, self.dim), (self.n // 2 + 0.5) * self._cell)
        corners = _cic_corners(pos, self._cell, self.n, self.dim, self.periodic)
        rho = _deposit(corners, np.ones(1), self.n, self.dim) / self._cell ** self.dim
        phi = self._to_real(self._phi_k(rho))
        return float(_gather(corners, phi)[0])

    # ---- public grid operations -------------------------------------------------
    def deposit(self, pos: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """CIC density (mass per cell volume) on the n^dim grid; signed if weights are."""
        corners = _cic_corners(pos, self._cell, self.n, self.dim, self.periodic)
        rho = _deposit(corners, weights, self.n, self.dim) / self._cell ** self.dim
        return rho.astype(self.dtype, copy=False)

    def solve_potential(self, rho: np.ndarray) -> np.ndarray:
        """Potential of the (signed) density on the unpadded grid."""
        return self._to_real(self._phi_k(rho)).astype(self.dtype, copy=False)

    def gradient(self, phi: np.ndarray) -> list:
        """-dPhi/dx_k grids (acceleration of a unit positive mass).  Spectral for
        periodic BC; isolated BC uses second-order finite differences (the solver's
        own force path differentiates spectrally on the padded grid instead)."""
        phi = np.asarray(phi, dtype=np.float64)
        if self.periodic:
            fk = np.fft.rfftn(phi)
            grids = self._acc_from_k(fk)
        else:
            grads = np.gradient(phi, self._cell, edge_order=2)
            grids = [-g for g in grads]
        return [g.astype(self.dtype, copy=False) for g in grids]

    def interpolate(self, grids, pos: np.ndarray) -> np.ndarray:
        """CIC gather: list of grids -> (N, len(grids)); single grid -> (N,)."""
        corners = _cic_corners(pos, self._cell, self.n, self.dim, self.periodic)
        if isinstance(grids, np.ndarray) and grids.ndim == self.dim:
            return _gather(corners, grids).astype(self.dtype, copy=False)
        out = np.empty((corners[0][0].shape[0], len(grids)), dtype=np.float64)
        for k, g in enumerate(grids):
            out[:, k] = _gather(corners, g)
        return out.astype(self.dtype, copy=False)

    def field_from_density(self, rho: np.ndarray):
        """(acc grids, phi grid) of a static density, e.g. an analytic annulus."""
        fk = self._phi_k(rho)
        acc = [g.astype(self.dtype, copy=False) for g in self._acc_from_k(fk)]
        return acc, self._to_real(fk).astype(self.dtype, copy=False)

    def forces(self, pos: np.ndarray, w: np.ndarray, s: np.ndarray):
        """(acc (N,dim) including the sign s_i, phi (N,) unsigned potential)."""
        pos = np.asarray(pos)
        npart = pos.shape[0]
        w = np.asarray(w, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        if w.shape != (npart,) or s.shape != (npart,):
            raise ValueError("w and s must have shape (N,)")
        acc = np.zeros((npart, self.dim), dtype=self.dtype)
        phi = np.zeros(npart, dtype=self.dtype)
        if npart == 0:
            return acc, phi
        corners = _cic_corners(pos, self._cell, self.n, self.dim, self.periodic)
        rho = _deposit(corners, s * w, self.n, self.dim) / self._cell ** self.dim
        fk = self._phi_k(rho)
        for d, grid in enumerate(self._acc_from_k(fk)):
            acc[:, d] = (s * _gather(corners, grid)).astype(self.dtype)
        phi[:] = _gather(corners, self._to_real(fk)).astype(self.dtype)
        return acc, phi

    def self_energy(self, w: np.ndarray) -> float:
        """0.5 * phi_self * sum w^2 with phi_self the measured potential of a unit
        particle at its own (cell-centre) position; subtract from 0.5 sum s w phi."""
        w = np.asarray(w, dtype=np.float64)
        return 0.5 * self._phi_self * float(np.sum(w * w))

    @property
    def phi_self_unit(self) -> float:
        return self._phi_self

    def __repr__(self) -> str:
        return (f"PMSolver(dim={self.dim}, grid={self.n}, box={self.box:g}, bc={self.bc}, "
                f"law={self.force_law}, eps={self.eps:g}, cell={self._cell:.4g})")


# ---------------------------------------------------------------------------
# Standalone density grid for diagnostics
# ---------------------------------------------------------------------------

def density_grid(pos: np.ndarray, weights: np.ndarray, box: float, ngrid: int, dim: int,
                 periodic: bool = True, origin=None) -> np.ndarray:
    """CIC density (mass per cell volume) at arbitrary resolution.  With `origin`
    the grid covers [origin, origin + box); non-periodic grids drop particles
    outside that range instead of clipping them."""
    pos = np.asarray(pos, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if origin is not None:
        pos = pos - np.asarray(origin, dtype=np.float64)[None, :]
    cell = box / ngrid
    if not periodic:
        inside = np.all((pos >= 0.0) & (pos < box), axis=1)
        pos, weights = pos[inside], weights[inside]
    if pos.shape[0] == 0:
        return np.zeros((ngrid,) * dim)
    corners = _cic_corners(pos, cell, ngrid, dim, periodic)
    return _deposit(corners, weights, ngrid, dim) / cell ** dim


if __name__ == '__main__':  # minimal self-test
    rng = np.random.default_rng(0)
    for dim in (2, 3):
        n = 32
        pm = PMSolver(dim, n, 1.0, 1.0, 'r2', 'periodic')
        N = 500
        pos = rng.random((N, dim))
        w = rng.random(N)
        s = np.where(np.arange(N) < N // 2, 1.0, -1.0)
        a, p = pm.forces(pos, w, s)
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(p))
        # Newton's third law for |m| a (no self-force with identical kernels)
        assert np.abs((w[:, None] * a).sum(0)).max() < 1e-9 * np.abs(a).max(), dim
        assert np.isfinite(pm.self_energy(w))
        rho = density_grid(pos, w, 1.0, n, dim)
        assert np.isclose(rho.sum() * (1.0 / n) ** dim, w.sum())
    # isolated 2-D: two unit masses 8 cells apart -> |a| ~ G/r^2 (sheet law), opposite
    # signs repel; consistency between interpolate(gradient(phi)) and forces()
    pm = PMSolver(2, 64, 1.0, 1.0, 'r2', 'isolated', eps=0.0)
    r = 8 * pm.cell
    pos = np.array([[0.5, 0.5], [0.5 + r, 0.5]])
    a, p = pm.forces(pos, np.ones(2), np.ones(2))
    assert a[0, 0] > 0 and abs(a[0, 0] * r * r - 1.0) < 0.1, a
    a2, _ = pm.forces(pos, np.ones(2), np.array([1.0, -1.0]))
    assert a2[0, 0] < 0 and a2[1, 0] > 0
    ag, pg = pm.field_from_density(pm.deposit(pos, np.ones(2)))
    assert np.allclose(pm.interpolate(ag, pos), a, rtol=1e-6, atol=1e-9)
    # log law kernel: force ~ G/r at 8 cells (periodic, images weak at 1/8 box)
    pm = PMSolver(2, 128, 1.0, 1.0, 'log', 'periodic')
    r = 8 * pm.cell
    a, _ = pm.forces(np.array([[0.5, 0.5], [0.5 + r, 0.5]]), np.ones(2), np.ones(2))
    assert abs(a[0, 0] * r - 1.0) < 0.15, a
    print("forces_pm self-test OK")
