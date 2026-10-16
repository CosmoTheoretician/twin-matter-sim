"""forces_pm_torch.py -- PyTorch (ROCm / "cuda" device) port of forces_pm.PMSolver.

The torch solver is built *from* a CPU PMSolver: it copies that solver's
spectral kernel, its Nyquist-zeroed wave-number arrays and its measured self
potential to the device, so the recipe is exactly the one of forces_pm:

  * CIC deposit of the signed density on the node-centred n^dim grid
    (torch index_add_ instead of np.bincount),
  * one rfftn, multiplication by the kernel (periodic: -2piG/|k|, -2piG/k^2 or
    -4piG/k^2 with the k = 0 mode removed; isolated: Hockney-Eastwood zero-padded
    doubled grid convolved with the softened real-space Green function),
  * acceleration grids by the spectral gradient -i k Phi_k (Nyquist zeroed),
  * CIC gather with the same weights (no self-force, |m|-momentum conserved),
  * a_i = s_i * (-grad Phi)(x_i), phi_i the unsigned potential.

Everything is computed in the requested dtype (float32 by default on the GPU,
float64 for validation); the CPU solver always works in float64, so in float64
the two agree to round-off (checked by the self-test and by gputest).
Only numpy, torch and forces_pm are imported.
"""
from __future__ import annotations

import itertools

import numpy as np

try:
    import torch
    _TORCH_ERR = ''
except Exception as _exc:            # ImportError or a broken runtime
    torch = None
    _TORCH_ERR = repr(_exc)

from forces_pm import PMSolver


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _torch_dtype(name):
    if torch is None:
        raise RuntimeError(f"forces_pm_torch needs PyTorch (import failed: {_TORCH_ERR})")
    if isinstance(name, torch.dtype):
        return name
    key = str(np.dtype(name).name) if not isinstance(name, str) else name
    table = {'float32': torch.float32, 'float64': torch.float64,
             'f4': torch.float32, 'f8': torch.float64}
    if key not in table:
        raise ValueError(f"unsupported dtype {name!r}: use 'float32' or 'float64'")
    return table[key]


def _complex_of(dtype):
    return torch.complex64 if dtype == torch.float32 else torch.complex128


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------

class PMSolverTorch:
    """Device-resident particle-mesh solver with the interface of forces_pm.PMSolver.
    All public methods accept numpy arrays or torch tensors and return torch
    tensors on `device` in `dtype` (acc includes the sign s_i; phi unsigned)."""

    def __init__(self, cpu: PMSolver, device, dtype):
        if torch is None:
            raise RuntimeError(f"forces_pm_torch needs PyTorch (import failed: {_TORCH_ERR})")
        if not isinstance(cpu, PMSolver):
            raise TypeError("cpu must be a forces_pm.PMSolver")
        self.cpu = cpu                                   # reference solver (gputest)
        self.device = torch.device(device)
        self.dtype = _torch_dtype(dtype)
        self.cdtype = _complex_of(self.dtype)
        self.dim, self.n, self.nw = int(cpu.dim), int(cpu.n), int(cpu.nw)
        self.box, self._cell = float(cpu.box), float(cpu.cell)
        self.G, self.force_law, self.bc = float(cpu.G), cpu.force_law, cpu.bc
        self.periodic = bool(cpu.periodic)
        self.eps, self.eps_eff = float(cpu.eps), float(cpu.eps_eff)
        self.phi_self_unit = float(cpu.phi_self_unit)
        kern = np.ascontiguousarray(np.asarray(cpu._kernel, dtype=np.complex128))
        self._kernel = torch.from_numpy(kern).to(device=self.device, dtype=self.cdtype)
        self._kd = [torch.from_numpy(np.ascontiguousarray(np.asarray(k, dtype=np.float64)))
                    .to(device=self.device, dtype=self.dtype) for k in cpu._kd]
        # -i k, broadcastable, complex: the spectral derivative factors
        self._ikd = [torch.from_numpy(np.ascontiguousarray(-1j * np.asarray(k, dtype=np.complex128)))
                     .to(device=self.device, dtype=self.cdtype) for k in cpu._kd]
        self._grid_shape = (self.n,) * self.dim
        self._work_shape = (self.nw,) * self.dim

    @classmethod
    def from_config(cls, dim, grid, box, G, force_law, bc, eps, device, dtype):
        cpu = PMSolver(dim, grid, box, G, force_law, bc=bc, eps=eps, dtype=np.float64)
        return cls(cpu, device, dtype)

    # ---- conversions ----------------------------------------------------------
    @property
    def cell(self) -> float:
        return self._cell

    def _tensor(self, x, dtype=None):
        dt = self.dtype if dtype is None else dtype
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=dt)
        return torch.as_tensor(np.ascontiguousarray(np.asarray(x)), dtype=dt, device=self.device)

    # ---- CIC on the device ----------------------------------------------------
    def _cic_corners(self, pos):
        """(idx (2^dim, N) int64 flat node indices, wgt (2^dim, N) weights),
        replicating forces_pm._cic_corners: nodes at i*cell, periodic wrap or
        isolated clamp into [0, (n-1)*cell)."""
        pos = self._tensor(pos)
        if pos.dim() != 2 or pos.shape[1] != self.dim:
            raise ValueError(f"pos must have shape (N, {self.dim})")
        n = self.n
        u = pos / self._cell
        if self.periodic:
            u = torch.remainder(u, float(n))
            u = torch.where(u >= n, u - n, u)              # round-off landing on n
        else:
            u = torch.clamp(u, 0.0, (n - 1.0) - 1e-9)
        i0 = torch.floor(u).to(torch.int64)
        # identical to the CPU code in float64; in float32 the clamp bound rounds to
        # n-1 exactly, so keep the base node at n-2 (f = 1) rather than n-1 (ii = n)
        i0 = torch.clamp(i0, 0, n - 1 if self.periodic else n - 2)
        f = u - i0.to(u.dtype)
        npart = u.shape[0]
        idx_list, wgt_list = [], []
        for corner in itertools.product((0, 1), repeat=self.dim):
            idx = torch.zeros(npart, dtype=torch.int64, device=self.device)
            wgt = torch.ones(npart, dtype=self.dtype, device=self.device)
            for d in range(self.dim):
                ii = i0[:, d] + corner[d]
                if self.periodic:
                    ii = torch.where(ii >= n, ii - n, ii)
                idx = idx * n + ii
                wgt = wgt * (f[:, d] if corner[d] else 1.0 - f[:, d])
            idx_list.append(idx)
            wgt_list.append(wgt)
        return torch.stack(idx_list, 0), torch.stack(wgt_list, 0)

    def _deposit(self, idx, wgt, weights):
        """Density (mass per cell volume) on the n^dim grid."""
        weights = self._tensor(weights)
        flat = torch.zeros(self.n ** self.dim, dtype=self.dtype, device=self.device)
        flat.index_add_(0, idx.reshape(-1), (wgt * weights[None, :]).reshape(-1))
        return flat.reshape(self._grid_shape) / self._cell ** self.dim

    def _gather(self, idx, wgt, grid):
        return (wgt * grid.reshape(-1)[idx]).sum(0)

    # ---- FFT pipeline -----------------------------------------------------------
    def _phi_k(self, rho):
        rho = self._tensor(rho)
        if tuple(rho.shape) != self._grid_shape:
            raise ValueError(f"rho must have shape {self._grid_shape}")
        if not self.periodic:
            work = torch.zeros(self._work_shape, dtype=self.dtype, device=self.device)
            work[(slice(0, self.n),) * self.dim] = rho
            rho = work
        return self._kernel * torch.fft.rfftn(rho)

    def _to_real(self, fk):
        out = torch.fft.irfftn(fk, s=self._work_shape)
        if not self.periodic:
            out = out[(slice(0, self.n),) * self.dim]
        return out

    def _acc_from_k(self, fk):
        return [self._to_real(ik * fk) for ik in self._ikd]

    # ---- public grid operations ---------------------------------------------------
    def deposit(self, pos, weights):
        """CIC density (mass per cell volume) on the n^dim grid; signed if weights are."""
        idx, wgt = self._cic_corners(pos)
        return self._deposit(idx, wgt, weights)

    def solve_potential(self, rho):
        """Potential of the (signed) density on the unpadded grid."""
        return self._to_real(self._phi_k(rho))

    def gradient(self, phi):
        """-dPhi/dx_k grids: spectral for periodic BC, second-order finite
        differences for isolated BC (same as the CPU solver's gradient())."""
        phi = self._tensor(phi)
        if self.periodic:
            return self._acc_from_k(torch.fft.rfftn(phi))
        grads = torch.gradient(phi, spacing=self._cell, edge_order=2)
        return [-g for g in grads]

    def interpolate(self, grids, pos):
        """CIC gather: list of grids -> (N, len(grids)); single grid -> (N,)."""
        idx, wgt = self._cic_corners(pos)
        single = ((torch.is_tensor(grids) and grids.dim() == self.dim)
                  or (isinstance(grids, np.ndarray) and grids.ndim == self.dim))
        if single:
            return self._gather(idx, wgt, self._tensor(grids))
        out = torch.empty((idx.shape[1], len(grids)), dtype=self.dtype, device=self.device)
        for k, g in enumerate(grids):
            out[:, k] = self._gather(idx, wgt, self._tensor(g))
        return out

    def field_from_density(self, rho):
        """(acc grids, phi grid) of a static density (rigid ring precompute)."""
        fk = self._phi_k(rho)
        return self._acc_from_k(fk), self._to_real(fk)

    def forces(self, pos, w, s):
        """(acc (N,dim) including the sign s_i, phi (N,) unsigned potential)."""
        pos, w, s = self._tensor(pos), self._tensor(w), self._tensor(s)
        npart = pos.shape[0]
        if w.shape != (npart,) or s.shape != (npart,):
            raise ValueError("w and s must have shape (N,)")
        acc = torch.zeros((npart, self.dim), dtype=self.dtype, device=self.device)
        phi = torch.zeros(npart, dtype=self.dtype, device=self.device)
        if npart == 0:
            return acc, phi
        idx, wgt = self._cic_corners(pos)
        rho = self._deposit(idx, wgt, s * w)
        fk = self._phi_k(rho)
        del rho
        for d, ik in enumerate(self._ikd):              # one acc grid at a time (memory)
            grid = self._to_real(ik * fk)
            acc[:, d] = s * self._gather(idx, wgt, grid)
            del grid
        phi[:] = self._gather(idx, wgt, self._to_real(fk))
        return acc, phi

    def self_energy(self, w) -> float:
        """0.5 * phi_self * sum w^2 (phi_self measured by the CPU solver)."""
        if torch.is_tensor(w):
            ssq = float((w.detach().to(torch.float64) ** 2).sum().item())
        else:
            w = np.asarray(w, dtype=np.float64)
            ssq = float(np.sum(w * w))
        return 0.5 * self.phi_self_unit * ssq

    def __repr__(self) -> str:
        return (f"PMSolverTorch(dim={self.dim}, grid={self.n}, box={self.box:g}, bc={self.bc}, "
                f"law={self.force_law}, eps={self.eps:g}, cell={self._cell:.4g}, "
                f"device={self.device}, dtype={self.dtype})")


if __name__ == '__main__':  # self-test: agreement with the numpy solver
    import sys
    if torch is None:
        print("torch not installed, forces_pm_torch self-test skipped:", _TORCH_ERR)
        sys.exit(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cpu':
        print("no GPU: checking the torch solver on the CPU device")
    rng = np.random.default_rng(0)

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-300))

    for dim in (2, 3):
        for bc in ('periodic', 'isolated'):
            for law in (('r2', 'log') if dim == 2 else ('r2',)):
                n = 64 if dim == 2 else 24
                cpu = PMSolver(dim, n, 1.0, 1.0, law, bc, eps=0.0)
                pt = PMSolverTorch(cpu, dev, 'float64')
                N = 500
                pos = rng.random((N, dim))
                w = rng.random(N)
                s = np.where(np.arange(N) < N // 2, 1.0, -1.0)
                a_c, p_c = cpu.forces(pos, w, s)
                a_t, p_t = pt.forces(pos, w, s)
                a_t, p_t = a_t.cpu().numpy(), p_t.cpu().numpy()
                assert a_t.shape == (N, dim) and p_t.shape == (N,)
                assert rel(a_t, a_c) < 1e-10, (dim, bc, law, rel(a_t, a_c))
                assert rel(p_t, p_c) < 1e-10, (dim, bc, law, rel(p_t, p_c))
                if bc == 'periodic':
                    assert np.abs((w[:, None] * a_t).sum(0)).max() < 1e-9 * np.abs(a_t).max()
                rho_c = cpu.deposit(pos, s * w)
                rho_t = pt.deposit(pos, s * w)
                assert rel(rho_t.cpu().numpy(), rho_c) < 1e-10
                assert abs(float(rho_t.sum()) * cpu.cell ** dim - float(np.sum(s * w))) < 1e-9
                phi_c = cpu.solve_potential(rho_c)
                phi_t = pt.solve_potential(rho_t)
                assert rel(phi_t.cpu().numpy(), phi_c) < 1e-10
                g_c = cpu.gradient(phi_c)
                g_t = pt.gradient(phi_t)
                for gc, gt in zip(g_c, g_t):
                    assert rel(gt.cpu().numpy(), gc) < 1e-10, (dim, bc, law)
                i_c = cpu.interpolate(g_c, pos)
                i_t = pt.interpolate(g_t, pos)
                assert rel(i_t.cpu().numpy(), i_c) < 1e-10
                assert rel(pt.interpolate(phi_t, pos).cpu().numpy(), cpu.interpolate(phi_c, pos)) < 1e-10
                ag_t, pg_t = pt.field_from_density(rho_t)
                ag_c, pg_c = cpu.field_from_density(rho_c)
                assert rel(pg_t.cpu().numpy(), pg_c) < 1e-10
                assert rel(pt.interpolate(ag_t, pos).cpu().numpy(), cpu.interpolate(ag_c, pos)) < 1e-10
                assert np.isclose(pt.self_energy(w), cpu.self_energy(w))
                assert np.isclose(pt.self_energy(torch.as_tensor(w)), cpu.self_energy(w))
                a0, p0 = pt.forces(np.zeros((0, dim)), np.zeros(0), np.zeros(0))
                assert a0.shape == (0, dim) and p0.shape == (0,)
    # float32 path: same recipe, round-off level agreement
    cpu = PMSolver(2, 128, 1.0, 1.0, 'r2', 'periodic')
    pt32 = PMSolverTorch.from_config(2, 128, 1.0, 1.0, 'r2', 'periodic', 0.0, dev, 'float32')
    N = 2000
    pos = rng.random((N, 2))
    w = np.where(np.arange(N) < N // 2, 1.0 / N, 64.0 / N)
    s = np.where(np.arange(N) < N // 2, 1.0, -1.0)
    a_c, p_c = cpu.forces(pos, w, s)
    a_t, p_t = pt32.forces(pos, w, s)
    assert a_t.dtype == torch.float32 and p_t.dtype == torch.float32
    assert rel(a_t.cpu().numpy(), a_c) < 1e-4, rel(a_t.cpu().numpy(), a_c)
    assert rel(p_t.cpu().numpy() - p_t.cpu().numpy().mean(), p_c - p_c.mean()) < 1e-4
    # isolated float32: edge clamp must not produce out-of-range nodes
    pt32i = PMSolverTorch.from_config(2, 64, 1.0, 1.0, 'r2', 'isolated', 0.0, dev, 'float32')
    edge = np.array([[0.0, 0.0], [0.999999, 0.999999], [1.0 - 1e-9, 0.5]])
    a_e, p_e = pt32i.forces(edge, np.ones(3), np.ones(3))
    assert torch.isfinite(a_e).all() and torch.isfinite(p_e).all()
    print(repr(pt32))
    print("forces_pm_torch self-test OK")
