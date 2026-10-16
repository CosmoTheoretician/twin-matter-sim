"""integrator_torch.py -- GPU (PyTorch / ROCm) counterpart of integrator.py.

Same conventions as integrator.py:
  * ParticleSet / GPUParticles .vel holds the comoving velocity dx/dt;
  * kinematic expansion (A15) through u = a^2 x':  du/dt = g_c a^(1-p),
    dx/dt = u/a^2, trapezoidal <a^(1-p)> over each kick;
  * ForceEvaluatorTorch returns the signed acceleration (species sign applied,
    rigid external field added as s_i * ext.accel) and the unsigned potential
    (external potential added), storing the external part in last_phi_ext.

The evaluator accepts either a device-resident GPUParticles (tensors in, tensors
out -- the time loop) or an ordinary numpy ParticleSet (uploads, solves,
downloads), so initial_conditions.make_galaxy_ic works unchanged on the GPU.
The rigid external fields of external_potentials get torch adapters with the
same convention (acceleration of a unit positive mass); the ring's grids are
precomputed once on the CPU and uploaded once.
"""
from __future__ import annotations

import math
import time

import numpy as np

try:
    import torch
    _TORCH_ERR = ''
except Exception as _exc:            # ImportError or a broken runtime
    torch = None
    _TORCH_ERR = repr(_exc)

from config import RunConfig, Derived
from particles import ParticleSet
from forces_pm import PMSolver
from external_potentials import (ExternalField, NoField, RigidRingField, RigidHaloField,
                                 make_external_field)
from gpu_backend import GPUParticles, require_gpu, torch_dtype
from forces_pm_torch import PMSolverTorch
from integrator import Expansion


def _need_torch() -> None:
    if torch is None:
        raise RuntimeError(f"integrator_torch needs PyTorch (import failed: {_TORCH_ERR})")


def _as_tensor(x, device, dtype):
    return torch.as_tensor(np.ascontiguousarray(np.asarray(x, dtype=np.float64)),
                           dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# torch adapters of the rigid external fields
# ---------------------------------------------------------------------------

class TorchGridField:
    """RigidRingField on the device: the ring's acc/phi grids are uploaded once
    and interpolated with the torch PM solver (same CIC gather as the CPU)."""
    name = 'rigid-repulsive'

    def __init__(self, ring: RigidRingField, pm_t: PMSolverTorch):
        if ring.pm.n != pm_t.n or ring.pm.dim != pm_t.dim or ring.pm.bc != pm_t.bc \
                or abs(ring.pm.box - pm_t.box) > 1e-12 * max(1.0, pm_t.box):
            raise ValueError("rigid ring grid does not match the torch PM solver")
        self.ring, self.pm_t = ring, pm_t
        self.acc_grids = [_as_tensor(g, pm_t.device, pm_t.dtype) for g in ring.acc_grids]
        self.phi_grid = _as_tensor(ring.phi_grid, pm_t.device, pm_t.dtype)

    def accel(self, pos):
        return self.pm_t.interpolate(self.acc_grids, pos)

    def potential(self, pos):
        return self.pm_t.interpolate(self.phi_grid, pos)


class TorchHaloField:
    """RigidHaloField on the device: Phi = 0.5 vc^2 ln(r^2 + rc^2), minimum image
    when the halo was built with a box."""
    name = 'rigid-halo'

    def __init__(self, halo: RigidHaloField, device, dtype):
        self.halo = halo
        self.vc, self.rc = float(halo.vc), float(halo.rc)
        self.box = None if halo.box is None else float(halo.box)
        self.centre = _as_tensor(halo.centre, device, dtype)

    def _sep(self, pos):
        d = pos - self.centre[None, :pos.shape[1]]
        if self.box is not None:
            d = d - self.box * torch.round(d / self.box)
        return d, (d * d).sum(1)

    def accel(self, pos):
        d, r2 = self._sep(pos)
        return -(self.vc ** 2) * d / (r2 + self.rc ** 2)[:, None]

    def potential(self, pos):
        _, r2 = self._sep(pos)
        return 0.5 * self.vc ** 2 * torch.log(r2 + self.rc ** 2)


def wrap_torch_field(ext: ExternalField, pm_t: PMSolverTorch, device, dtype):
    """Torch adapter of a rigid field; None for NoField."""
    if ext is None or ext.is_null:
        return None
    if isinstance(ext, RigidRingField):
        return TorchGridField(ext, pm_t)
    if isinstance(ext, RigidHaloField):
        return TorchHaloField(ext, device, dtype)
    raise TypeError(f"no torch adapter for external field {ext!r}")


# ---------------------------------------------------------------------------
# force evaluator
# ---------------------------------------------------------------------------

class ForceEvaluatorTorch:
    """PM three-law force on the device plus an optional rigid external field.
    Same attributes as integrator.ForceEvaluator (method, pm, ext, G, eps, box,
    periodic, n_evals, last_wall, total_wall, last_phi_ext, eps_eff,
    self_energy_fn).  __call__ accepts a GPUParticles (tensors returned) or a
    numpy ParticleSet (numpy arrays of the set's dtype returned)."""

    def __init__(self, cfg: RunConfig, der: Derived, pm_t: PMSolverTorch, ext: ExternalField):
        _need_torch()
        self.cfg, self.der = cfg, der
        self.method = 'pm'
        self.pm = pm_t
        self.ext = ext if ext is not None else NoField()
        self.device, self.dtype = pm_t.device, pm_t.dtype
        self.ext_t = wrap_torch_field(self.ext, pm_t, self.device, self.dtype)
        self.G = der.G
        self.eps = max(cfg.eps, 0.0)
        self.force_law = cfg.force_law
        self.periodic = cfg.bc == 'periodic'
        self.box = der.box if self.periodic else None
        self.n_evals = 0
        self.last_wall = 0.0
        self.total_wall = 0.0
        self.last_phi_ext = None
        self._cuda = self.device.type == 'cuda'

    @property
    def eps_eff(self) -> float:
        return float(self.der.eps_abs)

    @property
    def self_energy_fn(self):
        return self.pm.self_energy

    def _sync(self) -> None:
        if self._cuda:
            torch.cuda.synchronize()

    def forces_device(self, gp: GPUParticles):
        """(acc (N,dim) signed, phi (N,) unsigned) tensors on the device."""
        self._sync()
        t0 = time.perf_counter()
        acc, phi = self.pm.forces(gp.pos, gp.w, gp.s)
        if gp.n and self.ext_t is not None:
            a_ext = self.ext_t.accel(gp.pos).to(acc.dtype)
            phi_ext = self.ext_t.potential(gp.pos).to(phi.dtype)
            acc = acc + gp.s.to(acc.dtype)[:, None] * a_ext
            phi = phi + phi_ext
            self.last_phi_ext = phi_ext
        else:
            self.last_phi_ext = torch.zeros(gp.n, dtype=phi.dtype, device=self.device)
        self._sync()
        self.last_wall = time.perf_counter() - t0
        self.total_wall += self.last_wall
        self.n_evals += 1
        return acc, phi

    def __call__(self, ps):
        if isinstance(ps, GPUParticles):
            return self.forces_device(ps)
        gp = GPUParticles.from_host(ps, self.device, self.dtype)
        acc, phi = self.forces_device(gp)
        npdt = ps.pos.dtype if ps.pos.dtype.kind == 'f' else np.dtype(np.float64)
        self.last_phi_ext = self.last_phi_ext.detach().cpu().numpy().astype(npdt, copy=False)
        return (acc.detach().cpu().numpy().astype(npdt, copy=False),
                phi.detach().cpu().numpy().astype(npdt, copy=False))

    def __repr__(self) -> str:
        return f"ForceEvaluatorTorch({self.pm!r}, ext={self.ext.describe()}, n_evals={self.n_evals})"


def build_gpu_forces(cfg: RunConfig, der: Derived):
    """(ForceEvaluatorTorch, PMSolverTorch, ExternalField) for a resolved GPU config.
    The CPU PMSolver (float64) supplies the kernel, the self potential and the
    rigid-ring precompute; the torch solver works in cfg.dtype on the GPU."""
    device = require_gpu()
    if der.method != 'pm':
        raise ValueError("the GPU path implements the particle-mesh solver only")
    dtype = cfg.dtype if cfg.dtype in ('float32', 'float64') else 'float32'
    pm_cpu = PMSolver(der.dim, der.grid, der.box, der.G, cfg.force_law, bc=cfg.bc,
                      eps=max(cfg.eps, 0.0), dtype=np.float64)
    ext = make_external_field(cfg, der, pm_cpu)
    pm_t = PMSolverTorch(pm_cpu, device, torch_dtype(dtype))
    return ForceEvaluatorTorch(cfg, der, pm_t, ext), pm_t, ext


# ---------------------------------------------------------------------------
# leapfrog pieces on the device (mirroring integrator.py)
# ---------------------------------------------------------------------------

def kick_t(gp: GPUParticles, acc, dt: float, a_old: float, a_new: float, p: int) -> None:
    """In-place velocity update: v += acc dt when a_old = a_new = 1, otherwise
    u = a_old^2 v + acc dt <a^(1-p)>, v = u / a_new^2 (tensor dtype)."""
    acc = acc.to(dtype=gp.vel.dtype)
    if a_old == 1.0 and a_new == 1.0:
        gp.vel.add_(acc, alpha=float(dt))
        return
    q = 1 - int(p)
    fac = 0.5 * (a_old ** q + a_new ** q)
    u = (a_old * a_old) * gp.vel + acc * (dt * fac)
    gp.vel.copy_(u / (a_new * a_new))


def wrap_periodic_t(pos, box: float):
    """Coordinates mapped into [0, box) with the same guards as particles.wrap_periodic."""
    out = torch.remainder(pos, box)
    out = torch.where(out >= box, out - box, out)
    out = torch.where(out < 0, out + box, out)
    return out


def drift_t(gp: GPUParticles, dt: float, box) -> None:
    """In-place x += v dt, periodic wrap if box is given."""
    gp.pos.add_(gp.vel, alpha=float(dt))
    if box is not None:
        gp.pos.copy_(wrap_periodic_t(gp.pos, float(box)))


def choose_dt_t(acc, eps_eff: float, eta: float, dt_max: float, dt_min: float) -> float:
    """eta*sqrt(eps_eff/|a|_max) clipped to [dt_min, dt_max]; one host sync."""
    if acc is None or acc.numel() == 0 or eps_eff <= 0 or eta <= 0:
        return float(dt_max)
    a2 = float((acc * acc).sum(1).max().item())
    if not math.isfinite(a2) or a2 <= 0:
        return float(dt_max)
    dt = eta * math.sqrt(eps_eff / math.sqrt(a2))
    return float(min(max(dt, dt_min), dt_max))


def leapfrog_step_t(gp: GPUParticles, acc, phi, t: float, dt: float, force, expansion: Expansion,
                    p: int, box):
    """One KDK step from t to t+dt on the device; returns (acc_new, phi_new, t+dt)."""
    a0 = expansion.a(t)
    a_half = expansion.a(t + 0.5 * dt)
    a1 = expansion.a(t + dt)
    kick_t(gp, acc, 0.5 * dt, a0, a_half, p)
    drift_t(gp, dt, box)
    acc_new, phi_new = force(gp)
    kick_t(gp, acc_new, 0.5 * dt, a_half, a1, p)
    return acc_new, phi_new, t + dt


def integrate_t(gp: GPUParticles, force, expansion: Expansion, p: int, box, t: float, dt: float,
                n_steps: int, acc=None, phi=None):
    """Convenience loop without diagnostics; returns (acc, phi, t)."""
    if acc is None:
        acc, phi = force(gp)
    for _ in range(int(n_steps)):
        acc, phi, t = leapfrog_step_t(gp, acc, phi, t, dt, force, expansion, p, box)
    return acc, phi, t


if __name__ == '__main__':  # self-test
    import sys
    if torch is None:
        print("torch not installed, integrator_torch self-test skipped:", _TORCH_ERR)
        sys.exit(0)
    from config import resolve_defaults, derive
    from integrator import ForceEvaluator, leapfrog_step
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu':
        print("no GPU: checking the torch integrator on the CPU device")

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-300))

    def energy(ps, phi, force):
        ke = 0.5 * float(np.sum(ps.w * np.einsum('ij,ij->i', ps.vel, ps.vel)))
        return ke + 0.5 * float(np.sum(ps.s * ps.w * phi)) - force.self_energy_fn(ps.w)

    # ---- small LSS box: conservation of E and P_abs with the torch solver ------
    cfg = resolve_defaults(RunConfig('lss', npos=200, nneg=200, method='pm', grid=32, dt=0.01))
    der = derive(cfg)
    rng = np.random.default_rng(3)
    n = 400
    s = np.where(np.arange(n) < 200, 1.0, -1.0)
    w = np.where(s > 0, der.w_pos, der.w_neg)
    vel = rng.normal(0.0, 1.0, (n, 2)) * np.where(s > 0, der.sigma_pos, der.sigma_neg)[:, None]
    vel -= (w[:, None] * vel).sum(0) / w.sum()
    ps = ParticleSet(rng.random((n, 2)), vel, w, s, (s < 0).astype(np.int32))
    pm_cpu = PMSolver(2, 32, der.box, der.G, 'r2', 'periodic', eps=0.0)
    pm_t = PMSolverTorch(pm_cpu, dev, 'float64')
    force = ForceEvaluatorTorch(cfg, der, pm_t, NoField())
    assert force.method == 'pm' and force.box == der.box and force.periodic
    exp0 = Expansion(False, 0.0, 1.0)
    gp = GPUParticles.from_host(ps, dev, 'float64')
    acc, phi = force(gp)
    assert torch.is_tensor(acc) and acc.shape == (n, 2) and force.n_evals == 1
    h0 = gp.to_host()
    E0, P0 = energy(h0, phi.cpu().numpy(), force), (h0.w[:, None] * h0.vel).sum(0)
    dt = der.dt_abs
    acc, phi, t = integrate_t(gp, force, exp0, der.force_exponent_p, der.box, 0.0, dt, 100, acc, phi)
    h1 = gp.to_host()
    E1, P1 = energy(h1, phi.cpu().numpy(), force), (h1.w[:, None] * h1.vel).sum(0)
    assert abs(E1 - E0) < 5e-2 * abs(E0), (E0, E1)
    assert np.abs(P1 - P0).max() < 1e-9 * float(np.sum(h1.w * np.hypot(h1.vel[:, 0], h1.vel[:, 1])))
    assert np.all((h1.pos >= 0) & (h1.pos < der.box)) and force.n_evals == 101 and np.isclose(t, 100 * dt)
    # ---- numpy-call path agrees with the CPU solver ---------------------------
    force_c = ForceEvaluator(cfg, der, pm_cpu, NoField())
    a_c, p_c = force_c(h1)
    a_n, p_n = force(h1)
    assert isinstance(a_n, np.ndarray) and a_n.dtype == h1.pos.dtype and isinstance(force.last_phi_ext, np.ndarray)
    assert rel(a_n, a_c) < 1e-10 and rel(p_n, p_c) < 1e-10
    # ---- one leapfrog step vs integrator.leapfrog_step (float64), no expansion ---
    ps_c = ps.copy()
    acc_c, phi_c = force_c(ps_c)
    gp = GPUParticles.from_host(ps, dev, 'float64')
    acc_t, phi_t = force(gp)
    tc, tt = 0.0, 0.0
    for _ in range(5):
        acc_c, phi_c, tc = leapfrog_step(ps_c, acc_c, phi_c, tc, dt, force_c, exp0, der.force_exponent_p, der.box)
        acc_t, phi_t, tt = leapfrog_step_t(gp, acc_t, phi_t, tt, dt, force, exp0, der.force_exponent_p, der.box)
    hg = gp.to_host()
    assert rel(hg.pos, ps_c.pos) < 1e-10 and rel(hg.vel, ps_c.vel) < 1e-10 and tc == tt
    assert rel(acc_t.cpu().numpy(), acc_c) < 1e-10 and rel(phi_t.cpu().numpy(), phi_c) < 1e-10
    # ---- with expansion switched on across the step ----------------------------
    ex = Expansion(True, 2.0 / 3.0, 0.5 * der.t_ref)
    ps_c = ps.copy()
    acc_c, phi_c = force_c(ps_c)
    gp = GPUParticles.from_host(ps, dev, 'float64')
    acc_t, phi_t = force(gp)
    tc = tt = 0.5 * der.t_ref
    for _ in range(5):
        acc_c, phi_c, tc = leapfrog_step(ps_c, acc_c, phi_c, tc, dt, force_c, ex, der.force_exponent_p, der.box)
        acc_t, phi_t, tt = leapfrog_step_t(gp, acc_t, phi_t, tt, dt, force, ex, der.force_exponent_p, der.box)
    assert ex.a(tt) > 1.0
    hg = gp.to_host()
    assert rel(hg.pos, ps_c.pos) < 1e-10 and rel(hg.vel, ps_c.vel) < 1e-10
    # ---- kick with zero force conserves u = a^2 v; adaptive step; wrap --------
    g1 = GPUParticles.from_host(ParticleSet(np.zeros((1, 2)), np.array([[1.0, 0.0]]), np.ones(1),
                                            np.ones(1), np.zeros(1)), dev, 'float64')
    kick_t(g1, torch.zeros((1, 2), dtype=torch.float64, device=dev), 1.0, 1.0, 4.0, 2)
    assert np.isclose(float(g1.vel[0, 0]), 1.0 / 16.0)
    assert choose_dt_t(torch.zeros((3, 2), device=dev), 0.01, 0.1, 0.5, 1e-4) == 0.5
    assert np.isclose(choose_dt_t(torch.tensor([[4.0, 0.0]], device=dev), 1.0, 0.2, 10.0, 1e-4), 0.1)
    assert choose_dt_t(torch.tensor([[1e8, 0.0]], device=dev), 1.0, 0.2, 10.0, 1e-3) == 1e-3
    wp = wrap_periodic_t(torch.tensor([[-1e-12, 1.0], [1.5, -0.25]], dtype=torch.float64, device=dev), 1.0)
    assert bool(((wp >= 0) & (wp < 1.0)).all()) and np.isclose(float(wp[1, 0]), 0.5)
    # ---- rigid external fields: torch adapters vs the numpy fields -------------
    for control in ('rigid-repulsive', 'rigid-halo'):
        cg = resolve_defaults(RunConfig('galaxy', method='pm', grid=64, control=control, ndisk=300))
        dg = derive(cg)
        pmg = PMSolver(2, 64, dg.box, dg.G, 'r2', 'periodic')
        ext = make_external_field(cg, dg, pmg)
        pmg_t = PMSolverTorch(pmg, dev, 'float64')
        fe_t = ForceEvaluatorTorch(cg, dg, pmg_t, ext)
        fe_c = ForceEvaluator(cg, dg, pmg, ext)
        m = 300
        sg = np.where(np.arange(m) < 200, 1.0, -1.0)
        psg = ParticleSet(rng.random((m, 2)) * dg.box, np.zeros((m, 2)), np.full(m, 1e-3), sg,
                          (sg < 0).astype(np.int32))
        a_c, p_c = fe_c(psg)
        a_t, p_t = fe_t(psg)
        assert rel(a_t, a_c) < 1e-9 and rel(p_t, p_c) < 1e-9, control
        assert rel(fe_t.last_phi_ext, fe_c.last_phi_ext) < 1e-9
        assert wrap_torch_field(NoField(), pmg_t, dev, torch.float64) is None
    # ---- float32 container path runs and stays close ----------------------------
    pm32 = PMSolverTorch(pm_cpu, dev, 'float32')
    f32 = ForceEvaluatorTorch(cfg, der, pm32, NoField())
    g32 = GPUParticles.from_host(ps, dev, 'float32')
    a32, p32 = f32(g32)
    assert a32.dtype == torch.float32 and rel(a32.cpu().numpy(), force_c(ps)[0]) < 1e-4
    integrate_t(g32, f32, exp0, der.force_exponent_p, der.box, 0.0, dt, 3, a32, p32)
    assert torch.isfinite(g32.pos).all() and torch.isfinite(g32.vel).all()
    print(repr(force))
    print("integrator_torch self-test OK")
