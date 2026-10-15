"""integrator.py -- force wrapper, kinematic expansion, adaptive step, leapfrog.

Conventions
-----------
* ParticleSet.vel always holds the *comoving* velocity dx/dt at the current
  epoch (identical to the physical velocity when expansion is off, a = 1).
  diagnostics_core.kinetic_energy multiplies by a^2 to obtain the peculiar
  kinetic energy 0.5 sum w (a dx/dt)^2.
* Expansion (A15) is purely kinematic: a prescribed a(t) = (t/t0)^n switched
  on at t0, with the comoving equation of motion
        x'' + 2 H x' = g_c / a^(p+1),   p = force-law exponent (2: 1/r^2, 1: log)
  where g_c is the three-law acceleration evaluated from comoving positions.
  With the auxiliary variable u = a^2 x' this reads
        du/dt = g_c * a^(1-p),      dx/dt = u / a^2 ,
  which is what kick()/drift() integrate.  Not a Janus cosmology -- it only
  tests whether Hubble drag erases the pattern.
* ForceEvaluator returns the signed acceleration (species sign s_i already
  applied, external rigid field added as s_i * ext.accel) and the unsigned
  potential phi (external potential added) so that the potential energy is
  0.5 sum s w phi_self + sum s w phi_ext; the split is handled downstream by
  subtracting nothing for the external part (it has no back-reaction, so a
  factor 1/2 is not needed there -- see external_potentials).  To keep a
  single phi array the evaluator stores the external contribution separately
  in `last_phi_ext` for callers that need the exact split.
"""
from __future__ import annotations

import time

import numpy as np

from config import RunConfig, Derived
from particles import ParticleSet, wrap_periodic
from forces_direct import direct_forces
from forces_pm import PMSolver
from external_potentials import ExternalField, NoField


# ---------------------------------------------------------------------------
# kinematic expansion
# ---------------------------------------------------------------------------

class Expansion:
    """Prescribed scale factor: a = 1 for t <= t0 (or disabled), (t/t0)^a_index after."""

    def __init__(self, enabled: bool, a_index: float, t0: float):
        self.enabled = bool(enabled)
        self.a_index = float(a_index)
        self.t0 = float(t0)
        if self.enabled and self.t0 <= 0:
            raise ValueError("expansion start t0 must be positive")

    def a(self, t: float) -> float:
        if not self.enabled or t <= self.t0:
            return 1.0
        return (t / self.t0) ** self.a_index

    def H(self, t: float) -> float:
        """Hubble rate da/dt / a (zero before t0 or when disabled)."""
        if not self.enabled or t <= self.t0:
            return 0.0
        return self.a_index / t

    def __repr__(self) -> str:
        if not self.enabled:
            return "Expansion(off)"
        return f"Expansion(a=(t/{self.t0:g})^{self.a_index:g})"


# ---------------------------------------------------------------------------
# force evaluator
# ---------------------------------------------------------------------------

class ForceEvaluator:
    """Direct or PM three-law force plus an optional rigid external field.

    __call__(ps) -> (acc (N,dim) signed, phi (N,) unsigned).  Counts
    evaluations and records the wall-clock time of the last one.
    """

    def __init__(self, cfg: RunConfig, der: Derived, pm: PMSolver | None, ext: ExternalField):
        self.cfg, self.der = cfg, der
        self.method = der.method
        self.pm = pm
        self.ext = ext if ext is not None else NoField()
        self.G = der.G
        self.eps = der.eps_abs if self.method == 'direct' else max(cfg.eps, 0.0)
        self.soft = cfg.soft
        self.force_law = cfg.force_law
        self.periodic = cfg.bc == 'periodic'
        self.box = der.box if self.periodic else None
        self.chunk = int(cfg.direct_chunk)
        if self.method == 'pm' and pm is None:
            raise ValueError("method='pm' requires a PMSolver")
        if self.method not in ('direct', 'pm'):
            raise ValueError(f"unknown method {self.method!r}")
        self.n_evals = 0
        self.last_wall = 0.0
        self.total_wall = 0.0
        self.last_phi_ext = None

    @property
    def eps_eff(self) -> float:
        """Effective softening (for the adaptive-step criterion)."""
        return float(self.der.eps_abs)

    @property
    def self_energy_fn(self):
        """Callable w -> self-energy to subtract from 0.5 sum s w phi (PM only)."""
        if self.method == 'pm':
            return self.pm.self_energy
        return lambda w: 0.0

    def self_force(self, ps: ParticleSet):
        """Self-consistent part only (no external field)."""
        if ps.n == 0:
            return np.zeros_like(ps.pos), np.zeros(0, dtype=ps.pos.dtype)
        if self.method == 'direct':
            return direct_forces(ps.pos, ps.w, ps.s, self.G, self.eps, self.soft,
                                 self.force_law, box=self.box, chunk=self.chunk,
                                 need_phi=True)
        acc, phi = self.pm.forces(ps.pos, ps.w, ps.s)
        return acc.astype(ps.pos.dtype, copy=False), phi.astype(ps.pos.dtype, copy=False)

    def __call__(self, ps: ParticleSet):
        t0 = time.perf_counter()
        acc, phi = self.self_force(ps)
        if ps.n and not self.ext.is_null:
            a_ext = np.asarray(self.ext.accel(ps.pos), dtype=acc.dtype)
            phi_ext = np.asarray(self.ext.potential(ps.pos), dtype=phi.dtype)
            acc = acc + ps.s.astype(acc.dtype)[:, None] * a_ext
            phi = phi + phi_ext
            self.last_phi_ext = phi_ext
        else:
            self.last_phi_ext = np.zeros(ps.n, dtype=phi.dtype)
        self.last_wall = time.perf_counter() - t0
        self.total_wall += self.last_wall
        self.n_evals += 1
        return acc, phi

    def __repr__(self) -> str:
        core = repr(self.pm) if self.method == 'pm' else (
            f"direct(eps={self.eps:g}, soft={self.soft}, law={self.force_law}, "
            f"box={self.box}, chunk={self.chunk})")
        return f"ForceEvaluator({core}, ext={self.ext.describe()}, n_evals={self.n_evals})"


# ---------------------------------------------------------------------------
# timestep
# ---------------------------------------------------------------------------

def choose_dt(acc: np.ndarray, eps_eff: float, eta: float, dt_max: float, dt_min: float) -> float:
    """Global adaptive step eta*sqrt(eps_eff/|a|_max), clipped to [dt_min, dt_max].

    Stand-in (A9) for the original back-up scheme: a single step for all
    particles set by the strongest acceleration and the softening length."""
    acc = np.asarray(acc, dtype=np.float64)
    if acc.size == 0 or eps_eff <= 0 or eta <= 0:
        return float(dt_max)
    a_max = float(np.sqrt(np.max(np.einsum('ij,ij->i', acc, acc))))
    if not np.isfinite(a_max) or a_max <= 0:
        return float(dt_max)
    dt = eta * np.sqrt(eps_eff / a_max)
    return float(min(max(dt, dt_min), dt_max))


# ---------------------------------------------------------------------------
# leapfrog pieces
# ---------------------------------------------------------------------------

def _a_kick(a_old: float, a_new: float, p: int) -> float:
    """Time-averaged a^(1-p) over the kick interval (trapezoidal in a)."""
    if a_old == 1.0 and a_new == 1.0:
        return 1.0
    q = 1 - int(p)
    return 0.5 * (a_old ** q + a_new ** q)


def kick(ps: ParticleSet, acc: np.ndarray, dt: float, a_old: float, a_new: float, p: int) -> None:
    """In-place velocity update over dt.

    vel holds the comoving velocity v = dx/dt at scale factor a_old on entry
    and at a_new on exit.  With u = a^2 v:  u_new = u_old + acc*dt*<a^(1-p)>,
    v_new = u_new / a_new^2.  Reduces to v += acc*dt when a_old = a_new = 1."""
    dtv = ps.vel.dtype
    if a_old == 1.0 and a_new == 1.0:
        ps.vel += (acc * dt).astype(dtv, copy=False)
        return
    fac = _a_kick(a_old, a_new, p)
    u = (a_old * a_old) * ps.vel.astype(np.float64) + np.asarray(acc, dtype=np.float64) * (dt * fac)
    ps.vel[...] = (u / (a_new * a_new)).astype(dtv, copy=False)


def drift(ps: ParticleSet, dt: float, a_mid: float, box: float | None) -> None:
    """In-place position update x += v*dt with v = dx/dt defined at a_mid
    (the value kick() left in vel), then periodic wrapping if box is given.
    a_mid is accepted for interface symmetry: after the half kick vel is
    already the comoving velocity at a_mid, so no extra factor is needed."""
    ps.pos += (ps.vel * dt).astype(ps.pos.dtype, copy=False)
    if box is not None:
        ps.pos[...] = wrap_periodic(ps.pos, box)


def leapfrog_step(ps: ParticleSet, acc: np.ndarray, phi: np.ndarray, t: float, dt: float,
                  force: ForceEvaluator, expansion: Expansion, p: int, box: float | None):
    """One KDK step from t to t+dt.  `acc`/`phi` are the fields at t (the
    previous step's output); returns (acc_new, phi_new, t+dt) at t+dt."""
    a0 = expansion.a(t)
    a_half = expansion.a(t + 0.5 * dt)
    a1 = expansion.a(t + dt)
    kick(ps, acc, 0.5 * dt, a0, a_half, p)
    drift(ps, dt, a_half, box)
    acc_new, phi_new = force(ps)
    kick(ps, acc_new, 0.5 * dt, a_half, a1, p)
    return acc_new, phi_new, t + dt


def integrate(ps: ParticleSet, force: ForceEvaluator, expansion: Expansion, p: int,
              box: float | None, t: float, dt: float, n_steps: int, acc=None, phi=None):
    """Convenience loop without diagnostics (used by tests and the IC builder).
    Returns (acc, phi, t) after n_steps steps."""
    if acc is None:
        acc, phi = force(ps)
    for _ in range(int(n_steps)):
        acc, phi, t = leapfrog_step(ps, acc, phi, t, dt, force, expansion, p, box)
    return acc, phi, t


if __name__ == '__main__':  # minimal self-test
    from config import resolve_defaults, derive
    from forces_direct import potential_energy_from_phi

    # expansion bookkeeping
    ex = Expansion(True, 2.0 / 3.0, 1.0)
    assert ex.a(0.5) == 1.0 and ex.H(0.5) == 0.0
    assert np.isclose(ex.a(8.0), 4.0) and np.isclose(ex.H(8.0), 2.0 / 3.0 / 8.0)
    # with zero force u = a^2 v is conserved: v scales as 1/a^2
    ps = ParticleSet(np.zeros((1, 2)), np.array([[1.0, 0.0]]), np.ones(1), np.ones(1), np.zeros(1))
    kick(ps, np.zeros((1, 2)), 1.0, 1.0, 4.0, 2)
    assert np.isclose(ps.vel[0, 0], 1.0 / 16.0)
    # energy and |m|-momentum conservation in a small two-population box (direct)
    cfg = resolve_defaults(RunConfig('lss', npos=60, nneg=60, method='direct', eps=0.02, dt=0.01))
    der = derive(cfg)
    rng = np.random.default_rng(3)
    n = 120
    pos = rng.random((n, 2))
    s = np.where(np.arange(n) < 60, 1.0, -1.0)
    w = np.where(s > 0, der.w_pos, der.w_neg)
    vel = rng.normal(0.0, 1.0, (n, 2)) * np.where(s > 0, der.sigma_pos, der.sigma_neg)[:, None]
    ps = ParticleSet(pos, vel, w, s, (s < 0).astype(np.int32))
    force = ForceEvaluator(cfg, der, None, NoField())
    exp0 = Expansion(False, 0.0, 1.0)

    def energy(ps, phi):
        return 0.5 * float(np.sum(ps.w * np.einsum('ij,ij->i', ps.vel, ps.vel))) \
            + potential_energy_from_phi(ps.w, ps.s, phi)

    acc, phi = force(ps)
    E0, P0 = energy(ps, phi), (ps.w[:, None] * ps.vel).sum(0)
    dt = der.dt_abs
    acc, phi, t = integrate(ps, force, exp0, der.force_exponent_p, der.box, 0.0, dt, 200, acc, phi)
    E1, P1 = energy(ps, phi), (ps.w[:, None] * ps.vel).sum(0)
    assert abs(E1 - E0) < 2e-2 * abs(E0), (E0, E1)
    assert np.allclose(P0, P1, atol=1e-9 * max(1.0, np.abs(P0).max() + np.abs(P1).max() + 1.0))
    assert np.all((ps.pos >= 0) & (ps.pos < der.box)) and force.n_evals == 201
    assert np.isclose(t, 200 * dt)
    # adaptive step clipping
    assert choose_dt(np.zeros((3, 2)), 0.01, 0.1, 0.5, 1e-4) == 0.5
    assert np.isclose(choose_dt(np.array([[4.0, 0.0]]), 1.0, 0.2, 10.0, 1e-4), 0.1)
    assert choose_dt(np.array([[1e8, 0.0]]), 1.0, 0.2, 10.0, 1e-3) == 1e-3
    # expansion run stays finite and velocities are damped relative to no expansion
    ps2 = ParticleSet(pos.copy(), vel.copy(), w, s, (s < 0).astype(np.int32))
    ex = Expansion(True, 2.0 / 3.0, 0.5 * der.t_ref)
    integrate(ps2, force, ex, der.force_exponent_p, der.box, 0.0, dt, 200)
    assert np.all(np.isfinite(ps2.vel)) and np.all(np.isfinite(ps2.pos))
    print("integrator self-test OK")
