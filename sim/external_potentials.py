"""external_potentials.py -- rigid (non-responding) external fields for the
galaxy control runs.

Three options selected by cfg.control (the live twin ring, cfg.control='live',
needs no external field and maps to NoField):

  none             : bare disk, no twin matter at all;
  rigid-repulsive  : a *static* annulus of negative mass with the same density
                     profile and total |M| as the live twin ring (A12).  Its
                     field is computed once on a PM grid from the analytic
                     annulus density (CIC-free, node-sampled with a one-cell
                     edge ramp) and then only interpolated, so it never responds
                     to the disk.  Answers "is a live ring needed, or does any
                     confining potential do the job?";
  rigid-halo       : a static attractive logarithmic halo
                     Phi = 0.5 vc^2 ln(r^2 + rc^2) (Ostriker-Peebles comparison).

Convention: accel(pos) and potential(pos) are the acceleration and potential
felt by a *unit positive mass*.  The caller (integrator.ForceEvaluator)
multiplies accel by the particle sign s_i, exactly as for the self-consistent
field, so ordinary matter is repelled by the rigid ring while any twin particle
would be attracted to it.  potential() has the same meaning as the unsigned
phi of forces_direct/forces_pm (U_ext = sum_i s_i w_i phi_ext(x_i), no 1/2
because the external field has no back-reaction).

Note on periodic boxes: a static negative annulus in a periodic box has a net
monopole; the periodic PM kernel removes the k = 0 mode (Jeans swindle, A3),
so the rigid ring is the same background-subtracted object the live ring is.
"""
from __future__ import annotations

import math

import numpy as np

from config import RunConfig, Derived
from forces_pm import PMSolver


# ---------------------------------------------------------------------------
# base class and the trivial field
# ---------------------------------------------------------------------------

class ExternalField:
    """Interface: accel(pos)->(N,dim), potential(pos)->(N,), name:str."""
    name: str = 'base'

    def accel(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def potential(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def is_null(self) -> bool:
        return False

    def describe(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.describe()})"


class NoField(ExternalField):
    name = 'none'

    def accel(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos)
        return np.zeros_like(pos)

    def potential(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos)
        return np.zeros(pos.shape[0], dtype=pos.dtype)

    @property
    def is_null(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _min_image(d: np.ndarray, box: float | None) -> np.ndarray:
    if box is None:
        return d
    return d - box * np.round(d / box)


def _edge_ramp(r: np.ndarray, rin: float, rout: float, width: float) -> np.ndarray:
    """1 inside [rin, rout], 0 outside, linear ramp of `width` across each edge
    (anti-aliasing of the hard annulus edge on the grid)."""
    w = max(width, 1e-12)
    inner = np.clip((r - rin) / w + 0.5, 0.0, 1.0) if rin > 0 else np.ones_like(r)
    outer = np.clip((rout - r) / w + 0.5, 0.0, 1.0)
    return inner * outer


# ---------------------------------------------------------------------------
# rigid repulsive ring
# ---------------------------------------------------------------------------

class RigidRingField(ExternalField):
    """Static annulus rin < R < rout of negative mass -|mass_abs| centred on
    `centre`.  2-D: uniform surface density.  3-D: cylindrical annulus with a
    sech^2(z/zscale) vertical profile (zscale in length units).  The grid
    density is normalised numerically so the deposited mass is exactly
    -mass_abs whatever the resolution; acc/phi grids are stored once."""
    name = 'rigid-repulsive'

    def __init__(self, pm: PMSolver, mass_abs: float, rin: float, rout: float,
                 centre: np.ndarray, dim: int, zscale: float = 0.1):
        if dim != pm.dim:
            raise ValueError(f"dim={dim} does not match the PM solver (dim={pm.dim})")
        if rin < 0 or rout <= rin:
            raise ValueError("need 0 <= rin < rout")
        if mass_abs < 0:
            raise ValueError("mass_abs is an unsigned mass")
        self.pm, self.dim = pm, int(dim)
        self.mass_abs, self.rin, self.rout = float(mass_abs), float(rin), float(rout)
        self.zscale = float(zscale)
        self.centre = np.asarray(centre, dtype=np.float64).reshape(dim)
        self.box = pm.box if pm.periodic else None
        if dim == 2:
            self.rho_analytic = mass_abs / (math.pi * (rout ** 2 - rin ** 2))
        else:
            # mass per unit volume at the midplane: Sigma / (2 zscale) for sech^2
            self.rho_analytic = mass_abs / (math.pi * (rout ** 2 - rin ** 2) * 2.0 * zscale)
        self.rho = self._annulus_density()
        self.acc_grids, self.phi_grid = pm.field_from_density(self.rho)
        self.acc_grids = [np.asarray(g, dtype=np.float64) for g in self.acc_grids]
        self.phi_grid = np.asarray(self.phi_grid, dtype=np.float64)

    def _annulus_density(self) -> np.ndarray:
        pm = self.pm
        n, cell = pm.n, pm.cell
        x = np.arange(n, dtype=np.float64) * cell          # node positions
        axes = []
        for k in range(self.dim):
            d = _min_image(x - self.centre[k], self.box)
            shape = [1] * self.dim
            shape[k] = n
            axes.append(d.reshape(shape))
        r = np.sqrt(axes[0] ** 2 + axes[1] ** 2)           # cylindrical radius
        w = _edge_ramp(r, self.rin, self.rout, cell)
        if self.dim == 3:
            zs = max(self.zscale, 0.5 * cell)              # unresolved layer -> one cell
            w = w / np.cosh(np.clip(axes[2] / zs, -30.0, 30.0)) ** 2
        total = float(w.sum()) * cell ** self.dim
        if total <= 0:
            raise ValueError("rigid ring does not intersect the PM grid (box too small?)")
        return (-self.mass_abs / total) * w

    def accel(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos)
        if pos.shape[0] == 0:
            return np.zeros_like(pos)
        return self.pm.interpolate(self.acc_grids, pos).astype(pos.dtype, copy=False)

    def potential(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos)
        if pos.shape[0] == 0:
            return np.zeros(0, dtype=pos.dtype)
        return self.pm.interpolate(self.phi_grid, pos).astype(pos.dtype, copy=False)

    @property
    def deposited_mass(self) -> float:
        return float(self.rho.sum()) * self.pm.cell ** self.dim

    def describe(self) -> str:
        return (f"rigid repulsive annulus {self.rin:g}-{self.rout:g}, |M| = {self.mass_abs:g}, "
                f"grid {self.pm.n}^{self.dim}, bc = {self.pm.bc}"
                + (f", zscale = {self.zscale:g}" if self.dim == 3 else ""))


# ---------------------------------------------------------------------------
# rigid attractive logarithmic halo
# ---------------------------------------------------------------------------

class RigidHaloField(ExternalField):
    """Phi = 0.5 vc^2 ln(r^2 + rc^2): flat rotation curve v -> vc for r >> rc.
    `box` (optional) switches on minimum-image distances in a periodic box."""
    name = 'rigid-halo'

    def __init__(self, vc: float, rc: float, centre: np.ndarray, box: float | None = None):
        if vc < 0 or rc <= 0:
            raise ValueError("need vc >= 0 and rc > 0")
        self.vc, self.rc = float(vc), float(rc)
        self.centre = np.asarray(centre, dtype=np.float64)
        self.box = None if box is None else float(box)

    def _sep(self, pos: np.ndarray):
        pos = np.asarray(pos, dtype=np.float64)
        d = _min_image(pos - self.centre[None, :pos.shape[1]], self.box)
        return d, np.einsum('ij,ij->i', d, d)

    def accel(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos)
        d, r2 = self._sep(pos)
        a = -(self.vc ** 2) * d / (r2 + self.rc ** 2)[:, None]
        return a.astype(pos.dtype, copy=False)

    def potential(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos)
        _, r2 = self._sep(pos)
        return (0.5 * self.vc ** 2 * np.log(r2 + self.rc ** 2)).astype(pos.dtype, copy=False)

    def circular_velocity(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        return self.vc * r / np.sqrt(r * r + self.rc ** 2)

    def describe(self) -> str:
        return f"rigid logarithmic halo vc = {self.vc:g}, rc = {self.rc:g}"


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def make_external_field(cfg: RunConfig, der: Derived, pm: PMSolver | None) -> ExternalField:
    """Select the rigid field from cfg.control.  'live' and 'none' need nothing.
    'rigid-repulsive' needs a PM grid: the run's own solver when method='pm',
    otherwise a private one (same grid/box/bc/law as the run) is built; raises
    if that is impossible."""
    control = cfg.control
    dim = der.dim
    centre = np.full(dim, 0.5 * der.box)
    box = der.box if cfg.bc == 'periodic' else None
    if control in ('live', 'none'):
        return NoField()
    if control == 'rigid-halo':
        return RigidHaloField(cfg.halo_vc, cfg.halo_rc, centre, box=box)
    if control == 'rigid-repulsive':
        mass_abs = cfg.twin_ratio * 1.0                      # M_disk = 1 (A14)
        if mass_abs <= 0:
            return NoField()
        if pm is None:
            if der.method != 'direct':
                raise RuntimeError("control='rigid-repulsive' needs a PM grid but none was "
                                   "supplied and method is not 'direct'")
            if der.grid < 4 or der.box <= 0:
                raise RuntimeError("cannot build a private PM grid for the rigid ring "
                                   f"(grid={der.grid}, box={der.box})")
            pm = PMSolver(dim, der.grid, der.box, der.G, cfg.force_law, bc=cfg.bc,
                          eps=max(cfg.eps, 0.0), dtype=np.float64)
        rin, rout = cfg.twin_rin * cfg.rd, cfg.twin_rout * cfg.rd
        if rout > 0.5 * der.box and cfg.bc == 'periodic':
            raise RuntimeError(f"rigid ring outer radius {rout:g} exceeds half the box "
                               f"{0.5 * der.box:g}: the annulus would overlap its images")
        return RigidRingField(pm, mass_abs, rin, rout, centre, dim, zscale=cfg.zscale * cfg.rd)
    raise ValueError(f"unknown control {control!r}")


if __name__ == '__main__':  # minimal self-test
    pm = PMSolver(2, 128, 16.0, 1.0, 'r2', 'periodic')
    c = np.array([8.0, 8.0])
    ring = RigidRingField(pm, 1.0, 4.5, 7.0, c, 2)
    assert abs(ring.deposited_mass + 1.0) < 1e-9
    # symmetry at the centre; confinement: positive mass inside the annulus is pushed
    # inward (sheet law: a negative ring repels interior matter toward the centre)
    a0 = ring.accel(c[None, :])
    assert np.abs(a0).max() < 1e-3 * np.abs(ring.acc_grids[0]).max(), a0
    p = np.array([[8.0 + 3.5, 8.0], [8.0, 8.0 - 3.5]])
    a = ring.accel(p)
    assert a[0, 0] < 0 and a[1, 1] > 0, a
    # potential of a negative ring is a hill: higher inside than far outside
    assert ring.potential(c[None, :])[0] > ring.potential(np.array([[0.5, 0.5]]))[0]
    # halo: flat rotation curve, inward acceleration, periodic image handling
    halo = RigidHaloField(1.0, 1.0, c, box=16.0)
    ah = halo.accel(np.array([[9.0, 8.0], [8.0, 11.0]]))
    assert np.isclose(ah[0, 0], -0.5) and np.isclose(ah[1, 1], -3.0 / 10.0), ah
    assert np.isclose(halo.accel(np.array([[0.5, 8.0]]))[0, 0], 8.5 / (8.5 ** 2 + 1.0))
    assert np.isclose(halo.circular_velocity(np.array([100.0]))[0], 1.0, rtol=1e-3)
    # factory on a resolved galaxy config with method='direct' builds a private grid
    from config import RunConfig, resolve_defaults, derive
    cfg = resolve_defaults(RunConfig('galaxy', method='direct', control='rigid-repulsive', grid=64))
    f = make_external_field(cfg, derive(cfg), None)
    assert isinstance(f, RigidRingField) and f.pm.n == 64
    cfg3 = resolve_defaults(RunConfig('galaxy', dim=3, method='pm', control='rigid-repulsive', grid=32))
    f3 = make_external_field(cfg3, derive(cfg3), None) if False else None  # method='pm' needs a solver
    pm3 = PMSolver(3, 32, 16.0, 1.0, 'r2', 'periodic')
    r3 = RigidRingField(pm3, 1.0, 4.5, 7.0, np.full(3, 8.0), 3, zscale=0.1)
    assert abs(r3.deposited_mass + 1.0) < 1e-9 and r3.accel(np.full((1, 3), 8.0)).shape == (1, 3)
    assert make_external_field(resolve_defaults(RunConfig('galaxy', control='none')),
                               derive(RunConfig('galaxy', control='none')), None).is_null
    print("external_potentials self-test OK")
