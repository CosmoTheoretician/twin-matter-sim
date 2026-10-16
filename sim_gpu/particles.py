"""particles.py -- particle container and small array helpers.

A ParticleSet carries positions, velocities, unsigned weights w = |m|, signs
s = +/-1 and integer species ids (0 = ordinary matter, 1 = twin matter).
The signed mass is s*w.  All helpers are pure numpy; no other project module
is imported so this file sits at the bottom of the dependency graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleSet:
    pos: np.ndarray       # (N, dim)
    vel: np.ndarray       # (N, dim)
    w: np.ndarray         # (N,) unsigned weights |m| > 0
    s: np.ndarray         # (N,) signs +1 / -1
    species: np.ndarray   # (N,) int: 0 ordinary, 1 twin

    def __post_init__(self):
        self.pos = np.asarray(self.pos)
        self.vel = np.asarray(self.vel)
        self.w = np.asarray(self.w)
        self.s = np.asarray(self.s)
        self.species = np.asarray(self.species, dtype=np.int32)
        if self.pos.ndim != 2:
            raise ValueError("pos must have shape (N, dim)")
        n = self.pos.shape[0]
        if self.vel.shape != self.pos.shape:
            raise ValueError(f"vel shape {self.vel.shape} != pos shape {self.pos.shape}")
        for name in ('w', 's', 'species'):
            a = getattr(self, name)
            if a.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
        if n and not np.all(np.abs(self.s) == 1):
            raise ValueError("signs must be +1 or -1")
        if n and np.any(self.w < 0):
            raise ValueError("weights w = |m| must be non-negative")

    # ---- sizes -----------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.pos.shape[0])

    @property
    def dim(self) -> int:
        return int(self.pos.shape[1])

    @property
    def npos(self) -> int:
        return int(np.count_nonzero(self.s > 0))

    @property
    def nneg(self) -> int:
        return int(np.count_nonzero(self.s < 0))

    @property
    def dtype(self):
        return self.pos.dtype

    # ---- masks and derived arrays -----------------------------------------
    @property
    def pos_mask(self) -> np.ndarray:
        return self.s > 0

    @property
    def neg_mask(self) -> np.ndarray:
        return self.s < 0

    def species_mask(self, k: int) -> np.ndarray:
        return self.species == k

    @property
    def signed_mass(self) -> np.ndarray:
        return self.s * self.w

    @property
    def total_abs_mass(self) -> float:
        return float(self.w.sum())

    @property
    def total_signed_mass(self) -> float:
        return float(self.signed_mass.sum())

    # ---- copies -------------------------------------------------------------
    def copy(self) -> "ParticleSet":
        return ParticleSet(self.pos.copy(), self.vel.copy(), self.w.copy(),
                           self.s.copy(), self.species.copy())

    def astype(self, dtype) -> "ParticleSet":
        dt = np.dtype(dtype)
        return ParticleSet(self.pos.astype(dt, copy=True), self.vel.astype(dt, copy=True),
                           self.w.astype(dt, copy=True), self.s.astype(dt, copy=True),
                           self.species.copy())

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (f"ParticleSet(n={self.n}, dim={self.dim}, npos={self.npos}, nneg={self.nneg}, "
                f"species={sorted(set(self.species.tolist())) if self.n else []}, dtype={self.dtype})")


# ---- periodic helpers ---------------------------------------------------------

def wrap_periodic(pos: np.ndarray, box: float) -> np.ndarray:
    """Map coordinates into [0, box) along every axis (new array)."""
    out = np.mod(pos, box)
    # guard against -0.0 or round-off landing exactly on `box` in float32
    out[out >= box] -= box
    out[out < 0] += box
    return out


def min_image(d: np.ndarray, box: float) -> np.ndarray:
    """Separation vectors mapped to the nearest periodic image, in [-box/2, box/2)."""
    return d - box * np.round(d / box)


# ---- set algebra ----------------------------------------------------------------

def concat(a: ParticleSet, b: ParticleSet) -> ParticleSet:
    if a.n == 0:
        return b.copy()
    if b.n == 0:
        return a.copy()
    if a.dim != b.dim:
        raise ValueError(f"dimension mismatch: {a.dim} vs {b.dim}")
    dt = np.result_type(a.pos.dtype, b.pos.dtype)
    return ParticleSet(np.concatenate([a.pos, b.pos]).astype(dt, copy=False),
                       np.concatenate([a.vel, b.vel]).astype(dt, copy=False),
                       np.concatenate([a.w, b.w]).astype(dt, copy=False),
                       np.concatenate([a.s, b.s]).astype(dt, copy=False),
                       np.concatenate([a.species, b.species]))


def subset(ps: ParticleSet, mask: np.ndarray) -> ParticleSet:
    """Copy of the particles selected by a boolean mask or an index array."""
    idx = np.asarray(mask)
    return ParticleSet(ps.pos[idx].copy(), ps.vel[idx].copy(), ps.w[idx].copy(),
                       ps.s[idx].copy(), ps.species[idx].copy())


def empty(dim: int, dtype=np.float64) -> ParticleSet:
    dt = np.dtype(dtype)
    return ParticleSet(np.zeros((0, dim), dt), np.zeros((0, dim), dt), np.zeros(0, dt),
                       np.zeros(0, dt), np.zeros(0, np.int32))


# ---- centre of |m|-weighted mass --------------------------------------------------

def centre_of_mass(ps: ParticleSet, mask=None, box=None) -> np.ndarray:
    """|m|-weighted centre of the (optionally masked) particles.

    With `box` given the centre is periodic-aware: each coordinate is mapped to an
    angle theta = 2 pi x / box, the weighted mean of (cos, sin) is taken and the
    resulting angle mapped back into [0, box) (circular mean).  This is well
    defined for a clump straddling a box edge and reduces to the ordinary mean for
    a compact distribution far from the edges.  If the weighted vector sum vanishes
    (perfectly uniform ring) the plain mean is returned for that axis.
    """
    if mask is None:
        pos, w = ps.pos, ps.w
    else:
        pos, w = ps.pos[mask], ps.w[mask]
    if pos.shape[0] == 0:
        return np.full(ps.dim, np.nan)
    wsum = float(w.sum())
    if wsum <= 0:
        return pos.mean(axis=0).astype(np.float64)
    if box is None:
        return (w[:, None] * pos).sum(axis=0) / wsum
    theta = 2.0 * np.pi * pos.astype(np.float64) / box
    c = (w[:, None] * np.cos(theta)).sum(axis=0) / wsum
    sn = (w[:, None] * np.sin(theta)).sum(axis=0) / wsum
    out = np.empty(pos.shape[1], dtype=np.float64)
    for k in range(pos.shape[1]):
        if c[k] ** 2 + sn[k] ** 2 < 1e-24:
            out[k] = float((w * pos[:, k]).sum() / wsum)
        else:
            out[k] = (np.arctan2(sn[k], c[k]) / (2.0 * np.pi)) * box
    return wrap_periodic(out, box)


def velocity_centre(ps: ParticleSet, mask=None) -> np.ndarray:
    """|m|-weighted mean velocity (P_abs / sum |m|)."""
    vel, w = (ps.vel, ps.w) if mask is None else (ps.vel[mask], ps.w[mask])
    if vel.shape[0] == 0 or w.sum() <= 0:
        return np.zeros(ps.dim)
    return (w[:, None] * vel).sum(axis=0) / float(w.sum())


def remove_bulk_velocity(ps: ParticleSet, mask=None) -> np.ndarray:
    """Subtract the |m|-weighted mean velocity in place; returns the removed vector."""
    v0 = velocity_centre(ps, mask)
    if mask is None:
        ps.vel -= v0.astype(ps.vel.dtype)
    else:
        ps.vel[mask] -= v0.astype(ps.vel.dtype)
    return v0


def radii(ps: ParticleSet, centre: np.ndarray, box: float | None = None,
          mask=None) -> np.ndarray:
    """Distances from `centre` (minimum-image if box is given)."""
    pos = ps.pos if mask is None else ps.pos[mask]
    d = pos - np.asarray(centre, dtype=pos.dtype)[None, :]
    if box is not None:
        d = min_image(d, box)
    return np.sqrt((d * d).sum(axis=1))
