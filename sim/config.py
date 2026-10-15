"""config.py -- single source of truth for run parameters, defaults, derived
physical quantities, unit conventions and the explicit assumption list.

Units (A14): G = 1.  LSS: box side L = 1, total positive mass M_+ = 1.
Galaxy: disk mass M_disk = 1, disk scale length R_d = 1.
Sentinel values -1 / 'auto' mean "fill in resolve_defaults()".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, asdict, replace

# I0(1)K0(1) - I1(1)K1(1): Freeman (1970) exponential-disk rotation curve at R = 2 R_d
_FREEMAN_AT_2RD = 0.192849


@dataclass
class RunConfig:
    scenario: str
    dim: int = 2
    method: str = 'auto'
    bc: str = 'periodic'
    force_law: str = 'r2'
    soft: str = 'plummer'
    eps: float = -1.0
    grid: int = -1
    dt: float = -1.0            # fraction of the reference time
    adaptive: bool = False
    eta: float = 0.1
    tmax: float = -1.0          # reference times
    snap_every: float = -1.0    # reference times
    log_every: int = 10         # steps
    structure_every: float = 0.5  # reference times
    dtype: str = 'float64'
    direct_chunk: int = 1024
    seed: int = 1
    outdir: str = ''
    quiet: bool = False
    # LSS
    npos: int = 5000
    nneg: int = 5000
    mass_ratio: float = 64.0
    vel_ratio: float = 4.0
    lambda_j_neg: float = 0.125   # lambda_J(negative) / L
    ic: str = 'poisson'
    ic_noise: float = 0.0
    neg_only: bool = False
    pos_only: bool = False
    expansion: bool = False
    a_index: float = 0.6667
    t0_exp: float = 1.0           # in t_J,neg
    structure_grid: int = 128
    clump_threshold: float = 5.0
    clump_min_cells: int = 4
    # galaxy
    ndisk: int = 5000
    ntwin: int = 5000
    rd: float = 1.0
    rmax: float = 4.0
    q: float = 1.5
    zscale: float = 0.1
    twin_ratio: float = 1.0
    twin_rin: float = 4.5
    twin_rout: float = 7.0
    twin_sigma: float = 0.5
    twin_rot: float = 0.0
    control: str = 'live'
    halo_vc: float = 1.0
    halo_rc: float = 1.0
    box: float = -1.0
    label: str = ''


@dataclass
class Derived:
    G: float
    box: float
    dim: int
    method: str
    w_pos: float
    w_neg: float
    sigma_pos: float
    sigma_neg: float
    lambda_j_pos: float
    lambda_j_neg: float
    t_ref: float
    t_ref_name: str
    dt_abs: float
    tmax_abs: float
    n_steps: int
    snap_every_abs: float
    structure_every_abs: float
    eps_abs: float
    grid: int
    force_exponent_p: int
    vc_ref: float
    t_rot_ref: float


SWEEPABLE: dict = {
    'npos': int, 'nneg': int, 'n': int, 'eps': float, 'dt': float, 'grid': int,
    'tmax': float, 'mass_ratio': float, 'vel_ratio': float, 'lambda_j_neg': float,
    'ic': str, 'method': str, 'bc': str, 'force_law': str, 'dim': int,
    'expansion': bool, 'ndisk': int, 'ntwin': int, 'q': float, 'twin_ratio': float,
    'twin_sigma': float, 'twin_rot': float, 'control': str, 'box': float,
}

_CHOICES = {
    'scenario': ('lss', 'galaxy'), 'dim': (2, 3), 'method': ('auto', 'direct', 'pm'),
    'bc': ('periodic', 'isolated'), 'force_law': ('r2', 'log'), 'soft': ('plummer', 'cutin'),
    'dtype': ('float64', 'float32'), 'ic': ('poisson', 'glass', 'lattice'),
    'control': ('live', 'none', 'rigid-repulsive', 'rigid-halo'),
}


def parse_param_value(param: str, text: str):
    """Convert one CLI token to the python type of SWEEPABLE[param] (bool-safe)."""
    typ = SWEEPABLE.get(param, str)
    if typ is bool:
        return str(text).strip().lower() in ('1', 'true', 'yes', 'on')
    return typ(text)


def _n_total(c: RunConfig) -> int:
    if c.scenario == 'lss':
        return (0 if c.neg_only else c.npos) + (0 if c.pos_only else c.nneg)
    return c.ndisk + (c.ntwin if c.control == 'live' else 0)


def _validate(c: RunConfig) -> None:
    for name, allowed in _CHOICES.items():
        if getattr(c, name) not in allowed:
            raise ValueError(f"{name}={getattr(c, name)!r} not in {allowed}")
    if c.force_law == 'log' and c.dim != 2:
        raise ValueError("force_law='log' (true 2-D gravity) is only defined for dim=2")
    if c.neg_only and c.pos_only:
        raise ValueError("neg_only and pos_only are mutually exclusive")
    if c.scenario == 'lss' and _n_total(c) <= 0:
        raise ValueError("no particles: check npos/nneg/neg_only/pos_only")
    if c.scenario == 'galaxy' and c.ndisk <= 0:
        raise ValueError("ndisk must be positive")
    if c.vel_ratio <= 0 or c.mass_ratio < 0 or c.lambda_j_neg <= 0:
        raise ValueError("vel_ratio, lambda_j_neg must be > 0 and mass_ratio >= 0")
    if c.twin_rout <= c.twin_rin:
        raise ValueError("twin_rout must exceed twin_rin")


def resolve_defaults(cfg: RunConfig) -> RunConfig:
    """Return a copy with every -1/'auto' sentinel replaced by the scenario default."""
    c = replace(cfg)
    _validate(c)
    lss = c.scenario == 'lss'
    if c.method == 'auto':
        c.method = 'direct' if _n_total(c) <= 4000 else 'pm'
    if c.grid < 0:
        c.grid = 256 if c.dim == 2 else 64
    if c.box < 0:
        c.box = 1.0 if lss else 16.0 * c.rd
    if c.eps < 0:
        if c.method == 'direct':
            c.eps = 0.002 * c.box if lss else 0.05 * c.rd
        else:
            c.eps = 0.0          # PM: 0 means "cell size only"
    if c.dt < 0:
        c.dt = 0.02 if lss else 0.005
    if c.tmax < 0:
        c.tmax = 20.0 if lss else 10.0
    if c.snap_every < 0:
        c.snap_every = 1.0 if lss else 0.5
    if c.structure_every < 0:
        c.structure_every = 0.5
    if c.dt <= 0 or c.tmax <= 0:
        raise ValueError("dt and tmax must be positive")
    return c


def jeans_length(sigma: float, density: float, G: float, dim: int, force_law: str) -> float:
    """Jeans length from the linear dispersion relation of the chosen (dim, law).
    sheet (2-D, 1/r^2):  omega^2 = k^2 s^2 - 2 pi G Sigma |k|  -> lambda_J = s^2/(G Sigma)
    2-D log gravity:     omega^2 = k^2 s^2 - 2 pi G rho       -> lambda_J = s sqrt(2 pi/(G rho))
    3-D:                 omega^2 = k^2 s^2 - 4 pi G rho       -> lambda_J = s sqrt(pi/(G rho))
    """
    if density <= 0 or G <= 0:
        return float('inf')
    if dim == 2 and force_law == 'r2':
        return sigma * sigma / (G * density)
    if dim == 2:
        return sigma * math.sqrt(2.0 * math.pi / (G * density))
    return sigma * math.sqrt(math.pi / (G * density))


def sigma_from_jeans_length(lam: float, density: float, G: float, dim: int, force_law: str) -> float:
    """Inverse of jeans_length: dispersion giving Jeans length lam."""
    if dim == 2 and force_law == 'r2':
        return math.sqrt(lam * G * density)
    if dim == 2:
        return lam * math.sqrt(G * density / (2.0 * math.pi))
    return lam * math.sqrt(G * density / math.pi)


def derive(cfg: RunConfig) -> Derived:
    c = resolve_defaults(cfg)
    G = 1.0
    p = 1 if (c.force_law == 'log' and c.dim == 2) else 2
    if c.scenario == 'lss':
        L = c.box
        vol = L ** c.dim
        M_pos, M_neg = 1.0, c.mass_ratio * 1.0          # A6: density ratio
        w_pos = M_pos / max(c.npos, 1)
        w_neg = M_neg / max(c.nneg, 1)
        rho_pos, rho_neg = M_pos / vol, M_neg / vol
        lam_neg = c.lambda_j_neg * L                      # A4
        sigma_neg = sigma_from_jeans_length(lam_neg, rho_neg, G, c.dim, c.force_law)
        sigma_pos = sigma_neg / c.vel_ratio               # A5
        lam_pos = jeans_length(sigma_pos, rho_pos, G, c.dim, c.force_law)
        t_ref, t_ref_name = lam_neg / sigma_neg, 't_J,neg'
        vc_ref, t_rot_ref = 0.0, 0.0
    else:
        M_disk = 1.0
        w_pos = M_disk / max(c.ndisk, 1)
        w_neg = c.twin_ratio * M_disk / max(c.ntwin, 1)
        x = c.rmax / c.rd
        f_trunc = 1.0 - math.exp(-x) * (1.0 + x)         # mass fraction inside rmax
        # analytic Freeman v_c at 2 R_d of the (truncation-normalised) exponential disk alone;
        # the IC uses the *measured* field (A11); this one only sets reference times.
        vc_ref = math.sqrt(2.0 * G * M_disk * _FREEMAN_AT_2RD / (c.rd * f_trunc))
        t_rot_ref = 2.0 * math.pi * 2.0 * c.rd / vc_ref
        t_ref, t_ref_name = t_rot_ref, 'T_rot(2R_d)'
        sigma_pos = 0.0                                   # set from Toomre Q in the IC
        sigma_neg = c.twin_sigma * vc_ref                 # A12
        if c.dim == 2:
            ring_vol = math.pi * (c.twin_rout ** 2 - c.twin_rin ** 2)
        else:
            ring_vol = 4.0 / 3.0 * math.pi * (c.twin_rout ** 3 - c.twin_rin ** 3)
        rho_ring = c.twin_ratio * M_disk / ring_vol
        lam_neg = jeans_length(sigma_neg, rho_ring, G, c.dim, c.force_law) if c.twin_ratio > 0 else 0.0
        lam_pos = 0.0
    if c.method == 'pm':
        cell = c.box / c.grid
        eps_abs = max(cell, c.eps)
    else:
        eps_abs = c.eps
    dt_abs = c.dt * t_ref
    tmax_abs = c.tmax * t_ref
    n_steps = int(math.ceil(c.tmax / c.dt - 1e-9))
    return Derived(G=G, box=c.box, dim=c.dim, method=c.method, w_pos=w_pos, w_neg=w_neg,
                   sigma_pos=sigma_pos, sigma_neg=sigma_neg, lambda_j_pos=lam_pos,
                   lambda_j_neg=lam_neg, t_ref=t_ref, t_ref_name=t_ref_name, dt_abs=dt_abs,
                   tmax_abs=tmax_abs, n_steps=n_steps, snap_every_abs=c.snap_every * t_ref,
                   structure_every_abs=c.structure_every * t_ref, eps_abs=eps_abs, grid=c.grid,
                   force_exponent_p=p, vc_ref=vc_ref, t_rot_ref=t_rot_ref)


def assumptions(cfg: RunConfig, der: Derived) -> list:
    """Human-readable A1..A17, instantiated with the actual numbers of this run."""
    c = resolve_defaults(cfg)
    lss = c.scenario == 'lss'
    unit = 'L' if lss else 'R_d'
    A = []
    # A1 force law
    if c.dim == 3:
        law = "3-D Newtonian 1/r^2 force (FFT kernel -4piG/k^2)"
    elif c.force_law == 'r2':
        law = "1/r^2 force in the plane (razor-thin sheet, 'flat gas'; FFT kernel -2piG/|k|)"
    else:
        law = "logarithmic (true 2-D) gravity, force 1/r (FFT kernel -2piG/k^2) -- STRESS TEST"
    A.append(f"A1 Force law: {law}. Pages say only 'sum of Newtonian forces' for mass points in a plane.")
    # A2 boundaries
    if c.bc == 'periodic':
        bcs = ("periodic box; direct sum uses the minimum-image rule (the 'dashed rectangle' tiling), "
               "PM uses exact periodic images")
    else:
        bcs = "isolated (open) boundaries; PM via zero-padded doubled grid, direct sum without images"
    A.append(f"A2 Boundary conditions: {bcs}; box = {c.box:g} {unit}, method = {c.method}. "
             + ("Pages are silent for LSS (torus only mentioned for the spiral run)." if lss
                else "Pages: computational space closed on a flat torus."))
    # A3 net mass
    if lss:
        msign = 1.0 - c.mass_ratio
        A.append(f"A3 Net signed mass M_+ - |M_-| = {msign:g} (box not neutral); "
                 + ("k=0 mode removed (Jeans swindle: uniform background subtracted); the minimum-image "
                    "direct sum does this implicitly by symmetry." if c.bc == 'periodic'
                    else "isolated BC: no background subtraction, the net monopole acts on all particles."))
    else:
        A.append(f"A3 Net signed mass M_disk - |M_twin| = {1.0 - (c.twin_ratio if c.control == 'live' else 0.0):g}; "
                 + ("k=0 mode removed in the periodic solver (Jeans swindle)." if c.bc == 'periodic'
                    else "isolated BC: full monopole retained."))
    # A4 velocity scale
    if lss:
        A.append(f"A4 Absolute velocity scale set by lambda_J(neg)/L = {c.lambda_j_neg:g}: "
                 f"sigma_- = {der.sigma_neg:.4g}, sigma_+ = {der.sigma_pos:.4g} per component; "
                 f"lambda_J,+ = {der.lambda_j_pos:.4g} L, lambda_J,- = {der.lambda_j_neg:.4g} L "
                 f"(dispersion relation of dim={c.dim}, law={c.force_law}); t_J,neg = {der.t_ref:.4g} "
                 f"(code units). Pages give only ratios ('temperature' unspecified).")
    else:
        A.append(f"A4 Velocity scale: analytic Freeman v_c(2R_d) = {der.vc_ref:.4g} of the truncated exponential "
                 f"disk alone -> T_rot(2R_d) = {der.t_rot_ref:.4g} used as reference time; actual disk "
                 f"velocities come from the measured field (A11).")
    # A5
    A.append(f"A5 '<V_twin> = 4<V>' read as a mean-speed ratio => sigma_-/sigma_+ = vel_ratio = {c.vel_ratio:g} "
             f"per component (mean speed proportional to sigma for a Maxwellian)."
             + ("" if lss else " (LSS parameter; unused for the galaxy.)"))
    # A6
    A.append(f"A6 '|M_twin| = 64 M' implemented as a density ratio mass_ratio = {c.mass_ratio:g} "
             f"(total |M_-| / M_+); particle weights w_+ = {der.w_pos:.4g}, w_- = {der.w_neg:.4g} "
             f"(N_+ = {c.npos}, N_- = {c.nneg}); identical to a per-particle ratio when N_+ = N_-."
             + ("" if lss else " (LSS parameter; galaxy uses twin_ratio, A12.)"))
    # A7
    A.append(f"A7 Initial positions: ic = {c.ic}, jitter = {c.ic_noise:g} mean spacings"
             + (" (Poisson shot noise seeds the instability)" if c.ic == 'poisson' else "")
             + f"; seed = {c.seed}. Pages say only 'uniform'."
             + (" Single-population control: " + ('negatives only' if c.neg_only else 'positives only') + "."
                if (c.neg_only or c.pos_only) else ""))
    # A8 softening
    if c.method == 'direct':
        soft = f"direct sum, {c.soft} softening eps = {der.eps_abs:.4g} {unit}"
        if c.soft == 'cutin':
            soft += " (force zeroed below eps, the pages' 'cut-in')"
    else:
        soft = (f"PM grid {c.grid}^{c.dim}, cell = {c.box / c.grid:.4g} {unit}"
                + (f", Gaussian k-space smoothing eps = {c.eps:.4g}" if c.eps > 0 else ", no extra smoothing")
                + f"; effective softening = {der.eps_abs:.4g}")
    A.append(f"A8 Softening: {soft}. Pages mention a cut-in only for cluster runs.")
    # A9 timestep
    A.append(f"A9 Timestep: leapfrog KDK, dt = {c.dt:g} {der.t_ref_name} = {der.dt_abs:.4g} code units, "
             f"{der.n_steps} steps"
             + (f"; adaptive global step eta*sqrt(eps/a_max) with eta = {c.eta:g} as stand-in for the "
                f"back-up scheme (dt above is the cap)" if c.adaptive else " (fixed)")
             + "; dtype = " + c.dtype + ". Pages: 'small enough that halving it changes nothing'.")
    # A10
    A.append(f"A10 Run length tmax = {c.tmax:g} {der.t_ref_name} = {der.tmax_abs:.4g} code units; "
             f"snapshots every {c.snap_every:g}, structure every {c.structure_every:g} {der.t_ref_name}. "
             "Pages: 'remarkably stable' / 'impressive number of turns' (no duration given).")
    # A11 disk
    if lss:
        A.append("A11 Galaxy disk: not applicable to the LSS scenario.")
    else:
        A.append(f"A11 Galaxy disk: {'2-D' if c.dim == 2 else '3-D sech^2 (zscale=%g R_d)' % c.zscale} "
                 f"exponential disk, R_d = {c.rd:g}, truncated at {c.rmax:g} (= {c.rmax / c.rd:g} R_d), "
                 f"N_disk = {c.ndisk}, Toomre Q = {c.q:g}; v_c measured from the actual initial force field "
                 "(including the twin ring's repulsion), epicyclic dispersions with asymmetric-drift correction; "
                 "where the net radial force points outward v_c is undefined and this is logged, not patched. "
                 "Pages: 'rotating central clump', IC analytic solution never published.")
    # A12 ring
    if lss:
        A.append("A12 Twin ring: not applicable to the LSS scenario.")
    else:
        A.append(f"A12 Twin ring: control = {c.control}; "
                 + (f"annulus {c.twin_rin:g}-{c.twin_rout:g} R_d of negative mass, |M_twin| = twin_ratio x M_disk "
                    f"= {c.twin_ratio:g}, N_twin = {c.ntwin} (w = {der.w_neg:.4g}), Maxwellian sigma = "
                    f"{c.twin_sigma:g} v_c,ref = {der.sigma_neg:.4g}, rotation fraction {c.twin_rot:g}; ring "
                    f"Jeans length (self-gravity of the ring alone) = {der.lambda_j_neg:.4g} R_d vs ring width "
                    f"{c.twin_rout - c.twin_rin:g} R_d -- fragmentation is a measured outcome, not assumed "
                    "absent." if c.control == 'live' else
                    {'none': "bare disk, no twin matter.",
                     'rigid-repulsive': f"static repulsive annulus {c.twin_rin:g}-{c.twin_rout:g} R_d with "
                                        f"|M| = {c.twin_ratio:g} M_disk (same density as the live ring).",
                     'rigid-halo': f"static attractive logarithmic halo v_c = {c.halo_vc:g}, r_c = {c.halo_rc:g} "
                                   "(Ostriker-Peebles comparison)."}[c.control]))
    # A13
    A.append(f"A13 Box: {c.box:g} {unit} ({c.bc}); " + ("the pages give no torus size." if not lss else
             "LSS box is the unit of length."))
    # A14
    A.append("A14 Units: G = 1; " + ("L = 1, M_+ = 1 (total positive mass)." if lss
                                     else "M_disk = 1, R_d = 1."))
    # A15
    if c.expansion:
        A.append(f"A15 Expansion: kinematic comoving frame a(t) = (t/t0)^{c.a_index:g} switched on at "
                 f"t0 = {c.t0_exp:g} {der.t_ref_name}; equations x'' + 2H x' = g_c / a^{der.force_exponent_p + 1} "
                 f"(p = {der.force_exponent_p}). Not a Janus cosmology; tests only whether Hubble drag erases the pattern.")
    else:
        A.append("A15 Expansion: none (as in the original runs: 'we do not know how').")
    # A16
    A.append(f"A16 Dimension: {c.dim}-D" + (f"; PM {c.grid}^3 with 1/k^2 kernel, same IC recipe; {_n_total(c)} "
             "particles in 3-D are sparse -- N sweeps are mandatory before concluding." if c.dim == 3
             else " (pages: 3-D 'beyond our capabilities')."))
    # A17
    A.append("A17 Potential energy: " + ("PM self-energy subtracted using the measured potential of a unit "
             "particle at a cell centre; residual position dependence appears as noise, not drift."
             if c.method == 'pm' else "direct pair sum, no self-interaction term.")
             + " Conserved diagnostics: E = KE + U and |m|-weighted momentum P_abs = sum |m| v; the signed "
             "momentum sum m v is NOT conserved (cross-sign pairs receive parallel impulses) and its drift is "
             "expected physics.")
    return A


def cfg_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


def cfg_from_dict(d: dict) -> RunConfig:
    names = {f.name for f in fields(RunConfig)}
    kw = {k: v for k, v in d.items() if k in names}
    if 'scenario' not in kw:
        raise ValueError("config dict lacks 'scenario'")
    return RunConfig(**kw)


def cfg_with(cfg: RunConfig, **overrides) -> RunConfig:
    """Copy with changes (sweeps). 'n' sets both populations of the current scenario."""
    ov = dict(overrides)
    if 'n' in ov:
        n = int(ov.pop('n'))
        if cfg.scenario == 'lss':
            ov.update(npos=n, nneg=n)
        else:
            ov.update(ndisk=n, ntwin=n)
    names = {f.name for f in fields(RunConfig)}
    bad = set(ov) - names
    if bad:
        raise ValueError(f"unknown config field(s): {sorted(bad)}")
    return replace(cfg, **ov)


def derived_to_dict(der: Derived) -> dict:
    return asdict(der)
