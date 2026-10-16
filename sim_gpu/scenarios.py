"""scenarios.py -- run orchestration for both scenarios.

Builds the force evaluator (PM or direct plus rigid external field), the
initial condition, integrates with diagnostics / snapshots / structure
measurements at the configured cadence, evaluates the PRE-REGISTERED criteria
(thresholds fixed as module constants, never adjusted by the code) and writes
config.json, assumptions.txt, diagnostics.csv, structure.csv, summary.json
and the plots.  The time loop is shared by the LSS and galaxy scenarios.
With cfg.device == 'gpu' the force evaluator and the time loop are delegated
to integrator_torch / run_loop_gpu (particles resident on the GPU, host copies
only at diagnostic, structure and snapshot cadence); everything else -- initial
conditions, criteria, summaries, plots -- is reused unchanged.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from config import (RunConfig, Derived, resolve_defaults, derive, assumptions, cfg_to_dict,
                    derived_to_dict)
from particles import ParticleSet, centre_of_mass
from forces_pm import PMSolver
from external_potentials import ExternalField, make_external_field
from integrator import ForceEvaluator, Expansion, choose_dt, leapfrog_step
from initial_conditions import make_lss_ic, make_galaxy_ic
from diagnostics_core import diagnostics_row, CsvLogger, write_json
from structure_stats import lss_structure_row, disk_structure_row
from plotting import (snapshot, plot_conservation, plot_lss_structure, plot_disk_modes,
                      plot_rotation_curve)

# ---- pre-registered thresholds (notes, section 3) ---------------------------
LSS_SEG_THRESHOLD = -0.3     # segregation index below this = cells
LSS_SEG_TIME = 3.0           # ... by this many t_J,neg
LSS_STABLE_T0, LSS_STABLE_T1 = 5.0, 20.0
LSS_STABLE_CHANGE = 0.20     # |dn/n| of negative clump count between T0 and T1
LSS_PERSISTENCE = 0.7
GAL_SPIRAL_A2 = 0.1          # outer A2 threshold
GAL_SPIRAL_ROT = 8           # ... sustained for >= 8 of 10 rotations
GAL_BAR_A2 = 0.2
GAL_BRAKING = 0.05           # disk L_z decrease fraction
NOISE_BAND = 0.05            # "ambiguous (within noise)" half-width for index-like criteria
_AX = 'xyz'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _say(cfg: RunConfig, msg: str) -> None:
    if not cfg.quiet:
        print("[run] " + msg, file=sys.stderr, flush=True)


def _f(v) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _col(rows: list, key: str) -> np.ndarray:
    return np.array([_f(r.get(key)) for r in rows], dtype=np.float64)


def _last(rows: list, key: str) -> float:
    for r in reversed(rows):
        v = _f(r.get(key))
        if np.isfinite(v):
            return v
    return float('nan')


def _first(rows: list, key: str) -> float:
    for r in rows:
        v = _f(r.get(key))
        if np.isfinite(v):
            return v
    return float('nan')


def _at_time(rows: list, t_abs: float, tol: float):
    if not rows:
        return None
    ts = _col(rows, 't')
    i = int(np.nanargmin(np.abs(ts - t_abs)))
    return rows[i] if abs(ts[i] - t_abs) <= tol else None


def _max_abs(rows: list, key: str) -> float:
    v = np.abs(_col(rows, key))
    return float(np.nanmax(v)) if np.isfinite(v).any() else float('nan')


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')


def _outdir(cfg: RunConfig) -> str:
    out = cfg.outdir or os.path.join('out', f"{cfg.scenario}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    return out


def _plot(cfg, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except Exception as exc:  # a broken figure must not lose a finished run
        _say(cfg, f"plot {fn.__name__} failed: {exc!r}")


# ---------------------------------------------------------------------------
# forces and metadata
# ---------------------------------------------------------------------------

def build_forces(cfg: RunConfig, der: Derived):
    """(ForceEvaluator, PMSolver | None, ExternalField) for the resolved config.
    On the GPU: (ForceEvaluatorTorch, PMSolverTorch, ExternalField)."""
    if cfg.device == 'gpu':
        from integrator_torch import build_gpu_forces
        return build_gpu_forces(cfg, der)
    pm = None
    if der.method == 'pm':
        pm = PMSolver(der.dim, der.grid, der.box, der.G, cfg.force_law, bc=cfg.bc,
                      eps=max(cfg.eps, 0.0), dtype=np.dtype(cfg.dtype))
    ext = make_external_field(cfg, der, pm)
    return ForceEvaluator(cfg, der, pm, ext), pm, ext


def write_run_metadata(cfg: RunConfig, der: Derived, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    write_json(os.path.join(outdir, 'config.json'),
               {'config': cfg_to_dict(cfg), 'derived': derived_to_dict(der)})
    with open(os.path.join(outdir, 'assumptions.txt'), 'w') as fh:
        fh.write(f"Assumptions in force for this run ({cfg.scenario}, seed {cfg.seed})\n\n")
        for line in assumptions(cfg, der):
            fh.write(line + "\n\n")


# ---------------------------------------------------------------------------
# generic time loop
# ---------------------------------------------------------------------------

def run_loop(cfg: RunConfig, der: Derived, ps: ParticleSet, force: ForceEvaluator, outdir: str,
             structure_fn, snapshot_kwargs: dict) -> dict:
    """Leapfrog KDK loop with logging at cfg.log_every steps, snapshots every
    der.snap_every_abs, structure rows every der.structure_every_abs.
    structure_fn(ps, t, state) -> (row, state).  snapshot_kwargs keys: centre_fn
    (ps -> centre), r_escape, relative (bool), extent, title.
    cfg.device == 'gpu' delegates to run_loop_gpu (same outputs)."""
    if cfg.device == 'gpu':
        from run_loop_gpu import run_loop_gpu
        return run_loop_gpu(cfg, der, ps, force, outdir, structure_fn, snapshot_kwargs)
    box_wrap = der.box if cfg.bc == 'periodic' else None
    p = der.force_exponent_p
    expansion = Expansion(cfg.expansion, cfg.a_index, cfg.t0_exp * der.t_ref)
    centre_fn = snapshot_kwargs.get('centre_fn')
    r_escape = snapshot_kwargs.get('r_escape')
    relative = bool(snapshot_kwargs.get('relative', False))
    extent, title = snapshot_kwargs.get('extent'), snapshot_kwargs.get('title', '')
    self_energy = force.self_energy_fn(ps.w)
    diag_log = CsvLogger(os.path.join(outdir, 'diagnostics.csv'))
    struct_log = CsvLogger(os.path.join(outdir, 'structure.csv'))
    diag_rows, struct_rows = [], []
    tmax, dt_max = der.tmax_abs, der.dt_abs
    dt_min = 1e-3 * dt_max
    wall0 = time.perf_counter()
    acc, phi = force(ps)
    t, step = 0.0, 0
    dt = choose_dt(acc, force.eps_eff, cfg.eta, dt_max, dt_min) if cfg.adaptive else dt_max
    dts = [dt]
    state = [None, 0.0, 0]          # structure state, E0, P0 placeholders
    E0, P0 = None, None

    def centre():
        return centre_fn(ps) if centre_fn is not None else None

    def log_diag(fix_zero=False):
        c = centre()
        row = diagnostics_row(ps, phi, t, step, dt, expansion.a(t), self_energy, E0, P0, centre=c,
                              r_escape=r_escape, wall=time.perf_counter() - wall0, box=box_wrap,
                              phi_ext=force.last_phi_ext)
        if fix_zero:
            row['dE_rel'], row['dP_abs_rel'] = 0.0, 0.0
        row['t_ref_units'] = t / der.t_ref
        row['n_force_evals'] = force.n_evals
        diag_log.log(row)
        diag_rows.append(row)
        return row

    def do_structure():
        row, state[0] = structure_fn(ps, t, state[0])
        row['t_ref_units'] = t / der.t_ref
        struct_log.log(row)
        struct_rows.append(row)

    def do_snapshot(k):
        c = centre() if relative else None
        snapshot(ps, der.box, t / der.t_ref, der.t_ref_name,
                 os.path.join(outdir, f'snap_{k:04d}.png'), der.dim, centre=c,
                 extent=extent if relative else None, title=title)

    row0 = log_diag()
    E0 = row0['E']
    P0 = np.array([row0[f'p_abs_{_AX[k]}'] for k in range(der.dim)])
    diag_rows.pop()                                     # re-log with the drifts defined as 0
    diag_log.close()
    diag_log = CsvLogger(os.path.join(outdir, 'diagnostics.csv'))
    log_diag(fix_zero=True)
    do_structure()
    do_snapshot(0)
    next_snap, next_struct, k_snap = der.snap_every_abs, der.structure_every_abs, 1
    n_est = int(math.ceil(tmax / dt_max))
    _say(cfg, f"integrating to t = {cfg.tmax:g} {der.t_ref_name} ({n_est} steps of {dt_max:.4g}"
              f"{', adaptive cap' if cfg.adaptive else ''}), method = {der.method}, N = {ps.n}")
    report_every = max(cfg.log_every, n_est // 20)
    while t < tmax - 1e-9 * tmax:
        if cfg.adaptive:
            dt = choose_dt(acc, force.eps_eff, cfg.eta, dt_max, dt_min)
        dt = min(dt, tmax - t)
        dts.append(dt)
        acc, phi, t = leapfrog_step(ps, acc, phi, t, dt, force, expansion, p, box_wrap)
        step += 1
        last = t >= tmax - 1e-9 * tmax
        if step % cfg.log_every == 0 or last:
            row = log_diag()
            if step % report_every == 0 or last:
                _say(cfg, f"step {step} t = {t / der.t_ref:.3f} {der.t_ref_name} dE/E = {row['dE_rel']:.2e} "
                          f"dP_abs = {row['dP_abs_rel']:.1e} wall = {row['wall']:.0f}s")
        if t >= next_struct - 0.5 * dt or last:
            do_structure()
            while next_struct <= t + 0.5 * dt:
                next_struct += der.structure_every_abs
        if t >= next_snap - 0.5 * dt or last:
            do_snapshot(k_snap)
            k_snap += 1
            while next_snap <= t + 0.5 * dt:
                next_snap += der.snap_every_abs
    diag_log.close()
    struct_log.close()
    wall = time.perf_counter() - wall0
    _say(cfg, f"done: {step} steps, {force.n_evals} force evaluations, {wall:.1f} s "
              f"({force.total_wall:.1f} s in forces)")
    return {'final': diag_rows[-1], 'diag_rows': diag_rows, 'structure_rows': struct_rows,
            'wall_time': wall, 'force_wall': force.total_wall, 'n_steps': step,
            'n_force_evals': force.n_evals, 'dt_min_used': float(min(dts)),
            'dt_max_used': float(max(dts)), 'n_snapshots': k_snap}


# ---------------------------------------------------------------------------
# pre-registered criteria
# ---------------------------------------------------------------------------

def _verdict_below(value: float, thr: float, band: float, short_msg=None) -> str:
    """'met' if value < thr - band, 'not met' if value > thr + band, else ambiguous."""
    if short_msg:
        return f"ambiguous ({short_msg})"
    if not np.isfinite(value):
        return "ambiguous (not measurable)"
    if value < thr - band:
        return "met"
    if value > thr + band:
        return "not met"
    return "ambiguous (within noise)"


def lss_criteria(structure_rows: list, diag_rows: list, der: Derived) -> dict:
    tr = der.t_ref
    tol = 0.51 * der.structure_every_abs
    rows = structure_rows
    t_end = _last(rows, 't') / tr if rows else float('nan')
    out = {}
    # --- cells form: segregation < -0.3 by 3 t_J,neg --------------------------
    r3 = _at_time(rows, LSS_SEG_TIME * tr, tol)
    seg3 = _f(r3.get('segregation')) if r3 else float('nan')
    ts, seg = _col(rows, 't') / tr, _col(rows, 'segregation')
    below = ts[np.isfinite(seg) & (seg < LSS_SEG_THRESHOLD)]
    t_first = float(below.min()) if below.size else float('nan')
    short = None if (np.isfinite(t_end) and t_end >= LSS_SEG_TIME * 0.99) else \
        f"run shorter than {LSS_SEG_TIME:g} {der.t_ref_name}"
    out['cells_form'] = {'threshold': LSS_SEG_THRESHOLD, 'by_t_ref': LSS_SEG_TIME,
                         'segregation_at_3': seg3, 't_first_below_threshold': t_first,
                         'segregation_final': _last(rows, 'segregation'),
                         'segregation_min': float(np.nanmin(seg)) if np.isfinite(seg).any() else float('nan'),
                         'verdict': _verdict_below(seg3, LSS_SEG_THRESHOLD, NOISE_BAND, short)}
    # --- stable clumps: n_clumps,neg change < 20 % between 5 and 20 t_J ---------
    r5 = _at_time(rows, LSS_STABLE_T0 * tr, tol)
    t1 = LSS_STABLE_T1 if (np.isfinite(t_end) and t_end >= LSS_STABLE_T1 * 0.99) else t_end
    r20 = _at_time(rows, t1 * tr, tol) if np.isfinite(t1) else None
    n5 = _f(r5.get('n_clumps_neg')) if r5 else float('nan')
    n20 = _f(r20.get('n_clumps_neg')) if r20 else float('nan')
    change = (n20 - n5) / n5 if (np.isfinite(n5) and n5 > 0 and np.isfinite(n20)) else float('nan')
    pers = _last(rows, 'persistence')
    ns = _col(rows, 'n_clumps_neg')
    late = (ts >= LSS_STABLE_T0) & np.isfinite(ns)
    if r5 is None:
        verdict = f"ambiguous (run shorter than {LSS_STABLE_T0:g} {der.t_ref_name})"
    elif not np.isfinite(change):
        verdict = "ambiguous (not measurable)"
    elif abs(change) < LSS_STABLE_CHANGE and pers > LSS_PERSISTENCE:
        verdict = "met" if t1 >= LSS_STABLE_T1 * 0.99 else f"met up to {t1:.1f} {der.t_ref_name} only (run shorter than 20)"
    elif abs(change) > LSS_STABLE_CHANGE + 0.1 or pers < LSS_PERSISTENCE - 0.1:
        verdict = "not met"
    else:
        verdict = "ambiguous (within noise)"
    out['stable_clumps'] = {'change_threshold': LSS_STABLE_CHANGE, 'window_t_ref': [LSS_STABLE_T0, t1],
                            'n_clumps_at_5': n5, 'n_clumps_at_end_window': n20,
                            'n_clumps_change_5_20': change,
                            'n_clumps_trend_per_t_ref': (float(np.polyfit(ts[late], ns[late], 1)[0])
                                                         if late.sum() >= 3 else float('nan')),
                            'n_clumps_max': float(np.nanmax(ns)) if np.isfinite(ns).any() else float('nan'),
                            'verdict': verdict}
    out['persistence'] = {'threshold': LSS_PERSISTENCE, 'final': pers,
                          'reference_t_ref': _last(rows, 'persistence_ref_t') / tr,
                          'min_after_reference': (float(np.nanmin(_col(rows, 'persistence')))
                                                  if np.isfinite(_col(rows, 'persistence')).any() else float('nan')),
                          'verdict': ("met" if pers > LSS_PERSISTENCE + NOISE_BAND else
                                      "not met" if pers < LSS_PERSISTENCE - NOISE_BAND else
                                      "ambiguous (within noise)") if np.isfinite(pers)
                          else "ambiguous (not measurable)"}
    out['note'] = ("Thresholds fixed before any run; 'reproducible' (seed-to-seed std of n_clumps < 20 % "
                   "of the mean) is evaluated by the ensemble command.  Positive-only / negative-only "
                   "controls have no segregation index by construction.")
    out['metrics'] = {
        'n_clumps_neg_final': _last(rows, 'n_clumps_neg'), 'n_clumps_pos_final': _last(rows, 'n_clumps_pos'),
        'n_clumps_at_5': n5, 'n_clumps_change_5_20': change, 'segregation_final': _last(rows, 'segregation'),
        'segregation_at_3': seg3, 'persistence_final': pers, 'phi_over_d': _last(rows, 'phi_over_d'),
        'd_cell': _last(rows, 'd_cell'), 'mass_mean': _last(rows, 'mass_mean'), 'mass_std': _last(rows, 'mass_std'),
        'diam_mean': _last(rows, 'diam_mean'), 'diam_std': _last(rows, 'diam_std'),
        'mass_frac_clumps_neg': _last(rows, 'mass_frac_clumps_neg'),
        'lambda_peak_pos': _last(rows, 'lambda_peak_pos'), 'lambda_peak_neg': _last(rows, 'lambda_peak_neg'),
        'void_frac_pos': _last(rows, 'void_frac_pos'), 'void_frac_neg': _last(rows, 'void_frac_neg'),
        'dE_rel_max': _max_abs(diag_rows, 'dE_rel'), 'dP_abs_rel_max': _max_abs(diag_rows, 'dP_abs_rel'),
        'dP_signed_rel_final': _last(diag_rows, 'dP_signed_rel'), 'virial_final': _last(diag_rows, 'virial'),
        'sigma_pos_final': _last(diag_rows, 'sigma_pos'), 'sigma_neg_final': _last(diag_rows, 'sigma_neg'),
    }
    return out


def galaxy_criteria(structure_rows: list, diag_rows: list, der: Derived, rc_info: dict) -> dict:
    rows = structure_rows
    T = der.t_ref
    ts = _col(rows, 't') / T
    sp, bar = _col(rows, 'spiral_A2'), _col(rows, 'bar_A2')
    n_rot = int(math.floor(float(np.nanmax(ts)) + 1e-6)) if np.isfinite(ts).any() else 0
    means = []
    for k in range(n_rot):
        m = (ts >= k) & (ts < k + 1)
        means.append(_nanmean(sp[m]) if m.any() else float('nan'))
    means = np.array(means, dtype=np.float64)
    n_eval = int(np.isfinite(means).sum())
    n_spiral = int(np.sum(means > GAL_SPIRAL_A2))
    if n_eval < 10:
        v = (f"ambiguous ({n_eval} rotations evaluated, criterion needs 10; "
             f"{n_spiral}/{n_eval} rotations above A2 = {GAL_SPIRAL_A2:g})")
    elif n_spiral >= GAL_SPIRAL_ROT:
        v = "met"
    elif n_spiral >= GAL_SPIRAL_ROT - 2:
        v = "ambiguous (within noise)"
    else:
        v = "not met"
    late = ts >= max(float(np.nanmax(ts)) - 5.0, 0.5 * float(np.nanmax(ts))) if n_rot else np.zeros(ts.shape, bool)
    out = {'sustained_spiral': {'threshold_A2': GAL_SPIRAL_A2, 'rotations_required': GAL_SPIRAL_ROT,
                                'rotations_evaluated': n_eval, 'rotations_with_spiral': n_spiral,
                                'spiral_A2_per_rotation': means.tolist(), 'spiral_A2_initial': _first(rows, 'spiral_A2'),
                                'spiral_A2_mean_late': _nanmean(sp[late]), 'spiral_A2_final': _last(rows, 'spiral_A2'),
                                'verdict': v}}
    bar_late = _nanmean(bar[late])
    out['bar'] = {'threshold_A2': GAL_BAR_A2, 'bar_A2_initial': _first(rows, 'bar_A2'),
                  'bar_A2_mean_late': bar_late, 'bar_A2_final': _last(rows, 'bar_A2'),
                  'verdict': ("met" if bar_late > GAL_BAR_A2 + 0.02 else "not met" if bar_late < GAL_BAR_A2 - 0.02
                              else "ambiguous (within noise)") if np.isfinite(bar_late) else "ambiguous (not measurable)",
                  'note': "a surviving bar is not the same as surviving spiral arms (inner vs outer A2)"}
    Ld0, Ld1 = _first(rows, 'Lz_disk'), _last(rows, 'Lz_disk')
    Lt0, Lt1 = _first(rows, 'Lz_twin_abs'), _last(rows, 'Lz_twin_abs')
    has_twin = int(rc_info.get('n_twin', 0)) > 0
    dLd = (Ld1 - Ld0) / abs(Ld0) if (np.isfinite(Ld0) and Ld0 != 0) else float('nan')
    dLt = (Lt1 - Lt0) / abs(Ld0) if (np.isfinite(Ld0) and Ld0 != 0 and np.isfinite(Lt0)) else float('nan')
    if not np.isfinite(dLd):
        v = "ambiguous (not measurable)"
    elif dLd < -GAL_BRAKING:
        if has_twin and np.isfinite(dLt) and dLt > 0 and dLt >= 0.5 * abs(dLd):
            v = "met"
        elif has_twin:
            v = "ambiguous (disk L_z fell but the twin ring did not gain a matching amount: escape or numerics)"
        else:
            v = "not applicable (no live twin ring to receive angular momentum; disk L_z change is numerical/escape)"
    elif dLd < -GAL_BRAKING + 0.02:
        v = "ambiguous (within noise)"
    else:
        v = "not met"
    out['braking'] = {'threshold': GAL_BRAKING, 'Lz_disk_initial': Ld0, 'Lz_disk_final': Ld1, 'Lz_disk_change': dLd,
                      'Lz_twin_abs_initial': Lt0, 'Lz_twin_abs_final': Lt1,
                      'Lz_twin_change_in_units_of_Lz_disk0': dLt,
                      'L_abs_drift_rel': ((_last(diag_rows, 'L_abs') - _first(diag_rows, 'L_abs')) / abs(_first(diag_rows, 'L_abs'))
                                          if _first(diag_rows, 'L_abs') not in (0.0,) and np.isfinite(_first(diag_rows, 'L_abs')) else float('nan')),
                      'vrot_disk_initial': _first(rows, 'vrot_disk'), 'vrot_disk_final': _last(rows, 'vrot_disk'),
                      'verdict': v}
    nct = _col(rows, 'n_clumps_twin')
    last2 = ts >= float(np.nanmax(ts)) - 2.0 if n_rot else np.zeros(ts.shape, bool)
    med = float(np.nanmedian(nct[last2])) if np.isfinite(nct[last2]).any() else float('nan')
    out['ring_fragmentation'] = {
        'n_clumps_twin_median_last_2_rotations': med, 'n_clumps_twin_final': _last(rows, 'n_clumps_twin'),
        'n_clumps_twin_max': float(np.nanmax(nct)) if np.isfinite(nct).any() else float('nan'),
        'twin_clump_mass_frac_final': _last(rows, 'twin_clump_mass_frac'),
        'verdict': ("not applicable (no live twin ring)" if not has_twin else
                    "present" if med >= 2 else "absent" if med == 0 else
                    "ambiguous (single clump / intermittent)" if np.isfinite(med) else "ambiguous (not measurable)"),
        'note': ("'tiling artefact confirmed' requires fragmentation present with bc=periodic and absent with "
                 "bc=isolated at the same N, eps, dt (convergence / sweep bc); a single run cannot decide it. "
                 f"Ring Jeans length {der.lambda_j_neg:.3g} vs width {(rc_info.get('twin_rout', 7.0) - rc_info.get('twin_rin', 4.5)):.3g}.")}
    fu = _f(rc_info.get('frac_undefined'))
    out['vc_undefined_fraction'] = {
        'fraction': fu, 'n_undefined': rc_info.get('n_undefined'), 'vc_at_2rd': _f(rc_info.get('vc_at_2rd')),
        'vc_ref_analytic': der.vc_ref,
        'verdict': ("rotating equilibrium defined everywhere" if fu == 0 else
                    "minor: < 5 % of disk without defined v_c" if fu < 0.05 else
                    "warning: confining ring repulsion exceeds disk gravity over part of the disk; "
                    "the set-up is not a rotating equilibrium there") if np.isfinite(fu) else "not measured"}
    out['metrics'] = {
        'spiral_A2_mean_late': _nanmean(sp[late]), 'spiral_A2_final': _last(rows, 'spiral_A2'),
        'bar_A2_final': _last(rows, 'bar_A2'), 'bar_A2_mean_late': bar_late,
        'rotations_with_spiral': n_spiral, 'rotations_evaluated': n_eval,
        'Lz_disk_change': dLd, 'Lz_twin_change': dLt, 'n_clumps_twin_final': _last(rows, 'n_clumps_twin'),
        'twin_clump_mass_frac_final': _last(rows, 'twin_clump_mass_frac'),
        'pitch_deg_mean': _nanmean(_col(rows, 'pitch_deg')[late]), 'omega_p_mean_late': _nanmean(_col(rows, 'omega_p')[late]),
        'Q_at_2rd_initial': _first(rows, 'Q_at_2rd'), 'Q_at_2rd_final': _last(rows, 'Q_at_2rd'),
        'escaped_frac_final': _last(rows, 'escaped_frac'), 'vrot_disk_change': (
            (_last(rows, 'vrot_disk') - _first(rows, 'vrot_disk')) / abs(_first(rows, 'vrot_disk'))
            if np.isfinite(_first(rows, 'vrot_disk')) and _first(rows, 'vrot_disk') != 0 else float('nan')),
        'vc_undefined_fraction': fu, 'dE_rel_max': _max_abs(diag_rows, 'dE_rel'),
        'dP_abs_rel_max': _max_abs(diag_rows, 'dP_abs_rel'), 'dP_signed_rel_final': _last(diag_rows, 'dP_signed_rel'),
    }
    return out


# ---------------------------------------------------------------------------
# scenario drivers
# ---------------------------------------------------------------------------

def _summary(cfg, der, outdir, result, crit, extra=None) -> dict:
    metrics = dict(crit.pop('metrics', {}))
    metrics['wall_time'] = result['wall_time']
    s = {'scenario': cfg.scenario, 'label': cfg.label, 'seed': cfg.seed, 'outdir': outdir,
         'method': der.method, 'device': cfg.device, 'dim': der.dim, 'n_steps': result['n_steps'],
         'n_force_evals': result['n_force_evals'], 'wall_time': result['wall_time'],
         'force_wall': result['force_wall'], 'dt_used': [result['dt_min_used'], result['dt_max_used']],
         'criteria': crit, 'metrics': metrics, 'final_diagnostics': result['final'],
         'config': cfg_to_dict(cfg), 'derived': derived_to_dict(der)}
    if result.get('gpu_memory') is not None:
        s['gpu_memory'] = result.get('gpu_memory')
    if extra:
        s.update(extra)
    s.update(metrics)                       # flat access for sweeps / ensembles
    write_json(os.path.join(outdir, 'summary.json'), s)
    return s


def run_lss(cfg: RunConfig) -> dict:
    cfg = resolve_defaults(cfg)
    der = derive(cfg)
    outdir = _outdir(cfg)
    write_run_metadata(cfg, der, outdir)
    rng = np.random.default_rng(cfg.seed)
    force, pm, ext = build_forces(cfg, der)
    ps = make_lss_ic(cfg, der, rng)
    t_persist = der.t_ref * (LSS_STABLE_T0 if cfg.tmax >= LSS_STABLE_T0 else 0.5 * cfg.tmax)

    def structure_fn(ps, t, state):
        state = state or {'ref_grid': None, 'ref_t': float('nan')}
        row, cur = lss_structure_row(ps, t, der.box, cfg.structure_grid, der.dim, cfg.clump_threshold,
                                     cfg.clump_min_cells, state['ref_grid'])
        if state['ref_grid'] is None and t >= t_persist - 1e-6 * der.t_ref:
            state['ref_grid'], state['ref_t'] = cur, t
            row['persistence'] = 1.0
        row['persistence_ref_t'] = state['ref_t']
        return row, state

    result = run_loop(cfg, der, ps, force, outdir, structure_fn,
                      {'title': f"LSS {cfg.mass_ratio:g}/{cfg.vel_ratio:g} {cfg.ic}", 'relative': False})
    crit = lss_criteria(result['structure_rows'], result['diag_rows'], der)
    _plot(cfg, plot_conservation, os.path.join(outdir, 'diagnostics.csv'),
          os.path.join(outdir, 'conservation.png'), cfg.expansion)
    _plot(cfg, plot_lss_structure, os.path.join(outdir, 'structure.csv'),
          os.path.join(outdir, 'structure.png'), der.t_ref_name, t_ref=der.t_ref)
    s = _summary(cfg, der, outdir, result, crit, {'n_pos': ps.npos, 'n_neg': ps.nneg,
                                                   'external_field': ext.describe()})
    _say(cfg, "LSS verdicts: " + ", ".join(f"{k} = {v['verdict']}" for k, v in crit.items()
                                          if isinstance(v, dict) and 'verdict' in v))
    return s


def run_galaxy(cfg: RunConfig) -> dict:
    cfg = resolve_defaults(cfg)
    der = derive(cfg)
    outdir = _outdir(cfg)
    write_run_metadata(cfg, der, outdir)
    rng = np.random.default_rng(cfg.seed)
    force, pm, ext = build_forces(cfg, der)
    ps, rc = make_galaxy_ic(cfg, der, rng, lambda _ps: force)
    rc['twin_rin'], rc['twin_rout'] = cfg.twin_rin * cfg.rd, cfg.twin_rout * cfg.rd
    write_json(os.path.join(outdir, 'rotation_curve.json'), rc)
    _plot(cfg, plot_rotation_curve, rc, os.path.join(outdir, 'rotation_curve.png'))
    box_wrap = der.box if cfg.bc == 'periodic' else None
    ring_grid = int(min(max(cfg.structure_grid, 64), 128 if der.dim == 2 else 64))
    centre_fn = (lambda p: centre_of_mass(p, p.species == 0, box_wrap))

    def structure_fn(ps, t, state):
        return disk_structure_row(ps, t, centre_fn(ps), cfg.rd, cfg.rmax, der.box, der.G, state,
                                  ring_grid, cfg.clump_threshold, cfg.clump_min_cells)

    result = run_loop(cfg, der, ps, force, outdir, structure_fn,
                      {'centre_fn': centre_fn, 'r_escape': 1.5 * cfg.rmax, 'relative': True,
                       'extent': 0.5 * der.box, 'title': f"disk + {cfg.control} twin"})
    crit = galaxy_criteria(result['structure_rows'], result['diag_rows'], der, rc)
    _plot(cfg, plot_conservation, os.path.join(outdir, 'diagnostics.csv'),
          os.path.join(outdir, 'conservation.png'), False)
    _plot(cfg, plot_disk_modes, os.path.join(outdir, 'structure.csv'), os.path.join(outdir, 'modes.png'),
          t_ref=der.t_ref, t_ref_name=der.t_ref_name)
    ic_info = {k: rc.get(k) for k in ('vc_at_2rd', 'vc_max', 't_rot_at_2rd', 'sigma_R_at_2rd',
                                      'frac_undefined', 'n_undefined', 'n_sigma_clipped', 'n_disk', 'n_twin')}
    s = _summary(cfg, der, outdir, result, crit, {'initial_condition': ic_info, 'external_field': ext.describe(),
                                                   'control': cfg.control})
    _say(cfg, "galaxy verdicts: " + ", ".join(f"{k} = {v['verdict']}" for k, v in crit.items()
                                             if isinstance(v, dict) and 'verdict' in v))
    return s


def run_simulation(cfg: RunConfig) -> dict:
    if cfg.scenario == 'lss':
        return run_lss(cfg)
    if cfg.scenario == 'galaxy':
        return run_galaxy(cfg)
    raise ValueError(f"unknown scenario {cfg.scenario!r}")


if __name__ == '__main__':  # minimal smoke test (tiny runs)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        s1 = run_simulation(RunConfig('lss', npos=300, nneg=300, method='pm', grid=32, tmax=1.0,
                                      snap_every=0.5, structure_grid=32, quiet=True,
                                      outdir=os.path.join(td, 'lss')))
        assert os.path.exists(os.path.join(td, 'lss', 'summary.json')) and 'cells_form' in s1['criteria']
        assert s1['device'] == 'cpu' and 'gpu_memory' not in s1
        s2 = run_simulation(RunConfig('galaxy', ndisk=400, ntwin=300, method='pm', grid=32, tmax=0.3,
                                      snap_every=0.15, structure_every=0.1, structure_grid=32, quiet=True,
                                      outdir=os.path.join(td, 'gal')))
        assert 'sustained_spiral' in s2['criteria'] and os.path.exists(os.path.join(td, 'gal', 'modes.png'))
        s3 = run_simulation(RunConfig('galaxy', ndisk=200, control='rigid-repulsive', method='direct', grid=32,
                                      tmax=0.1, structure_every=0.05, snap_every=0.1, quiet=True,
                                      outdir=os.path.join(td, 'gal2')))
        assert s3['criteria']['ring_fragmentation']['verdict'].startswith('not applicable')
    print("scenarios smoke test OK")
