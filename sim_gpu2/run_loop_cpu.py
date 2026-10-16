"""run_loop_cpu.py -- the CPU (numpy) time loop of scenarios.run_loop, moved out
of scenarios.py unchanged in its numerics, plus a timing breakdown and an
optional final-state dump.

run_loop_cpu(cfg, der, ps, force, outdir, structure_fn, snapshot_kwargs) has
the signature, the cadence logic, the CSV re-logging of the first row (drifts
defined as 0), the log messages and the returned dict keys of the previous
scenarios.run_loop:

  * leapfrog KDK (integrator.leapfrog_step) with the kinematic expansion and
    the optional adaptive global step (integrator.choose_dt),
  * diagnostics rows (diagnostics_core.diagnostics_row) every cfg.log_every
    steps, structure rows (structure_fn) every der.structure_every_abs,
    snapshots (plotting.snapshot) every der.snap_every_abs, all at the last
    step too.

Additions (no change to any number written to diagnostics.csv / structure.csv):

  * a timing.Timers instance: the whole leapfrog step (which contains the
    force evaluation) is timed as 'integration', diagnostics_row + CSV as
    'diagnostics', structure_fn + CSV as 'structure', the snapshot as
    'snapshots'; the force time is the evaluator's total_wall accumulated
    since the loop start, so the breakdown reports forces separately and
    'integration' as step time minus forces.  The result dict gains
    'timing' = {'breakdown': rows, 'text': aligned table, 'sections': raw};
  * when cfg.dump_final is set, final_state.npz (pos, vel, w, s, species) is
    written to outdir at the end of the run.

The GPU loop (run_loop_gpu) mirrors this file with device-resident particles.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from config import RunConfig, Derived
from particles import ParticleSet
from integrator import ForceEvaluator, Expansion, choose_dt, leapfrog_step
from diagnostics_core import diagnostics_row, CsvLogger
from plotting import snapshot
from timing import Timers, format_breakdown

_AX = 'xyz'
TIMING_TITLE = 'timing breakdown'


def _say(cfg: RunConfig, msg: str) -> None:
    if not cfg.quiet:
        print("[run] " + msg, file=sys.stderr, flush=True)


def dump_final_state(ps: ParticleSet, path: str) -> str:
    """Write pos, vel, w, s, species of a host particle set to an .npz file."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez(path, pos=np.asarray(ps.pos), vel=np.asarray(ps.vel), w=np.asarray(ps.w),
             s=np.asarray(ps.s), species=np.asarray(ps.species))
    return path


# ---------------------------------------------------------------------------
# the time loop
# ---------------------------------------------------------------------------

def run_loop_cpu(cfg: RunConfig, der: Derived, ps: ParticleSet, force: ForceEvaluator, outdir: str,
                 structure_fn, snapshot_kwargs: dict) -> dict:
    """Leapfrog KDK loop with logging at cfg.log_every steps, snapshots every
    der.snap_every_abs, structure rows every der.structure_every_abs.
    structure_fn(ps, t, state) -> (row, state).  snapshot_kwargs keys: centre_fn
    (ps -> centre), r_escape, relative (bool), extent, title.
    Returns {'final', 'diag_rows', 'structure_rows', 'wall_time', 'force_wall',
    'n_steps', 'n_force_evals', 'dt_min_used', 'dt_max_used', 'n_snapshots',
    'timing'} (+ 'final_state' path when cfg.dump_final)."""
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
    timers = Timers()
    force_wall0 = float(force.total_wall)                  # forces accumulated before the loop
    with timers.section('integration'):
        acc, phi = force(ps)
    t, step = 0.0, 0
    dt = choose_dt(acc, force.eps_eff, cfg.eta, dt_max, dt_min) if cfg.adaptive else dt_max
    dts = [dt]
    state = [None, 0.0, 0]          # structure state, E0, P0 placeholders
    E0, P0 = None, None

    def centre():
        return centre_fn(ps) if centre_fn is not None else None

    def log_diag(fix_zero=False):
        with timers.section('diagnostics'):
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
        with timers.section('structure'):
            row, state[0] = structure_fn(ps, t, state[0])
            row['t_ref_units'] = t / der.t_ref
            struct_log.log(row)
            struct_rows.append(row)

    def do_snapshot(k):
        with timers.section('snapshots'):
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
        with timers.section('integration'):
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
    dumped = None
    if cfg.dump_final:
        dumped = dump_final_state(ps, os.path.join(outdir, 'final_state.npz'))
        _say(cfg, f"final particle state written to {dumped}")
    wall = time.perf_counter() - wall0
    forces_loop = float(force.total_wall) - force_wall0
    rows = timers.breakdown(total=wall, forces=forces_loop)
    text = format_breakdown(rows, TIMING_TITLE)
    _say(cfg, f"done: {step} steps, {force.n_evals} force evaluations, {wall:.1f} s "
              f"({force.total_wall:.1f} s in forces)")
    result = {'final': diag_rows[-1], 'diag_rows': diag_rows, 'structure_rows': struct_rows,
              'wall_time': wall, 'force_wall': force.total_wall, 'n_steps': step,
              'n_force_evals': force.n_evals, 'dt_min_used': float(min(dts)),
              'dt_max_used': float(max(dts)), 'n_snapshots': k_snap,
              'timing': {'breakdown': rows, 'text': text, 'sections': timers.to_dict(),
                         'forces_in_loop': forces_loop}}
    if dumped is not None:
        result['final_state'] = dumped
    return result


# ---------------------------------------------------------------------------
# smoke test: the tiny LSS / galaxy loops of the old scenarios smoke test
# ---------------------------------------------------------------------------

def _check_timing(res: dict) -> None:
    tm = res['timing']
    assert set(tm) >= {'breakdown', 'text', 'sections'}
    keys = [r['key'] for r in tm['breakdown']]
    assert keys == ['forces', 'integration', 'diagnostics', 'structure', 'snapshots', 'other', 'total'], keys
    by = {r['key']: r for r in tm['breakdown']}
    wall = res['wall_time']
    parts = sum(by[k]['seconds'] for k in ('forces', 'integration', 'diagnostics', 'structure', 'snapshots'))
    assert parts <= wall * (1.0 + 1e-9) + 1e-6, (parts, wall)
    assert all(by[k]['seconds'] >= 0.0 for k in by) and by['total']['seconds'] == wall
    assert abs(sum(r['seconds'] for r in tm['breakdown'][:-1]) - wall) < 1e-9
    assert by['forces']['seconds'] > 0.0 and by['forces']['seconds'] <= res['force_wall'] + 1e-12
    assert by['snapshots']['seconds'] > 0.0 and by['diagnostics']['seconds'] > 0.0
    assert tm['text'].splitlines()[0] == TIMING_TITLE and tm['text'].splitlines()[-1].startswith('total')
    assert tm['sections']['counts']['snapshots'] == res['n_snapshots']


if __name__ == '__main__':
    import tempfile
    from config import resolve_defaults, derive
    from forces_pm import PMSolver
    from external_potentials import make_external_field
    from initial_conditions import make_lss_ic, make_galaxy_ic
    from structure_stats import lss_structure_row, disk_structure_row
    from particles import centre_of_mass

    def build(cfg, der):
        pm = None
        if der.method == 'pm':
            pm = PMSolver(der.dim, der.grid, der.box, der.G, cfg.force_law, bc=cfg.bc,
                          eps=max(cfg.eps, 0.0), dtype=np.dtype(cfg.dtype))
        return ForceEvaluator(cfg, der, pm, make_external_field(cfg, der, pm))

    with tempfile.TemporaryDirectory() as td:
        # ---- LSS loop with the final dump ---------------------------------------------
        cfg = resolve_defaults(RunConfig('lss', npos=300, nneg=300, method='pm', grid=32, tmax=1.0,
                                         snap_every=0.5, structure_grid=32, quiet=True, dump_final=True))
        der = derive(cfg)
        force = build(cfg, der)
        ps = make_lss_ic(cfg, der, np.random.default_rng(cfg.seed))

        def structure_fn(p_, t, state):
            state = state or {'ref_grid': None}
            row, cur = lss_structure_row(p_, t, der.box, cfg.structure_grid, der.dim,
                                         cfg.clump_threshold, cfg.clump_min_cells, state['ref_grid'])
            if state['ref_grid'] is None:
                state['ref_grid'] = cur
                row['persistence'] = 1.0
            return row, state

        out = os.path.join(td, 'lss')
        os.makedirs(out)
        res = run_loop_cpu(cfg, der, ps, force, out, structure_fn, {'title': 'smoke', 'relative': False})
        keys = {'final', 'diag_rows', 'structure_rows', 'wall_time', 'force_wall', 'n_steps',
                'n_force_evals', 'dt_min_used', 'dt_max_used', 'n_snapshots', 'timing', 'final_state'}
        assert set(res) == keys, set(res)
        assert abs(res['n_steps'] - der.n_steps) <= 1 and res['n_force_evals'] == res['n_steps'] + 1
        assert res['diag_rows'][0]['dE_rel'] == 0.0 and res['diag_rows'][0]['dP_abs_rel'] == 0.0
        assert res['final']['step'] == res['n_steps'] and np.isfinite(res['final']['E'])
        assert abs(res['final']['dP_abs_rel']) < 1e-8 and res['n_snapshots'] == 3
        assert len(res['structure_rows']) == 3 and 'n_clumps_neg' in res['structure_rows'][0]
        assert res['structure_rows'][2]['persistence'] <= 1.0 + 1e-12
        for f in ('diagnostics.csv', 'structure.csv', 'snap_0000.png', 'snap_0002.png'):
            assert os.path.getsize(os.path.join(out, f)) > 100, f
        _check_timing(res)
        with np.load(res['final_state']) as fs:
            assert set(fs.files) == {'pos', 'vel', 'w', 's', 'species'}
            assert np.array_equal(fs['pos'], ps.pos) and np.array_equal(fs['vel'], ps.vel)
            assert np.array_equal(fs['species'], ps.species) and fs['s'].shape == (ps.n,)
        assert res['final_state'] == os.path.join(out, 'final_state.npz')
        # ---- galaxy loop (relative view, centre function, r_escape), no dump ------------
        cfg = resolve_defaults(RunConfig('galaxy', ndisk=400, ntwin=300, method='pm', grid=32, tmax=0.3,
                                         snap_every=0.15, structure_every=0.1, structure_grid=32, quiet=True))
        der = derive(cfg)
        force = build(cfg, der)
        ps, rc = make_galaxy_ic(cfg, der, np.random.default_rng(cfg.seed), lambda _ps: force)
        n_before = force.n_evals
        box_wrap = der.box if cfg.bc == 'periodic' else None
        ring_grid = int(min(max(cfg.structure_grid, 64), 128 if der.dim == 2 else 64))
        centre_fn = (lambda p_: centre_of_mass(p_, p_.species == 0, box_wrap))

        def gal_structure_fn(p_, t, state):
            return disk_structure_row(p_, t, centre_fn(p_), cfg.rd, cfg.rmax, der.box, der.G, state,
                                      ring_grid, cfg.clump_threshold, cfg.clump_min_cells)

        out = os.path.join(td, 'gal')
        os.makedirs(out)
        res2 = run_loop_cpu(cfg, der, ps, force, out, gal_structure_fn,
                            {'centre_fn': centre_fn, 'r_escape': 1.5 * cfg.rmax, 'relative': True,
                             'extent': 0.5 * der.box, 'title': 'disk + live twin'})
        assert 'final_state' not in res2 and not os.path.exists(os.path.join(out, 'final_state.npz'))
        assert res2['n_force_evals'] == n_before + res2['n_steps'] + 1
        assert 'bar_A2' in res2['structure_rows'][0] and len(res2['structure_rows']) == 4
        assert np.isfinite(res2['final']['vrot_mean_pos']) and np.isfinite(res2['final']['escaped_frac_pos'])
        assert res2['n_snapshots'] == 3 and os.path.getsize(os.path.join(out, 'snap_0002.png')) > 1000
        _check_timing(res2)
        # forces measured before the loop (IC) are excluded from the breakdown's force row
        by = {r['key']: r for r in res2['timing']['breakdown']}
        assert abs(by['forces']['seconds'] - res2['timing']['forces_in_loop']) < 1e-12
        assert res2['timing']['forces_in_loop'] < res2['force_wall']
        print(res2['timing']['text'])
    print("run_loop_cpu smoke test OK")
