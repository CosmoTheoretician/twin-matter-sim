"""run_loop_gpu.py -- the time loop of scenarios.run_loop with the particles
resident on the GPU (PyTorch / ROCm, exposed as the 'cuda' device) and NO host
copies of the particle arrays during the run.

The initial condition (built on the CPU exactly as before) is uploaded once to
a GPUParticles; positions, velocities, accelerations and potentials then stay
on the device for the whole run.  Everything that used to need a numpy
ParticleSet is now evaluated on the device:

  * diagnostics rows  : diagnostics_torch.diagnostics_row_t (tensors in, one
                        small float64 vector out per row; same keys, same
                        numerical definitions as diagnostics_core),
  * structure rows    : structure_fn(gp, t, state) receives the GPUParticles
                        (scenarios passes structure_torch.lss_structure_row_t /
                        disk_structure_row_t; only a smoothed grid crosses the
                        bus for scipy's clump labelling),
  * centre of the view: centre_fn(gp) (diagnostics_torch.centre_of_mass_t, a
                        device tensor accepted by every consumer),
  * snapshots         : snapshot_torch.snapshot_t -- density maps built on the
                        device, scatter panels drawn from a fixed random
                        subsample (cfg.scatter_subsample per population, seeded
                        with cfg.seed, built once and stated in the panel).

The only GPUParticles.to_host() of a run is the optional final dump
(cfg.dump_final -> final_state.npz) after the last step.  Cadences, CSV
contents, log messages and the returned dict are those of the CPU loop
(run_loop_cpu), plus 'gpu_memory' and 'timing' (forces = the evaluator's
total_wall accumulated during the loop, integration = timed leapfrog step
minus forces, energy diagnostics, structure statistics, snapshots, other;
the timers synchronise the device around every section).  Integration: torch
leapfrog KDK (integrator_torch) with the kinematic expansion and the adaptive
global step of integrator.py.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from config import RunConfig, Derived
from particles import ParticleSet
from gpu_backend import GPUParticles, gpu_memory_report, torch_dtype, device_name, torch
from integrator import Expansion
from integrator_torch import ForceEvaluatorTorch, choose_dt_t, leapfrog_step_t
from diagnostics_core import CsvLogger
from diagnostics_torch import diagnostics_row_t
from snapshot_torch import snapshot_t, ScatterSubsample
from timing import Timers, format_breakdown
from run_loop_cpu import dump_final_state, TIMING_TITLE

_AX = 'xyz'


def _say(cfg: RunConfig, msg: str) -> None:
    if not cfg.quiet:
        print("[gpu] " + msg, file=sys.stderr, flush=True)


def run_loop_gpu(cfg: RunConfig, der: Derived, ps: ParticleSet, force: ForceEvaluatorTorch,
                 outdir: str, structure_fn, snapshot_kwargs: dict) -> dict:
    """Device-resident leapfrog KDK loop; same inputs, outputs and cadences as
    run_loop_cpu.  structure_fn(gp, t, state) -> (row, state) and the optional
    snapshot_kwargs['centre_fn'](gp) receive the GPUParticles.  snapshot_kwargs
    keys: centre_fn, r_escape, relative (bool), extent, title.  Returns the
    run_loop dict plus 'gpu_memory', 'timing' and 'scatter_subsample'
    (+ 'final_state' path when cfg.dump_final)."""
    box_wrap = der.box if cfg.bc == 'periodic' else None
    p = der.force_exponent_p
    expansion = Expansion(cfg.expansion, cfg.a_index, cfg.t0_exp * der.t_ref)
    centre_fn = snapshot_kwargs.get('centre_fn')
    r_escape = snapshot_kwargs.get('r_escape')
    relative = bool(snapshot_kwargs.get('relative', False))
    extent, title = snapshot_kwargs.get('extent'), snapshot_kwargs.get('title', '')
    self_energy = force.self_energy_fn(ps.w)                 # host weights, before the upload
    dtype = torch_dtype(cfg.dtype)
    cuda = force.device.type == 'cuda'
    sync = torch.cuda.synchronize if cuda else None
    timers = Timers(sync=sync)
    wall0 = timers.wall0
    gp = GPUParticles.from_host(ps, force.device, dtype)    # the only upload of the run
    with timers.section('snapshots'):
        sub = ScatterSubsample(gp, cfg.scatter_subsample, cfg.seed)
    diag_log = CsvLogger(os.path.join(outdir, 'diagnostics.csv'))
    struct_log = CsvLogger(os.path.join(outdir, 'structure.csv'))
    diag_rows, struct_rows = [], []
    tmax, dt_max = der.tmax_abs, der.dt_abs
    dt_min = 1e-3 * dt_max
    force_wall0 = float(force.total_wall)                   # forces accumulated before the loop
    with timers.section('integration'):
        acc, phi = force(gp)                                # tensors on the device
    _say(cfg, f"device = {device_name()} ({force.device}), dtype = {cfg.dtype}, N = {gp.n}, "
              f"PM {der.grid}^{der.dim}, first force evaluation {1e3 * force.last_wall:.1f} ms; "
              f"scatter subsample {sub.n_pos} + / {sub.n_neg} - (seed {cfg.seed}); "
              + gpu_memory_report()['text'])
    t, step = 0.0, 0
    dt = choose_dt_t(acc, force.eps_eff, cfg.eta, dt_max, dt_min) if cfg.adaptive else dt_max
    dts = [dt]
    state = [None]                                          # structure state
    E0, P0 = None, None

    def centre():
        return centre_fn(gp) if centre_fn is not None else None

    def log_diag(fix_zero=False):
        with timers.section('diagnostics'):
            c = centre()
            row = diagnostics_row_t(gp, phi, t, step, dt, expansion.a(t), self_energy, E0, P0, centre=c,
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
            row, state[0] = structure_fn(gp, t, state[0])
            row['t_ref_units'] = t / der.t_ref
            struct_log.log(row)
            struct_rows.append(row)

    def do_snapshot(k):
        with timers.section('snapshots'):
            c = centre() if relative else None
            snapshot_t(gp, der.box, t / der.t_ref, der.t_ref_name,
                       os.path.join(outdir, f'snap_{k:04d}.png'), der.dim, centre=c,
                       extent=extent if relative else None, title=title, subsample=sub)
            import numpy as _np
            _np.savez_compressed(os.path.join(outdir, f'pts_{k:04d}.npz'), t=t / der.t_ref, box=der.box,
                                 pos=sub.positions(gp, 'pos', centre=c, box=der.box).astype(_np.float32),
                                 neg=sub.positions(gp, 'neg', centre=c, box=der.box).astype(_np.float32))

    row0 = log_diag()
    E0 = row0['E']
    P0 = np.array([row0[f'p_abs_{_AX[k]}'] for k in range(der.dim)])
    diag_rows.pop()                                         # re-log with the drifts defined as 0
    diag_log.close()
    diag_log = CsvLogger(os.path.join(outdir, 'diagnostics.csv'))
    log_diag(fix_zero=True)
    do_structure()
    do_snapshot(0)
    next_snap, next_struct, k_snap = der.snap_every_abs, der.structure_every_abs, 1
    n_est = int(math.ceil(tmax / dt_max))
    _say(cfg, f"integrating to t = {cfg.tmax:g} {der.t_ref_name} ({n_est} steps of {dt_max:.4g}"
              f"{', adaptive cap' if cfg.adaptive else ''}), method = pm on {force.device}, N = {gp.n}, "
              f"{expansion!r}")
    report_every = max(cfg.log_every, n_est // 20)
    while t < tmax - 1e-9 * tmax:
        if cfg.adaptive:
            dt = choose_dt_t(acc, force.eps_eff, cfg.eta, dt_max, dt_min)
        dt = min(dt, tmax - t)
        dts.append(dt)
        with timers.section('integration'):
            acc, phi, t = leapfrog_step_t(gp, acc, phi, t, dt, force, expansion, p, box_wrap)
        step += 1
        last = t >= tmax - 1e-9 * tmax
        if step % cfg.log_every == 0 or last:
            row = log_diag()
            if step % report_every == 0 or last:
                _say(cfg, f"step {step} t = {t / der.t_ref:.3f} {der.t_ref_name} dE/E = {row['dE_rel']:.2e} "
                          f"dP_abs = {row['dP_abs_rel']:.1e} wall = {row['wall']:.0f}s "
                          f"({1e3 * force.last_wall:.1f} ms/force)")
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
    if cfg.dump_final:                                      # the only full host copy of the run
        if sync is not None:
            sync()
        dumped = dump_final_state(gp.to_host(), os.path.join(outdir, 'final_state.npz'))
        _say(cfg, f"final particle state written to {dumped}")
    wall = time.perf_counter() - wall0
    forces_loop = float(force.total_wall) - force_wall0
    rows = timers.breakdown(total=wall, forces=forces_loop)
    text = format_breakdown(rows, TIMING_TITLE)
    mem = gpu_memory_report()
    _say(cfg, f"done: {step} steps, {force.n_evals} force evaluations, {wall:.1f} s "
              f"({forces_loop:.1f} s in forces during the loop, {force.total_wall:.1f} s in total); "
              f"{mem['text']}")
    for ln in text.splitlines():
        _say(cfg, "  " + ln)
    result = {'final': diag_rows[-1], 'diag_rows': diag_rows, 'structure_rows': struct_rows,
              'wall_time': wall, 'force_wall': force.total_wall, 'n_steps': step,
              'n_force_evals': force.n_evals, 'dt_min_used': float(min(dts)),
              'dt_max_used': float(max(dts)), 'n_snapshots': k_snap, 'gpu_memory': mem,
              'timing': {'breakdown': rows, 'text': text, 'sections': timers.to_dict(),
                         'forces_in_loop': forces_loop},
              'scatter_subsample': {'pos': sub.n_pos, 'neg': sub.n_neg, 'seed': sub.seed,
                                    'requested': cfg.scatter_subsample}}
    if dumped is not None:
        result['final_state'] = dumped
    return result


# ---------------------------------------------------------------------------
# smoke test: tiny loops on the device without any host copy of the particles
# ---------------------------------------------------------------------------

def _check_timing(res: dict) -> None:
    tm = res['timing']
    keys = [r['key'] for r in tm['breakdown']]
    assert keys == ['forces', 'integration', 'diagnostics', 'structure', 'snapshots', 'other', 'total'], keys
    by = {r['key']: r for r in tm['breakdown']}
    wall = res['wall_time']
    parts = sum(by[k]['seconds'] for k in ('forces', 'integration', 'diagnostics', 'structure', 'snapshots'))
    assert parts <= wall * (1.0 + 1e-9) + 1e-6, (parts, wall)
    assert all(by[k]['seconds'] >= 0.0 for k in by) and by['total']['seconds'] == wall
    assert abs(sum(r['seconds'] for r in tm['breakdown'][:-1]) - wall) < 1e-9
    assert by['forces']['seconds'] > 0.0 and abs(by['forces']['seconds'] - tm['forces_in_loop']) < 1e-12
    assert by['snapshots']['seconds'] > 0.0 and by['diagnostics']['seconds'] > 0.0
    assert by['structure']['seconds'] > 0.0
    assert tm['text'].splitlines()[0] == TIMING_TITLE and tm['text'].splitlines()[-1].startswith('total')
    assert tm['sections']['counts']['snapshots'] == res['n_snapshots'] + 1    # + the subsample build


if __name__ == '__main__':
    import tempfile
    if torch is None:
        print("torch not installed, run_loop_gpu smoke test skipped")
        sys.exit(0)
    from config import resolve_defaults, derive
    from forces_pm import PMSolver
    from forces_pm_torch import PMSolverTorch
    from external_potentials import NoField
    from initial_conditions import make_lss_ic
    from structure_torch import lss_structure_row_t
    from diagnostics_torch import centre_of_mass_t
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cpu':
        print("no GPU: running the smoke test with the torch solver on the CPU device")

    def build(cfg):
        der = derive(cfg)
        pm_cpu = PMSolver(der.dim, der.grid, der.box, der.G, cfg.force_law, bc=cfg.bc,
                          eps=max(cfg.eps, 0.0), dtype=np.float64)
        pm_t = PMSolverTorch(pm_cpu, dev, cfg.dtype)
        return der, ForceEvaluatorTorch(cfg, der, pm_t, NoField())

    def lss_structure(cfg, der):
        def structure_fn(g, t, state):
            assert isinstance(g, GPUParticles)
            state = state or {'ref_grid': None}
            row, cur = lss_structure_row_t(g, t, der.box, cfg.structure_grid, der.dim,
                                           cfg.clump_threshold, cfg.clump_min_cells, state['ref_grid'])
            if state['ref_grid'] is None:
                state['ref_grid'] = cur
                row['persistence'] = 1.0
            return row, state
        return structure_fn

    counter = {'n': 0}
    _orig_to_host = GPUParticles.to_host

    def _counted_to_host(self, dtype=None):
        counter['n'] += 1
        return _orig_to_host(self, dtype)

    GPUParticles.to_host = _counted_to_host
    try:
        with tempfile.TemporaryDirectory() as td:
            # ---- 2-D LSS, float32, no dump: not a single host copy ---------------------
            cfg = resolve_defaults(RunConfig('lss', npos=300, nneg=300, method='pm', grid=32, tmax=0.5,
                                             snap_every=0.25, structure_every=0.25, log_every=5,
                                             structure_grid=32, scatter_subsample=100, quiet=True,
                                             device='gpu'))
            der, force = build(cfg)
            ps = make_lss_ic(cfg, der, np.random.default_rng(cfg.seed))
            assert ps.pos.dtype == np.float32
            counter['n'] = 0
            res = run_loop_gpu(cfg, der, ps, force, td, lss_structure(cfg, der),
                               {'title': 'smoke', 'relative': False})
            assert counter['n'] == 0, counter
            keys = {'final', 'diag_rows', 'structure_rows', 'wall_time', 'force_wall', 'n_steps',
                    'n_force_evals', 'dt_min_used', 'dt_max_used', 'n_snapshots', 'gpu_memory', 'timing',
                    'scatter_subsample'}
            assert keys <= set(res) and 'final_state' not in res, set(res)
            assert abs(res['n_steps'] - der.n_steps) <= 1 and res['n_force_evals'] == res['n_steps'] + 1
            assert res['diag_rows'][0]['dE_rel'] == 0.0 and res['diag_rows'][0]['dP_abs_rel'] == 0.0
            assert res['final']['step'] == res['n_steps'] and np.isfinite(res['final']['E'])
            assert isinstance(res['final']['n_pos'], int) and res['final']['n_pos'] == 300
            assert abs(res['final']['dP_abs_rel']) < 1e-4 and res['n_snapshots'] == 3
            assert len(res['structure_rows']) == 3 and 'n_clumps_neg' in res['structure_rows'][0]
            assert res['structure_rows'][0]['persistence'] == 1.0
            assert res['structure_rows'][2]['persistence'] <= 1.0 + 1e-12
            assert res['scatter_subsample'] == {'pos': 100, 'neg': 100, 'seed': cfg.seed, 'requested': 100}
            for f in ('diagnostics.csv', 'structure.csv', 'snap_0000.png', 'snap_0002.png'):
                assert os.path.getsize(os.path.join(td, f)) > 100, f
            assert not os.path.exists(os.path.join(td, 'final_state.npz'))
            _check_timing(res)
            # ---- expansion + adaptive step + float64 + relative view + final dump -------
            cfg2 = resolve_defaults(RunConfig('lss', npos=200, nneg=200, method='pm', grid=32, tmax=0.2,
                                              snap_every=0.1, structure_every=0.1, log_every=2,
                                              expansion=True, t0_exp=0.05, adaptive=True, dtype='float64',
                                              structure_grid=32, quiet=True, device='gpu', dump_final=True))
            der2, force2 = build(cfg2)
            ps2 = make_lss_ic(cfg2, der2, np.random.default_rng(2))
            assert ps2.pos.dtype == np.float64
            sub = os.path.join(td, 'exp')
            os.makedirs(sub)
            counter['n'] = 0
            res2 = run_loop_gpu(cfg2, der2, ps2, force2, sub, lambda g, t, s: ({'t': t, 'n': g.n}, s),
                                {'title': 'exp', 'relative': True, 'extent': 0.3, 'r_escape': 0.3,
                                 'centre_fn': lambda g: centre_of_mass_t(g, g.s > 0, der2.box)})
            assert counter['n'] == 1, counter                      # the final dump only
            assert res2['final']['a'] > 1.0 and res2['dt_min_used'] <= res2['dt_max_used'] <= der2.dt_abs
            assert np.isfinite(res2['final']['E']) and os.path.exists(os.path.join(sub, 'snap_0001.png'))
            assert np.isfinite(res2['final']['escaped_frac_pos']) and np.isfinite(res2['final']['L_pos'])
            assert res2['final_state'] == os.path.join(sub, 'final_state.npz')
            with np.load(res2['final_state']) as fs:
                assert set(fs.files) == {'pos', 'vel', 'w', 's', 'species'}
                assert fs['pos'].shape == (400, 2) and fs['pos'].dtype == np.float64
                assert np.all((fs['pos'] >= 0) & (fs['pos'] < der2.box)) and np.all(np.isfinite(fs['vel']))
                assert np.array_equal(fs['species'], ps2.species) and np.allclose(fs['w'], ps2.w)
            _check_timing(res2)
            # ---- 3-D LSS with snapshots (projections + slice), float32, no host copy ------
            cfg3 = resolve_defaults(RunConfig('lss', npos=400, nneg=400, dim=3, method='pm', grid=16,
                                              tmax=0.2, snap_every=0.1, structure_every=0.1, log_every=5,
                                              structure_grid=16, scatter_subsample=150, quiet=True,
                                              device='gpu'))
            der3, force3 = build(cfg3)
            ps3 = make_lss_ic(cfg3, der3, np.random.default_rng(3))
            sub3 = os.path.join(td, 'd3')
            os.makedirs(sub3)
            counter['n'] = 0
            res3 = run_loop_gpu(cfg3, der3, ps3, force3, sub3, lss_structure(cfg3, der3),
                                {'title': '3-D', 'relative': False})
            assert counter['n'] == 0, counter
            assert res3['n_snapshots'] == 3 and 'L_pos_z' in res3['final'] and 'p_abs_z' in res3['final']
            assert np.isfinite(res3['final']['E']) and abs(res3['final']['dP_abs_rel']) < 1e-4
            assert len(res3['structure_rows']) == 3 and np.isfinite(res3['structure_rows'][-1]['lambda_peak_neg'])
            for f in ('snap_0000.png', 'snap_0002.png'):
                assert os.path.getsize(os.path.join(sub3, f)) > 1000, f
            assert not os.path.exists(os.path.join(sub3, 'final_state.npz'))
            _check_timing(res3)
    finally:
        GPUParticles.to_host = _orig_to_host
    print(res['gpu_memory']['text'])
    print(res2['timing']['text'])
    print("run_loop_gpu smoke test OK")
