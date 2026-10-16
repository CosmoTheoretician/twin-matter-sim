"""run_loop_gpu.py -- the time loop of scenarios.run_loop with the particles
resident on the GPU (PyTorch / ROCm, exposed as the 'cuda' device).

The initial condition (built on the CPU exactly as before) is uploaded once to
a GPUParticles; positions, velocities, accelerations and potentials then stay
on the device for the whole run.  Host copies are made only when something
needs them: a diagnostics row (every cfg.log_every steps), a structure row
(every der.structure_every_abs), a snapshot (every der.snap_every_abs) and the
final step -- the same cadence logic as the CPU loop, with one host copy per
step at most (shared by whatever is due at that step).  The ordinary numpy
ParticleSet handed back at those moments feeds the unchanged diagnostics,
structure and plotting code, so diagnostics.csv, structure.csv, the snapshots
and the returned result dict are identical in form to the CPU run (plus a
'gpu_memory' entry).  Integration: torch leapfrog KDK (integrator_torch) with
the kinematic expansion and the adaptive global step of integrator.py.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from config import RunConfig, Derived
from particles import ParticleSet
from gpu_backend import GPUParticles, gpu_memory_report, torch_dtype, device_name
from integrator import Expansion
from integrator_torch import ForceEvaluatorTorch, choose_dt_t, leapfrog_step_t
from diagnostics_core import diagnostics_row, CsvLogger
from plotting import snapshot

_AX = 'xyz'


def _say(cfg: RunConfig, msg: str) -> None:
    if not cfg.quiet:
        print("[gpu] " + msg, file=sys.stderr, flush=True)


def _to_numpy(x, dtype):
    """Tensor (any device) or array -> numpy array of `dtype`; None stays None."""
    if x is None:
        return None
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def run_loop_gpu(cfg: RunConfig, der: Derived, ps: ParticleSet, force: ForceEvaluatorTorch,
                 outdir: str, structure_fn, snapshot_kwargs: dict) -> dict:
    """Device-resident leapfrog KDK loop; same inputs, outputs and cadences as
    scenarios.run_loop.  structure_fn(ps, t, state) -> (row, state) receives a
    host ParticleSet.  snapshot_kwargs keys: centre_fn (ps -> centre), r_escape,
    relative (bool), extent, title.  Returns the run_loop dict plus 'gpu_memory'."""
    box_wrap = der.box if cfg.bc == 'periodic' else None
    p = der.force_exponent_p
    expansion = Expansion(cfg.expansion, cfg.a_index, cfg.t0_exp * der.t_ref)
    centre_fn = snapshot_kwargs.get('centre_fn')
    r_escape = snapshot_kwargs.get('r_escape')
    relative = bool(snapshot_kwargs.get('relative', False))
    extent, title = snapshot_kwargs.get('extent'), snapshot_kwargs.get('title', '')
    self_energy = force.self_energy_fn(ps.w)                 # host weights
    dtype = torch_dtype(cfg.dtype)
    npdt = np.dtype(cfg.dtype)
    gp = GPUParticles.from_host(ps, force.device, dtype)    # the only upload of the run
    diag_log = CsvLogger(os.path.join(outdir, 'diagnostics.csv'))
    struct_log = CsvLogger(os.path.join(outdir, 'structure.csv'))
    diag_rows, struct_rows = [], []
    tmax, dt_max = der.tmax_abs, der.dt_abs
    dt_min = 1e-3 * dt_max
    wall0 = time.perf_counter()
    acc, phi = force(gp)                                    # tensors on the device
    _say(cfg, f"device = {device_name()} ({force.device}), dtype = {cfg.dtype}, N = {gp.n}, "
              f"PM {der.grid}^{der.dim}, first force evaluation {1e3 * force.last_wall:.1f} ms; "
              + gpu_memory_report()['text'])
    t, step = 0.0, 0
    dt = choose_dt_t(acc, force.eps_eff, cfg.eta, dt_max, dt_min) if cfg.adaptive else dt_max
    dts = [dt]
    state = [None]                                          # structure state
    E0, P0 = None, None
    cache = {'step': -1, 'ps': None, 'phi': None, 'phi_ext': None}

    def host():
        """Host copies of the current state, made once per step when something is due."""
        if cache['step'] != step:
            cache['ps'] = gp.to_host()
            cache['phi'] = _to_numpy(phi, npdt)
            cache['phi_ext'] = _to_numpy(force.last_phi_ext, npdt)
            cache['step'] = step
        return cache['ps'], cache['phi'], cache['phi_ext']

    def centre(hps):
        return centre_fn(hps) if centre_fn is not None else None

    def log_diag(fix_zero=False):
        hps, hphi, hphi_ext = host()
        c = centre(hps)
        row = diagnostics_row(hps, hphi, t, step, dt, expansion.a(t), self_energy, E0, P0, centre=c,
                              r_escape=r_escape, wall=time.perf_counter() - wall0, box=box_wrap,
                              phi_ext=hphi_ext)
        if fix_zero:
            row['dE_rel'], row['dP_abs_rel'] = 0.0, 0.0
        row['t_ref_units'] = t / der.t_ref
        row['n_force_evals'] = force.n_evals
        diag_log.log(row)
        diag_rows.append(row)
        return row

    def do_structure():
        hps, _, _ = host()
        row, state[0] = structure_fn(hps, t, state[0])
        row['t_ref_units'] = t / der.t_ref
        struct_log.log(row)
        struct_rows.append(row)

    def do_snapshot(k):
        hps, _, _ = host()
        c = centre(hps) if relative else None
        snapshot(hps, der.box, t / der.t_ref, der.t_ref_name,
                 os.path.join(outdir, f'snap_{k:04d}.png'), der.dim, centre=c,
                 extent=extent if relative else None, title=title)

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
    wall = time.perf_counter() - wall0
    mem = gpu_memory_report()
    _say(cfg, f"done: {step} steps, {force.n_evals} force evaluations, {wall:.1f} s "
              f"({force.total_wall:.1f} s in forces); {mem['text']}")
    return {'final': diag_rows[-1], 'diag_rows': diag_rows, 'structure_rows': struct_rows,
            'wall_time': wall, 'force_wall': force.total_wall, 'n_steps': step,
            'n_force_evals': force.n_evals, 'dt_min_used': float(min(dts)),
            'dt_max_used': float(max(dts)), 'n_snapshots': k_snap, 'gpu_memory': mem}


if __name__ == '__main__':  # guarded smoke test: a tiny LSS loop on the device
    import tempfile
    try:
        import torch
    except Exception as exc:
        print("torch not installed, run_loop_gpu smoke test skipped:", repr(exc))
        sys.exit(0)
    from config import resolve_defaults, derive
    from forces_pm import PMSolver
    from forces_pm_torch import PMSolverTorch
    from external_potentials import NoField
    from initial_conditions import make_lss_ic
    from structure_stats import lss_structure_row
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cpu':
        print("no GPU: running the smoke test with the torch solver on the CPU device")

    def build(cfg):
        der = derive(cfg)
        pm_cpu = PMSolver(der.dim, der.grid, der.box, der.G, cfg.force_law, bc=cfg.bc,
                          eps=max(cfg.eps, 0.0), dtype=np.float64)
        pm_t = PMSolverTorch(pm_cpu, dev, cfg.dtype)
        return der, ForceEvaluatorTorch(cfg, der, pm_t, NoField())

    with tempfile.TemporaryDirectory() as td:
        cfg = resolve_defaults(RunConfig('lss', npos=300, nneg=300, method='pm', grid=32, tmax=0.5,
                                         snap_every=0.25, structure_every=0.25, log_every=5,
                                         structure_grid=32, quiet=True, device='gpu'))
        der, force = build(cfg)
        ps = make_lss_ic(cfg, der, np.random.default_rng(cfg.seed))
        assert ps.pos.dtype == np.float32

        def structure_fn(hps, t, state):
            row, _ = lss_structure_row(hps, t, der.box, cfg.structure_grid, der.dim,
                                       cfg.clump_threshold, cfg.clump_min_cells, None)
            return row, state

        res = run_loop_gpu(cfg, der, ps, force, td, structure_fn, {'title': 'smoke', 'relative': False})
        keys = {'final', 'diag_rows', 'structure_rows', 'wall_time', 'force_wall', 'n_steps',
                'n_force_evals', 'dt_min_used', 'dt_max_used', 'n_snapshots', 'gpu_memory'}
        assert keys <= set(res), set(res)
        assert abs(res['n_steps'] - der.n_steps) <= 1 and res['n_force_evals'] == res['n_steps'] + 1
        assert res['diag_rows'][0]['dE_rel'] == 0.0 and res['diag_rows'][0]['dP_abs_rel'] == 0.0
        assert res['final']['step'] == res['n_steps'] and np.isfinite(res['final']['E'])
        assert abs(res['final']['dP_abs_rel']) < 1e-4 and res['n_snapshots'] == 3
        assert len(res['structure_rows']) == 3 and 'n_clumps_neg' in res['structure_rows'][0]
        for f in ('diagnostics.csv', 'structure.csv', 'snap_0000.png', 'snap_0002.png'):
            assert os.path.getsize(os.path.join(td, f)) > 100, f
        # expansion + adaptive step + float64 + relative view with a centre function
        cfg2 = resolve_defaults(RunConfig('lss', npos=200, nneg=200, method='pm', grid=32, tmax=0.2,
                                          snap_every=0.1, structure_every=0.1, log_every=2, expansion=True,
                                          t0_exp=0.05, adaptive=True, dtype='float64', structure_grid=32,
                                          quiet=True, device='gpu'))
        der2, force2 = build(cfg2)
        ps2 = make_lss_ic(cfg2, der2, np.random.default_rng(2))
        sub = os.path.join(td, 'exp')
        os.makedirs(sub)
        res2 = run_loop_gpu(cfg2, der2, ps2, force2, sub, lambda h, t, s: ({'t': t, 'n': h.n}, s),
                            {'title': 'exp', 'relative': True, 'extent': 0.3,
                             'centre_fn': lambda h: np.full(h.dim, 0.5 * der2.box)})
        assert res2['final']['a'] > 1.0 and res2['dt_min_used'] <= res2['dt_max_used'] <= der2.dt_abs
        assert np.isfinite(res2['final']['E']) and os.path.exists(os.path.join(sub, 'snap_0001.png'))
    print(res['gpu_memory']['text'])
    print("run_loop_gpu smoke test OK")
