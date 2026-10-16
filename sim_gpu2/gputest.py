"""gputest.py -- the `gputest` command: CPU vs GPU agreement, speed and memory.

The GPU path (forces_pm_torch.PMSolverTorch, integrator_torch) is meant to be
a drop-in replacement for the validated numpy particle-mesh solver, so before
any long GPU run it is worth measuring how closely it reproduces the CPU
reference on identical random particles:

  * compare_forces   : same positions, weights (w_+ = 1/N, w_- = 64/N) and signs
                       on both solvers, 2-D and 3-D, periodic and isolated
                       boundaries (and the logarithmic law in 2-D); reports the
                       Frobenius relative error of the accelerations, the worst
                       per-particle error (normalised by the rms acceleration),
                       the relative error of the potential (mean removed), the
                       |m|-weighted momentum residual (must be round-off: no
                       self-force), the median time per force evaluation on both
                       devices, the speed-up and the peak GPU memory;
  * compare_leapfrog : n_steps of KDK leapfrog with both integrators, expansion
                       off and on, rms position difference relative to the rms
                       displacement and the relative energy drift of each;
  * run_gputest      : the table printed by `python main.py gputest`, plus the
                       device name, the torch version, the memory estimate of
                       the default 2-D and 3-D LSS runs and a verdict line.

Expected: float64 agrees to ~1e-12 (TOL_F64 = 1e-9 is generous), float32 to
~1e-6 (TOL_F32 = 1e-4).  Exceeding the tolerance prints a warning, never an
exception, so a broken ROCm runtime is diagnosed rather than hidden.
"""
from __future__ import annotations

import sys
import time

import numpy as np

try:
    import torch
    _TORCH_ERR = ''
except Exception as _exc:            # ImportError or a broken runtime
    torch = None
    _TORCH_ERR = repr(_exc)

from config import RunConfig, resolve_defaults, derive
from particles import ParticleSet, min_image
from forces_pm import PMSolver
from external_potentials import NoField
from integrator import Expansion, ForceEvaluator, leapfrog_step
from gpu_backend import (require_gpu, device_name, torch_dtype, gpu_memory_report,
                         estimate_gpu_memory, GPUParticles)
from forces_pm_torch import PMSolverTorch
from integrator_torch import ForceEvaluatorTorch, leapfrog_step_t, kick_t, drift_t  # noqa: F401

TOL_F64 = 1e-9          # float64 GPU vs float64 CPU: round-off of the FFT pipeline
TOL_F32 = 1e-4          # float32 GPU vs float64 CPU
MASS_RATIO = 64.0       # |M_-| / M_+ of the test particles (A6)
_GB = float(2 ** 30)
_HEADER = ("dim  bc        law  dtype    N        grid  rel_err_acc  max_rel_err  rel_err_phi  "
           "momentum   t_cpu[s]   t_gpu[s]   speedup  peak_GB")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _need_torch() -> None:
    if torch is None:
        raise RuntimeError(f"gputest needs PyTorch (import failed: {_TORCH_ERR})")


def _sync(device) -> None:
    if device is not None and torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def _random_particles(n: int, dim: int, rng, w_pos=None, w_neg=None):
    """Uniform positions in the unit box, first half positive (w_pos, default 1/N),
    second half negative (w_neg, default 64/N)."""
    n = int(n)
    pos = rng.random((n, dim))
    s = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    wp = 1.0 / n if w_pos is None else float(w_pos)
    wn = MASS_RATIO / n if w_neg is None else float(w_neg)
    w = np.where(s > 0, wp, wn)
    return pos, w, s


def _median_time(fn, n_timing: int, device) -> float:
    """Median wall time of n_timing calls after one warm-up (synchronised)."""
    fn()
    _sync(device)
    times = []
    for _ in range(max(1, int(n_timing))):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _norm(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def _energy(ps: ParticleSet, phi, a: float, self_energy: float) -> float:
    v = np.asarray(ps.vel, np.float64)
    w = np.asarray(ps.w, np.float64)
    ke = 0.5 * a * a * float(np.sum(w * np.einsum('ij,ij->i', v, v)))
    pe = 0.5 * float(np.sum(np.asarray(ps.s, np.float64) * w * np.asarray(phi, np.float64)))
    return ke + pe - float(self_energy)


def _n_total(c: RunConfig) -> int:
    return (0 if c.neg_only else c.npos) + (0 if c.pos_only else c.nneg)


def _fmt_row(r: dict) -> str:
    return (f"{r['dim']:<4d} {r['bc']:<9s} {r['force_law']:<4s} {r['dtype']:<8s} {r['n']:<8d} "
            f"{r['grid']:<5d} {r['rel_err_acc']:<12.3e} {r['max_rel_err_acc']:<12.3e} "
            f"{r['rel_err_phi']:<12.3e} {r['momentum']:<10.2e} {r['cpu_time_per_eval']:<10.4f} "
            f"{r['gpu_time_per_eval']:<10.4f} {r['speedup']:<8.1f} {r['peak_gpu_bytes'] / _GB:.3f}")


# ---------------------------------------------------------------------------
# force comparison
# ---------------------------------------------------------------------------

def compare_forces(dim: int, n_part: int, grid: int, bc: str, force_law: str, dtype_name: str,
                   device, seed: int, n_timing: int = 5) -> dict:
    """CPU PMSolver (float64 reference) vs PMSolverTorch (dtype_name) on the same
    random particles; returns the error metrics, timings and peak GPU memory."""
    _need_torch()
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    pos, w, s = _random_particles(n_part, dim, rng)
    cpu = PMSolver(dim, grid, 1.0, 1.0, force_law, bc=bc, eps=0.0, dtype=np.float64)
    a_cpu, phi_cpu = cpu.forces(pos, w, s)
    a_cpu, phi_cpu = np.asarray(a_cpu, np.float64), np.asarray(phi_cpu, np.float64)
    t_cpu = _median_time(lambda: cpu.forces(pos, w, s), n_timing, None)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    pt = PMSolverTorch(cpu, device, dtype_name)
    tdt = torch_dtype(dtype_name)
    pos_t = torch.as_tensor(pos, dtype=tdt, device=device)
    w_t = torch.as_tensor(w, dtype=tdt, device=device)
    s_t = torch.as_tensor(s, dtype=tdt, device=device)
    a_t, p_t = pt.forces(pos_t, w_t, s_t)
    _sync(device)
    t_gpu = _median_time(lambda: pt.forces(pos_t, w_t, s_t), n_timing, device)
    a_gpu = a_t.detach().cpu().numpy().astype(np.float64)
    phi_gpu = p_t.detach().cpu().numpy().astype(np.float64)
    diff = a_gpu - a_cpu
    norm_cpu = max(_norm(a_cpu), 1e-300)
    rms_a = max(float(np.sqrt(np.mean(np.einsum('ij,ij->i', a_cpu, a_cpu)))), 1e-300)
    per_particle = np.sqrt(np.einsum('ij,ij->i', diff, diff))
    pc = phi_cpu - phi_cpu.mean()
    pg = phi_gpu - phi_gpu.mean()
    P = (w[:, None] * a_gpu).sum(0)
    scale = max(float(np.sum(w * np.sqrt(np.einsum('ij,ij->i', a_gpu, a_gpu)))), 1e-300)
    peak = int(torch.cuda.max_memory_allocated()) if device.type == 'cuda' else 0
    return {'dim': int(dim), 'bc': bc, 'force_law': force_law, 'dtype': dtype_name,
            'n': int(n_part), 'grid': int(grid),
            'rel_err_acc': _norm(diff) / norm_cpu,
            'max_rel_err_acc': float(per_particle.max()) / rms_a,
            'rel_err_phi': _norm(pg - pc) / max(_norm(pc), 1e-300),
            'momentum': _norm(P) / scale,
            'cpu_time_per_eval': t_cpu, 'gpu_time_per_eval': t_gpu,
            'speedup': t_cpu / max(t_gpu, 1e-12), 'peak_gpu_bytes': peak}


# ---------------------------------------------------------------------------
# leapfrog comparison
# ---------------------------------------------------------------------------

def compare_leapfrog(dim: int, n_part: int, grid: int, dtype_name: str, device,
                     n_steps: int = 20, seed: int = 1) -> dict:
    """n_steps of KDK with ForceEvaluator + leapfrog_step (CPU, float64) and with
    ForceEvaluatorTorch + leapfrog_step_t (device, dtype_name), expansion off and
    on (t0 = 0.5 n_steps dt).  Per case: rms position difference relative to the
    rms displacement, relative velocity difference, energy drift of each path."""
    _need_torch()
    device = torch.device(device)
    npos = int(n_part) // 2
    cfg = resolve_defaults(RunConfig('lss', npos=npos, nneg=int(n_part) - npos, dim=dim, grid=grid,
                                     method='pm', dtype=dtype_name, device='gpu', quiet=True))
    der = derive(cfg)
    rng = np.random.default_rng(seed)
    pos, w, s = _random_particles(n_part, dim, rng, der.w_pos, der.w_neg)
    vel = rng.normal(size=(int(n_part), dim)) * np.where(s > 0, der.sigma_pos, der.sigma_neg)[:, None]
    vel -= (w[:, None] * vel).sum(0) / w.sum()
    ps0 = ParticleSet(pos, vel, w, s, (s < 0).astype(np.int32))
    pm_cpu = PMSolver(dim, grid, der.box, der.G, cfg.force_law, bc=cfg.bc, eps=0.0, dtype=np.float64)
    force_c = ForceEvaluator(cfg, der, pm_cpu, NoField())
    force_t = ForceEvaluatorTorch(cfg, der, PMSolverTorch(pm_cpu, device, dtype_name), NoField())
    dt, p, box = der.dt_abs, der.force_exponent_p, der.box
    se = force_c.self_energy_fn(ps0.w)
    out = {}
    for name, ex in (('no_expansion', Expansion(False, 0.0, 1.0)),
                     ('expansion', Expansion(True, cfg.a_index, 0.5 * n_steps * dt))):
        ps_c = ps0.copy()
        acc_c, phi_c = force_c(ps_c)
        E0_c = _energy(ps_c, phi_c, 1.0, se)
        t = 0.0
        t0 = time.perf_counter()
        for _ in range(int(n_steps)):
            acc_c, phi_c, t = leapfrog_step(ps_c, acc_c, phi_c, t, dt, force_c, ex, p, box)
        wall_c = time.perf_counter() - t0
        E1_c = _energy(ps_c, phi_c, ex.a(t), se)
        gp = GPUParticles.from_host(ps0, device, dtype_name)
        acc_t, phi_t = force_t(gp)
        E0_g = _energy(gp.to_host('float64'), phi_t.detach().cpu().numpy(), 1.0, se)
        tt = 0.0
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(int(n_steps)):
            acc_t, phi_t, tt = leapfrog_step_t(gp, acc_t, phi_t, tt, dt, force_t, ex, p, box)
        _sync(device)
        wall_g = time.perf_counter() - t0
        h = gp.to_host('float64')
        E1_g = _energy(h, phi_t.detach().cpu().numpy(), ex.a(tt), se)
        disp = min_image(np.asarray(ps_c.pos, np.float64) - pos, box)
        diff = min_image(h.pos - np.asarray(ps_c.pos, np.float64), box)
        rms_disp = max(float(np.sqrt(np.mean(np.einsum('ij,ij->i', disp, disp)))), 1e-300)
        rms_diff = float(np.sqrt(np.mean(np.einsum('ij,ij->i', diff, diff))))
        out[name] = {'pos_rel_diff': rms_diff / rms_disp,
                     'vel_rel_diff': _norm(h.vel - ps_c.vel) / max(_norm(ps_c.vel), 1e-300),
                     'dE_rel_cpu': (E1_c - E0_c) / max(abs(E0_c), 1e-300),
                     'dE_rel_gpu': (E1_g - E0_g) / max(abs(E0_g), 1e-300),
                     'a_final': ex.a(tt), 'wall_cpu': wall_c, 'wall_gpu': wall_g,
                     'n_steps': int(n_steps), 'dt': dt, 'dim': int(dim), 'dtype': dtype_name,
                     'n': int(n_part), 'grid': int(grid)}
    return out


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------

def run_gputest(n_part: int = 100000, grid2: int = 256, grid3: int = 64,
                dtypes=('float32', 'float64'), seed: int = 1, quiet: bool = False,
                device=None, n_leapfrog=None) -> dict:
    """Print the comparison table, the device/torch info, the memory estimates of
    the default LSS runs and the verdict; returns {'rows', 'leapfrog', 'ok', ...}."""
    _need_torch()
    dev = require_gpu() if device is None else torch.device(device)
    dtypes = tuple(dtypes)

    def say(*lines):
        if not quiet:
            for ln in lines:
                print(ln)

    say(f"=== gputest: CPU (numpy float64) vs GPU ({dev}) ===",
        f"device = {device_name() if dev.type == 'cuda' else dev}, torch {torch.__version__}",
        gpu_memory_report()['text'])
    estimates = {}
    for dim in (3, 2):
        c = resolve_defaults(RunConfig('lss', dim=dim, device='gpu'))
        est = estimate_gpu_memory(c, derive(c), _n_total(c))
        estimates[f'lss_{dim}d_default'] = est
        say(f"--- default LSS {dim}-D run on the GPU ---", est['text'])
    rows = []
    print(_HEADER)
    for dim in (2, 3):
        grid = int(grid2 if dim == 2 else grid3)
        for bc in ('periodic', 'isolated'):
            laws = ('r2', 'log') if (dim == 2 and bc == 'periodic') else ('r2',)
            for law in laws:
                for dt in dtypes:
                    row = compare_forces(dim, n_part, grid, bc, law, dt, dev, seed)
                    rows.append(row)
                    print(_fmt_row(row), flush=True)
    n_leap = int(n_leapfrog) if n_leapfrog is not None else int(min(n_part, 20000))
    leap = []
    say("--- leapfrog: CPU vs GPU after 20 steps (rms pos diff / rms displacement, dE/|E|) ---")
    for dim in (2, 3):
        grid = int(grid2 if dim == 2 else grid3)
        for dt in dtypes:
            res = compare_leapfrog(dim, n_leap, grid, dt, dev, n_steps=20, seed=seed)
            for name, r in res.items():
                leap.append(dict(r, case=name))
                say(f"  {dim}-D {dt:<8s} {name:<13s} N = {r['n']:<7d} pos diff = {r['pos_rel_diff']:.2e}  "
                    f"vel diff = {r['vel_rel_diff']:.2e}  dE/E cpu = {r['dE_rel_cpu']:+.2e}  "
                    f"gpu = {r['dE_rel_gpu']:+.2e}  a = {r['a_final']:.3f}  "
                    f"wall cpu/gpu = {r['wall_cpu']:.2f}/{r['wall_gpu']:.2f} s")
    ok = True
    verdicts = []
    for dt in dtypes:
        tol = TOL_F64 if dt == 'float64' else TOL_F32
        sel = [r for r in rows if r['dtype'] == dt]
        if not sel:
            continue
        worst = max(max(r['rel_err_acc'], r['rel_err_phi'], r['momentum']) for r in sel)
        worst_max = max(r['max_rel_err_acc'] for r in sel)
        if np.isfinite(worst) and worst < tol:
            msg = (f"{dt} GPU agrees with CPU to < {10 ** np.ceil(np.log10(max(worst, 1e-300))):.0e} "
                   f"(worst relative error {worst:.1e}, worst per-particle {worst_max:.1e}, tolerance {tol:g})")
        else:
            ok = False
            msg = (f"WARNING: {dt} GPU/CPU discrepancy {worst:.1e} exceeds the tolerance {tol:g} "
                   f"(worst per-particle {worst_max:.1e}): do not trust GPU runs in {dt} until this is fixed")
        verdicts.append(msg)
    best = max(rows, key=lambda r: r['speedup'])
    verdicts.append(f"best speed-up {best['speedup']:.1f}x ({best['dim']}-D {best['bc']} {best['dtype']}, "
                    f"{1e3 * best['gpu_time_per_eval']:.1f} ms per force evaluation on the GPU)")
    print("--- verdict ---")
    for v in verdicts:
        print("  " + v)
    say(gpu_memory_report()['text'])
    return {'rows': rows, 'leapfrog': leap, 'ok': ok, 'verdicts': verdicts, 'estimates': estimates,
            'device': str(dev), 'device_name': device_name(), 'torch_version': str(torch.__version__)}


if __name__ == '__main__':  # self-test with small sizes
    if torch is None:
        print("torch not installed, gputest self-test skipped:", _TORCH_ERR)
        sys.exit(0)
    try:
        require_gpu()
        dev = None
    except RuntimeError as exc:
        print(f"no GPU ({str(exc).splitlines()[0]}); running the gputest on the torch CPU device")
        dev = 'cpu'
    out = run_gputest(n_part=20000, grid2=64, grid3=24, dtypes=('float32', 'float64'), seed=1,
                      device=dev, n_leapfrog=4000)
    assert out['rows'] and all(np.isfinite(r['rel_err_acc']) for r in out['rows'])
    assert len(out['rows']) == 2 * (3 + 2) and len(out['leapfrog']) == 8
    assert out['ok'], out['verdicts']
    f64 = [r for r in out['rows'] if r['dtype'] == 'float64']
    assert max(r['rel_err_acc'] for r in f64) < TOL_F64 and max(r['momentum'] for r in f64) < TOL_F64
    for r in out['leapfrog']:
        tol = 1e-8 if r['dtype'] == 'float64' else 1e-2
        assert r['pos_rel_diff'] < tol and np.isfinite(r['dE_rel_gpu']), r
    print("gputest self-test OK")
