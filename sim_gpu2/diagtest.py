"""diagtest.py -- the `diagtest` command: CPU vs GPU agreement of every
diagnostic, structure and snapshot-map quantity.

The GPU time loop (run_loop_gpu) never copies the particles to the host: the
energy / momentum sums (diagnostics_torch), the density grids and grid
statistics (grids_torch, structure_torch) and the snapshot maps
(snapshot_torch) are all evaluated on the device.  Before trusting a long run
this command checks, on identical particle states, that the device versions
reproduce the validated numpy path:

  * three states built with numpy (seeded): a 2-D LSS box (N/2 positive
    uniform + N/2 negative, half of them in Gaussian clumps, one straddling
    the periodic edge; weights 1/N and 64/N; Maxwellian velocities), the same
    in 3-D, and a 2-D galaxy (disk + live twin ring on a 16 R_d box from
    initial_conditions.make_galaxy_ic with the CPU PM force);
  * per state: phi from the CPU PMSolver (plus a random phi_ext for the
    galaxy); CPU values from diagnostics_core.diagnostics_row (two rows, so
    dE_rel / dP_abs_rel are exercised), particles.centre_of_mass (circular
    mean), structure_stats.lss_structure_row x 2 (reference grid from the
    first call) or disk_structure_row x 2 (omega_p), forces_pm.density_grid +
    smooth_grid, power_spectrum / peak_scale and the 2-D / 3-D snapshot
    density maps, projections and slab map exactly as plotting draws them;
    GPU values from GPUParticles.from_host in float64 and float32 with the
    torch modules on the real device;
  * compare(cpu, gpu): per key |a - b| / max(|b|, floor) (arrays: max over the
    cells, nan patterns must agree, nan == nan); a table per state and dtype,
    the worst value, a verdict per dtype (TOL_F64 = 1e-9 is pass/fail, the
    float32 tolerance TOL_F32 = 2e-3 is informational: a float32 particle
    state legitimately differs), the row timings and the GPU memory report.
"""
from __future__ import annotations

import math
import sys
from time import perf_counter

import numpy as np

try:
    import torch
    _TORCH_ERR = ''
except Exception as _exc:            # ImportError or a broken runtime
    torch = None
    _TORCH_ERR = repr(_exc)

from config import RunConfig, resolve_defaults, derive
from particles import ParticleSet, min_image, centre_of_mass
from forces_pm import PMSolver, density_grid
from external_potentials import NoField
from integrator import ForceEvaluator
from initial_conditions import make_galaxy_ic
from diagnostics_core import diagnostics_row, kinetic_energy, potential_energy
from structure_stats import (lss_structure_row, disk_structure_row, smooth_grid, power_spectrum,
                             peak_scale, segregation_index, void_fraction, pattern_persistence)
from plotting_3d import _projected_density, _log_image
from gpu_backend import require_gpu, device_name, gpu_memory_report, synchronize, GPUParticles
from diagnostics_torch import diagnostics_row_t, centre_of_mass_t
from structure_torch import lss_structure_row_t, disk_structure_row_t
from grids_torch import (density_grid_t, gaussian_filter_t, power_spectrum_t, peak_scale_t,
                         segregation_index_t, void_fraction_t, pattern_persistence_t, slab_density_t)
from snapshot_torch import species_view_t, density_map_t, projection_maps_t

TOL_F64 = 1e-9          # float64 GPU vs float64 CPU: round-off of sums, CIC and FFTs
TOL_F32 = 2e-3          # float32 particle state vs float64 CPU (informational)
DT, T1, A1 = 0.01, 0.25, 1.2      # dt, time and scale factor of the second rows
_PROJ = ((0, 1), (0, 2), (1, 2))
_AX = 'xyz'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _need_torch() -> None:
    if torch is None:
        raise RuntimeError(f"diagtest needs PyTorch (import failed: {_TORCH_ERR})")


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _fin(x) -> float:
    x = float(x)
    return x if math.isfinite(x) else float('inf')


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        fin = v[np.isfinite(v)]
        mx = float(np.abs(fin).max()) if fin.size else float('nan')
        return f"[{'x'.join(str(d) for d in v.shape)}] max|.| {mx:.6g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.10g}"


def _scales(vs: float, ls: float) -> dict:
    """Reference magnitudes for keys whose values cancel to round-off (momenta of
    a bulk-velocity-free state, pattern speed of an identical state, ...)."""
    sc = {}
    for k in ('p_abs_x', 'p_abs_y', 'p_abs_z', 'p_signed_x', 'p_signed_y', 'p_signed_z',
              'p_abs_norm', 'p_signed_norm'):
        sc[k] = vs
    for k in ('L_abs', 'L_signed', 'L_pos', 'L_neg', 'L_pos_z', 'L_neg_z', 'Lz_disk', 'Lz_twin_abs'):
        sc[k] = ls
    for k in ('dP_abs_rel', 'dP_signed_rel', 'omega_p', 'escaped_frac', 'escaped_frac_pos',
              'escaped_frac_neg', 'bar_A2', 'spiral_A2', 'A1', 'A3', 'A4', 'twin_clump_mass_frac',
              'segregation', 'persistence', 'void_frac_pos', 'void_frac_neg', 'void_pos', 'void_neg',
              'vrot_disk', 'vrot_mean_pos', 'Q_at_2rd', 'pitch_deg'):
        sc[k] = 1.0
    sc['phase2_deg'], sc['phase2_state'] = 180.0, math.pi
    return sc


def _rel(a, b, floor: float) -> float:
    """max |a - b| / max(max |b|, floor); inf on shape / nan-pattern mismatch."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf')
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return float('inf')
    if (~nb).sum() == 0:
        return 0.0
    diff = float(np.abs(a[~na] - b[~nb]).max())
    return diff / max(float(np.abs(b[~nb]).max()), floor)


def compare(cpu: dict, gpu: dict, floor: float = 1e-12, scales=None) -> dict:
    """Per-key relative difference of the GPU value a to the CPU reference b:
    |a - b| / max(|b|, floor) (arrays: max over cells, nan == nan, mismatching
    nan patterns -> inf); `scales` maps a key suffix to a reference magnitude
    used instead of the floor; keys missing on the GPU side -> inf."""
    scales = scales or {}
    out = {}
    for key, b in cpu.items():
        if key not in gpu:
            out[key] = float('inf')
            continue
        sc = max(float(scales.get(key.split('/')[-1], floor)), floor)
        out[key] = _rel(gpu[key], b, sc)
    return out


# ---------------------------------------------------------------------------
# particle states
# ---------------------------------------------------------------------------

def _finish_state(name, kind, ps, box, phi, phi_ext, self_energy, centre, r_escape, struct_grid,
                  ring_grid, extent, snap_grid, G=1.0, rd=1.0, rmax=4.0) -> dict:
    w, v = np.asarray(ps.w, np.float64), np.asarray(ps.vel, np.float64)
    vmag = np.sqrt((v * v).sum(1))
    vs = float(np.sum(w * vmag))
    P0 = 0.5 * (w[:, None] * v).sum(0)
    P0[0] += 0.01 * vs                       # a drift of order 1 % of sum |m||v|
    E0 = 0.9 * (kinetic_energy(ps) + potential_energy(ps, phi, self_energy, phi_ext=phi_ext))
    r = min_image(np.asarray(ps.pos, np.float64) - centre[None, :], box)
    ls = float(np.sum(w * np.sqrt((r * r).sum(1)) * vmag))
    return dict(name=name, kind=kind, ps=ps, dim=ps.dim, box=float(box), phi=phi, phi_ext=phi_ext,
                self_energy=float(self_energy), centre=np.asarray(centre, np.float64),
                r_escape=float(r_escape), struct_grid=int(struct_grid), ring_grid=int(ring_grid),
                extent=float(extent), snap_grid=int(snap_grid), E0=float(E0), P0=P0,
                scales=_scales(vs, ls), threshold=5.0, min_cells=4, G=float(G), rd=float(rd),
                rmax=float(rmax))


def make_lss_state(n: int, dim: int, grid: int, seed: int, snap_grid: int) -> dict:
    rng = np.random.default_rng(seed + dim)
    n = int(n)
    h = n // 2
    m = n - h
    pos = rng.random((n, dim))
    nclump, kc = m // 2, 6
    cen = rng.random((kc, dim))
    cen[0] = 0.0                                 # a clump straddling the periodic edge
    pos[h:h + nclump] = np.mod(cen[np.arange(nclump) % kc] + 0.03 * rng.normal(size=(nclump, dim)), 1.0)
    s = np.where(np.arange(n) < h, 1.0, -1.0)
    w = np.where(s > 0, 1.0 / n, 64.0 / n)
    vel = rng.normal(size=(n, dim)) * np.where(s > 0, 0.1, 0.4)[:, None]
    ps = ParticleSet(pos, vel, w, s, (s < 0).astype(np.int32))
    pm = PMSolver(dim, grid, 1.0, 1.0, 'r2', 'periodic')
    _, phi = pm.forces(ps.pos, ps.w, ps.s)
    return _finish_state(f'{dim}-D LSS box', 'lss', ps, 1.0, np.asarray(phi, np.float64), None,
                         pm.self_energy(ps.w), np.full(dim, 0.5), 0.3, grid, grid, 0.3, snap_grid)


def make_galaxy_state(n: int, grid: int, seed: int, snap_grid: int) -> dict:
    rng = np.random.default_rng(seed + 100)
    h = int(n) // 2
    cfg = resolve_defaults(RunConfig('galaxy', ndisk=h, ntwin=int(n) - h, method='pm', grid=grid,
                                     seed=seed, quiet=True))
    der = derive(cfg)
    pm = PMSolver(2, grid, der.box, der.G, cfg.force_law, bc=cfg.bc, eps=max(cfg.eps, 0.0))
    force = ForceEvaluator(cfg, der, pm, NoField())
    ps, _ = make_galaxy_ic(cfg, der, rng, lambda _p: force)
    _, phi = pm.forces(ps.pos, ps.w, ps.s)
    phi_ext = 0.1 * rng.normal(size=ps.n)
    centre = centre_of_mass(ps, ps.species == 0, der.box)
    sg = max(8, int(grid) // 2)
    return _finish_state('2-D galaxy (disk + live ring)', 'galaxy', ps, der.box, np.asarray(phi, np.float64),
                         phi_ext, pm.self_energy(ps.w), centre, 1.5 * cfg.rmax, sg,
                         min(max(sg, 64), 128), 0.5 * der.box, snap_grid, G=der.G, rd=cfg.rd, rmax=cfg.rmax)


# ---------------------------------------------------------------------------
# CPU reference values
# ---------------------------------------------------------------------------

def _put(out: dict, prefix: str, row: dict) -> None:
    for k, v in row.items():
        out[f'{prefix}/{k}'] = v


def _cpu_maps(st: dict, out: dict) -> None:
    ps, dim, box, c, ext, ng = st['ps'], st['dim'], st['box'], st['centre'], st['extent'], st['snap_grid']
    n3 = max(8, ng // 2)
    for name, m in (('pos', ps.s > 0), ('neg', ps.s < 0)):
        pos, w = np.asarray(ps.pos[m], np.float64), np.asarray(ps.w[m], np.float64)
        rho = density_grid(pos, w, box, ng, dim, periodic=True)
        out[f'snap/abs_{name}'] = _log_image(rho.sum(axis=2) if dim == 3 else rho)
        d = min_image(pos - c[None, :], box)
        rho = density_grid(d, w, 2.0 * ext, ng, dim, periodic=False, origin=np.full(dim, -ext))
        out[f'snap/rel_{name}'] = _log_image(rho.sum(axis=2) if dim == 3 else rho)
        if dim == 3:
            for a, b in _PROJ:
                out[f'snap/proj_{_AX[a]}{_AX[b]}_{name}'] = _projected_density(pos, w, (a, b), 0.0, box, True, n3)
            sel = np.abs(min_image(pos[:, 2] - 0.5 * box, box)) < 0.05 * box
            out[f'snap/slab_{name}'] = density_grid(pos[sel, :2], w[sel], box, ng, 2, periodic=True)
            out[f'snap/slab_n_{name}'] = int(sel.sum())


def cpu_values(st: dict):
    """(dict of every CPU quantity, seconds spent in the four rows)."""
    ps, dim, box, c = st['ps'], st['dim'], st['box'], st['centre']
    kw = dict(centre=c, r_escape=st['r_escape'], wall=0.0, box=box, phi_ext=st['phi_ext'])
    t0 = perf_counter()
    row0 = diagnostics_row(ps, st['phi'], 0.0, 0, DT, 1.0, st['self_energy'], None, None, **kw)
    row1 = diagnostics_row(ps, st['phi'], T1, 1, DT, A1, st['self_energy'], st['E0'], st['P0'], **kw)
    if st['kind'] == 'lss':
        a = (box, st['struct_grid'], dim, st['threshold'], st['min_cells'])
        s0, cur = lss_structure_row(ps, 0.0, *a, None)
        s1, _ = lss_structure_row(ps, T1, *a, cur)
    else:
        a = (c, st['rd'], st['rmax'], box, st['G'])
        b = (st['ring_grid'], st['threshold'], st['min_cells'])
        s0, sta = disk_structure_row(ps, 0.0, *a, None, *b)
        s1, sta = disk_structure_row(ps, T1, *a, sta, *b)
        s1['phase2_state'] = sta['phase2']
    t_rows = perf_counter() - t0
    out = {}
    for pre, row in (('diag0', row0), ('diag1', row1), ('struct0', s0), ('struct1', s1)):
        _put(out, pre, row)
    out['com/pos_box'] = centre_of_mass(ps, ps.s > 0, box)
    out['com/neg_box'] = centre_of_mass(ps, ps.s < 0, box)
    out['com/all_free'] = centre_of_mass(ps)
    sm = {}
    for name, m in (('pos', ps.s > 0), ('neg', ps.s < 0)):
        rho = density_grid(ps.pos[m], ps.w[m], box, st['struct_grid'], dim, periodic=True)
        sm[name] = smooth_grid(rho, 1.0, True)
        k, P = power_spectrum(sm[name], box)
        out[f'grid/rho_{name}'], out[f'grid/smooth_{name}'], out[f'grid/pk_{name}'] = rho, sm[name], P
        out[f'grid/lambda_peak_{name}'] = peak_scale(k, P)
        out[f'grid/void_{name}'] = void_fraction(sm[name])
    out['grid/segregation'] = segregation_index(sm['pos'], sm['neg'])
    out['grid/persistence'] = pattern_persistence(sm['neg'], sm['pos'])
    _cpu_maps(st, out)
    return out, t_rows


# ---------------------------------------------------------------------------
# GPU values
# ---------------------------------------------------------------------------

def _gpu_maps(st: dict, gp, out: dict) -> None:
    dim, box, c, ext, ng = st['dim'], st['box'], st['centre'], st['extent'], st['snap_grid']
    n3 = max(8, ng // 2)
    for name in ('pos', 'neg'):
        pos, w = species_view_t(gp, name, None, box)
        out[f'snap/abs_{name}'] = _np(density_map_t(pos, w, dim, 0.0, box, True, ng))
        posr, wr = species_view_t(gp, name, c, box)
        out[f'snap/rel_{name}'] = _np(density_map_t(posr, wr, dim, -ext, ext, False, ng))
        if dim == 3:
            maps = projection_maps_t(pos, w, 0.0, box, True, n3)
            for a, b in _PROJ:
                out[f'snap/proj_{_AX[a]}{_AX[b]}_{name}'] = _np(maps[(a, b)])
            rho, nsl = slab_density_t(pos, w, 0.0, box, True, 0.5 * box, 0.05 * box, ng)
            out[f'snap/slab_{name}'], out[f'snap/slab_n_{name}'] = _np(rho), int(nsl)


def gpu_values(st: dict, dev, dtype: str):
    """(dict with the keys of cpu_values, seconds spent in the four rows on the device)."""
    ps, dim, box, c = st['ps'], st['dim'], st['box'], st['centre']
    gp = GPUParticles.from_host(ps, dev, dtype)
    phi_t = torch.as_tensor(st['phi'], device=dev).to(gp.dtype)
    phi_ext_t = None if st['phi_ext'] is None else torch.as_tensor(st['phi_ext'], device=dev).to(gp.dtype)
    kw = dict(centre=c, r_escape=st['r_escape'], wall=0.0, box=box, phi_ext=phi_ext_t)
    diagnostics_row_t(gp, phi_t, 0.0, 0, DT, 1.0, st['self_energy'], None, None, **kw)   # warm-up
    synchronize(dev)
    t0 = perf_counter()
    row0 = diagnostics_row_t(gp, phi_t, 0.0, 0, DT, 1.0, st['self_energy'], None, None, **kw)
    row1 = diagnostics_row_t(gp, phi_t, T1, 1, DT, A1, st['self_energy'], st['E0'], st['P0'], **kw)
    if st['kind'] == 'lss':
        a = (box, st['struct_grid'], dim, st['threshold'], st['min_cells'])
        s0, cur = lss_structure_row_t(gp, 0.0, *a, None)
        s1, _ = lss_structure_row_t(gp, T1, *a, cur)
    else:
        a = (c, st['rd'], st['rmax'], box, st['G'])
        b = (st['ring_grid'], st['threshold'], st['min_cells'])
        s0, sta = disk_structure_row_t(gp, 0.0, *a, None, *b)
        s1, sta = disk_structure_row_t(gp, T1, *a, sta, *b)
        s1['phase2_state'] = sta['phase2']
    synchronize(dev)
    t_rows = perf_counter() - t0
    out = {}
    for pre, row in (('diag0', row0), ('diag1', row1), ('struct0', s0), ('struct1', s1)):
        _put(out, pre, row)
    out['com/pos_box'] = _np(centre_of_mass_t(gp, gp.s > 0, box))
    out['com/neg_box'] = _np(centre_of_mass_t(gp, gp.s < 0, box))
    out['com/all_free'] = _np(centre_of_mass_t(gp))
    sm = {}
    for name, m in (('pos', gp.s > 0), ('neg', gp.s < 0)):
        rho = density_grid_t(gp.pos[m], gp.w[m], box, st['struct_grid'], dim, periodic=True)
        sm[name] = gaussian_filter_t(rho, 1.0, True)
        k, P = power_spectrum_t(sm[name], box)
        out[f'grid/rho_{name}'], out[f'grid/smooth_{name}'], out[f'grid/pk_{name}'] = _np(rho), _np(sm[name]), _np(P)
        out[f'grid/lambda_peak_{name}'] = peak_scale_t(k, P)
        out[f'grid/void_{name}'] = float(void_fraction_t(sm[name]))
    out['grid/segregation'] = float(segregation_index_t(sm['pos'], sm['neg']))
    out['grid/persistence'] = float(pattern_persistence_t(sm['neg'], sm['pos']))
    _gpu_maps(st, gp, out)
    return out, t_rows


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------

def _table(name: str, dtype: str, cpu: dict, gpu: dict, rel: dict, tol: float) -> None:
    print(f"--- {name} / {dtype} (tolerance {tol:g}) ---")
    print(f"{'key':<30s} {'cpu':>30s} {'gpu':>30s} {'rel diff':>10s}")
    for k in cpu:
        flag = '' if _fin(rel[k]) < tol else '  <-- above tolerance'
        print(f"{k:<30s} {_fmt(cpu[k]):>30s} {_fmt(gpu.get(k, 'missing')):>30s} {rel[k]:>10.2e}{flag}")


def run_diagtest(n: int = 200000, grid: int = 256, dtypes=('float64', 'float32'), seed: int = 1,
                 device=None, quiet: bool = False, grid3=None, snap_grid: int = 200) -> dict:
    """Build the three states, compare every CPU and GPU quantity, print the tables
    (unless quiet), the worst differences, the row timings, the verdicts and the
    memory report; returns {'states', 'rows', 'ok', 'verdicts', 'timings', ...}.
    grid3 (3-D PM / structure grid) defaults to grid // 4."""
    _need_torch()
    dev = require_gpu() if device is None else torch.device(device)
    dtypes = tuple(dtypes)
    grid3 = max(8, int(grid) // 4) if grid3 is None else int(grid3)

    def say(*lines):
        if not quiet:
            for ln in lines:
                print(ln, flush=True)

    say(f"=== diagtest: CPU (numpy float64) vs GPU ({dev}) diagnostics, structure and snapshot maps ===",
        f"device = {device_name() if dev.type == 'cuda' else dev}, torch {torch.__version__}, "
        f"N = {int(n)} per state, grids {grid}^2 / {grid3}^3, snapshot maps {snap_grid}^2",
        gpu_memory_report()['text'])
    t0 = perf_counter()
    states = [make_lss_state(n, 2, grid, seed, snap_grid), make_lss_state(n, 3, grid3, seed, snap_grid),
              make_galaxy_state(n, grid, seed, snap_grid)]
    say(f"states built in {perf_counter() - t0:.1f} s")
    rows, timings = [], []
    for st in states:
        cpu, t_cpu = cpu_values(st)
        tm = {'state': st['name'], 'cpu': t_cpu, 'gpu': {}}
        for dt in dtypes:
            gpu, t_gpu = gpu_values(st, dev, dt)
            tm['gpu'][dt] = t_gpu
            rel = compare(cpu, gpu, scales=st['scales'])
            tol = TOL_F64 if dt == 'float64' else TOL_F32
            worst_key = max(rel, key=lambda k: _fin(rel[k]))
            worst = _fin(rel[worst_key])
            over = [k for k in rel if _fin(rel[k]) >= tol]
            if not quiet:
                _table(st['name'], dt, cpu, gpu, rel, tol)
            print(f"  {st['name']} / {dt}: worst relative difference {worst:.2e} ({worst_key}); "
                  f"{len(over)} of {len(rel)} keys above {tol:g}; rows cpu {t_cpu:.3f} s, gpu {t_gpu:.3f} s",
                  flush=True)
            rows.append({'state': st['name'], 'dtype': dt, 'worst': worst, 'worst_key': worst_key,
                         'n_keys': len(rel), 'n_over': len(over), 'over': over[:30], 'tol': tol,
                         'cpu_time': t_cpu, 'gpu_time': t_gpu, 'rel': {k: float(v) for k, v in rel.items()}})
        timings.append(tm)
    say("--- timings of the diagnostics + structure rows (two rows each) ---")
    for tm in timings:
        say(f"  {tm['state']:<32s} cpu {tm['cpu']:8.3f} s" + "".join(
            f"   gpu {dt} {t:8.3f} s (x{tm['cpu'] / max(t, 1e-12):.1f})" for dt, t in tm['gpu'].items()))
    ok, verdicts = True, []
    for dt in dtypes:
        sel = [r for r in rows if r['dtype'] == dt]
        worst = max(r['worst'] for r in sel)
        bad = sorted({k for r in sel for k in r['over']})
        if dt == 'float64':
            if worst < TOL_F64:
                verdicts.append(f"float64 GPU diagnostics agree with the CPU to {worst:.1e} "
                                f"(tolerance {TOL_F64:g}): OK")
            else:
                ok = False
                verdicts.append(f"WARNING: float64 GPU/CPU discrepancy {worst:.1e} exceeds {TOL_F64:g} in "
                                f"{', '.join(bad[:8])}{' ...' if len(bad) > 8 else ''}: do not trust GPU "
                                "diagnostics until this is fixed")
        else:
            verdicts.append(f"float32 (informational): worst relative difference {worst:.1e} vs {TOL_F32:g}"
                            + (f"; above it in {', '.join(bad[:8])}{' ...' if len(bad) > 8 else ''}" if bad else ""))
    if 'float64' not in dtypes:
        verdicts.append("float64 not tested: no pass/fail verdict")
    print("--- verdict ---")
    for v in verdicts:
        print("  " + v)
    mem = gpu_memory_report()
    print(mem['text'])
    info = [{'name': s['name'], 'kind': s['kind'], 'dim': s['dim'], 'n': s['ps'].n,
             'struct_grid': s['struct_grid'], 'ring_grid': s['ring_grid'], 'snap_grid': s['snap_grid']}
            for s in states]
    return {'states': info, 'rows': rows, 'ok': ok, 'verdicts': verdicts, 'timings': timings,
            'device': str(dev), 'gpu_memory': mem}


if __name__ == '__main__':  # self-test with small sizes
    if torch is None:
        print("torch not installed, diagtest self-test skipped:", _TORCH_ERR)
        sys.exit(0)
    nan, inf = float('nan'), float('inf')
    r = compare({'a': 1.0, 'b': nan, 'c': np.array([1.0, nan]), 'd': 0.0, 'e': 2, 'f': np.array([2.0, 4.0])},
                {'a': 1.0 + 1e-12, 'b': nan, 'c': np.array([1.0 + 1e-13, nan]), 'd': 1e-15, 'e': 2,
                 'f': np.array([2.0, 4.0 + 4e-10])})
    assert 0.9e-12 < r['a'] < 1.2e-12 and r['b'] == 0.0 and 0.9e-13 < r['c'] < 1.2e-13
    assert abs(r['d'] - 1e-3) < 1e-9 and r['e'] == 0.0 and abs(r['f'] - 1e-10) < 1e-20
    assert compare({'x': 1.0}, {})['x'] == inf and compare({'x': nan}, {'x': 1.0})['x'] == inf
    assert compare({'x': np.zeros(3)}, {'x': np.zeros(2)})['x'] == inf
    assert compare({'q/p_abs_x': 1e-18}, {'q/p_abs_x': 2e-18}, scales={'p_abs_x': 1.0})['q/p_abs_x'] == 1e-18
    assert compare({'x': np.array([1.0, nan])}, {'x': np.array([nan, 1.0])})['x'] == inf
    try:
        require_gpu()
        dev = None
    except RuntimeError as exc:
        print(f"no GPU ({str(exc).splitlines()[0]}); running the diagtest on the torch CPU device")
        dev = 'cpu'
    out = run_diagtest(n=5000, grid=64, grid3=24, dtypes=('float64', 'float32'), seed=1, device=dev,
                       quiet=True)
    assert out['ok'], out['verdicts']
    assert len(out['states']) == 3 and len(out['rows']) == 6 and len(out['timings']) == 3
    assert [s['dim'] for s in out['states']] == [2, 3, 2] and out['states'][2]['kind'] == 'galaxy'
    f64 = [r for r in out['rows'] if r['dtype'] == 'float64']
    assert len(f64) == 3 and max(r['worst'] for r in f64) < TOL_F64, [(r['worst'], r['worst_key']) for r in f64]
    assert all(r['n_over'] == 0 for r in f64)
    for r in out['rows']:
        assert r['n_keys'] > 100 and math.isfinite(r['worst']) and set(r['rel']) >= {'diag1/dE_rel', 'grid/pk_neg'}
        assert abs(r['rel']['diag0/wall']) == 0.0 and r['cpu_time'] > 0 and r['gpu_time'] > 0
    assert any(k.startswith('snap/proj_xy') for k in f64[1]['rel']) and 'struct1/omega_p' in f64[2]['rel']
    print("diagtest self-test OK")
