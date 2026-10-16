"""sweeps.py -- parameter sweeps, seed ensembles and the fixed convergence suite.

Each variant runs in its own sub-folder (full output of scenarios.run_simulation),
its summary.json is reduced to a fixed list of metrics (flatten_summary), the
rows are written to CSV and compared in one figure.  Nothing here selects,
filters or re-tunes: every requested value is run and reported, failed runs
appear as rows with status='failed' and the error text, and the only verdict
computed here is the pre-registered 'reproducible' criterion of the ensemble
(seed-to-seed std of n_clumps_neg < 20 % of the mean) plus a plain
"max relative change < CONVERGENCE_CHANGE" flag per parameter of the
convergence suite, both with their thresholds written next to the numbers.
Sub-folders that already contain a summary.json for an identical resolved
configuration are reused (restartable long sweeps).
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np

from config import (RunConfig, SWEEPABLE, parse_param_value, resolve_defaults, derive,
                    cfg_with, cfg_to_dict)
from scenarios import run_simulation
from diagnostics_core import CsvLogger, write_json
from plotting import plot_sweep, plot_ensemble

SUMMARY_METRICS_LSS = ['n_clumps_neg_final', 'n_clumps_change_5_20', 'segregation_final',
                       'persistence_final', 'phi_over_d', 'lambda_peak_pos', 'dE_rel_max',
                       'dP_abs_rel_max', 'wall_time']
SUMMARY_METRICS_GALAXY = ['spiral_A2_mean_late', 'bar_A2_final', 'rotations_with_spiral',
                          'Lz_disk_change', 'Lz_twin_change', 'n_clumps_twin_final',
                          'pitch_deg_mean', 'dE_rel_max', 'wall_time']
VERDICT_KEYS = {'lss': ['cells_form', 'stable_clumps', 'persistence'],
                'galaxy': ['sustained_spiral', 'bar', 'braking', 'ring_fragmentation',
                           'vc_undefined_fraction']}
# pre-registered thresholds (notes, section 3) -- fixed, never adjusted by the code
ENSEMBLE_CV_THRESHOLD = 0.20     # 'reproducible': std/mean of n_clumps_neg_final (LSS)
ENSEMBLE_CV_BAND = 0.05          # ambiguous (within noise) half-width around the threshold
CONVERGENCE_CHANGE = 0.20        # 'converged' flag: max |m - m_ref|/|m_ref| below this
CONVERGENCE_FACTORS = {'dt': [1.0, 0.5, 0.25], 'eps': [2.0, 1.0, 0.5], 'grid': [0.5, 1.0, 2.0]}
CONVERGENCE_N = 2000             # direct-vs-PM comparison at N = 2000 + 2000
_RELATIVE = ('dt', 'eps', 'grid')   # convergence-plan values are multipliers of the base


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _say(quiet: bool, msg: str) -> None:
    if not quiet:
        print("[sweep] " + msg, file=sys.stderr, flush=True)


def _f(v) -> float:
    try:
        if isinstance(v, (bool, np.bool_)) or v is None:
            return float('nan')
        x = float(v)
        return x if np.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _slug(v) -> str:
    s = str(v)
    for a, b in (('.', 'p'), ('-', 'm'), ('/', '_'), (' ', '')):
        s = s.replace(a, b)
    return s


def metrics_for(scenario: str) -> list:
    if scenario == 'lss':
        return list(SUMMARY_METRICS_LSS)
    if scenario == 'galaxy':
        return list(SUMMARY_METRICS_GALAXY)
    raise ValueError(f"unknown scenario {scenario!r}")


def flatten_summary(summary: dict, scenario: str) -> dict:
    """Fixed metric list (from summary['metrics'], falling back to the flat copy)
    plus the verdict strings of the pre-registered criteria."""
    met = summary.get('metrics') or {}
    out = {m: _f(met.get(m, summary.get(m))) for m in metrics_for(scenario)}
    crit = summary.get('criteria') or {}
    for k in VERDICT_KEYS[scenario]:
        v = crit.get(k)
        out['verdict_' + k] = str(v.get('verdict', '')) if isinstance(v, dict) else ''
    return out


def _same_config(stored: dict, cfg: RunConfig) -> bool:
    try:
        want = cfg_to_dict(resolve_defaults(cfg))
    except Exception:
        return False
    skip = {'outdir', 'label', 'quiet'}
    return all(stored.get(k) == v for k, v in want.items() if k not in skip)


def _run_one(cfg: RunConfig, quiet: bool):
    """(summary or None, error string or '').  Reuses an existing summary.json
    when the stored resolved configuration is identical."""
    path = os.path.join(cfg.outdir, 'summary.json')
    if os.path.exists(path):
        try:
            with open(path) as fh:
                old = json.load(fh)
            if _same_config(old.get('config') or {}, cfg):
                _say(quiet, f"reusing {path}")
                return old, ''
        except Exception:
            pass
    try:
        return run_simulation(cfg), ''
    except Exception as exc:            # report the failure as a row; never hide it
        _say(quiet, f"run in {cfg.outdir} FAILED: {exc!r}")
        if not quiet:
            traceback.print_exc()
        return None, repr(exc)


def _row(scenario: str, head: dict, summary, error: str) -> dict:
    row = dict(head)
    row['status'] = 'ok' if summary is not None else 'failed'
    row['error'] = error
    if summary is not None:
        row.update(flatten_summary(summary, scenario))
        row['outdir'] = summary.get('outdir', row.get('outdir', ''))
    else:
        row.update({m: float('nan') for m in metrics_for(scenario)})
        row.update({'verdict_' + k: '' for k in VERDICT_KEYS[scenario]})
    return row


def _write_rows(rows: list, path: str) -> None:
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    lg = CsvLogger(path)
    for r in rows:
        lg.log({c: r.get(c, '') for c in cols})
    lg.close()


def _stats(rows: list, metrics: list) -> dict:
    """Nan-aware mean / std / std-over-mean / count per metric (all values kept)."""
    out = {}
    for m in metrics:
        y = np.array([_f(r.get(m)) for r in rows], dtype=np.float64)
        ok = np.isfinite(y)
        n = int(ok.sum())
        mu = float(np.mean(y[ok])) if n else float('nan')
        sd = float(np.std(y[ok])) if n > 1 else (0.0 if n == 1 else float('nan'))
        out[m] = {'mean': mu, 'std': sd, 'n': n,
                  'std_over_mean': (sd / abs(mu)) if (n and mu != 0) else float('nan')}
    return out


def _logx(values: list) -> bool:
    x = [_f(v) for v in values]
    if any(not np.isfinite(v) or v <= 0 for v in x) or len(x) < 2:
        return False
    return max(x) / min(x) >= 8.0


def _plot(quiet, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except Exception as exc:
        _say(quiet, f"plot {fn.__name__} failed: {exc!r}")


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

def parse_values(param: str, values: str) -> list:
    """'a,b,c' -> typed list according to SWEEPABLE[param] (bool-safe)."""
    if param not in SWEEPABLE:
        raise ValueError(f"'{param}' is not sweepable; choose from {sorted(SWEEPABLE)}")
    toks = [t.strip() for t in str(values).split(',') if t.strip()]
    if not toks:
        raise ValueError("no values given")
    return [parse_param_value(param, t) for t in toks]


def run_sweep(base: RunConfig, param: str, values: list, outdir: str) -> list:
    """One run per value in outdir/<param>_<value>/; writes sweep_<param>.csv and
    sweep_<param>.png.  Every value is run and reported; failed runs are rows
    with status='failed'."""
    if param not in SWEEPABLE:
        raise ValueError(f"'{param}' is not sweepable")
    os.makedirs(outdir, exist_ok=True)
    quiet = base.quiet
    csv_path = os.path.join(outdir, f'sweep_{param}.csv')
    rows = []
    for i, v in enumerate(values):
        sub = os.path.join(outdir, f"{param}_{_slug(v)}")
        head = {param: v, 'seed': base.seed, 'outdir': sub}
        _say(quiet, f"sweep {param} = {v}  ({i + 1}/{len(values)}) -> {sub}")
        try:
            cfg = cfg_with(base, **{param: v, 'outdir': sub, 'label': f"{param}={v}"})
            summary, err = _run_one(cfg, quiet)
        except Exception as exc:
            summary, err = None, repr(exc)
        rows.append(_row(base.scenario, head, summary, err))
        _write_rows(rows, csv_path)
    _plot(quiet, plot_sweep, rows, param, metrics_for(base.scenario),
          os.path.join(outdir, f'sweep_{param}.png'), logx=_logx(values))
    write_json(os.path.join(outdir, f'sweep_{param}.json'),
               {'param': param, 'values': values, 'rows': rows,
                'base_config': cfg_to_dict(base), 'note': 'all values reported, none selected'})
    _say(quiet, f"sweep over {param} done: {sum(r['status'] == 'ok' for r in rows)}/{len(rows)} runs ok")
    return rows


# ---------------------------------------------------------------------------
# ensembles
# ---------------------------------------------------------------------------

def run_ensemble(base: RunConfig, seeds: list, outdir: str) -> dict:
    """Same configuration for every seed in outdir/seed_<k>/; ensemble.csv holds
    the per-seed rows followed by mean / std / std_over_mean rows, ensemble.png
    the scatter, ensemble.json the statistics and the pre-registered
    'reproducible' verdict (LSS: std/mean of n_clumps_neg_final < 0.20)."""
    os.makedirs(outdir, exist_ok=True)
    quiet = base.quiet
    scenario = base.scenario
    metrics = metrics_for(scenario)
    rows = []
    csv_path = os.path.join(outdir, 'ensemble.csv')
    for i, s in enumerate(seeds):
        sub = os.path.join(outdir, f'seed_{int(s)}')
        _say(quiet, f"ensemble seed {s}  ({i + 1}/{len(seeds)}) -> {sub}")
        try:
            cfg = cfg_with(base, seed=int(s), outdir=sub, label=f"seed={s}")
            summary, err = _run_one(cfg, quiet)
        except Exception as exc:
            summary, err = None, repr(exc)
        rows.append(_row(scenario, {'seed': int(s), 'outdir': sub}, summary, err))
        _write_rows(rows, csv_path)
    stats = _stats(rows, metrics)
    table = list(rows)
    for key in ('mean', 'std', 'std_over_mean'):
        table.append({'seed': key, 'outdir': '', 'status': 'summary', 'error': '',
                      **{m: stats[m][key] for m in metrics}})
    _write_rows(table, csv_path)
    _plot(quiet, plot_ensemble, rows, metrics, os.path.join(outdir, 'ensemble.png'))
    # pre-registered reproducibility criterion (LSS clump count); reported, not tuned
    key = 'n_clumps_neg_final' if scenario == 'lss' else 'spiral_A2_mean_late'
    cv, n_ok = stats[key]['std_over_mean'], stats[key]['n']
    if scenario != 'lss':
        verdict = (f"not pre-registered for the galaxy (std/mean of {key} = {cv:.3g} "
                   f"reported for information)")
    elif n_ok < 2:
        verdict = "ambiguous (fewer than 2 successful seeds)"
    elif not np.isfinite(cv):
        verdict = "ambiguous (not measurable)"
    elif cv < ENSEMBLE_CV_THRESHOLD - ENSEMBLE_CV_BAND:
        verdict = "met"
    elif cv > ENSEMBLE_CV_THRESHOLD + ENSEMBLE_CV_BAND:
        verdict = "not met"
    else:
        verdict = "ambiguous (within noise)"
    result = {'scenario': scenario, 'seeds': [int(s) for s in seeds], 'n_ok': int(sum(r['status'] == 'ok' for r in rows)),
              'rows': rows, 'stats': stats,
              'reproducible': {'metric': key, 'std_over_mean': cv, 'threshold': ENSEMBLE_CV_THRESHOLD,
                               'band': ENSEMBLE_CV_BAND, 'n_seeds_used': n_ok, 'verdict': verdict},
              'verdict_counts': {k: _count_verdicts(rows, 'verdict_' + k) for k in VERDICT_KEYS[scenario]},
              'base_config': cfg_to_dict(base)}
    write_json(os.path.join(outdir, 'ensemble.json'), result)
    _say(quiet, f"ensemble done: {key} = {stats[key]['mean']:.4g} +- {stats[key]['std']:.3g} "
                f"(std/mean = {cv:.3g}); reproducible: {verdict}")
    return result


def _count_verdicts(rows: list, key: str) -> dict:
    out = {}
    for r in rows:
        v = r.get(key, '') or ('(failed)' if r.get('status') == 'failed' else '')
        out[v] = out.get(v, 0) + 1
    return out


# ---------------------------------------------------------------------------
# convergence suite
# ---------------------------------------------------------------------------

def convergence_plan(scenario: str) -> list:
    """Fixed plan: ('dt', [1, 1/2, 1/4]), ('eps', [2, 1, 1/2]), ('grid', [1/2, 1, 2])
    as multipliers of the base run's values (grid: PM only), ('method',
    ['direct', 'pm']) at N = 2000 + 2000, and for the galaxy ('bc',
    ['periodic', 'isolated'])."""
    plan = [(p, list(CONVERGENCE_FACTORS[p])) for p in _RELATIVE]
    plan.append(('method', ['direct', 'pm']))
    if scenario == 'galaxy':
        plan.append(('bc', ['periodic', 'isolated']))
    return plan


def _variant(base: RunConfig, res: RunConfig, der, param: str, value, sub: str):
    """Config for one convergence variant and the absolute value it represents."""
    label = f"{param}={value}"
    if param == 'dt':
        return cfg_with(base, dt=float(value) * res.dt, outdir=sub, label=label), float(value) * res.dt
    if param == 'eps':
        if float(value) == 1.0:
            return cfg_with(base, outdir=sub, label=label), der.eps_abs
        return cfg_with(base, eps=float(value) * der.eps_abs, outdir=sub, label=label), float(value) * der.eps_abs
    if param == 'grid':
        g = max(8, int(round(float(value) * res.grid)))
        return cfg_with(base, grid=g, method='pm', outdir=sub, label=label), g
    if param == 'method':
        return cfg_with(base, n=CONVERGENCE_N, method=str(value), outdir=sub, label=label), str(value)
    if param == 'bc':
        return cfg_with(base, bc=str(value), outdir=sub, label=label), str(value)
    raise ValueError(f"convergence plan does not know {param!r}")


def run_convergence(base: RunConfig, outdir: str) -> dict:
    """Runs the fixed plan around the base run (reference in outdir/reference/),
    writes convergence.csv (all rows), convergence_<param>.png per parameter
    and convergence.json with the max relative change of every metric with
    respect to the reference (flag 'converged' = change < CONVERGENCE_CHANGE)."""
    os.makedirs(outdir, exist_ok=True)
    quiet = base.quiet
    scenario = base.scenario
    metrics = metrics_for(scenario)
    res = resolve_defaults(base)
    der = derive(base)
    plan = convergence_plan(scenario)
    _say(quiet, f"convergence suite: reference run (method={res.method}, dt={res.dt:g}, "
                f"eps_abs={der.eps_abs:.4g}, grid={res.grid}) -> {outdir}/reference")
    ref_cfg = cfg_with(base, outdir=os.path.join(outdir, 'reference'), label='reference')
    ref_summary, ref_err = _run_one(ref_cfg, quiet)
    ref_row = _row(scenario, {'param': 'reference', 'value': '', 'abs_value': '', 'seed': base.seed,
                              'outdir': ref_cfg.outdir}, ref_summary, ref_err)
    rows, spread, skipped = [ref_row], {}, []
    csv_path = os.path.join(outdir, 'convergence.csv')
    for param, values in plan:
        if param == 'grid' and res.method != 'pm':
            skipped.append({'param': param, 'reason': f"base method is {res.method}: no grid to vary"})
            _say(quiet, "grid: skipped (base run is direct summation)")
            continue
        prow = []
        for v in values:
            sub = os.path.join(outdir, f"{param}_{_slug(v)}")
            head = {'param': param, 'value': v, 'abs_value': '', 'seed': base.seed, 'outdir': sub}
            same_as_ref = ((param in _RELATIVE and float(v) == 1.0) or
                           (param == 'bc' and str(v) == res.bc))
            if same_as_ref and ref_summary is not None:
                head['outdir'] = ref_cfg.outdir
                head['abs_value'] = {'dt': res.dt, 'eps': der.eps_abs, 'grid': res.grid, 'bc': res.bc}[param]
                prow.append(_row(scenario, head, ref_summary, ''))
                continue
            _say(quiet, f"convergence {param} = {v} -> {sub}")
            try:
                cfg, absval = _variant(base, res, der, param, v, sub)
                head['abs_value'] = absval
                summary, err = _run_one(cfg, quiet)
            except Exception as exc:
                summary, err = None, repr(exc)
            prow.append(_row(scenario, head, summary, err))
            rows_all = rows + prow
            _write_rows(rows_all, csv_path)
        rows.extend(prow)
        spread[param] = _spread(prow, ref_row, metrics)
        _plot(quiet, plot_sweep, prow, 'value', metrics,
              os.path.join(outdir, f'convergence_{param}.png'), logx=False)
    _write_rows(rows, csv_path)
    result = {'scenario': scenario, 'plan': [(p, v) for p, v in plan], 'skipped': skipped,
              'reference': ref_row, 'rows': rows, 'spread': spread,
              'converged_threshold': CONVERGENCE_CHANGE,
              'note': ("'converged' flags max |m - m_ref| / |m_ref| < threshold per parameter; the "
                       "method comparison is made at N = %d + %d, so its reference is the 'pm' row, "
                       "and the bc comparison tests the tiling-artefact attribution (ring "
                       "fragmentation present with periodic, absent with isolated)." % (CONVERGENCE_N, CONVERGENCE_N)),
              'base_config': cfg_to_dict(res), 'derived_reference': {'dt_abs': der.dt_abs, 'eps_abs': der.eps_abs,
                                                                     'grid': der.grid, 'method': der.method}}
    write_json(os.path.join(outdir, 'convergence.json'), result)
    for p, sp in spread.items():
        worst = max(((m, d['max_rel_change']) for m, d in sp.items() if np.isfinite(d['max_rel_change'])),
                    key=lambda t: t[1], default=('-', float('nan')))
        _say(quiet, f"{p}: largest relative change {worst[1]:.3g} in {worst[0]}; "
                    f"{sum(d['converged'] is True for d in sp.values())}/{len(sp)} metrics below {CONVERGENCE_CHANGE:g}")
    return result


def _spread(prow: list, ref_row: dict, metrics: list) -> dict:
    """Per metric: values across the variants, max |change| relative to the
    reference row of this parameter set (the 'pm' row for the method set,
    otherwise the global reference) and the converged flag."""
    out = {}
    pm_rows = [r for r in prow if r.get('param') == 'method' and r.get('value') == 'pm']
    ref = pm_rows[0] if pm_rows else ref_row
    for m in metrics:
        vals = [_f(r.get(m)) for r in prow]
        r0 = _f(ref.get(m))
        fin = [v for v in vals if np.isfinite(v)]
        if m == 'wall_time' or not fin or not np.isfinite(r0):
            change = float('nan')
        elif r0 == 0:
            change = float(max(abs(v) for v in fin))
        else:
            change = float(max(abs(v - r0) for v in fin) / abs(r0))
        out[m] = {'values': vals, 'reference': r0, 'max_rel_change': change,
                  'converged': (bool(change < CONVERGENCE_CHANGE) if np.isfinite(change) else None)}
    return out


if __name__ == '__main__':  # minimal smoke test (tiny runs)
    import tempfile
    assert parse_values('mass_ratio', '1, 4,16') == [1.0, 4.0, 16.0]
    assert parse_values('ic', 'poisson,glass') == ['poisson', 'glass']
    assert parse_values('expansion', 'true,0') == [True, False]
    assert convergence_plan('galaxy')[-1][0] == 'bc' and len(convergence_plan('lss')) == 4
    with tempfile.TemporaryDirectory() as td:
        base = RunConfig('lss', npos=200, nneg=200, method='pm', grid=32, tmax=0.6, snap_every=0.3,
                         structure_grid=32, quiet=True)
        rows = run_sweep(base, 'mass_ratio', [4.0, 64.0], os.path.join(td, 'sw'))
        assert len(rows) == 2 and os.path.exists(os.path.join(td, 'sw', 'sweep_mass_ratio.png'))
        assert all(r['status'] == 'ok' for r in rows) and 'verdict_cells_form' in rows[0]
        ens = run_ensemble(base, [1, 2], os.path.join(td, 'en'))
        assert ens['n_ok'] == 2 and 'verdict' in ens['reproducible']
        rows2 = run_sweep(base, 'ic', ['poisson', 'nonsense'], os.path.join(td, 'sw2'))
        assert rows2[1]['status'] == 'failed' and rows2[0]['status'] == 'ok'
        base_g = RunConfig('galaxy', ndisk=300, ntwin=200, method='pm', grid=32, tmax=0.2, snap_every=0.1,
                           structure_every=0.1, structure_grid=32, quiet=True)
        fl = flatten_summary(run_simulation(cfg_with(base_g, outdir=os.path.join(td, 'g'))), 'galaxy')
        assert set(SUMMARY_METRICS_GALAXY) <= set(fl) and 'verdict_braking' in fl
    print("sweeps smoke test OK")
