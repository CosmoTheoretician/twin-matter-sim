#!/usr/bin/env python3
"""main.py -- command-line entry point of the twin-matter N-body reproduction.

    python main.py lss [options]            faithful LSS run (defaults = the 1992 set-up)
    python main.py galaxy [options]         disk + live twin ring on a torus
    python main.py sweep {lss|galaxy} --param P --values a,b,c
    python main.py ensemble {lss|galaxy} --seeds K
    python main.py convergence {lss|galaxy}
    python main.py assumptions {lss|galaxy} [options]   (print the assumption list only)

Every departure from the original set-up is an explicit flag; the defaults are
the faithful reproduction.  Before dispatching, the resolved configuration,
the assumption list (A1..A17) and runtime warnings (direct summation at large N,
sparse 3-D runs, ring overlapping its periodic images, ...) are printed.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import fields

from config import (RunConfig, SWEEPABLE, resolve_defaults, derive, assumptions,
                    cfg_to_dict, cfg_with)
from scenarios import run_simulation
from sweeps import (parse_values, run_sweep, run_ensemble, run_convergence,
                    convergence_plan, metrics_for)

SCENARIOS = ('lss', 'galaxy')
_DEFAULTS = {f.name: f.default for f in fields(RunConfig)}
_DIRECT_WARN_SECONDS = 120.0


# ---------------------------------------------------------------------------
# option groups
# ---------------------------------------------------------------------------

def add_common_options(p) -> None:
    g = p.add_argument_group('common options')
    g.add_argument('--outdir', default='', help='output folder (default out/<scenario>_<timestamp>)')
    g.add_argument('--seed', type=int, default=1, help='random seed (default 1)')
    g.add_argument('--dim', type=int, choices=(2, 3), default=2, help='2 (original) or 3 (A16)')
    g.add_argument('--method', choices=('auto', 'direct', 'pm'), default='auto',
                   help='auto = direct if N_total <= 4000 else pm; direct is the original O(N^2) algorithm')
    g.add_argument('--bc', choices=('periodic', 'isolated'), default='periodic',
                   help='periodic torus (original) or isolated box (tiling-artefact test, A2/A13)')
    g.add_argument('--force-law', choices=('r2', 'log'), default='r2',
                   help='1/r^2 sheet (original reading) or true 2-D logarithmic gravity (A1)')
    g.add_argument('--soft', choices=('plummer', 'cutin'), default='plummer',
                   help="direct softening: Plummer or the pages' cut-in (A8)")
    g.add_argument('--eps', type=float, default=-1.0,
                   help='softening: direct 0.002 L (LSS) / 0.05 R_d (galaxy); PM 0 = cell size')
    g.add_argument('--grid', type=int, default=-1, help='PM grid per side (256 in 2-D, 64 in 3-D)')
    g.add_argument('--dt', type=float, default=-1.0,
                   help='timestep as fraction of the reference time (0.02 LSS, 0.005 galaxy; A9)')
    g.add_argument('--adaptive', action='store_true', help='global adaptive step eta*sqrt(eps/a_max)')
    g.add_argument('--eta', type=float, default=0.1, help='adaptive-step accuracy parameter')
    g.add_argument('--tmax', type=float, default=-1.0,
                   help='run length in reference times (20 t_J,neg / 10 rotations; A10)')
    g.add_argument('--snap-every', type=float, default=-1.0, help='snapshot cadence (reference times)')
    g.add_argument('--log-every', type=int, default=10, help='diagnostics cadence (steps)')
    g.add_argument('--structure-every', type=float, default=0.5, help='structure cadence (reference times)')
    g.add_argument('--dtype', choices=('float64', 'float32'), default='float64')
    g.add_argument('--direct-chunk', type=int, default=1024, help='target chunk of the direct sum')
    g.add_argument('--label', default='', help='free label written to summary.json')
    g.add_argument('--quiet', action='store_true')


def add_lss_options(p) -> None:
    g = p.add_argument_group('LSS options (Landsheat 1992 set-up)')
    g.add_argument('--npos', type=int, default=5000, help='ordinary-matter particles')
    g.add_argument('--nneg', type=int, default=5000, help='twin-matter particles')
    g.add_argument('--mass-ratio', type=float, default=64.0, help='|M_-| / M_+ (density ratio, A6)')
    g.add_argument('--vel-ratio', type=float, default=4.0, help='sigma_- / sigma_+ (A5)')
    g.add_argument('--lambda-j-neg', type=float, default=0.125,
                   help='lambda_J(negative) / L: sets the absolute velocity scale (A4)')
    g.add_argument('--ic', choices=('poisson', 'glass', 'lattice'), default='poisson',
                   help='initial positions (A7)')
    g.add_argument('--ic-noise', type=float, default=0.0, help='jitter for glass/lattice (mean spacings)')
    g.add_argument('--neg-only', action='store_true', help='control: twin population alone')
    g.add_argument('--pos-only', action='store_true', help='control: ordinary population alone')
    g.add_argument('--expansion', action='store_true', help='kinematic comoving frame (A15)')
    g.add_argument('--a-index', type=float, default=0.6667, help='a(t) ~ t^n exponent')
    g.add_argument('--t0-exp', type=float, default=1.0, help='expansion start (t_J,neg)')
    g.add_argument('--structure-grid', type=int, default=128, help='density grid for clump statistics')
    g.add_argument('--clump-threshold', type=float, default=5.0, help='clump threshold (x mean density)')
    g.add_argument('--clump-min-cells', type=int, default=4)


def add_galaxy_options(p) -> None:
    g = p.add_argument_group('galaxy options (Petit-Landsheat 1992 set-up)')
    g.add_argument('--ndisk', type=int, default=5000)
    g.add_argument('--ntwin', type=int, default=5000)
    g.add_argument('--rd', type=float, default=1.0, help='disk scale length (unit)')
    g.add_argument('--rmax', type=float, default=4.0, help='disk truncation radius (R_d)')
    g.add_argument('--q', type=float, default=1.5, help='Toomre Q of the disk (A11)')
    g.add_argument('--zscale', type=float, default=0.1, help='disk thickness (R_d, 3-D only)')
    g.add_argument('--twin-ratio', type=float, default=1.0, help='|M_twin| / M_disk (A12)')
    g.add_argument('--twin-rin', type=float, default=4.5, help='ring inner radius (R_d)')
    g.add_argument('--twin-rout', type=float, default=7.0, help='ring outer radius (R_d)')
    g.add_argument('--twin-sigma', type=float, default=0.5, help='ring dispersion / v_c,ref')
    g.add_argument('--twin-rot', type=float, default=0.0, help='ring rotation fraction of v_c,ref')
    g.add_argument('--control', choices=('live', 'none', 'rigid-repulsive', 'rigid-halo'), default='live',
                   help='live twin ring (original) or rigid control fields')
    g.add_argument('--halo-vc', type=float, default=1.0)
    g.add_argument('--halo-rc', type=float, default=1.0)
    g.add_argument('--box', type=float, default=-1.0, help='box side (R_d; default 16, A13)')


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='main.py', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', metavar='command')
    sub.required = True
    s = sub.add_parser('lss', help='LSS two-population box')
    add_common_options(s)
    add_lss_options(s)
    s = sub.add_parser('galaxy', help='disk + twin ring')
    add_common_options(s)
    add_galaxy_options(s)
    for name, help_ in (('sweep', 'one run per parameter value'),
                        ('ensemble', 'same run over several seeds'),
                        ('convergence', 'fixed dt/eps/grid/method/bc suite'),
                        ('assumptions', 'print the assumption list, do not run')):
        s = sub.add_parser(name, help=help_)
        s.add_argument('scenario', choices=SCENARIOS)
        if name == 'sweep':
            s.add_argument('--param', required=True, choices=sorted(SWEEPABLE), metavar='P',
                           help='one of ' + ', '.join(sorted(SWEEPABLE)))
            s.add_argument('--values', required=True, help='comma-separated values')
        if name == 'ensemble':
            s.add_argument('--seeds', type=int, default=4, help='number of seeds (seed, seed+1, ...)')
        add_common_options(s)
        add_lss_options(s)
        add_galaxy_options(s)
    return p


def config_from_args(args, scenario: str) -> RunConfig:
    """RunConfig from the namespace; options absent for this subcommand take the dataclass default."""
    kw = {'scenario': scenario}
    for name, default in _DEFAULTS.items():
        if name != 'scenario':
            kw[name] = getattr(args, name, default)
    return RunConfig(**kw)


# ---------------------------------------------------------------------------
# warnings and printing
# ---------------------------------------------------------------------------

def _n_total(c: RunConfig) -> int:
    if c.scenario == 'lss':
        return (0 if c.neg_only else c.npos) + (0 if c.pos_only else c.nneg)
    return c.ndisk + (c.ntwin if c.control == 'live' else 0)


def _direct_seconds(n: int, dim: int, dtype: str) -> float:
    """Rough numpy single-core estimate of one direct force evaluation (~3 s at N = 10^4, 2-D, float64)."""
    return 3.0e-8 * float(n) ** 2 * (1.0 if dtype == 'float64' else 0.55) * (1.0 if dim == 2 else 1.4)


def _fmt_time(sec: float) -> str:
    if sec < 90:
        return f'{sec:.0f} s'
    if sec < 5400:
        return f'{sec / 60:.0f} min'
    return f'{sec / 3600:.1f} h'


def runtime_warnings(cfg: RunConfig, der) -> list:
    """Human-readable warnings for a resolved configuration (never fatal)."""
    w = []
    n = _n_total(cfg)
    lss = cfg.scenario == 'lss'
    if der.method == 'direct':
        per = _direct_seconds(n, cfg.dim, cfg.dtype)
        tot = per * (der.n_steps + 1)
        if tot > _DIRECT_WARN_SECONDS:
            w.append(f"direct summation with N = {n}: ~{per:.2g} s per force evaluation, "
                     f"~{_fmt_time(tot)} for {der.n_steps} steps on one core"
                     + ("" if cfg.dtype == 'float32' else "; --dtype float32 roughly halves it")
                     + "; --method pm gives the same physics at ~10 ms/step.")
    elif cfg.method == 'auto' and lss and n > 4000:
        w.append(f"method auto -> pm for N = {n}; the original O(N^2) minimum-image algorithm is "
                 "--method direct (slow but faithful).")
    if cfg.dim == 3 and n < 50000:
        w.append(f"{n} particles in 3-D are very sparse (A16): sweep --n before concluding anything.")
    if cfg.adaptive:
        w.append("adaptive global step: a stand-in for the original back-up scheme (A9), not identical to it.")
    if der.method == 'pm' and 0 < cfg.eps < cfg.box / cfg.grid:
        w.append(f"PM eps = {cfg.eps:g} is below the cell size {cfg.box / cfg.grid:.4g}: the cell dominates.")
    if lss:
        if cfg.bc == 'isolated':
            w.append("LSS with isolated boundaries: the net monopole (|M_-| = 64 M_+) is retained and the "
                     "uniform box is not an equilibrium; this is a stress test, not the original set-up.")
        if cfg.expansion:
            w.append("expansion is a kinematic a(t) only (A15), not a Janus cosmology.")
        if cfg.neg_only or cfg.pos_only:
            w.append("single-population control: no segregation index, cell criteria are not applicable.")
        if cfg.ic in ('glass', 'lattice') and cfg.ic_noise == 0.0:
            w.append(f"{cfg.ic} initial positions with zero noise: structure can only be seeded by round-off; "
                     "compare with --ic-noise > 0.")
    else:
        rout = cfg.twin_rout * cfg.rd
        if cfg.control in ('live', 'rigid-repulsive') and cfg.bc == 'periodic' and rout > 0.5 * cfg.box:
            w.append(f"twin ring outer radius {rout:g} exceeds half the box {0.5 * cfg.box:g}: the ring "
                     "overlaps its periodic images (rigid-repulsive will refuse; live runs are artefact-dominated).")
        if cfg.control == 'live' and cfg.twin_ratio <= 0:
            w.append("control = live with twin_ratio = 0: no twin particles, equivalent to --control none.")
        if cfg.expansion:
            w.append("expansion with the galaxy scenario is not part of the original set-up.")
        if cfg.dim == 3 and cfg.control == 'rigid-halo':
            w.append("rigid halo is spherical in 3-D (logarithmic potential in r).")
    return w


def _print(quiet: bool, *lines) -> None:
    if not quiet:
        for ln in lines:
            print(ln)


def print_run_header(cfg: RunConfig, der, quiet: bool, title: str) -> list:
    """Print resolved settings, the assumption list and warnings; returns the warnings."""
    warns = runtime_warnings(cfg, der)
    _print(quiet, f"=== {title} ===",
           f"scenario = {cfg.scenario}, dim = {cfg.dim}, method = {der.method}, bc = {cfg.bc}, "
           f"law = {cfg.force_law}, N = {_n_total(cfg)}, seed = {cfg.seed}",
           f"t_ref = {der.t_ref:.4g} ({der.t_ref_name}), dt = {cfg.dt:g} t_ref = {der.dt_abs:.4g}, "
           f"tmax = {cfg.tmax:g} t_ref, {der.n_steps} steps, eps_eff = {der.eps_abs:.4g}, grid = {der.grid}",
           "--- assumptions in force (also written to assumptions.txt) ---")
    for a in assumptions(cfg, der):
        _print(quiet, "  " + a)
    if warns:
        print("--- runtime warnings ---")
        for x in warns:
            print("  WARNING: " + x)
    return warns


def _default_outdir(command: str, scenario: str) -> str:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    name = f"{scenario}_{stamp}" if command in SCENARIOS else f"{command}_{scenario}_{stamp}"
    return os.path.join('out', name)


def _print_verdicts(summary: dict) -> None:
    crit = summary.get('criteria') or {}
    print("--- pre-registered criteria ---")
    for k, v in crit.items():
        if isinstance(v, dict) and 'verdict' in v:
            print(f"  {k:24s}: {v['verdict']}")
    met = summary.get('metrics') or {}
    keys = [m for m in metrics_for(summary.get('scenario', 'lss')) if m in met]
    print("--- metrics ---")
    for m in keys:
        v = met[m]
        print(f"  {m:24s}: {v:.4g}" if isinstance(v, (int, float)) else f"  {m:24s}: {v}")
    print(f"outputs in {summary.get('outdir', '')}")


def _print_rows(rows: list, key: str, metrics: list) -> None:
    cols = [key] + metrics + ['status']
    widths = {c: max(len(c), 10) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, '')
            s = f"{v:.4g}" if isinstance(v, float) else str(v)
            cells.append(s.ljust(widths[c]))
        print("  ".join(cells))


def _variant_warnings(base: RunConfig, param: str, values: list) -> list:
    """Union of warnings over the sweep values (values that fail validation are reported as such)."""
    seen, out = set(), []
    for v in values:
        try:
            c = resolve_defaults(cfg_with(base, **{param: v}))
            ws = runtime_warnings(c, derive(c))
        except Exception as exc:
            ws = [f"{param} = {v!r} is not a valid configuration ({exc}); it will be reported as failed."]
        for x in ws:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cmd = args.command
    scenario = cmd if cmd in SCENARIOS else args.scenario
    cfg = config_from_args(args, scenario)
    quiet = cfg.quiet
    try:
        res = resolve_defaults(cfg)
        der = derive(res)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        if cmd == 'assumptions':
            print_run_header(res, der, False, f"assumptions for '{scenario}' with the given options")
            print("--- resolved configuration ---")
            for k, v in cfg_to_dict(res).items():
                print(f"  {k} = {v}")
            return 0
        if cmd in SCENARIOS:
            if not cfg.outdir:
                cfg = cfg_with(cfg, outdir=_default_outdir(cmd, scenario))
            os.makedirs(cfg.outdir, exist_ok=True)
            print_run_header(res, der, quiet, f"{scenario} run -> {cfg.outdir}")
            summary = run_simulation(cfg)
            _print_verdicts(summary)
            return 0
        outdir = cfg.outdir or _default_outdir(cmd, scenario)
        os.makedirs(outdir, exist_ok=True)
        base = cfg_with(cfg, outdir='')
        if cmd == 'sweep':
            values = parse_values(args.param, args.values)
            print_run_header(res, der, quiet, f"sweep of {args.param} over {values} -> {outdir}")
            extra = _variant_warnings(base, args.param, values)
            for x in extra:
                print("  WARNING (sweep variants): " + x)
            rows = run_sweep(base, args.param, values, outdir)
            print(f"--- sweep over {args.param} (all values reported, none selected) ---")
            _print_rows(rows, args.param, metrics_for(scenario))
            print(f"outputs in {outdir} (sweep_{args.param}.csv / .png)")
            return 0
        if cmd == 'ensemble':
            k = max(1, int(args.seeds))
            seeds = [cfg.seed + i for i in range(k)]
            print_run_header(res, der, quiet, f"ensemble over seeds {seeds} -> {outdir}")
            if der.method == 'direct' and _direct_seconds(_n_total(res), res.dim, res.dtype) * der.n_steps * k > _DIRECT_WARN_SECONDS:
                print(f"  WARNING: {k} direct-summation runs in sequence; consider --method pm.")
            result = run_ensemble(base, seeds, outdir)
            print("--- ensemble ---")
            _print_rows(result['rows'], 'seed', metrics_for(scenario))
            rep = result['reproducible']
            print(f"reproducible ({rep['metric']}, std/mean = {rep['std_over_mean']:.3g}, "
                  f"threshold {rep['threshold']:g}): {rep['verdict']}")
            for k2, cnt in result['verdict_counts'].items():
                print(f"  {k2}: {cnt}")
            print(f"outputs in {outdir} (ensemble.csv / ensemble.png / ensemble.json)")
            return 0
        if cmd == 'convergence':
            plan = convergence_plan(scenario)
            print_run_header(res, der, quiet, f"convergence suite {[p for p, _ in plan]} -> {outdir}")
            if der.method == 'direct':
                print("  NOTE: base method is direct: the grid set is skipped; the method set runs direct vs pm "
                      "at N = 2000 + 2000.")
            result = run_convergence(base, outdir)
            print("--- convergence (max relative change w.r.t. the reference) ---")
            thr = result['converged_threshold']
            for p, sp in result['spread'].items():
                line = ", ".join(f"{m}: {d['max_rel_change']:.3g}" for m, d in sp.items()
                                 if isinstance(d.get('max_rel_change'), float) and math.isfinite(d['max_rel_change']))
                nconv = sum(d['converged'] is True for d in sp.values())
                print(f"  {p:8s} [{nconv}/{len(sp)} metrics change < {thr:g}]  {line}")
            for sk in result.get('skipped', []):
                print(f"  {sk['param']}: skipped ({sk['reason']})")
            print(f"outputs in {outdir} (convergence.csv / convergence_<param>.png / convergence.json)")
            return 0
        parser.error(f"unknown command {cmd!r}")
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
