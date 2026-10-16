"""timing.py -- a small section timer shared by the CPU and GPU time loops.

Pure Python (no project imports, no torch): the loops wrap their leapfrog step,
diagnostics, structure statistics and snapshots in `with timers.section(name)`
blocks and print `format_breakdown(timers.breakdown(...))` at the end of the
run.  On the GPU the optional `sync` callable (torch.cuda.synchronize) is
called before every start and stop so that asynchronous device work is
attributed to the section that launched it.

Breakdown convention (same for both loops): the force evaluator keeps its own
accumulated `total_wall`; the loop times the whole leapfrog step as
'integration' and passes the force time to breakdown(forces=...), which then
reports forces separately and 'integration' as the timed step minus the forces.
'other' is whatever the wall clock saw outside the timed sections (initial
condition upload, CSV writes, bookkeeping), never negative.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

DEFAULT_SECTIONS = ('forces', 'integration', 'diagnostics', 'structure', 'snapshots')
_LABELS = {'forces': 'forces', 'integration': 'integration', 'diagnostics': 'energy diagnostics',
           'structure': 'structure statistics', 'snapshots': 'snapshots', 'other': 'other',
           'total': 'total'}


class Timers:
    """Accumulated wall-clock seconds per named section.

    sections : names known from the start (unknown names used later are added);
    sync     : optional callable invoked before each start/stop (GPU sync).
    """

    def __init__(self, sections=DEFAULT_SECTIONS, sync=None):
        self.sections = list(sections)
        self.seconds = {s: 0.0 for s in self.sections}
        self.counts = {s: 0 for s in self.sections}
        self.sync = sync
        self.wall0 = perf_counter()

    # ---- bookkeeping -----------------------------------------------------------
    def _ensure(self, name: str) -> None:
        if name not in self.seconds:
            self.sections.append(name)
            self.seconds[name] = 0.0
            self.counts[name] = 0

    def _sync(self) -> None:
        if self.sync is not None:
            self.sync()

    @contextmanager
    def section(self, name: str):
        """Context manager: `with timers.section('snapshots'): ...`."""
        self._ensure(name)
        self._sync()
        t0 = perf_counter()
        try:
            yield self
        finally:
            self._sync()
            self.seconds[name] += perf_counter() - t0
            self.counts[name] += 1

    def add(self, name: str, seconds: float) -> None:
        """Add externally measured seconds to a section."""
        self._ensure(name)
        self.seconds[name] += float(seconds)
        self.counts[name] += 1

    def elapsed(self, name: str) -> float:
        return float(self.seconds.get(name, 0.0))

    def total(self) -> float:
        """Wall-clock seconds since construction."""
        return perf_counter() - self.wall0

    def reset(self) -> None:
        for s in self.sections:
            self.seconds[s] = 0.0
            self.counts[s] = 0
        self.wall0 = perf_counter()

    # ---- reporting ---------------------------------------------------------------
    def breakdown(self, total=None, forces=None) -> list:
        """Ordered rows {key, name, seconds, percent} for forces, integration,
        energy diagnostics, structure statistics, snapshots, any extra sections,
        other = max(total - sum, 0) and a final 'total' row.

        total  : wall time to attribute (default: seconds since construction);
        forces : externally measured force time (e.g. ForceEvaluator.total_wall);
                 when given, 'integration' = timed step section - forces
                 (the step wraps the force evaluation); otherwise the 'forces'
                 section is used as timed and 'integration' is not corrected.
        """
        total = self.total() if total is None else float(total)
        if forces is None:
            f = self.elapsed('forces')
            integ = self.elapsed('integration')
        else:
            f = float(forces)
            integ = max(self.elapsed('integration') - f, 0.0)
        values = [('forces', f), ('integration', integ),
                  ('diagnostics', self.elapsed('diagnostics')),
                  ('structure', self.elapsed('structure')),
                  ('snapshots', self.elapsed('snapshots'))]
        for s in self.sections:
            if s not in DEFAULT_SECTIONS and s not in ('other', 'total'):
                values.append((s, self.elapsed(s)))
        accounted = sum(v for _, v in values)
        values.append(('other', max(total - accounted, 0.0)))
        values.append(('total', total))

        def pct(x):
            return 100.0 * x / total if total > 0 else 0.0

        return [{'key': k, 'name': _LABELS.get(k, k), 'seconds': float(v), 'percent': float(pct(v))}
                for k, v in values]

    def to_dict(self) -> dict:
        """Raw accumulated seconds and call counts (for summary.json)."""
        return {'sections': {s: float(self.seconds[s]) for s in self.sections},
                'counts': {s: int(self.counts[s]) for s in self.sections},
                'wall_total': float(self.total())}

    def __repr__(self) -> str:
        body = ", ".join(f"{s}={self.seconds[s]:.3f}s" for s in self.sections)
        return f"Timers({body}, total={self.total():.3f}s)"


def format_breakdown(rows: list, title: str = '') -> str:
    """Aligned text table 'forces   178.0 s   11.1 %' (one line per row)."""
    if not rows:
        return title or ''
    w = max(len(str(r['name'])) for r in rows)
    sw = max(len(f"{r['seconds']:.1f}") for r in rows)
    lines = [title] if title else []
    for r in rows:
        if r.get('key') == 'total':
            lines.append('-' * (w + sw + 13))
        lines.append(f"{r['name']:<{w}}  {r['seconds']:>{sw}.1f} s  {r['percent']:5.1f} %")
    return "\n".join(lines)


if __name__ == '__main__':  # self-test with sleeps
    import time
    calls = []
    tm = Timers(sync=lambda: calls.append(1))
    assert tm.sections == list(DEFAULT_SECTIONS) and tm.total() >= 0.0
    for _ in range(3):
        with tm.section('integration'):
            time.sleep(0.03)           # the step, including a pretend force evaluation
    with tm.section('diagnostics'):
        time.sleep(0.02)
    with tm.section('snapshots'):
        time.sleep(0.01)
    tm.add('structure', 0.005)
    with tm.section('plots'):          # unknown section is registered on the fly
        time.sleep(0.005)
    time.sleep(0.02)                   # untimed work -> 'other'
    assert len(calls) == 2 * 5 and tm.counts['integration'] == 3 and tm.counts['structure'] == 1
    assert 'plots' in tm.sections and tm.elapsed('plots') >= 0.004
    assert tm.elapsed('integration') >= 0.085 and tm.elapsed('diagnostics') >= 0.019
    # externally supplied force time: integration = step - forces
    rows = tm.breakdown(forces=0.05)
    keys = [r['key'] for r in rows]
    assert keys == ['forces', 'integration', 'diagnostics', 'structure', 'snapshots', 'plots', 'other', 'total']
    by = {r['key']: r for r in rows}
    assert by['forces']['seconds'] == 0.05
    assert abs(by['integration']['seconds'] - (tm.elapsed('integration') - 0.05)) < 1e-12
    assert by['diagnostics']['name'] == 'energy diagnostics' and by['structure']['name'] == 'structure statistics'
    assert by['other']['seconds'] >= 0.0 and by['other']['seconds'] >= 0.015
    assert abs(sum(r['seconds'] for r in rows[:-1]) - by['total']['seconds']) < 1e-9
    assert abs(sum(r['percent'] for r in rows[:-1]) - 100.0) < 1e-6 and abs(by['total']['percent'] - 100.0) < 1e-9
    # fixed total and no external forces: the 'forces' section is used uncorrected
    tm.add('forces', 0.01)
    rows2 = tm.breakdown(total=10.0)
    by2 = {r['key']: r for r in rows2}
    assert by2['forces']['seconds'] == 0.01 and by2['total']['seconds'] == 10.0
    assert abs(by2['integration']['seconds'] - tm.elapsed('integration')) < 1e-12
    assert abs(by2['total']['percent'] - 100.0) < 1e-9 and by2['other']['seconds'] > 9.0
    # a total smaller than the sections never yields negative 'other'
    assert {r['key']: r for r in tm.breakdown(total=0.0)}['other']['seconds'] == 0.0
    assert all(r['percent'] == 0.0 for r in tm.breakdown(total=0.0))
    txt = format_breakdown(rows, title='timing breakdown')
    lines = txt.splitlines()
    assert lines[0] == 'timing breakdown' and lines[1].startswith('forces') and ' s ' in lines[1]
    assert lines[1].rstrip().endswith('%') and lines[-1].startswith('total')
    assert len({ln.index(' s ') for ln in lines[1:] if ' s ' in ln}) == 1   # aligned columns
    assert format_breakdown([]) == ''
    d = tm.to_dict()
    assert set(d) == {'sections', 'counts', 'wall_total'} and d['counts']['diagnostics'] == 1
    assert d['sections']['snapshots'] >= 0.009 and d['wall_total'] >= 0.1
    tm.reset()
    assert tm.elapsed('integration') == 0.0 and tm.counts['integration'] == 0
    print(txt)
    print("timing self-test OK")