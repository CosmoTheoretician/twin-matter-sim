"""gpu_backend.py -- PyTorch/ROCm plumbing for the GPU port.

Everything that touches `torch` directly and is not a physics operation lives
here: the guarded import, the device/dtype resolution, the device-resident
particle container (GPUParticles) with explicit host <-> device copies, the
memory query of the card and a pure-Python estimate of the memory a run will
need (printed by main.py before a run starts, hard stop if it exceeds the
16 GB of the RX 9060 XT).

Under WSL with the ROCm build of PyTorch the AMD GPU is exposed through the
'cuda' device name, so every torch.cuda.* call below is the right one on an
AMD card.  No project import except particles (bottom of the dependency graph).
"""
from __future__ import annotations

import numpy as np

from particles import ParticleSet

try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ''
except Exception as _exc:          # ImportError, or a broken ROCm runtime at import time
    torch = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = repr(_exc)

_GB = float(2 ** 30)
_DEFAULT_TOTAL_BYTES = 16 * 2 ** 30          # RX 9060 XT (16 GB) when the card cannot be queried
_SAFETY = 1.5                                  # margin on the memory estimate


# ---------------------------------------------------------------------------
# availability, device, dtype
# ---------------------------------------------------------------------------

def torch_available() -> bool:
    return TORCH_AVAILABLE


def require_gpu():
    """torch.device('cuda') or a RuntimeError that says what is missing."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("--device gpu needs PyTorch (import torch failed: "
                           f"{TORCH_IMPORT_ERROR}); install the ROCm build of PyTorch in WSL "
                           "or run with --device cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--device gpu: torch.cuda.is_available() is False.  Under ROCm/WSL the "
                           "AMD GPU appears as the 'cuda' device; check that the ROCm build of "
                           "PyTorch is installed (not the CPU wheel), that the GPU is visible from "
                           "WSL and that the ROCm runtime matches it -- or run with --device cpu")
    return torch.device('cuda')


def device_name() -> str:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            return str(torch.cuda.get_device_name(0))
        except Exception:
            return 'cuda (name unavailable)'
    return 'n/a'


def torch_dtype(name):
    """'float32' / 'float64' (or a torch dtype, returned unchanged) -> torch dtype."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(f"torch is not available ({TORCH_IMPORT_ERROR})")
    if isinstance(name, torch.dtype):
        return name
    table = {'float32': torch.float32, 'float64': torch.float64,
             'f4': torch.float32, 'f8': torch.float64}
    key = str(np.dtype(name).name) if not isinstance(name, str) or name not in table else name
    if key not in table:
        raise ValueError(f"unsupported dtype {name!r}: use 'float32' or 'float64'")
    return table[key]


def numpy_dtype(tdt) -> np.dtype:
    """Inverse of torch_dtype."""
    if TORCH_AVAILABLE and tdt == torch.float32:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def synchronize(device=None) -> None:
    """torch.cuda.synchronize when `device` is a cuda device (no-op otherwise)."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        if device is None or torch.device(device).type == 'cuda':
            torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# memory of the card
# ---------------------------------------------------------------------------

def gpu_total_bytes() -> int:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            return int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            pass
    return int(_DEFAULT_TOTAL_BYTES)


def gpu_memory_report() -> dict:
    """allocated / reserved / max_allocated / total in bytes plus a one-line text in GB."""
    total = gpu_total_bytes()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        alloc = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        peak = int(torch.cuda.max_memory_allocated())
        text = (f"GPU memory: {alloc / _GB:.2f} GB allocated, {reserved / _GB:.2f} GB reserved, "
                f"peak {peak / _GB:.2f} GB, device total {total / _GB:.1f} GB")
    else:
        alloc = reserved = peak = 0
        text = f"GPU memory: n/a (no GPU), assumed device total {total / _GB:.1f} GB"
    return {'allocated': alloc, 'reserved': reserved, 'max_allocated': peak, 'total': total,
            'text': text}


# ---------------------------------------------------------------------------
# device-resident particles
# ---------------------------------------------------------------------------

class GPUParticles:
    """pos (N,dim), vel (N,dim), w (N,), s (N,) in one float dtype and species
    (N,) int32, all on one device.  Mirrors particles.ParticleSet; the host copy
    is only produced on request (to_host) at diagnostic / snapshot cadence."""

    def __init__(self, pos, vel, w, s, species):
        self.pos, self.vel, self.w, self.s, self.species = pos, vel, w, s, species
        if pos.dim() != 2:
            raise ValueError("pos must have shape (N, dim)")
        n = pos.shape[0]
        if vel.shape != pos.shape:
            raise ValueError(f"vel shape {tuple(vel.shape)} != pos shape {tuple(pos.shape)}")
        for name, a in (('w', w), ('s', s), ('species', species)):
            if tuple(a.shape) != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {tuple(a.shape)}")

    # ---- sizes ---------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.pos.shape[0])

    @property
    def dim(self) -> int:
        return int(self.pos.shape[1])

    @property
    def dtype(self):
        return self.pos.dtype

    @property
    def device(self):
        return self.pos.device

    # ---- host <-> device -------------------------------------------------------
    @classmethod
    def from_host(cls, ps: ParticleSet, device, dtype) -> "GPUParticles":
        dt = torch_dtype(dtype)
        dev = torch.device(device)

        def up(a, tdt):
            return torch.from_numpy(np.ascontiguousarray(a)).to(device=dev, dtype=tdt)

        return cls(up(ps.pos, dt), up(ps.vel, dt), up(ps.w, dt), up(ps.s, dt),
                   up(ps.species, torch.int32))

    def to_host(self, dtype=None) -> ParticleSet:
        """Fresh numpy copies (same float dtype as the device tensors unless overridden)."""
        npdt = numpy_dtype(self.dtype) if dtype is None else np.dtype(dtype)

        def down(t, dt_):
            return np.array(t.detach().cpu().numpy(), dtype=dt_, copy=True)

        return ParticleSet(down(self.pos, npdt), down(self.vel, npdt), down(self.w, npdt),
                           down(self.s, npdt), down(self.species, np.int32))

    def sync_from(self, ps: ParticleSet) -> None:
        """Overwrite pos/vel in place from a host set of the same size."""
        if ps.n != self.n or ps.dim != self.dim:
            raise ValueError(f"host set ({ps.n}, {ps.dim}) does not match device set ({self.n}, {self.dim})")
        self.pos.copy_(torch.from_numpy(np.ascontiguousarray(ps.pos)))
        self.vel.copy_(torch.from_numpy(np.ascontiguousarray(ps.vel)))

    def __repr__(self) -> str:
        return (f"GPUParticles(n={self.n}, dim={self.dim}, dtype={self.dtype}, "
                f"device={self.device})")


# ---------------------------------------------------------------------------
# memory estimate (pure Python, usable without torch)
# ---------------------------------------------------------------------------

def estimate_gpu_memory(cfg_resolved, der, n_total: int) -> dict:
    """Bytes needed on the device for a PM run of `n_total` particles.

    particles : pos, vel, acc (3*dim) + phi, w, s (3) floats per particle, plus the
                CIC temporaries 2^dim corners x (int64 index + float weight) and one
                flat scratch (2^dim x N floats) per deposit;
    grids     : periodic: (dim+3) real n^dim grids (rho, phi, dim acc grids, work)
                + 3 complex half-spectra of n^(dim-1) (n//2+1); isolated: x 2^dim
                (zero-padded doubled grid);
    fft       : workspace ~ 2 x the largest complex array;
    total     : (particles + grids + fft) x 1.5 safety factor.
    """
    dim = int(getattr(der, 'dim', getattr(cfg_resolved, 'dim', 2)))
    n = int(getattr(der, 'grid', getattr(cfg_resolved, 'grid', 256)))
    dtype = str(getattr(cfg_resolved, 'dtype', 'float32'))
    bc = str(getattr(cfg_resolved, 'bc', 'periodic'))
    item = 8 if dtype == 'float64' else 4
    corners = 2 ** dim
    npart = max(int(n_total), 0)
    part_arrays = npart * (3 * dim + 3) * item
    cic = npart * corners * (8 + item) + npart * corners * item
    particles = part_arrays + cic
    mult = 1 if bc == 'periodic' else corners
    ncells = n ** dim
    nhalf = n ** (dim - 1) * (n // 2 + 1)
    real_grids = (dim + 3) * ncells * item
    complex_grids = 3 * nhalf * 2 * item
    grids = (real_grids + complex_grids) * mult
    fft = 2 * nhalf * 2 * item * mult
    raw = particles + grids + fft
    total = int(_SAFETY * raw)
    limit = gpu_total_bytes()
    fits = total <= limit
    nw = n if bc == 'periodic' else 2 * n
    text = "\n".join([
        f"GPU memory estimate ({dtype}, N = {npart}, PM {n}^{dim}, bc = {bc}"
        f"{', FFT grid %d^%d' % (nw, dim) if bc != 'periodic' else ''}):",
        f"  particles : {particles / _GB:7.3f} GB  (arrays {part_arrays / _GB:.3f} GB + CIC temporaries "
        f"{cic / _GB:.3f} GB)",
        f"  grids     : {grids / _GB:7.3f} GB  ({dim + 3} real + 3 complex half-spectra"
        f"{', x %d padded' % mult if mult > 1 else ''})",
        f"  fft work  : {fft / _GB:7.3f} GB",
        f"  total     : {total / _GB:7.3f} GB  (x{_SAFETY:g} safety) of {limit / _GB:.1f} GB on the device -> "
        + ("fits" if fits else "DOES NOT FIT"),
    ])
    return {'particles': int(particles), 'cic': int(cic), 'grids': int(grids), 'fft': int(fft),
            'total': total, 'limit': int(limit), 'fits': bool(fits), 'text': text,
            'dtype': dtype, 'n_total': npart, 'grid': n, 'dim': dim, 'bc': bc}


def check_fits(est: dict) -> None:
    """RuntimeError with the summary when the estimate exceeds the device memory."""
    if not est.get('fits', True):
        raise RuntimeError(f"estimated GPU memory {est['total'] / _GB:.2f} GB exceeds the "
                           f"{est['limit'] / _GB:.1f} GB of the device: reduce --n, --grid or use "
                           f"--dtype float32\n{est['text']}")


if __name__ == '__main__':  # minimal self-test
    from types import SimpleNamespace as NS
    est = estimate_gpu_memory(NS(dtype='float32', bc='periodic'), NS(dim=3, grid=64), 100000)
    assert est['fits'] and est['total'] > 0 and 'fits' in est['text']
    check_fits(est)
    est2 = estimate_gpu_memory(NS(dtype='float32', bc='isolated'), NS(dim=3, grid=64), 100000)
    assert est2['grids'] == 8 * est['grids'] and est2['fft'] == 8 * est['fft']
    est64 = estimate_gpu_memory(NS(dtype='float64', bc='periodic'), NS(dim=3, grid=64), 100000)
    assert est64['grids'] == 2 * est['grids'] and est64['particles'] > est['particles']
    big = estimate_gpu_memory(NS(dtype='float64', bc='isolated'), NS(dim=3, grid=1024), 10 ** 7)
    assert not big['fits']
    try:
        check_fits(big)
        raise AssertionError("check_fits must raise")
    except RuntimeError as exc:
        assert 'exceeds' in str(exc) and 'DOES NOT FIT' in str(exc)
    # 2-D default LSS run must be tiny
    assert estimate_gpu_memory(NS(dtype='float32', bc='periodic'), NS(dim=2, grid=256), 10000)['total'] < 0.1 * _GB
    print(gpu_memory_report()['text'])
    if TORCH_AVAILABLE:
        rng = np.random.default_rng(0)
        n = 50
        s = np.where(np.arange(n) < 25, 1.0, -1.0)
        ps = ParticleSet(rng.random((n, 3)), rng.normal(size=(n, 3)), np.where(s > 0, 0.01, 0.64), s,
                         (s < 0).astype(np.int32))
        gp = GPUParticles.from_host(ps, 'cpu', 'float32')
        assert gp.n == n and gp.dim == 3 and gp.dtype == torch.float32 and gp.species.dtype == torch.int32
        back = gp.to_host()
        assert back.pos.dtype == np.float32 and np.allclose(back.pos, ps.pos, atol=1e-6)
        assert np.array_equal(back.species, ps.species) and np.allclose(back.s, ps.s)
        back64 = gp.to_host('float64')
        assert back64.vel.dtype == np.float64 and np.allclose(back64.vel, ps.vel, atol=1e-6)
        back.pos[0, 0] = 123.0                       # host copy is independent of the tensor
        assert float(gp.pos[0, 0]) != 123.0
        ps2 = ps.copy()
        ps2.pos[:] = 0.25
        ps2.vel[:] = -1.0
        gp.sync_from(ps2)
        assert float(gp.pos.max()) == 0.25 and float(gp.vel.min()) == -1.0
        assert torch_dtype('float64') == torch.float64 and torch_dtype(np.float32) == torch.float32
        assert numpy_dtype(torch.float32) == np.float32
        print("device:", device_name())
    else:
        print("torch not installed:", TORCH_IMPORT_ERROR)
    print("gpu_backend self-test OK")
