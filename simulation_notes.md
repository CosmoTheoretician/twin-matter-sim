# Notes

## 0. What is being reproduced

Two numerical "results" from the jp-petit.org pages, re-run under controlled conditions:

* **LSS (Landsheat, ~1992):** 2-D, two populations of 5 000 mass-points, |m̄| = 64 m, 2-D Maxwellian velocities with ⟨V̄⟩ = 4⟨V⟩, uniform initial distribution, no expansion. Claimed outcome: the heavier (negative) population clumps on its Jeans time, confines ordinary matter into cell walls, and the cellular pattern is "remarkably stable in time"; clump number/mass/diameter have small scatter across random seeds.
* **Barred spiral (Petit–Landsheat 1992):** 2-D, a rotating central clump of ordinary matter plus a confining ring of twin matter, computational domain closed on a flat torus (periodic tiling, minimum-image forces). Claimed outcome: a barred spiral stable over "an impressive number of turns", braking of the galaxy during spiral formation, morphology (spiral → ring) controlled by the twin/normal mass ratio, and twin-matter fragmentation attributed to the tiling (an artefact).

The default command-line settings *are* the faithful reproduction; every stress test is an explicit flag, so a user can see exactly which departure from the original setup changes the outcome.

## 1. Interaction law and conserved quantities (fixed by the pages)

Each particle carries an unsigned weight `w = |m|` and a sign `s = ±1`. Acceleration on particle *i* from *j*:

`a_i = −G s_i s_j w_j (r_i − r_j)/|r_ij|³`

* same sign → attraction (Newton), opposite sign → repulsion ("anti-Newton"), exactly the three-law scheme.
* Equivalent field form (used by the mesh solver): deposit the **signed** density ρ_s = Σ s_j w_j δ(x−x_j), solve Poisson once for Φ, then `a_i = −s_i ∇Φ(x_i)`. One FFT solve serves both populations.
* This scheme is Hamiltonian with pair potential `U_ij = −G s_i s_j w_i w_j / r_ij` and kinetic energy `Σ ½ w_i v_i²`. **Energy `E = KE + U` is conserved.**
* Newton's third law holds for `w·a`, not `m·a`: **`P_abs = Σ |m_i| v_i` is conserved; the signed momentum `Σ m_i v_i` is not** (cross-population pairs receive parallel impulses). The pages never discuss this. We log both and treat `P_abs` as the conservation diagnostic; drift of the signed momentum is expected physics, not a bug, and is labelled so in the output.
* The same holds for angular momentum (`L_abs` conserved, `L_signed` not), which is what makes the "braking of the galaxy" claim a meaningful, testable statement: disk `L_z` can decrease only if the twin ring's `|m|`-weighted `L_z` increases.

## 2. Assumptions where the pages are vague or silent (all written to `assumptions.txt` by every run)

| # | Item | Pages say | Assumption / default |
|---|------|-----------|----------------------|
| A1 | Force law in "2-D" | "sum of Newtonian forces" for mass points in a plane | 1/r² force (razor-thin sheet, "flat gas"); FFT kernel −2πG/|k| in 2-D. Option `--force-law log` for true 2-D (logarithmic) gravity as a stress test. |
| A2 | LSS boundary conditions | silent (torus is mentioned only for the spiral run) | periodic box; direct method uses the minimum-image rule (the "dashed rectangle" of the artefacts page), PM uses exact periodic images. |
| A3 | Net signed mass in a periodic box (|M̄| = 64 M ≠ 0) | silent | k = 0 mode removed (Jeans swindle: uniform background subtracted). The minimum-image direct sum does this implicitly by symmetry. |
| A4 | Absolute velocity scale ("temperature") | only ratios given | set by `--lambda-j-neg` = λ_J(negative)/L, default 0.125 (so λ_J,+ = L/2 for the default 64/4² ratio: positives marginally stable alone, negatives have ≈8 Jeans lengths per box side). λ_J from the linear dispersion relation of the chosen (dim, force law): sheet λ_J = σ²/(GΣ); 2-D log and 3-D: λ_J = σ√(π/(Gρ)) forms. Reference time t_J,neg = λ_J,neg/σ_−. |
| A5 | Meaning of "⟨V̄⟩ = 4⟨V⟩" | mean speed | σ_− = 4σ_+ per velocity component (mean speed ∝ σ for a Maxwellian). `--vel-ratio`. |
| A6 | "|M̄| = 64 M" with N_+ = N_− | per-particle masses | implemented as a **density** ratio (total |M_−| = 64 M_+), identical when N_+ = N_−; preserved if N differs. `--mass-ratio`. |
| A7 | Initial positions | "uniform" | Poisson random (shot noise seeds the instability). Options `--ic glass|lattice` with `--ic-noise` to test whether the pattern is discreteness-driven. |
| A8 | Softening | "cut-in" (force zeroed below a cutoff) mentioned for cluster runs; nothing for LSS | direct: Plummer by default (`--soft cutin` reproduces the cut-in); PM: cell size plus optional Gaussian k-space smoothing `--eps` in units of the box. |
| A9 | Timestep | "small enough that halving it changes nothing"; adaptive back-up scheme on Cray | fixed leapfrog (KDK), dt = 0.02 t_J,neg (LSS) or T_rot/200 (galaxy); `--adaptive` uses a global step η√(ε/a_max) as a stand-in for the back-up scheme. |
| A10 | Run length | "remarkably stable", "impressive number of turns" | LSS: 20 t_J,neg; galaxy: 10 rotations at 2 R_d. `--tmax` to extend. |
| A11 | Galaxy disk | "rotating central clump", analytic IC never published; Miyamoto–Nagai only in the semi-analytic rotation-curve section | 2-D exponential disk, R_d = 1, truncated at 4 R_d, Toomre Q = 1.5 (`--q`), circular velocity measured from the *actual* initial force field (including the twin ring's repulsion), epicyclic dispersions with asymmetric-drift correction. If the net radial force is outward anywhere, v_c is undefined there; this is logged, not patched. |
| A12 | Twin ring | "confining ring", density sketch only | annulus 4.5–7 R_d of negative mass, |M_twin| = `--twin-ratio` × M_disk (default 1.0), Maxwellian σ = `--twin-sigma` × v_c,max (default 0.5), rotation fraction `--twin-rot` (default 0). Not in equilibrium by construction; whether it fragments is one of the measured outcomes. |
| A13 | Galaxy box | torus, size not given | box 16 R_d periodic (`--box`), with `--bc isolated` (zero-padded Hockney–Eastwood solver) to test the tiling-artefact claim. |
| A14 | Units | none | G = 1; LSS: box L = 1, M_+ = 1; galaxy: M_disk = 1, R_d = 1. |
| A15 | Expansion | "we do not know how" | optional kinematic comoving frame with prescribed a(t) ∝ t^n (default n = 2/3), equations ẍ + 2Hẋ = g_c/a^{p+1} (p = force-law exponent). Not a Janus cosmology; just tests whether Hubble drag erases the pattern. |
| A16 | 3-D | "beyond our capabilities" | PM on 64³ (default) with 1/k² kernel, same IC recipe; 2 × 5 000 particles in 3-D are very sparse, so `--n` sweeps are mandatory before concluding anything. |
| A17 | Self-energy in PM potential energy | — | subtracted using the measured potential of a single unit particle at a cell centre; residual position dependence shows as small noise, not drift. |

## 3. Overall design

* **Particles:** `pos (N,dim)`, `vel`, `w`, `s`, `species`. Two species only (0 = ordinary, 1 = twin).
* **Force methods** (`--method`): 
  * `direct` — chunked O(N²) numpy pairwise sum, minimum image or open, Plummer or cut-in, 1/r² or log. This is the original algorithm. ~3 s per force evaluation for 10 000 particles in float64 (≈1 h for the faithful 1 000-step LSS run); float32 roughly halves it.
  * `pm` — CIC deposit, FFT Poisson solve (periodic analytic kernel, or isolated via doubled grid + real-space Green's function), spectral gradient, CIC interpolation. Identical deposit/interpolation kernels ⇒ no self-force, exact `P_abs` conservation up to round-off. 256² default in 2-D (~10 ms/step), 64³ default in 3-D. Allows N up to ~10⁶ in 2-D on a PC.
  * `auto` (default) — `direct` when N_total ≤ 4 000, else `pm`. The faithful 5 000 + 5 000 run therefore uses PM unless `--method direct` is given; the convergence suite compares the two.
* **Integrator:** leapfrog KDK, fixed or adaptive global dt; expansion variant with the auxiliary variable u = a²ẋ.
* **External rigid fields** for galaxy control runs: none (`--control none`), static repulsive ring with the same density as the live twin ring (`rigid-repulsive`), static attractive logarithmic halo (`rigid-halo`, Ostriker–Peebles comparison). The live-twin case is `--control live` (default).
* **Diagnostics every `--log-every` steps → `diagnostics.csv`:** KE, PE, E, relative energy drift, `P_abs`, `P_signed`, `L_abs`, `L_signed`, per-species σ and `L_z`, virial ratio, escaped fraction (galaxy), wall-clock.
* **Structure statistics → `structure.csv`:**
  * LSS: clump catalogue per species from a smoothed density grid + `scipy.ndimage.label` above a density threshold (count, mass, equivalent diameter φ and their std), cell size proxy d = mean nearest-neighbour distance between negative clump centroids, φ/d, segregation index (Pearson correlation of log-densities of the two species; strongly negative = cells), positive-matter void fraction, power-spectrum peak scale of each species, pattern persistence (correlation of ρ_−(t) with ρ_−(t_ref)).
  * Galaxy: Fourier amplitudes A_m(R) (m = 1…4) in radial bins, global bar strength (max A₂ inside 2 R_d), outer spiral strength (A₂ for R > 2 R_d), m = 2 phase → pitch angle and pattern speed, disk `L_z` and mean rotation speed (braking), twin ring clump count and `L_z` (angular-momentum transfer), Toomre Q from measured σ_R.
* **Pre-registered criteria** (fixed here, before any run, to avoid tuning): LSS "cells form" = segregation index < −0.3 by 3 t_J,neg; "stable" = n_clumps,neg changes by < 20 % between 5 and 20 t_J,neg and persistence > 0.7; "reproducible" = seed-to-seed std of n_clumps < 20 % of mean. Galaxy "sustained spiral" = outer A₂ > 0.1 for ≥ 8 of 10 rotations; "bar" = inner A₂ > 0.2; "braking" = disk `L_z` decreases by > 5 % while twin `L_z,abs` increases correspondingly; "tiling artefact confirmed" = ring fragmentation present with periodic BC and absent with isolated BC at the same N, eps, dt. `summary.json` reports the numbers and whether each criterion was met, with explicit "not met" / "ambiguous (within noise)" wording. Nothing in the code searches parameter space for a passing configuration; sweeps report all values.
* **Output:** `outdir/` contains `config.json`, `assumptions.txt`, `diagnostics.csv`, `structure.csv`, `summary.json`, `snap_XXXX.png` (two-panel species maps; 3-D: three projections), `conservation.png`, `structure.png` / `modes.png`, and for sweeps `sweep_<param>.csv` + `sweep_<param>.png`. Matplotlib Agg backend only.

## 4. Command-line interface

`python main.py <subcommand> [options]`

Subcommands: `lss`, `galaxy`, `sweep {lss|galaxy} --param P --values a,b,c`, `ensemble {lss|galaxy} --seeds K`, `convergence {lss|galaxy}`, `assumptions {lss|galaxy}` (prints the assumption list for the given options without running).

Common options: `--outdir PATH` (default `out/<scenario>_<timestamp>`), `--seed INT` (1), `--dim {2,3}` (2), `--method {auto,direct,pm}` (auto), `--bc {periodic,isolated}` (periodic), `--force-law {r2,log}` (r2), `--soft {plummer,cutin}` (plummer), `--eps FLOAT` (direct: 0.002 L for LSS, 0.05 R_d for galaxy; PM: 0 = cell size), `--grid INT` (256 in 2-D, 64 in 3-D), `--dt FLOAT` (fraction of reference time; 0.02 / 0.005), `--adaptive`, `--eta FLOAT` (0.1), `--tmax FLOAT` (reference times; 20 / 10), `--snap-every FLOAT` (1.0 / 0.5 reference times), `--log-every INT` (steps, 10), `--structure-every FLOAT` (reference times, 0.5), `--dtype {float64,float32}`, `--direct-chunk INT` (1024), `--quiet`.

LSS options: `--npos INT` (5000), `--nneg INT` (5000), `--mass-ratio FLOAT` (64), `--vel-ratio FLOAT` (4), `--lambda-j-neg FLOAT` (0.125), `--ic {poisson,glass,lattice}` (poisson), `--ic-noise FLOAT` (0.0, fraction of mean spacing), `--neg-only`, `--pos-only` (single-population controls), `--expansion`, `--a-index FLOAT` (0.6667), `--t0-exp FLOAT` (expansion start, in t_J,neg; 1.0), `--structure-grid INT` (128), `--clump-threshold FLOAT` (5 × mean density), `--clump-min-cells INT` (4).

Galaxy options: `--ndisk INT` (5000), `--ntwin INT` (5000), `--rd FLOAT` (1), `--rmax FLOAT` (4), `--q FLOAT` (1.5), `--zscale FLOAT` (0.1, 3-D only), `--twin-ratio FLOAT` (1.0), `--twin-rin FLOAT` (4.5), `--twin-rout FLOAT` (7.0), `--twin-sigma FLOAT` (0.5), `--twin-rot FLOAT` (0.0), `--control {live,none,rigid-repulsive,rigid-halo}` (live), `--halo-vc FLOAT` (1.0), `--halo-rc FLOAT` (1.0), `--box FLOAT` (16).

Sweepable parameters (`--param`): `npos`, `nneg`, `n` (both), `eps`, `dt`, `grid`, `tmax`, `mass_ratio`, `vel_ratio`, `lambda_j_neg`, `ic`, `method`, `bc`, `force_law`, `dim`, `expansion`, `ndisk`, `ntwin`, `q`, `twin_ratio`, `twin_sigma`, `twin_rot`, `control`, `box`. `convergence` runs a fixed set: dt × {1, ½, ¼}, eps × {2, 1, ½}, grid × {½, 1, 2}, method {direct, pm} at N = 2 000 + 2 000, and bc {periodic, isolated} for the galaxy.

## 5. Example commands

```
# faithful reproductions (defaults)
python main.py lss                                   # 2×5000, 2-D, 64/4, periodic, 20 t_J
python main.py lss --method direct --dtype float32   # the original O(N²) minimum-image algorithm (~30-60 min)
python main.py galaxy                                # disk + live twin ring, torus, 10 rotations
python main.py assumptions lss                       # print the assumption list only

# ensembles and controls
python main.py ensemble lss --seeds 8                # scatter of clump number/mass/diameter
python main.py lss --neg-only                        # do negatives alone give the same clump statistics?
python main.py galaxy --control none                 # bare disk (does a bar/spiral survive anyway?)
python main.py galaxy --control rigid-repulsive      # is a live twin ring needed, or any confining potential?
python main.py galaxy --control rigid-halo           # standard Ostriker–Peebles comparison

# stress tests
python main.py sweep lss --param n --values 1000,5000,20000,100000 --method pm
python main.py sweep lss --param mass_ratio --values 1,4,16,64,256
python main.py sweep lss --param ic --values poisson,glass
python main.py lss --tmax 100 --snap-every 5          # long run: do clumps keep merging?
python main.py lss --force-law log                    # true 2-D gravity instead of a flat sheet
python main.py lss --dim 3 --method pm --grid 64 --npos 50000 --nneg 50000
python main.py lss --expansion --tmax 40
python main.py sweep galaxy --param twin_ratio --values 0,0.25,0.5,1,2,4
python main.py sweep galaxy --param box --values 12,16,24,32          # tiling artefact?
python main.py galaxy --bc isolated --method pm                       # no tiling at all
python main.py sweep galaxy --param ndisk --values 2000,5000,20000,100000 --method pm
python main.py convergence galaxy
```

## 6. What would support or undermine the original claims

**LSS – supporting evidence:** negative-mass clumps appear on ≈ t_J,neg; segregation index drops below −0.3 with positives forming connected walls (low void fraction of walls, high void fraction inside cells); n_clumps,neg plateaus between 5 and 20 t_J,neg and persistence stays high; seed-to-seed scatter small; the same cell count survives N ↑, grid ↑, dt ↓, eps variation, direct vs PM, and glass IC with small noise; qualitatively similar bubbles in 3-D.

**LSS – undermining or ambiguous evidence:** n_clumps keeps falling (hierarchical merging — what a self-attracting population normally does; "remarkably stable" would then be a short-run impression); `--neg-only` gives the same clump statistics (positives are passive tracers of the complement, which is not wrong but means the cells carry no extra stabilising physics); cell number scales with N or with eps (discreteness/softening artefact); no structure from glass IC with zero noise and structure appearing only with noise ≥ Poisson level (pattern seeded by shot noise of 5 000 points — ambiguous, because a real universe has a perturbation spectrum, so we flag rather than condemn); strong dependence on the 1/r² vs log choice (then "2-D" results do not transfer); expansion wipes the pattern; 3-D results qualitatively different from 2-D.

**Galaxy – supporting evidence:** outer A₂ sustained ≥ 8 rotations with the live ring but decaying within 1–2 rotations for `--control none` and `rigid-repulsive`; disk `L_z` decreases while twin `L_z,abs` increases during arm growth; spiral → ring morphology as `twin_ratio` rises; ring fragmentation disappears with `--bc isolated` and larger boxes; results invariant to dt, eps, N.

**Galaxy – undermining or ambiguous evidence:** the bare disk also keeps a strong bar for 10 rotations (bars are generically long-lived; a surviving bar is not the same as surviving spiral arms, and the code separates inner A₂ from outer A₂); the rigid repulsive ring reproduces everything (then the effect is just "any confining potential", analogous to the halo stabilisation already known since 1973); ring fragmentation persists with isolated BC (Jeans instability of the self-attracting ring, not a tiling artefact, contradicting the page's attribution); arm strength changes with N at fixed physics (collisional heating/cooling of a 5 000-particle 2-D disk) or with eps/dt (numerical); v_c undefined over part of the disk because the ring's repulsion exceeds disk gravity (the "confining ring" set-up is then not a rotating equilibrium).

In all cases the summary reports the measured numbers against the pre-registered thresholds and states "met", "not met" or "ambiguous"; no parameter is adjusted by the code to reach a verdict.

```json
[
  {
    "file": "config.py",
    "purpose": "Single source of truth for all run parameters, defaults, derived physical quantities (particle weights, velocity dispersions from the Jeans-length parameter, reference times, absolute dt/tmax/step counts), unit conventions, and the explicit list of assumptions in force for a given configuration.",
    "interface": {
      "imports": [],
      "exposes": [
        "@dataclass class RunConfig: scenario:str, dim:int=2, method:str='auto', bc:str='periodic', force_law:str='r2', soft:str='plummer', eps:float=-1.0, grid:int=-1, dt:float=-1.0, adaptive:bool=False, eta:float=0.1, tmax:float=-1.0, snap_every:float=-1.0, log_every:int=10, structure_every:float=0.5, dtype:str='float64', direct_chunk:int=1024, seed:int=1, outdir:str='', quiet:bool=False, npos:int=5000, nneg:int=5000, mass_ratio:float=64.0, vel_ratio:float=4.0, lambda_j_neg:float=0.125, ic:str='poisson', ic_noise:float=0.0, neg_only:bool=False, pos_only:bool=False, expansion:bool=False, a_index:float=0.6667, t0_exp:float=1.0, structure_grid:int=128, clump_threshold:float=5.0, clump_min_cells:int=4, ndisk:int=5000, ntwin:int=5000, rd:float=1.0, rmax:float=4.0, q:float=1.5, zscale:float=0.1, twin_ratio:float=1.0, twin_rin:float=4.5, twin_rout:float=7.0, twin_sigma:float=0.5, twin_rot:float=0.0, control:str='live', halo_vc:float=1.0, halo_rc:float=1.0, box:float=-1.0, label:str=''",
        "@dataclass class Derived: G:float, box:float, dim:int, method:str, w_pos:float, w_neg:float, sigma_pos:float, sigma_neg:float, lambda_j_pos:float, lambda_j_neg:float, t_ref:float, t_ref_name:str, dt_abs:float, tmax_abs:float, n_steps:int, snap_every_abs:float, structure_every_abs:float, eps_abs:float, grid:int, force_exponent_p:int, vc_ref:float, t_rot_ref:float",
        "def resolve_defaults(cfg: RunConfig) -> RunConfig  # fills every -1/'auto' field (method by N, grid by dim, eps, dt, tmax, snap_every, box) and returns a new config",
        "def jeans_length(sigma: float, density: float, G: float, dim: int, force_law: str) -> float  # dispersion-relation Jeans length for (dim, law): sheet sigma^2/(G*Sigma); 2-D log and 3-D sigma*sqrt(pi/(G*rho))-type forms",
        "def derive(cfg: RunConfig) -> Derived",
        "def assumptions(cfg: RunConfig, der: Derived) -> list[str]  # human-readable list A1..A17 instantiated with the actual numbers",
        "def cfg_to_dict(cfg: RunConfig) -> dict",
        "def cfg_from_dict(d: dict) -> RunConfig",
        "def cfg_with(cfg: RunConfig, **overrides) -> RunConfig  # copy with changes, used by sweeps",
        "SWEEPABLE: dict[str, type]  # parameter name -> python type for CLI value parsing"
      ]
    }
  },
  {
    "file": "particles.py",
    "purpose": "Particle container (positions, velocities, unsigned weights |m|, signs, species ids) and small array helpers: periodic wrapping, species masks, concatenation, subsetting, centre of |m|-weighted mass.",
    "interface": {
      "imports": [],
      "exposes": [
        "@dataclass class ParticleSet: pos: np.ndarray, vel: np.ndarray, w: np.ndarray, s: np.ndarray, species: np.ndarray",
        "ParticleSet.n -> int; ParticleSet.dim -> int; ParticleSet.npos -> int; ParticleSet.nneg -> int",
        "ParticleSet.pos_mask -> np.ndarray[bool]; ParticleSet.neg_mask -> np.ndarray[bool]",
        "ParticleSet.signed_mass -> np.ndarray  # s*w",
        "ParticleSet.copy() -> ParticleSet",
        "ParticleSet.astype(dtype) -> ParticleSet",
        "def wrap_periodic(pos: np.ndarray, box: float) -> np.ndarray  # in [0, box)",
        "def min_image(d: np.ndarray, box: float) -> np.ndarray  # separation vectors to nearest image",
        "def concat(a: ParticleSet, b: ParticleSet) -> ParticleSet",
        "def subset(ps: ParticleSet, mask: np.ndarray) -> ParticleSet",
        "def centre_of_mass(ps: ParticleSet, mask=None, box=None) -> np.ndarray  # |m|-weighted; periodic-aware via circular mean when box given",
        "def empty(dim: int, dtype) -> ParticleSet"
      ]
    }
  },
  {
    "file": "forces_direct.py",
    "purpose": "Direct O(N^2) summation of the three-law force with the minimum-image rule (Petit's dashed-rectangle tiling) or open boundaries, Plummer or cut-in softening, 1/r^2 (sheet/3-D) or logarithmic (true 2-D) law; chunked over targets to bound memory. Returns signed accelerations and the unsigned potential at each particle.",
    "interface": {
      "imports": ["particles"],
      "exposes": [
        "def pair_kernel(r2: np.ndarray, eps: float, soft: str, force_law: str) -> tuple[np.ndarray, np.ndarray]  # returns (f_over_r, phi_unit): f_over_r such that acc = -G*s_i*s_j*w_j*f_over_r*d; phi_unit = potential of a unit positive mass; cut-in: force 0 and phi constant inside eps; log law: phi = ln r",
        "def direct_forces(pos: np.ndarray, w: np.ndarray, s: np.ndarray, G: float, eps: float, soft: str, force_law: str, box: float | None = None, chunk: int = 1024, need_phi: bool = True) -> tuple[np.ndarray, np.ndarray]  # (acc (N,dim) including s_i sign, phi (N,) unsigned potential from the signed density; phi zeros if need_phi False)",
        "def potential_energy_from_phi(w: np.ndarray, s: np.ndarray, phi: np.ndarray) -> float  # 0.5*sum(s*w*phi)",
        "def pair_count_estimate(n: int) -> float  # for runtime warnings"
      ]
    }
  },
  {
    "file": "forces_pm.py",
    "purpose": "FFT particle-mesh solver in 2-D and 3-D: CIC deposit of the signed density, Poisson solve with the periodic analytic kernel (-2piG/|k| sheet, -2piG/k^2 log, -4piG/k^2 3-D, k=0 removed) or isolated boundaries via a zero-padded doubled grid with a real-space Green's function, optional Gaussian k-space smoothing as softening, spectral gradient, CIC interpolation back to particles, self-energy estimate for the potential-energy diagnostic, and a generic density-grid routine reused by the structure statistics.",
    "interface": {
      "imports": ["particles"],
      "exposes": [
        "class PMSolver",
        "PMSolver.__init__(self, dim: int, grid: int, box: float, G: float, force_law: str, bc: str = 'periodic', eps: float = 0.0, dtype=np.float64)",
        "PMSolver.cell -> float  # cell size",
        "PMSolver.deposit(self, pos: np.ndarray, weights: np.ndarray) -> np.ndarray  # CIC density on grid^dim (signed if weights signed); periodic wrap or clip for isolated",
        "PMSolver.solve_potential(self, rho: np.ndarray) -> np.ndarray  # Phi on the (unpadded) grid",
        "PMSolver.gradient(self, phi: np.ndarray) -> list[np.ndarray]  # spectral -dPhi/dx_k grids (acceleration of a unit positive mass)",
        "PMSolver.interpolate(self, grids: list[np.ndarray] | np.ndarray, pos: np.ndarray) -> np.ndarray  # CIC gather, shape (N,dim) or (N,)",
        "PMSolver.forces(self, pos: np.ndarray, w: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]  # (acc (N,dim) including s_i sign, phi (N,) unsigned)",
        "PMSolver.field_from_density(self, rho: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]  # (acc grids, phi grid) for a static analytic density (used by external_potentials)",
        "PMSolver.self_energy(self, w: np.ndarray) -> float  # sum 0.5*w^2*phi_self of a unit particle at a cell centre, to subtract from PE",
        "def density_grid(pos: np.ndarray, weights: np.ndarray, box: float, ngrid: int, dim: int, periodic: bool = True, origin: np.ndarray | None = None) -> np.ndarray  # standalone CIC deposit for diagnostics at arbitrary resolution",
        "def green_function_real(dim: int, n: int, cell: float, G: float, force_law: str, eps: float) -> np.ndarray  # doubled-grid kernel for isolated BC"
      ]
    }
  },
  {
    "file": "external_potentials.py",
    "purpose": "Rigid (non-responding) external fields for the galaxy control runs: none, a static repulsive ring with the same density profile as the live twin ring (field precomputed on a PM grid from the analytic annulus density), and a static attractive logarithmic halo. Fields are expressed as the acceleration felt by a unit positive mass; species sign is applied by the caller.",
    "interface": {
      "imports": ["config", "forces_pm"],
      "exposes": [
        "class ExternalField  # base: accel(pos)->(N,dim), potential(pos)->(N,), name:str",
        "class NoField(ExternalField)",
        "class RigidRingField(ExternalField)",
        "RigidRingField.__init__(self, pm: PMSolver, mass_abs: float, rin: float, rout: float, centre: np.ndarray, dim: int, zscale: float = 0.1)  # deposits negative annulus density analytically on the grid, stores acc/phi grids",
        "class RigidHaloField(ExternalField)",
        "RigidHaloField.__init__(self, vc: float, rc: float, centre: np.ndarray)  # Phi = 0.5*vc^2*ln(r^2+rc^2)",
        "def make_external_field(cfg: RunConfig, der: Derived, pm: PMSolver | None) -> ExternalField  # selects by cfg.control; raises if control needs a PM grid and none is available (builds a private one for method='direct')"
      ]
    }
  },
  {
    "file": "integrator.py",
    "purpose": "Force evaluator wrapper (direct or PM plus optional external field, applying species signs), kinematic cosmic expansion (prescribed a(t), Hubble drag through u = a^2 x_dot), global adaptive timestep, and the leapfrog KDK step with periodic wrapping.",
    "interface": {
      "imports": ["config", "particles", "forces_direct", "forces_pm", "external_potentials"],
      "exposes": [
        "class Expansion",
        "Expansion.__init__(self, enabled: bool, a_index: float, t0: float)",
        "Expansion.a(self, t: float) -> float  # 1 for t<=t0 or disabled; (t/t0)^a_index after",
        "Expansion.H(self, t: float) -> float",
        "class ForceEvaluator",
        "ForceEvaluator.__init__(self, cfg: RunConfig, der: Derived, pm: PMSolver | None, ext: ExternalField)",
        "ForceEvaluator.__call__(self, ps: ParticleSet) -> tuple[np.ndarray, np.ndarray]  # (acc (N,dim) signed and including s*ext.accel, phi (N,) unsigned including ext.potential)",
        "ForceEvaluator.n_evals -> int; ForceEvaluator.last_wall -> float",
        "def choose_dt(acc: np.ndarray, eps_eff: float, eta: float, dt_max: float, dt_min: float) -> float  # eta*sqrt(eps_eff/|a|_max) clipped",
        "def kick(ps: ParticleSet, acc: np.ndarray, dt: float, a_old: float, a_new: float, p: int) -> None  # in place; expansion-aware via u=a^2 v, p = force-law exponent",
        "def drift(ps: ParticleSet, dt: float, a_mid: float, box: float | None) -> None  # in place, wraps if box",
        "def leapfrog_step(ps: ParticleSet, acc: np.ndarray, phi: np.ndarray, t: float, dt: float, force: ForceEvaluator, expansion: Expansion, p: int, box: float | None) -> tuple[np.ndarray, np.ndarray, float]  # KDK; returns (acc_new, phi_new, t+dt)"
      ]
    }
  },
  {
    "file": "initial_conditions.py",
    "purpose": "Initial conditions: LSS two-population uniform box (Poisson/glass/lattice positions, Maxwellian velocities from the derived dispersions, single-population controls, 2-D or 3-D), galaxy exponential disk (2-D or thick 3-D) with circular velocities measured from the actual initial force field and epicyclic dispersions at the requested Toomre Q, and the twin ring/shell with Maxwellian plus optional rotation.",
    "interface": {
      "imports": ["config", "particles", "integrator"],
      "exposes": [
        "def poisson_positions(n: int, dim: int, box: float, rng) -> np.ndarray",
        "def lattice_positions(n: int, dim: int, box: float, rng, noise: float) -> np.ndarray  # nearest square/cubic lattice with jitter = noise*spacing; n adjusted to a perfect power and reported",
        "def glass_positions(n: int, dim: int, box: float, rng, n_iter: int = 30, noise: float = 0.0) -> np.ndarray  # cKDTree-based repulsive relaxation of a Poisson set, then jitter",
        "def maxwellian_velocities(n: int, dim: int, sigma: float, rng) -> np.ndarray  # zero-mean per component",
        "def make_lss_ic(cfg: RunConfig, der: Derived, rng) -> ParticleSet  # species 0 then 1; honours neg_only/pos_only; removes net |m|-weighted drift",
        "def sample_exponential_disk(n: int, rd: float, rmax: float, rng) -> np.ndarray  # (n,2) radii by inverse-CDF with truncation",
        "def make_disk_positions(cfg: RunConfig, der: Derived, rng, centre: np.ndarray) -> ParticleSet  # velocities zero placeholder; 3-D adds sech^2 z with cfg.zscale*rd",
        "def make_twin_ring(cfg: RunConfig, der: Derived, rng, centre: np.ndarray) -> ParticleSet  # annulus/ellipsoidal shell of negative mass, Maxwellian sigma=twin_sigma*vc_ref plus twin_rot*v_circ tangential",
        "def measure_rotation_curve(ps: ParticleSet, force: ForceEvaluator, centre: np.ndarray, rbins: np.ndarray) -> dict  # azimuthally averaged radial acceleration on positive species -> vc(R) (nan where outward), Omega, kappa, Sigma(R)",
        "def set_disk_velocities(ps: ParticleSet, force: ForceEvaluator, cfg: RunConfig, der: Derived, rng, centre: np.ndarray) -> dict  # in place for species 0: v_phi from vc with asymmetric drift, sigma_R from Q, sigma_phi = sigma_R*kappa/(2 Omega); returns the rotation-curve dict and the fraction of disk with undefined vc",
        "def make_galaxy_ic(cfg: RunConfig, der: Derived, rng, force_builder) -> tuple[ParticleSet, dict]  # force_builder(ps)->ForceEvaluator; assembles disk+ring (or disk only), measures field, sets velocities, returns (ps, rotation-curve info)"
      ]
    }
  },
  {
    "file": "diagnostics_core.py",
    "purpose": "Conservation and bulk diagnostics (energies with self-energy subtraction, |m|-weighted and signed linear/angular momenta, per-species dispersions and angular momenta, virial ratio, escaped fraction) plus CSV/JSON writers.",
    "interface": {
      "imports": ["particles"],
      "exposes": [
        "def kinetic_energy(ps: ParticleSet, a: float = 1.0) -> float  # 0.5*sum(w*|v|^2)*a^2 (peculiar) ",
        "def potential_energy(ps: ParticleSet, phi: np.ndarray, self_energy: float = 0.0) -> float  # 0.5*sum(s*w*phi) - self_energy",
        "def momenta(ps: ParticleSet, centre: np.ndarray | None = None) -> dict  # keys: p_abs_x.., p_signed_x.., L_abs, L_signed, L_pos, L_neg (z-component in 2-D, |L| in 3-D)",
        "def species_stats(ps: ParticleSet, centre: np.ndarray | None = None, r_escape: float | None = None) -> dict  # sigma_pos, sigma_neg, vrot_mean_pos, r_rms_pos, r_rms_neg, escaped_frac_pos, escaped_frac_neg",
        "def diagnostics_row(ps: ParticleSet, phi: np.ndarray, t: float, step: int, dt: float, a: float, self_energy: float, E0: float | None, P0: np.ndarray | None, centre=None, r_escape=None, wall: float = 0.0) -> dict  # merges all of the above; adds dE_rel and dP_abs_rel, virial",
        "class CsvLogger",
        "CsvLogger.__init__(self, path: str)",
        "CsvLogger.log(self, row: dict) -> None  # header written from the first row; later rows reordered/filled with nan",
        "CsvLogger.close(self) -> None",
        "def write_json(path: str, obj) -> None  # numpy-safe",
        "def read_csv(path: str) -> dict[str, np.ndarray]"
      ]
    }
  },
  {
    "file": "structure_stats.py",
    "purpose": "Quantitative structure measures. LSS: clump catalogue per species via smoothed density grid + ndimage.label (count, mass, equivalent diameter, nearest-neighbour cell size d, phi/d), segregation index, void fraction, power-spectrum peak scale, pattern persistence. Galaxy: Fourier modes A_m(R), bar and outer spiral strength, m=2 phase -> pitch angle and pattern speed, disk and ring angular momenta (braking), ring clump count, measured Toomre Q.",
    "interface": {
      "imports": ["particles", "forces_pm"],
      "exposes": [
        "@dataclass class ClumpCatalog: n: int, masses: np.ndarray, diameters: np.ndarray, centroids: np.ndarray, labels: np.ndarray",
        "def smooth_grid(rho: np.ndarray, sigma_cells: float, periodic: bool) -> np.ndarray",
        "def find_clumps(rho: np.ndarray, box: float, threshold_factor: float, min_cells: int, periodic: bool) -> ClumpCatalog  # threshold relative to mean |density|; periodic label merging across edges",
        "def nn_distance(points: np.ndarray, box: float | None) -> np.ndarray  # cKDTree, periodic via tiling when box given",
        "def segregation_index(rho_pos: np.ndarray, rho_neg_abs: np.ndarray) -> float  # Pearson correlation of log(1+rho/mean)",
        "def void_fraction(rho: np.ndarray, thr_factor: float = 0.2) -> float",
        "def power_spectrum(rho: np.ndarray, box: float) -> tuple[np.ndarray, np.ndarray]  # (k, P(k)) shell-averaged",
        "def peak_scale(k: np.ndarray, pk: np.ndarray) -> float  # 2pi/k_peak with k=0 excluded",
        "def pattern_persistence(rho_ref: np.ndarray, rho_now: np.ndarray) -> float  # normalised cross-correlation",
        "def lss_structure_row(ps: ParticleSet, t: float, box: float, ngrid: int, dim: int, threshold: float, min_cells: int, ref_grid: np.ndarray | None) -> tuple[dict, np.ndarray]  # (row with n_clumps_neg, mass_mean/std, diam_mean/std, d_cell, phi_over_d, n_clumps_pos, segregation, void_frac_pos, lambda_peak_pos, lambda_peak_neg, persistence; current neg grid for use as ref)",
        "def fourier_modes(pos: np.ndarray, w: np.ndarray, centre: np.ndarray, rbins: np.ndarray, mmax: int = 4) -> tuple[np.ndarray, np.ndarray]  # A[m, bin] amplitudes normalised by bin mass, phases[m, bin]",
        "def bar_strength(A2: np.ndarray, rbins: np.ndarray, r_inner: float) -> float  # max A2 for R< r_inner",
        "def spiral_strength(A2: np.ndarray, rbins: np.ndarray, r_inner: float, r_outer: float) -> float  # mass-weighted mean A2 in the annulus",
        "def pitch_angle(phase2: np.ndarray, rbins: np.ndarray, r_inner: float, r_outer: float) -> float  # degrees from d(phi2/2)/d ln R by unwrapped least squares; nan if A2 too small",
        "def pattern_speed(phase_now: float, phase_prev: float, dt: float, m: int = 2) -> float",
        "def measured_toomre_q(ps: ParticleSet, centre: np.ndarray, rbins: np.ndarray, G: float) -> np.ndarray",
        "def disk_structure_row(ps: ParticleSet, t: float, centre: np.ndarray, rd: float, rmax: float, box: float, G: float, prev: dict | None, ring_grid_n: int, threshold: float, min_cells: int) -> tuple[dict, dict]  # (row with bar_A2, spiral_A2, A1, A3, A4, pitch_deg, omega_p, Lz_disk, Lz_twin_abs, vrot_disk, Q_at_2rd, n_clumps_twin, escaped_frac; state dict with phase/time for the next call)"
      ]
    }
  },
  {
    "file": "plotting.py",
    "purpose": "All figure output with the Agg backend: species snapshots (scatter + density panels, 3-D projections), conservation time series, LSS structure time series, disk mode time series, rotation curve, and sweep/ensemble comparison plots.",
    "interface": {
      "imports": ["particles", "forces_pm", "diagnostics_core"],
      "exposes": [
        "def snapshot(ps: ParticleSet, box: float, t: float, t_ref_name: str, path: str, dim: int, centre: np.ndarray | None = None, extent: float | None = None, title: str = '') -> None  # 2-D: two scatter panels (ordinary, twin) + two density panels; 3-D: xy/xz/yz projections per species",
        "def plot_conservation(csv_path: str, out_png: str, expansion: bool) -> None  # dE_rel, P_abs and P_signed drift, L_abs/L_signed, virial",
        "def plot_lss_structure(csv_path: str, out_png: str, t_ref_name: str) -> None",
        "def plot_disk_modes(csv_path: str, out_png: str) -> None  # bar/spiral A2, pitch, omega_p, Lz_disk vs Lz_twin, n_clumps_twin",
        "def plot_rotation_curve(rc: dict, out_png: str) -> None",
        "def plot_sweep(rows: list[dict], param: str, metrics: list[str], out_png: str, logx: bool = False) -> None",
        "def plot_ensemble(rows: list[dict], metrics: list[str], out_png: str) -> None"
      ]
    }
  },
  {
    "file": "scenarios.py",
    "purpose": "Run orchestration: build the force evaluator (PM or direct, external field), build ICs, integrate with logging/snapshots/structure measurements at the configured cadence, evaluate the pre-registered criteria, write config/assumptions/CSVs/summary.json/plots. Shared loop for both scenarios.",
    "interface": {
      "imports": ["config", "particles", "forces_pm", "external_potentials", "integrator", "initial_conditions", "diagnostics_core", "structure_stats", "plotting"],
      "exposes": [
        "def build_forces(cfg: RunConfig, der: Derived) -> tuple[ForceEvaluator, PMSolver | None, ExternalField]",
        "def write_run_metadata(cfg: RunConfig, der: Derived, outdir: str) -> None  # config.json, assumptions.txt",
        "def run_loop(cfg: RunConfig, der: Derived, ps: ParticleSet, force: ForceEvaluator, outdir: str, structure_fn, snapshot_kwargs: dict) -> dict  # generic time loop; structure_fn(ps, t, state)->(row, state); returns dict with final diagnostics row, structure rows, timing",
        "def lss_criteria(structure_rows: list[dict], diag_rows: list[dict], der: Derived) -> dict  # pre-registered: cells_form, stable_clumps, persistence, verdict strings 'met'/'not met'/'ambiguous'",
        "def galaxy_criteria(structure_rows: list[dict], diag_rows: list[dict], der: Derived, rc_info: dict) -> dict  # sustained_spiral, bar, braking, ring_fragmentation, vc_undefined_fraction, verdict strings",
        "def run_lss(cfg: RunConfig) -> dict  # returns summary (also written to summary.json)",
        "def run_galaxy(cfg: RunConfig) -> dict",
        "def run_simulation(cfg: RunConfig) -> dict  # dispatch on cfg.scenario"
      ]
    }
  },
  {
    "file": "sweeps.py",
    "purpose": "Parameter sweeps, seed ensembles and the fixed convergence suite: runs scenarios in subfolders, collects summary metrics into CSV, computes mean/std across seeds, and produces comparison plots. Reports all values without selection.",
    "interface": {
      "imports": ["config", "scenarios", "diagnostics_core", "plotting"],
      "exposes": [
        "SUMMARY_METRICS_LSS: list[str]  # n_clumps_neg_final, n_clumps_change_5_20, segregation_final, persistence_final, phi_over_d, lambda_peak_pos, dE_rel_max, dP_abs_rel_max, wall_time",
        "SUMMARY_METRICS_GALAXY: list[str]  # spiral_A2_mean_late, bar_A2_final, rotations_with_spiral, Lz_disk_change, Lz_twin_change, n_clumps_twin_final, pitch_deg_mean, dE_rel_max, wall_time",
        "def parse_values(param: str, values: str) -> list",
        "def run_sweep(base: RunConfig, param: str, values: list, outdir: str) -> list[dict]  # writes sweep_<param>.csv and .png",
        "def run_ensemble(base: RunConfig, seeds: list[int], outdir: str) -> dict  # per-seed rows + mean/std table ensemble.csv, ensemble.png",
        "def convergence_plan(scenario: str) -> list[tuple[str, list]]  # the fixed dt/eps/grid/method/bc sets",
        "def run_convergence(base: RunConfig, outdir: str) -> dict  # runs the plan, writes convergence.csv and one plot per parameter",
        "def flatten_summary(summary: dict, scenario: str) -> dict  # picks the metric list from summary.json content"
      ]
    }
  },
  {
    "file": "main.py",
    "purpose": "Command-line entry point: argparse with subcommands lss, galaxy, sweep, ensemble, convergence, assumptions; builds RunConfig from arguments, resolves defaults, prints the assumption list and runtime warnings (e.g. direct method at large N), creates the output folder and dispatches.",
    "interface": {
      "imports": ["config", "scenarios", "sweeps"],
      "exposes": [
        "def build_parser() -> argparse.ArgumentParser",
        "def add_common_options(p) -> None",
        "def add_lss_options(p) -> None",
        "def add_galaxy_options(p) -> None",
        "def config_from_args(args, scenario: str) -> RunConfig",
        "def main(argv: list[str] | None = None) -> int",
        "if __name__ == '__main__': sys.exit(main())"
      ]
    }
  }
]
```