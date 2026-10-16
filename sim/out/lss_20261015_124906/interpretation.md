# Review of run `mix_box16_expansion_long` (LSS scenario, seed 1)

**Setup recap (from `summary.json` / `assumptions.txt`):** 2‑D particle‑mesh, 128² grid, periodic unit box, 1.6 M particles per population, |M₋|/M₊ = 64, σ₋/σ₊ = 4, Poisson initial conditions with full shot noise, λ_J,neg = box/16, fixed leapfrog step 0.02 t_J,neg, kinematic expansion a ∝ t^{2/3} switched on at t = 1 t_J,neg, run to t = 60 t_J,neg (a_final ≈ 15.3). This is a single seed; everything below is subject to that.

---

## 1. What the images show over time

- **t = 0.** Both populations are uniform Poisson. The density panels are pure shot noise (log₁₀(1+ρ/⟨ρ⟩) ≈ 0.30 ± 0.04, i.e. ρ/⟨ρ⟩ = 1 ± ~10%, as expected for ~100 particles per grid cell).
- **t = 3.** Structure is already fully developed. Ordinary matter forms a closed cellular network of walls and nodes with ~50 cells of size ~0.1–0.15 (≈2 λ_J,neg). Twin matter sits inside the cells: ~40 compact round clumps plus a diffuse component. The two populations are visibly complementary.
- **t = 6–9.** Twin clumps become denser (peak contrast ρ/⟨ρ⟩ ≈ 50–80) and start merging; the ordinary‑matter cells enlarge, walls thin into filaments, and the nodes become the dominant ordinary‑matter structures. Sharp caustic‑like edges (multi‑stream folds) appear in the ordinary‑matter scatter plots.
- **t = 13–28.** Hierarchical coarsening: twin clumps merge from ~20 down to ~12, each surrounded by a puffy, hot particle halo in the scatter plots; a diffuse twin haze fills the box. Ordinary matter keeps a cellular topology but with 6–10 large cells, thick multi‑stream walls and clumpy nodes.
- **t = 32–60.** The twin population settles to ~7 clumps (one large merger product at ≈(0.55, 0.62), others at ≈(0.08, 0.35), (0.25, 0.10), (0.57, 0.83), (0.80, 0.93), (0.90, 0.93), (0.97, 0.07)); several of these stay within ~0.05 of their t = 32 positions through t = 60. Ordinary matter is a coarse web with 3–4 box‑scale voids, thick walls showing arcs/shells, and ~12–15 nodes. The gross topology changes slowly after t ≈ 35, but it is still evolving (mergers, node drift).

**Where things end up:** ~80% of twin mass is in a handful of compact clumps inside voids of the ordinary‑matter web; ordinary matter is in walls/nodes threading between them; a ~20% diffuse twin component permeates the walls.

---

## 2. Images vs. measured data

**Agreement**
- Clump counts: `n_clumps_neg` = 38 at t = 6, 14 at t = 20, 7 from t ≈ 42 onward; `n_clumps_pos` ≈ 35 → 14. Both match visual counts within a few.
- Timing: first clumps detected at t ≈ 2–2.5, maximum (53) at t = 4; images at t = 3 show a fully formed pattern.
- Mass concentration: `mass_mean` of twin clumps grows 0.3 → 7.3 (M₊ units), `mass_frac_clumps_neg` → 0.80, `void_frac_neg` → 0.78, twin colour‑bar maximum rises from 1.4 to 2.5 dex. All consistent with the merging, dense blobs in the panels.
- `diam_mean` ≈ 0.07 (≈9 grid cells) is the thresholded core; the scatter plots show larger (~0.12) halos, which is not a contradiction.
- Initial density‑map noise level agrees with the particle count per cell.

**Points to flag**
- The **segregation index** reaches its most negative value (−0.79) at t = 1.5, *before* any clump is detected, then weakens monotonically to −0.23 at the end. The "cells_form" criterion (−0.3 by t = 3) is therefore met essentially by the linear‑regime response of the two fields to each other's noise, and would **not** be met if evaluated at t = 60. The images explain the decline: thickening ordinary‑matter walls, hot twin halos, and the diffuse twin haze all reduce pixel‑level anti‑correlation even though the eye still sees complementary distributions. The criterion as fixed is thus a weak test.
- **Persistence** (correlation with the t = 5 field) decays to ~0 by t ≈ 32 and stays there. This is accurate but partly a consequence of the reference choice: t = 5 has 45 small clumps, and merging/drift necessarily decorrelates pixel maps. Visually the largest twin clumps and the void topology are recognisably the same from t ≈ 32 to 60; a sliding‑reference or clump‑tracking metric would likely show substantial late persistence. The data neither confirm nor deny this as logged.
- `lambda_peak_neg` = 1.0 and `lambda_peak_pos` = 0.5 at the end: the dominant power is at the box scale (see §4).
- `d_cell` ≈ 0.19 is smaller than the visual void size at t = 60 (~0.3–0.4); without the definition I cannot say whether this is a mismatch or a metric that measures clump spacing rather than void diameter.
- `dE_rel_max` = 65.6 looks alarming but is **uninterpretable here**: energy is not conserved in a comoving frame with Hubble drag, and the initial E of a Poisson box with the k = 0 mode removed is near zero, so the ratio is ill‑conditioned. `dP_abs_rel` ≈ 4×10⁻¹⁶ shows the force solver is symmetric to machine precision; the signed‑momentum drift (0.2%) is expected physics for cross‑sign pairs.

---

## 3. Outcome vs. the original claims

| Claim | Run supports | Run contradicts | Run cannot decide |
|---|---|---|---|
| **Cellular pattern** (twin clumps in cell centres, ordinary matter in walls) | Yes, robustly and early (t ≈ 3). The complementary web/clump morphology is exactly the qualitative picture. | — | Whether it resembles observed LSS (no statistical comparison; 2‑D; kinematic expansion only). |
| **"Heavier population clumps first and confines ordinary matter"** | Twin clumps are denser and hold more mass. | The ordering is not clean: clumps are detected in both populations at the same time (t ≈ 2–2.5, with *more* ordinary‑matter clumps initially), and the anti‑correlation is set up in the linear regime before either clumps. It looks like a coupled instability, not a two‑stage process. | — |
| **Stable clumps** | Clumps are persistent *objects*: they merge rather than dissolve, and their mass fraction grows. | The fixed criterion fails: the count drops 69% between t = 5 and 20 and keeps falling to 7 by t ≈ 42. The configuration is hierarchical, not stationary. | Whether a stable count exists in a larger box or without expansion. |
| **Long‑lived structure** | A cellular topology still exists at t = 60; the largest clumps and voids change little after t ≈ 35. | Persistence against t = 5 is ~0; the specific pattern is not preserved. | Whether the late slow‑down is intrinsic or caused by Hubble drag (a grows 5 → 15 over t = 13–60) and by the box being reached. This is the main open question. |
| **Clump‑statistic spreads** (Petit's quoted σ of number/mass/diameter) | — | — | Single seed; `mass_std/mass_mean` ≈ 0.5, `diam_std/diam_mean` ≈ 0.24 at the end but nothing to compare to. |

The run is at far higher resolution than the 1990s 5,000‑point experiments, so the qualitative morphology is clearly not a small‑N artefact. It says nothing about rotation curves, lensing, or 3‑D behaviour.

---

## 4. Numerical‑artefact indicators

- **Box‑size limitation (most serious).** Only 16 λ_J,neg per side. By t ≈ 30 the peak wavelength is ⅓–½ box and by the end it is the box itself; the t ≥ 32 panels show voids of ~0.4 box. Late‑time clump number (7) and "stability" are constrained by the periodic box, not by physics. Structures visibly wrap across edges (twin clumps at x ≈ 1, y ≈ 0.07 and y ≈ 0.93; ordinary walls crossing x = 0/1).
- **Expansion‑induced freezing.** The kinematic a ∝ t^{2/3} is not a Janus cosmology; its drag reduces velocity dispersions ~10× (σ₋ 2 → 0.2) and slows comoving growth. The late‑time quiescence may be largely this.
- **Shot‑noise seeding at grid scale.** Poisson ICs put white noise at all scales; λ_J,neg is only 8 cells and the first clumps are ~3 cells across. Early clump counts (and the 53 maximum) probably depend on grid and N.
- **Resolution of clump cores.** Softening = 1 cell (0.0078); clump cores are 2–3 cells and contrast reaches ρ/⟨ρ⟩ ≈ 300, where the local dynamical time is ~17× shorter than t_J,neg while dt is fixed at 0.02 t_J,neg. Core structure and merger timing are marginally resolved; a dt and grid convergence test is needed.
- **No obvious grid‑aligned pattern.** The t = 3 wall orientations are varied; I see at most a mild hint of axis alignment, not conclusive.
- **Energy diagnostic unusable** in this configuration (see §2); the clean momentum conservation is the only solver sanity check available.
- The multi‑stream arcs in the ordinary‑matter scatter plots are expected collisionless caustics, not artefacts.

---

## 5. Most informative next runs

1. **Same seed, expansion off** — separates intrinsic late‑time stability from Hubble‑drag freezing, and makes the energy diagnostic meaningful.
2. **Larger box (≥64 λ_J,neg, grid ≥512²)** — pushes the box‑scale crossing beyond t = 60 and gives enough cells/clumps for statistics (cell‑size distribution, clump mass function vs time).
3. **Seed ensemble** (the summary notes this facility exists) — required before any statement about clump number spreads or reproducibility.
4. **Controls: pos‑only and neg‑only** with identical Jeans parameters — tests whether the cross‑repulsion adds anything beyond a cold self‑gravitating web plus an independently merging hot population.
5. **Convergence: dt/2, grid 256², smoothed (glass or low‑noise) ICs** — checks the early clump count and merger timing.
6. **Better persistence metrics** — sliding‑window correlation and clump tracking (merger tree) to distinguish merging from dissolution, and a segregation index measured on smoothed fields so the late‑time wall/halo thickening is not misread as loss of segregation.
7. **Parameter scan in mass ratio and velocity ratio** — the pages assert morphology depends on these.
8. Eventually **3‑D**, which the original author himself identified as the untested regime.

**Bottom line for this single run:** the three‑law scheme does produce a cellular ordinary‑matter web with twin clumps in the voids, quickly and robustly. It does not produce a stationary set of clumps; instead the twin population merges hierarchically until the box scale is reached, and the apparent late‑time calm is confounded by box size and imposed expansion. Two of the three pre‑registered criteria fail as defined, but the definitions (reference time t = 5, pixel‑level segregation) are not ideal tests of what "stable" and "long‑lived" plausibly meant. The decisive runs are the no‑expansion and larger‑box comparisons.