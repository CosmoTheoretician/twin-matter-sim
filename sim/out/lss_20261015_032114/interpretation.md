# Review of run `lss_20261015_032114` (label `mix_box16`, seed 1)

**What was run.** 2‑D particle‑mesh (128² grid, periodic unit box, no expansion), 1.6 M positive + 1.6 M negative particles, |M₋|/M₊ = 64, σ₋/σ₊ = 4, Poisson initial positions, fixed leapfrog step dt = 0.02 t_J,neg, 1000 steps to t = 20 t_J,neg. Snapshots every t_J,neg (snap 10 is absent from the set I was given; I treat t = 10 as uninspected). This is **one seed**; the ensemble/seed‑scatter command has not been run, so nothing below should be read as reproducible until it is.

---

## 1. What the images show

| t / t_J,neg | Negative ("twin") population | Positive population | Scale of structure |
|---|---|---|---|
| 0 | Pure shot noise (±~10 % per cell) | Same | — |
| 1 | Mild blotchiness, blobs ~0.03–0.05 L | Already a fine honeycomb: small voids with thin walls; peak log(1+δ) ≈ 0.75 | ~0.05 L (data: λ_peak = 0.14) |
| 2 | ~30–40 compact round clumps sitting inside the positive voids | Clear **cellular network**: walls ~0.01–0.02 L thick around round voids ~0.1 L, dense nodes (~25) at wall junctions | ~0.1–0.15 L |
| 3 | ~25 larger clumps (0.08 L), peak δ ≈ 20 | Cells now 0.15–0.2 L; walls thicker, more filamentary; nodes dissolving | d_cell ≈ 0.14 |
| 4 | ~13–15 clumps, pairing up | Cells 0.25–0.3 L; positive matter broad filaments, some sharp caustic‑like sheets | λ_peak → 0.5–1.0 |
| 5 | 4 clumps (one near (0.55,0.65), two bottom‑left, one at the corner) | A single connected region filling the rest of the box | Box‑scale |
| 6–8 | 2 → 1 clump; by t = 8 one clump straddling the y = 0/1 boundary at x ≈ 0.3 | A horizontal band y ≈ 0.3–0.6 spanning the full x‑range | Box‑scale |
| 9–20 | **One clump** at (0.3, 0 ≡ 1), peak δ ≈ 40, surrounded by a broad diffuse halo of negative matter that covers the entire box at low density | Diffuse band y ≈ 0.35–0.65 with internal knots that drift and re‑form (e.g. (0.7,0.4) → (0.9,0.5) → (0.65,0.35)); peak δ drops from ~11 to ~6 | Frozen at box scale |

Qualitatively: a cellular phase appears fast (visible by t ≈ 1, fully developed t ≈ 2–3), the cells coarsen by hierarchical merging of the negative clumps (t ≈ 3–8), and the system then sits in the generic end‑state of a non‑expanding periodic box — one clump plus its complement. The negative clump's position does not change from t ≈ 9 to 20 (11 t_J,neg), but the positive matter never settles into compact objects; it is a hot, churning diffuse band.

## 2. Images vs. measured data

**Agreements**
- `cells_form`: segregation index goes from +0.02 (t=0) to −0.66 at t = 0.5 and −0.77 at t = 1.5, i.e. the two populations anti‑correlate almost immediately; the t = 1–2 images show exactly that (negative blobs inside positive voids). Criterion "met" is consistent with the pictures.
- Clump counts: n_clumps_neg = 23 (t = 2.5), 21 (3), 13 (4), 4 (5), 2 (6–7), 1 (≥7.5) tracks what one counts by eye at t = 3, 4, 5, 6, 8+. n_clumps_pos = 25 at t = 2 matches the ~25 dark wall nodes in that frame, and its collapse to 0–3 later matches the loss of compact positive structure.
- d_cell 0.13–0.16 (t = 2.5–4) → 0.23–0.30 (t = 4.5–5) → 0.54 (t = 5.5) → undefined; λ_peak → 1.0 (box) from t ≈ 4.5–5. Both agree with the visual coarsening and the box‑scale band/clump afterwards.
- `stable_clumps` "not met" (4 → 1, −75 %) and `persistence` "not met" (0.013 vs. t = 5 reference) both agree with the visible merging between t = 5 and 9.
- Colour‑bar maxima: negative peak rises monotonically (0.9 → 1.6 in log), positive peak falls (1.1 → ~0.8); consistent with σ₋ rising 2 → 10.6 and σ₊ rising 0.5 → 7.8 (×15) — the positive matter is being heated and dispersed, the negative matter is forming one deep well plus a hot halo (mass_frac_clumps_neg plateaus at 0.40, i.e. 60 % of negative mass is diffuse; void_frac_neg drops from 0.54 at t = 5 to ~0.02 later, matching the red scatter that covers the whole box at late times).

**Mismatches / caveats**
- At t = 2 the image shows ~35 distinct negative blobs but the metric reports 3. This is a threshold effect (5×mean density, ≥4 cells): clumps at t = 2 peak at δ ≈ 7, so almost none clear the cut. Clump counts before t ≈ 2.5 are therefore not meaningful, and `n_clumps_max = 23` is a lower bound on how many cells actually formed.
- `persistence` is anchored at t = 5, in the middle of the merging phase. From t ≈ 9 onward the negative clump is visually static; a reference at t ≈ 9 would probably pass the 0.7 threshold. The criterion was fixed in advance so it stands, but the interpretation is "the t = 5 configuration did not persist", not "nothing is long‑lived". What is long‑lived is the box‑scale end‑state (see §4).
- Energy: dE_rel reaches 8.6 % (4 % at the end). That is large for a run meant to test "stability". No dt or grid convergence test accompanies the run, so the pages' "halving dt changes nothing" criterion was not actually applied. |m|‑weighted momentum is conserved to 1e‑15 (good); signed‑momentum drift is expected physics, as the assumptions note.
- virial = 2.08: KE far exceeds |PE|/2. Partly ambiguous for a mixed‑sign system, but together with σ₊ × 15 it says the late‑time state is hot, not a relaxed bound configuration.

## 3. Comparison with the original claims

**Supported (within this single run and its assumptions)**
- A 2‑D cellular pattern with the heavier self‑attracting population at cell centres and ordinary matter in walls *does* form, quickly (t ≲ 2–3 t_J,neg) and without any tuning beyond the three‑law force and the 64/4 ratios. The nodes at wall junctions and the rounded voids look much like the description in Colloque_2001_1 §3.
- The mechanism is as the investigation anticipated: the dense negative population clumps first and sweeps positive matter into the interstices.

**Contradicted (within this run)**
- "Stable" clumps / long‑lived cellular structure: the cells are a **transient**. Clump number falls from ≥23 to 1 over ~6 t_J,neg by hierarchical merging; the cellular phase lasts ~2–3 Jeans times. The run shows no mechanism that halts merging.
- Petit's galaxy‑formation sketch (walls compress, cool, fragment): the positive nodes that exist at t = 2 dissolve; positive matter is heated to σ ≈ 15× its initial value and never re‑condenses. Of course the run has no cooling, so this contradicts only the *collisionless* version of the scenario.
- Reported clump statistics (std of number/mass/diameter across randomized runs): not reproduced here — single seed, and mass_std/diam_std = 0 at the end only because there is one clump.

**Cannot decide**
- Whether a cellular phase would be frozen in by cosmic expansion (none here, as in the original). In ΛCDM‑style boxes it is expansion that sets the scale at which merging stalls; without it, a single clump is the inevitable end‑state of *any* self‑gravitating periodic box, with or without twin matter. So the merging seen here is not evidence *against* the claim either.
- Whether the positive‑matter walls would fragment with cooling.
- Anything about 3‑D, rotation curves, lensing — out of scope of this scenario.
- Note a dimensional subtlety: in a razor‑thin 2‑D sheet with 1/r² forces, t_J ∝ σ/(GΣ), so t_J,pos = 16 t_J,neg (not the 8 quoted from the 3‑D √64 argument). The positive population's own Jeans time is nearly the whole run length, and its initial λ_J (0.25 L) is a quarter of the box. All positive‑matter structure here is therefore induced by the negative population; the run cannot speak to self‑gravitating fragmentation of ordinary matter at all.

## 4. Numerical / setup artefacts visible or inferable

1. **Box‑size domination from t ≈ 5.** λ_peak = 1.0 (= box) for both populations from t ≈ 4.5–5 onward; the final positive band is a pure k = (0, 1) mode periodic in x, and the single negative clump sits exactly on the y = 0/1 seam (same object seen at top and bottom of the scatter plots). Everything after t ≈ 5 is a statement about a 16 λ_J box, not about an infinite medium. The label `mix_box16` itself flags this.
2. **Periodic images.** Besides the seam clump, corner‑straddling negative clumps at t = 5 ((0,0)/(1,1) and (1,0.95)) are single objects counted/seen in pieces.
3. **Energy drift 8.6 %** with fixed dt = 0.02 t_J,neg and no softening beyond the 1/128 cell: by t ≳ 10 the negative velocity dispersion (~10.6) means particles cross ~0.9 grid cells per step (12.5 would be exactly one). This is a CFL‑type regime where force errors feed heating; the σ₊ × 15 heating is probably mostly physical (repulsive sweeping) but its magnitude is not trustworthy without a dt/grid convergence test.
4. **Shot‑noise seeding** (ic_noise = 0, pure Poisson) puts power at all scales including the grid scale; the first structures (t = 1, ~0.05 L) are only ~6 grid cells across and 8 cells is λ_J,neg itself. Resolution of the earliest cellular phase is marginal.
5. **No grid imprint seen**: no axis‑aligned streaks or checkerboard in any density map; the t = 0 maps show white noise at the expected amplitude (~100 particles/cell → ~10 %). Good.
6. **Non‑neutral box with Jeans swindle** (net signed mass −63, k = 0 removed). This is standard but means the "mean field" subtracted is dominated by the negative population; the implications for a net‑negative‑mass universe (which has a cosmological solution in Petit's framework) are not represented.
7. **Clump detector threshold** (5× mean) undercounts early clumps (t ≤ 2) and makes n_clumps_max underestimate the true peak number of cells.
8. Missing snap 10 — not an artefact, just an inspection gap.

## 5. Most informative next runs (in priority order)

1. **Seed ensemble** (≥5 seeds, same setup) — the pre‑registered "reproducible" criterion and Petit's own reported run‑to‑run scatter cannot be assessed from one seed.
2. **Box‑size scan at fixed λ_J,neg** (box = 32, 64 λ_J) and/or **expansion on** (`expansion: true`, a ∝ t^{2/3}). These two together decide the central question: is the cellular phase a transient that always coarsens to the box, or does a characteristic cell size/clump count freeze out? Measure n_clumps per unit area vs. time and d_cell(t) and see whether they collapse onto one curve.
3. **Convergence**: dt/2 and dt/4, grid 256², Plummer ε ≈ 1–2 cells. Require dE_rel ≲ 1 % before trusting late‑time dispersions and the merging rate.
4. **Controls already supported by the code**: `neg_only` and `pos_only`. If the negative‑only run shows the same clump mass function and merging timeline, the cellular appearance is just ordinary hierarchical clustering of the heavy population with the light population passively filling voids — which is still Petit's mechanism, but it tells you whether the cross‑repulsion adds anything dynamically (e.g. the positive heating).
5. **Parameter scan** of mass_ratio and vel_ratio (and Gaussian‑field ICs instead of Poisson) to see whether cell size scales with λ_J,neg as the Jeans argument predicts (currently d_cell ≈ 2–2.5 λ_J,neg at t = 2.5–4, a single data point).
6. **Metric upgrade**: track clumps across snapshots (lifetimes, merger tree) and compute persistence with a sliding reference, so that "stable" can be quantified independently of the merging phase.
7. Later, a **3‑D** run at lower N: the 2‑D sheet changes both the Jeans scalings (t_J ratio 16 vs. 8) and the merging dynamics, so no 2‑D result transfers directly.

**Bottom line.** This single run reproduces the *formation* of the cellular, segregated pattern described on the site, quickly and robustly, but not its *stability*: the pattern coarsens by merging within a few Jeans times and ends in the box‑limited one‑clump state that any non‑expanding periodic self‑gravitating box reaches. Because the run has no expansion, a 16 λ_J box, a single seed, and 8 % energy drift, it neither confirms nor refutes the long‑lived‑structure claim; the box‑size/expansion scan and the seed ensemble are what would settle it.