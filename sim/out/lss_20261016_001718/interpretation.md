# Review of run `lss / mix_box32_expansion_long` (seed 1)

**What was run.** 2‑D particle‑mesh (256² grid, periodic unit box), 6.4 M positive + 6.4 M negative particles, |M₋|/M₊ = 64, σ₋/σ₊ = 4, Poisson initial positions, 1/r² in‑plane force, fixed leapfrog step 0.02 t_J,neg, 3000 steps to 60 t_J,neg. The box is 32 twin Jeans lengths (λ_J,neg = 1/32) and only 8 positive Jeans lengths (λ_J,pos = 1/8) across. A kinematic comoving expansion a ∝ t^{2/3} is switched on at t = 1 t_J,neg and reaches a = 15.3 by the end. This is one seed; no ensemble statistics are in the data provided.

---

## 1. What the images show

**t = 0.** Both populations uniform; the density maps show pure shot noise at log₁₀(1+ρ/⟨ρ⟩) ≈ 0.30 ± 0.02, i.e. ρ = ⟨ρ⟩ with ~5 % Poisson scatter (~390 particles per 128‑grid cell). Consistent with the stated ICs.

**t = 3.** Structure is already fully developed in the positive population: a honeycomb of walls/filaments enclosing roughly 150–250 roughly circular cells of size ~0.04–0.08 L, with knots at the vertices. The twin population looks nearly uniform in the scatter plot (6.4 M points saturate the panel) but its density map shows a comparable number of small, dense knots (ρ/⟨ρ⟩ up to ~27) sitting inside the positive‑matter cells, connected by faint filaments. Positive cells and twin knots are spatially complementary — the pattern Petit describes.

**t = 6–9.** Twin knots become visible as compact clumps in the scatter plot (cores ~0.02–0.04 L) surrounded by sprays of diffuse particles; empty twin regions open up where positive‑matter knots sit. Positive matter thickens into walls, then thins into filaments; cells grow to ~0.1 L. Many small twin clumps have merged by t = 9.

**t = 13–28.** Hierarchical merging: cells enlarge to ~0.15–0.25 L, positive filaments show thin multi‑stream caustic sheets (concentric arcs around twin clumps), twin clumps merge into fewer, larger objects with increasingly extended diffuse envelopes. The twin envelopes begin to overlap the positive filaments in the scatter plots, though the twin *cores* still sit in positive‑matter holes.

**t = 32–60.** The comoving pattern essentially freezes. Snapshots at t = 35, 41, 47, 54, 60 are nearly indistinguishable: ~15–20 large positive cells (0.2–0.3 L; some span a third of the box), ~22 twin clump complexes, plus a few small isolated twin clumps (e.g. near (0.60, 0.51), (0.36, 0.22), (0.88, 0.77)) that persist at the same comoving position from t ≈ 9 to t = 60 without merging. Positive matter is in thin filaments with dense knots at intersections and almost nothing inside the cells; twin matter is in compact cores (ρ/⟨ρ⟩ ≈ 200) with broad low‑density halos.

**End state:** positive matter in a filamentary network, twin matter in isolated clumps at the network's cell centres, the two anti‑correlated at the core/void level but with substantial overlap between twin halos and positive filaments.

---

## 2. Images vs. measured data

| Quantity | Data | Images | Verdict |
|---|---|---|---|
| Initial density scatter | — | 0.30 ± 0.02 in log | consistent with Poisson shot noise |
| Onset of segregation | index −0.79 at t = 0.5, min −0.88 at t = 1.5 | no visible structure until t = 3 | not a contradiction: the index is amplitude‑independent and is picking up anti‑phased linear‑regime fluctuations before they are visible; the t = 3 image confirms them |
| n_clumps_neg | 0 at t ≤ 2.5, 11 at t = 3, peak 86 at t ≈ 7–8 | ~200 twin knots visible at t = 3; ~120–150 at t = 6–9 | **mismatch in level, not in timing**: the finder (128 grid, ρ > 5⟨ρ⟩, ≥ 4 cells) misses early knots smaller than ~0.02 L. Early clump counts are lower limits |
| n_clumps decline 86 → 22 (neg), 56 → 29 (pos) | monotonic from t ≈ 8 | merging visible in every interval | consistent; the decline is merging, not dissolution |
| mass_frac_clumps_neg = 0.78; mass_mean 2.26 (×M₊) × 22 ≈ 50 ≈ 0.78 × 64 | — | compact cores plus diffuse envelopes | consistent; ~22 % of twin mass is in the halos/sprays |
| diam_mean 0.048 ± 0.012 | — | cores ~0.03–0.06 L | consistent |
| void_frac_neg 0.77, void_frac_pos 0.45 | — | twin density map mostly white; pos map ~half white | consistent. (The twin *scatter* panels look much fuller because 6.4 M points saturate at low density — the density maps are the reliable guide) |
| λ_peak = 1/3 box for both populations at end (rising 0.06–0.09 → 0.1 → 0.2 → 0.25 → 0.33) | — | largest cells ~0.3 L | consistent — and a box‑size warning (see §4) |
| segregation weakening −0.88 → −0.27 | — | twin halos spreading over positive filaments; both fields dominated by empty cells | consistent; the index's exact definition matters (two mostly‑empty fields can lose anti‑correlation even if cores remain anti‑correlated) |
| persistence falls 1.0 (t = 5) → 0.09 (t = 20) → 0.017 (t = 30), then flat ~0.013–0.018 | — | images frozen after t ≈ 35 | consistent: the field decorrelates from the t = 5 reference by merging, then stops changing. A reference at t = 30 would almost certainly give persistence near 1 |
| σ₊: 0.354 → 0.047; σ₋: 1.41 → 0.146 | — | positive matter forms razor‑thin caustics (cold) | consistent with expansion cooling (v_pec ∝ 1/a would give 0.023 and 0.092; the excess is collapse heating) |
| dE_rel_max = 70 | — | — | **not interpretable**: KE + PE is not conserved in a comoving frame with Hubble drag (Layzer–Irvine term absent). This run therefore contains no usable check of integration accuracy. |P|‑weighted momentum is conserved to 10⁻¹⁶, as expected for a symmetric PM kernel |
| `stable_clumps` verdict "not met" | n(5→20) change = −19.4 % vs threshold 20 % | — | **marginal / ambiguous**: the stated 5→20 change is *inside* the threshold; the verdict must rest on the trend term (−1.1 clumps per t_J) or on the peak‑to‑end drop (86 → 50 = −42 % within the window). The criterion as applied should be stated precisely |

---

## 3. Comparison with the original claims

**Supported by this run (given the three‑law interaction and 2‑D):**
- A cellular pattern forms, fast (within ~3 t_J,neg, i.e. < 0.4 t_J,pos) and robustly: positive matter is confined to walls around twin clumps. The pre‑registered `cells_form` criterion is met with large margin (−0.68 at t = 3 vs threshold −0.3). The mechanism is as Petit describes: a twin density contrast of only a few percent exerts a force on positive particles 64× stronger than their own self‑gravity, so the light population is swept into walls while the heavy population is still barely visibly clumped.
- Twin clumps, once formed, are not destroyed: some small isolated ones survive at fixed comoving position for ~50 t_J,neg, and 78 % of the twin mass ends in bound clumps.
- The pattern survives a factor‑15 kinematic expansion (though see caveat below).
- Clump‑population scatter is finite and measurable (mass std/mean ≈ 0.57, diameter std/mean ≈ 0.25), which is the kind of statistic Petit reported.

**Contradicted or not supported:**
- "Stable" in the sense of a fixed clump population is **not** what happens: the clump count peaks at 86 around t ≈ 7–8 and declines by ~75 % to 22 through hierarchical merging; the cell scale grows by ×4. Both pre‑registered stability criteria (`stable_clumps`, `persistence`) fail, though the first is marginal on the stated numbers and the second is sensitive to the choice of reference time.
- The late‑time segregation is weak (−0.27): twin halos and positive filaments overlap substantially. The clean "twin inside, ordinary matter in walls" picture holds for the cores but not for the diffuse twin component.
- The eventual freezing of the pattern is caused by the imposed expansion (comoving forces scale as a⁻³ and peculiar velocities decay), not by a dynamical equilibrium. In the absence of expansion one would expect merging to continue until the box scale is reached. Petit's runs had no expansion and an unspecified duration; his "remarkably stable" may simply correspond to the t ≈ 5–10 window here, where the count changes slowly.

**Cannot decide from this run:**
- Whether the late‑time positive‑matter filament network is twin‑driven or partly positive‑matter self‑gravity: 60 t_J,neg = 7.5 t_J,pos, long enough for positive matter to collapse on its own. The t = 3 cells are unambiguously twin‑driven; the t = 60 network is not. The pos‑only control (mentioned in the notes but not supplied) is needed.
- Any resemblance to the observed cosmic web: no correlation function, power spectrum shape or void statistics are compared to data, and the box contains only ~15 cells at the end.
- 3‑D behaviour, dependence on mass/velocity ratios, reproducibility across seeds (single run), integration accuracy (energy diagnostic unusable), and anything about galaxy formation inside the walls.

---

## 4. Visual signs of numerical artefacts

- **Box‑size limitation (the main one).** λ_peak reaches 1/3 of the box in both populations by t ≈ 45 and the largest cells span ~0.3 L; the positive population has only 8 Jeans lengths across the box from the start. The final cell size, clump count and d_cell are set by the box plus the expansion freeze‑out, not by physics alone.
- **Periodic boundaries.** Filaments continue seamlessly across edges (e.g. the positive filament at x ≈ 0.3 crossing y = 0/1 at t ≥ 35) — expected, but with structures at 1/3 box the images of a clump interact with it.
- **No grid‑aligned features** are visible: filaments have arbitrary orientations, no preferred axes in clump positions. Good.
- **Thin caustic sheets** in positive matter at t ≥ 13 are about one structure‑grid cell wide. They are physically expected for cold collapse in 2‑D, but their thinness means they are at the resolution limit, and the extremely cold positive population (σ₊ → 0.047) is partly an artefact of the kinematic expansion recipe.
- **Diffuse twin sprays/halos** around clumps from t ≈ 9 onward could be genuine (hot population, mergers eject material) or numerical heating in cores with ρ/⟨ρ⟩ ≈ 200, where the local dynamical time (~0.07 t_J,neg) is covered by only ~3–4 fixed timesteps. Undetermined.
- **Clump‑finder resolution** undercounts early clumps by an order of magnitude (visible vs counted at t = 3).
- **White‑noise ICs**: Poisson shot noise seeds all scales equally; with no imposed spectrum the only length scales are λ_J,neg, λ_J,pos and the box, so cell sizes carry no cosmological information.
- **Expansion onset at t₀ = 1 t_J,neg** is an arbitrary choice; the whole non‑linear phase is then strongly damped (a goes 1 → 15), which is why the run is "long" in t_J but short in growth (ln a ≈ 2.7 e‑folds after t₀).
- **Energy**: dE_rel ≈ 70 is meaningless without a Layzer–Irvine correction; no convergence in dt or grid is demonstrated.

---

## 5. Most informative next runs

1. **No‑expansion twin (same seed, same ICs).** Determines whether freezing/merging and the clump count plateau are expansion effects or intrinsic. Likely already exists given the label; it should be reported alongside this one.
2. **Controls: pos‑only and neg‑only** with identical ICs. Decides whether the late filament network needs twin matter.
3. **Box‑size convergence**: 64 and 128 λ_J,neg boxes at fixed resolution per λ_J. If λ_peak keeps tracking the box, the "cell size" is unphysical; if it saturates, there is a real scale.
4. **Seed ensemble** (≥ 5–10 seeds) for n_clumps, d_cell, mass/diameter std — the reproducibility criterion that the notes say is evaluated separately.
5. **Numerical convergence**: dt/2, 512² grid, modest Plummer softening; report Layzer–Irvine energy residual, not raw dE.
6. **Better stability diagnostics**: track individual clumps (merger trees) to separate merging from dissolution; compute persistence with a sliding reference (e.g. t − 10) and report a decorrelation time; finer structure grid or lower min‑cells so early clumps are counted.
7. **Parameter sweep** in mass ratio (e.g. 1, 8, 64) and velocity ratio, to see whether the cellular morphology is generic to opposite‑sign repulsion or specific to Petit's chosen values.
8. **Quantitative statistics** (two‑point correlation / P(k) of each population, void‑size distribution) to permit any comparison with surveys or with ΛCDM 2‑D analogues, which Petit never provided.
9. **A small 3‑D run** at modest resolution, since 2‑D gravity (and the a⁻³ comoving scaling used here for a 2‑D sheet) does not transfer directly.

**Bottom line.** In this single 2‑D run, the three‑law interaction does produce Petit's cellular pattern quickly and robustly, and twin clumps that are not destroyed. It does *not* produce a stable clump population: the structure merges hierarchically up to the box scale and only stops because the imposed expansion damps the comoving dynamics. The pre‑registered "stable clumps" and "persistence" criteria fail, but one is marginal and the other depends on the reference epoch; neither distinguishes merging from dissolution. The result is box‑limited and lacks an energy check, so quantitative numbers (cell size, clump count) should not be taken as physical yet.