# Review of run `mix_n800k_g64` (LSS, seed 1)

Single run, seed 1, 2‑D PM, 64² force grid, 400k + 400k particles, |M₋|/M₊ = 64, σ₋/σ₊ = 4, no expansion, periodic unit box, 20 t_J,neg. Snapshot 10 is absent from the set supplied (t = 0…9, 11…20), so one frame of the late phase is interpolated below. Everything that follows is about this one realisation unless `structure.csv` says otherwise.

---

## 1. What the images show, in time order

**t = 0.** Both populations are Poisson‑uniform; density maps show only shot noise (log₁₀(1+ρ/⟨ρ⟩) within ±0.15 of the mean, i.e. the expected √N per‑cell noise).

**t = 1.** The *positive* population already shows a foam of small voids (~0.05–0.1 L) and ridges, with density contrast up to log 0.8 (ρ/⟨ρ⟩ ≈ 5). The *negative* population at the same time is only faintly structured (colourbar max 0.52, barely above noise). The cold positive gas (σ₊ = 0.71, λ_J,+ = 0.5 L, so it cannot be self‑collapsing at this scale) is reacting to the potential of the heavy negative shot noise: it flows out of incipient negative overdensities before those overdensities are themselves visible. The structure scale is roughly λ_J,neg = 0.125 L and below.

**t = 2.** This is the frame that most resembles the claim. Positive matter forms a connected network of walls and nodes around ~12–15 roughly circular voids of 0.15–0.3 L; negative matter sits in ~8–10 compact blobs located at void centres (e.g. (0.4,0.78), (0.65,0.8), (0.4,0.3), (0.2,0.05), (0.93,0.42)). Walls are thin (~0.02 L) and sharp‑edged.

**t = 3.** Negative blobs have merged into ~7 larger ones (diameter ~0.1–0.15 L); positive cells have correspondingly coarsened to ~0.3–0.4 L. Positive walls now show caustic‑like straight edges and tapered "lens" voids — the signature of cold shell‑crossing collapse in 2‑D. Positive peaks reach log ≈ 1.0 (10× mean).

**t = 4–5.** Hierarchical merging continues: 4 negative clumps at t = 4, 3 at t = 5 (two in the lower left plus one straddling the x = 0/1 boundary at y ≈ 0.43). Positive walls are now diffuse bands; the clear cellular network is gone, replaced by two or three big voids and broad overdense regions.

**t = 6–7.** The remaining negative clumps merge into one elongated structure along y ≈ 0.37 that spans half the box width, then relaxes to a single round clump at (0.25,0.37); a remnant at the right/left edge (x ≈ 1.0, y ≈ 0.35) is still visible at t = 7 and dissolves into the main clump by t ≈ 9 via the periodic boundary. Positive matter is now a diffuse sea (log ≈ 0.3–0.5 nearly everywhere) with one large void centred on the negative clump.

**t = 8–20.** The picture freezes. One negative clump at (0.22,0.37), peak log ≈ 1.4 (ρ/⟨ρ⟩ ≈ 25), embedded in a faint uniform negative haze; one positive void of radius ~0.25 L around it; elsewhere positive matter is a smooth hot background with weak, shifting overdensities (log 0.8–0.9, ρ/⟨ρ⟩ ≈ 6) in the upper right and at (0.7,0.15) that come and go between frames. No walls, no cells, no second void.

Net: structure appears at t ≈ 1, a genuine cellular pattern exists at t ≈ 2–3, it is destroyed by merging and heating between t ≈ 3 and 7, and from t ≈ 8 the system is a single negative clump plus a single positive void, static for ≥12 t_J,neg.

---

## 2. Images vs. measured data

| Quantity | Data | Image check |
|---|---|---|
| n_clumps_neg (5× threshold, ≥4 cells of 128²) | 0 until t=1.5; 5 (t=2), 8 (2.5), 7 (3), 5, 4, 3 (4.5–5), 1 from t=5.5 | Matches well: ~7 blobs at t=3, 4 at t=4, 3 at t=5, 1 from t=6/7. The count of 5 at exactly t=2 undercounts what the eye sees (~8–10); some blobs are just below 5× mean. |
| Segregation index | −0.50 at t=0.5, −0.67 at t=1, min −0.72 at t=1.5, relaxes to −0.5…−0.58 during merging (t 2–5.5), recovers to −0.70 by t=20 | Consistent: anticorrelation is strongest in the fine‑grained early foam, weakens while the pattern is reorganising, and is strong again in the final one‑clump/one‑void state. Note the "cells_form" trigger fires at t = 0.5, *before* anything cell‑like is visible; the index is a correlation coefficient and responds to low‑amplitude linear anticorrelation. |
| Final clump mass 25.5 (= 0.40 |M₋|), diameter 0.224 L, constant to ±0.5 % from t=8 to 20 | Red core diameter in the maps ≈ 0.2–0.25 L; essentially unchanged in every frame after t=8. Agrees. 60 % of negative mass is diffuse, which matches the visible red haze across the whole box. |
| void_frac_pos 0.22–0.23 (t ≥ 7) | White void of radius ~0.25 L ⇒ area ≈ 0.2. Agrees. |
| n_clumps_pos 0–3, flickering | The positive "clumps" after t=7 are the marginal overdensities (ρ/⟨ρ⟩ ≈ 5–6) in the upper right that hover at the threshold; the flicker is threshold noise, not real structure appearing and disappearing. |
| λ_peak = 1.0 L for both species from t=4 on | Confirms visually: the dominant structure is one feature per box. The run is box‑mode dominated from t ≈ 4. |
| persistence 0.57–0.63 (ref t=5) | Reference frame has 3 clumps; two vanish by t=6, so persistence drops to 0.57 at once and then creeps up as the survivor's footprint stabilises. The metric is reporting the t=5→6 merger, not instability afterwards. |
| σ₊: 0.71 → 6.09 (×8.6); σ₋: 2.83 → 10.2 (×3.6) | Explains the late images: the positive population's Jeans length has grown to ~4 L ≫ box, so it cannot form walls any more and sits as a hot pressure‑supported background with a hole punched by the negative clump. Whether the heating is all physical (release of binding energy, expulsion from repulsive wells) or partly numerical is not decidable from this run (see §4). |
| dE_rel max 5.2 %, final 0.8 %; P_abs conserved to 1e‑16; signed P drift 5e‑4 (expected) | No visible gross failure, but 5 % during the merger phase is marginal for a fixed dt = 0.02 t_J,neg: the clump core at 25× mean density has a local dynamical time ~5× shorter than t_J,neg, so the core is integrated at ~10 steps per local t_dyn. |
| virial 2.17, E_total > 0 | Not straightforwardly interpretable for a mixed‑sign system (KE and PE of the two species have different signs under the three‑law scheme); reported, not used. |

No mismatch between images and data that cannot be explained by threshold sensitivity. The data file also shows d_cell = 0.24 L and φ/d = 0.59 only at t = 5 (undefined afterwards with one clump), and mass_std = diam_std = 0 purely because n = 1.

---

## 3. Outcome vs. the original claims

**Supported (qualitatively, transiently):**
- "The heavier negative population clumps first and confines ordinary matter into cell walls." The mechanism operates exactly as described: at t ≈ 2–3 t_J,neg there are ~8 negative clumps at void centres and positive matter in a connected wall network, cell size ≈ 0.25–0.35 L ≈ 2–3 λ_J,neg. The predefined "cells form by 3 t_J" criterion is met (segregation −0.58 vs threshold −0.3).
- An interesting detail not in the pages: the positive population shows the pattern *before* the negative clumps are visible, because it is colder and responds more strongly to the same potential.

**Contradicted (in this setup):**
- "Stable 2‑D cellular structure" / "stable clumps". Clump number falls 8 → 1 within ~4 t_J,neg by hierarchical merging; the positive wall network survives only ~2 t_J,neg before dissolving into a hot background. Criteria "stable_clumps" (3 → 1 between t=5 and 20, −67 %) and "persistence" (0.62 < 0.7) are not met.
- What *is* stable is the end state: one clump holding 40 % of the negative mass, constant in mass and size for ≥12 t_J,neg, with a fixed positive void. But this is the fundamental mode of a periodic box with no expansion, so its stability says nothing about cosmological cells.

**Cannot decide:**
- Whether the cellular stage would freeze instead of merging if Hubble expansion were included (it damps merging in standard CDM runs) or if the box were much larger than λ_J,neg. Both were absent in Petit's original runs as well, so his "stable" pattern may simply be a snapshot in the t ≈ 2–3 window, or his runs may have been short; the pages give no duration.
- Whether positive‑matter heating by a factor ~9 is robust physics or inflated by the 5 % energy error and the unsoftened grid‑scale collapse of a cold population. This directly controls whether walls can persist.
- Any statistical comparison with observed structure (correlation functions, void sizes): nothing here is calibrated.
- Seed‑to‑seed variability: one seed; the "reproducible" criterion is deferred to the ensemble command and has not been evaluated.

---

## 4. Visual signs of numerical artefacts

1. **Box‑size domination.** λ_peak = 1.0 L from t = 4; one clump, one void; the t = 5–6 negative filament is aligned with the x‑axis and spans the box; the last merger (t ≈ 7–9) happens across the periodic edge at y ≈ 0.4. Everything after t ≈ 4 is a box mode, not a cosmological statement.
2. **Periodic‑boundary effects.** The edge remnant at x ≈ 1.0 (t = 5–9) is the image of structure at x ≈ 0; its absorption into the main clump is mediated by the periodic images.
3. **Resolution of walls.** Positive walls at t = 2 are ~0.02 L thick, comparable to the force cell (0.0156 L) and to the effective softening; wall thickness and peak density (log ≈ 1.0) are set by the grid, not by physics. The sharp caustic edges at t = 3 are physical for a cold sheet but are likewise under‑resolved.
4. **Energy error and heating.** 5 % peak energy error coincides with the merging/collapse phase; the fixed dt is coarse for the clump core. Positive σ grows ×8.6. Part of this may be integration error masquerading as physical heating.
5. **Shot noise / threshold flicker.** The positive clump count (0–3) and late persistence wobble are threshold noise on a background at ρ/⟨ρ⟩ ≈ 5–6. The 128² density maps show pixel noise everywhere; no grid‑aligned Moiré or checkerboard patterns are visible in the particle plots.
6. **Early trigger.** "cells_form" fires at t = 0.5 from a linear anticorrelation with no visible cells; the criterion is lenient in the direction of the claim.
7. (Minor) The output directory is date‑stamped 2026‑10‑15; as with the pages, treat it as a clock artefact.

---

## 5. Most informative next runs

1. **Seed ensemble** (≥5 seeds, same parameters): establishes n_clumps(t), cell size and merger time scatter; required before any statement about "the" outcome.
2. **Box‑size scaling**: λ_J,neg/L = 1/16, 1/32 at fixed grid resolution per λ_J (grid 128, 256). If late‑time n_clumps scales ∝ (L/λ_J)² and merging keeps going, the single clump is purely a box effect; if a characteristic clump mass/size freezes out, that is a physical result.
3. **Expansion on** (config already has `a_index 0.6667`): the only run that can test whether merging stalls and cells survive — the decisive discriminator for "stable cellular structure", and a deviation from Petit's own setup that must be stated as such.
4. **Convergence**: dt/2 and dt/4 with grid 128 and 256 (or adaptive η); report dE_rel and σ₊(t). If positive heating drops substantially, the wall dissolution here is partly numerical.
5. **Controls**: `pos_only` and `neg_only` with identical seeds, plus mass_ratio = 1 and vel_ratio = 1 variants, to isolate what the cross‑species repulsion adds beyond ordinary gravitational clustering of the heavy population.
6. **Metric fixes**: re‑evaluate persistence with reference at t = 8 (separates "clump count unstable" from "final clump unstable"); add a positive‑matter power spectrum / wall‑thickness vs grid diagnostic; make the cells_form trigger require a visible void fraction, not just anticorrelation.
7. **Direct‑sum cross‑check** at N ≈ 5k+5k (the original's size) to see whether a small, collisional run shows the same 2–4 t_J cellular window and the same merging, which is the closest thing to a replication of the 1992 picture.
8. Only after the above: 3‑D.

---

**Bottom line.** This single run reproduces the *mechanism* Petit describes — heavy negative clumps at void centres, ordinary matter squeezed into walls — as a transient lasting about 2–4 negative Jeans times. It does not reproduce a stable cellular pattern: the clumps merge hierarchically until one is left, the positive walls dissolve into a strongly heated background, and the late state is a single box‑scale mode. Whether stable cells could exist with expansion, in a larger box, or at better energy conservation is not decided by this run, and the measured persistence/clump criteria are shaped by where the reference time falls. The run supports the qualitative "cells form" claim, contradicts "stable long‑lived cells" under the stated assumptions, and leaves the cosmological relevance open pending the ensemble, box‑scaling and expansion tests.