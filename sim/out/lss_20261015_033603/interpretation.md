## 1. What the images show

**t = 0.** Both populations are Poisson-uniform. The density panels show only shot noise (log colourbar 0.13–0.48 ≈ ±1/√24 around log10 2 for ~24 particles per 128² cell).

**t = 1.** The *positive* (blue) population is already strongly structured: a fine foam of round holes ~0.05–0.1 L across, with the red (twin) population only mildly mottled. The blue foam is the light, cold population (σ₊ = 0.71) being pushed out of shot-noise overdensities of the 64× heavier twin population; the twins themselves, with 4× higher dispersion, are still sub-Jeans at these scales. The segregation index is already −0.50 at t = 0.5 and reaches its extreme (−0.72) at t = 1.5, i.e. the anti-correlation peaks *before* the twin clumps exist.

**t = 2–4.** Twin clumps condense at the physically expected scale (λ_J,neg = 0.125 L; ~13–14 clumps → mean spacing ≈ 0.27 L ≈ 2 λ_J). Blue matter is confined to walls forming a cellular network with roughly one twin clump per cell. This is the stage that most resembles the "2-D cellular structure" of the pages. Blue nodes at wall junctions begin to show compact knots; red clumps are circular with concentric ring/shell features (shell-crossing caustics of cold collapse).

**t = 5–9.** Twin clumps merge pairwise ((0.40,0.30)+(0.50,0.25); (0.75,0.46)+(0.95,0.42); (0.05,0.8)+(0.15,0.73)…). Blue walls thicken, cells enlarge, and sharp triangular/arc-shaped caustics appear at the nodes (e.g. (0.1,0.3), (0.55,0.6)), indicating that the positive population has started to collapse under its own gravity as Hubble drag cools it.

**t = 11–20.** Hierarchical coarsening continues: the twin field goes 8 → 7 → 5 → 4 clumps; the (0.40,0.77)+(0.65,0.80) pair merges by t = 13, the (0.2,0.4)+(0.35,0.3) pair by t = 18. The final twin clumps sit at ≈(0.15,0.83), (0.52,0.72), (0.80,0.48), (0.25,0.25), each ~0.1 L across with diffuse haloes and, in two cases, tidal tails (e.g. the stream trailing (0.55,0.72) at t = 19–20). The blue population ends as a network of 3–4 large voids (the largest ~0.35 L across, centred ≈(0.5,0.8)) with ~7 knots at junctions and long thin caustic arcs (the crescent at (0.4–0.6, 0.65) is a void-boundary shell crossing). Red clumps remain inside blue voids throughout — the two populations never mix.

Snapshot 10 is missing from the set; the sequence is otherwise continuous.

## 2. Images vs. measured data

Agreement:
- `n_clumps_neg` (0 → 13 at t = 3 → 14 at t = 4 → 4 at t ≥ 17.5) matches my visual counts within ±1–2 (ambiguity only for mid-merger pairs).
- `n_clumps_pos_final` = 7 matches the ~7 blue knots at t = 20; `n_clumps_pos` ≈ 10–14 at t = 3–5 matches the junction knots.
- `mass_frac_clumps_neg` rising to 0.76 and `mass_mean` growing 1.1 → 12.2 (M₊ = 1 units; 4 × 12.2 ≈ 49 ≈ 76 % of 64) is consistent with the visible mergers. The large final `mass_std` (7.4) reflects the one over-merged clump at (0.25,0.25).
- `void_frac_neg` → 0.77 vs `void_frac_pos` ≈ 0.57 (flat from t ≈ 4) matches the pictures: twins collapse into point-like objects, positives stay a space-filling web.
- `lambda_peak_pos` reaching 1.0 (box) by t = 18 and `lambda_peak_neg` = 0.5 matches the visible coarsening to ~box-scale voids.
- `cells_form` "met" (segregation −0.41 at t = 3) is visually obvious.

Points needing care:
- **Segregation drifting from −0.72 to −0.23** does *not* visually correspond to the populations mixing; at t = 20 every red clump still lies in a blue void. The decline is most likely a property of the index: once 76 % of twin mass sits in ~1 % of the area, the twin field's variance is dominated by a few spikes and a pixel cross-correlation is driven toward zero even with perfect anti-coincidence. I cannot confirm the exact definition from the files, so treat this as a probable metric effect rather than a physical one.
- **Persistence → −0.03** (reference t = 5) decays smoothly and monotonically, which the images confirm is *continuous coarsening* (merging and void growth), not disruption of the pattern. The metric answers "is the same pattern still there?" (no), not "is there still a cellular pattern?" (yes).
- **Clump diameter** grows 0.05 → 0.10 L *comoving* while a grows 1 → 7.4, i.e. the measured physical size grows ~×15. Virialised objects should shrink in comoving units. Either the threshold-based diameter is picking up merger envelopes/tails, or the twin clumps are being puffed up (mergers, Hubble drag freezing them before virialisation, or time-resolution heating). The images show the clumps visibly more diffuse at t = 20 than at t = 5, so this is a real open question; `virial_final` = 0.18 is global and does not settle it.
- **dE_rel_max = 30.8.** E = KE + U in a comoving frame with kinematic Hubble drag is not a conserved quantity (no Layzer–Irvine term), so this number cannot be read as an integration error — but it also means the run has *no* usable accuracy check. `dP_abs_rel` ~ 10⁻¹⁶ is guaranteed by PM symmetry and is uninformative.

## 3. Against the original claims

**Supported (qualitatively, single run):** Given the three-law interaction, a heavy self-attracting twin population and a light ordinary population separate spontaneously; by t ≈ 3–5 t_J,neg the ordinary matter forms walls around cells each containing one twin clump. Hubble drag (a ∝ t^2/3, ×7.4) does not erase this. The Jeans-scale ordering (twin clumps at ~2λ_J,neg) is as expected.

**Contradicted (in this run):** "Stable clumps" in the sense of a fixed population: the twin clump count falls 14 → 4 (−64 % over the t = 5–20 window), i.e. ordinary hierarchical merging, exactly what any self-gravitating population does. "Long-lived structure" in the sense of a frozen cell pattern: the pattern coarsens continuously (persistence ≈ 0). Individual clumps do persist as bound entities and merge rather than dissolve — so "stable" may be defensible if read as "clumps don't evaporate," but not as "the cellular pattern is stationary."

Note also a mechanistic nuance: the pages say the heavy population clumps *first* and then confines the light one. Here the light population structures first, as a passive response to shot noise in the heavy one; the twin clumps follow. Whether this ordering depends on N (shot-noise amplitude) is untested.

**Cannot decide:** (i) whether coarsening stops at some physical scale — by t ≈ 15 the dominant scales reach half-box to box, so the end state is box-limited; (ii) what Petit's 5,000-point, non-expanding runs would show, since no duration is given and the prettiest epoch here (t ≈ 3–6) could be where such runs were stopped; (iii) 3-D behaviour; (iv) reproducibility — this is one seed, and `mass_std/mass_mean` ≈ 0.6 at the end warns that clump statistics are noisy; (v) anything about observations — the run makes no quantitative contact with data.

## 4. Numerical-artefact signs

- **Box-size effects:** dominant from t ≈ 12 on. `lambda_peak_pos` = 1.0 L, `lambda_peak_neg` = 0.5 L, `d_cell` = 0.38 L, four clumps at ~0.4–0.5 L spacings, voids ~0.35 L. Twin clumps straddling x = 0/1 at t = 12–13 and blue walls wrapping through y = 0/1 are correct periodic behaviour but show the structure "knows" the box. Late-time results should not be trusted quantitatively.
- **Shot-noise seeding at the force-resolution scale:** `ic_noise = 0`, so the only seed is Poisson noise; the t = 1 blue foam has holes of 3–5 PM cells (cell = 1/64 L). Its scale and the −0.72 segregation extreme are plausibly set by N and grid, not physics. The later λ_J-scale twin clumps are physical.
- **Force vs. display resolution:** the structure grid (128) is finer than the force grid (64) and `eps` = 0. The razor-sharp blue caustics at t ≥ 15 are kinematic features advected in a grid-smoothed potential, fine as such, but twin clumps of ~0.1 L are only ~6 force cells across, so their internal dynamics and merger products are under-resolved.
- **Timestep:** dt = 0.02 t_J,neg fixed. My estimate for orbits at the softening scale inside an early twin clump (M ~ 1–3, r ~ 0.016–0.03) gives ~15–20 steps per orbit during t ≈ 1–4, before Hubble drag has weakened comoving forces; marginal, and a candidate cause of the diffuse clump haloes. No halving test was done.
- **No visible grid imprint** (no checkerboard or axis-aligned structure) in any density map; the ring structures around early twin clumps are shell crossings, not grid rings. Small orange specks in the blue scatter panels at t ≤ 2 are not explained by the labels and look like a plotting overlay; they do not affect the interpretation.
- **Initial segregation** of +0.027 at t = 0 is ~3× the expected shot-noise level for independent fields; minor, but worth checking that the two populations were drawn from independent streams.

## 5. Most informative next runs

1. **Same seed, `expansion = false`.** Isolates Hubble drag and, crucially, gives a run where dE_rel *is* a valid accuracy diagnostic.
2. **Seed ensemble (≥ 10)** to evaluate the pre-registered "reproducible" criterion and to put error bars on clump number/mass/diameter versus time — the quantity Petit reported.
3. **Box and resolution convergence:** box 2 L and 4 L at fixed particle density and fixed λ_J/L, PM 128 and 256, to see whether coarsening saturates at a physical scale or always runs to the box; `dt`/2 and a nonzero `eps` to test the marginal early time resolution.
4. **Decouple from discreteness:** N up/down by 4×, and `ic_noise > 0` with a prescribed spectrum, to test whether the early blue foam and the t = 1–2 segregation extreme are shot-noise artefacts.
5. **`neg_only` and `pos_only` controls** at identical parameters: the first shows whether the twin hierarchy is just ordinary clustering; the second (λ_J,pos = 0.5 L) tests the prediction that ordinary matter alone would form nothing on these time-scales, which is what makes the confinement mechanism non-trivial.
6. **Better diagnostics:** a merger tree per clump (merging vs. dissolution), per-clump virial ratio and comoving/physical radius, a topological persistence measure (void count/Euler characteristic) separate from positional persistence, and a Layzer–Irvine energy for expanding runs.
7. **Parameter sweep** over mass ratio and velocity ratio (64/4 are the pages' numbers) and the expansion index; then a low-N 3-D run.

**Bottom line:** this single run reproduces the qualitative cell-plus-clump picture at t ≈ 3–6 t_J,neg, but then shows the twin population merging hierarchically and the cell pattern coarsening until it is limited by the box. It supports "the pattern forms," contradicts "the pattern is stationary," and cannot yet say what happens on scales larger than the box, in 3-D, or across seeds.