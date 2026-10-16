## 0. What this run is (and is not)

Before interpreting the pictures: `config.pos_only = true`, `n_neg = 0`. This is the **positive-only control** of the LSS scenario, with kinematic expansion switched on (a ∝ t^{2/3} from t₀ = 1 t_J,neg, reaching a = 15.3 at the end), 6.4 M particles, PM 256², periodic unit box = 32 λ_J,neg = 8 λ_J,pos. It contains no twin matter, so by construction it cannot test the cellular-structure or segregation claims; it establishes what ordinary matter does on its own under these settings. The right-hand panels ("no particles") are empty for that reason, not because twin matter disappeared. Single run, seed 1.

## 1. What the images show over time

- **Top-left (particle scatter):** a uniform solid blue square at every time. With 6.4 M points this panel saturates and carries no information; it only confirms nothing has left the box and no large voids opened.
- **Bottom-left (density, log10(1+ρ/⟨ρ⟩)):** at t = 0 a salt-and-pepper field centred on 0.301 (= log10 2, i.e. ρ/⟨ρ⟩ ≈ 1). The 128² structure grid holds ~390 particles per cell, so Poisson shot noise is ~5 %, giving ±0.01 in the plotted quantity — exactly the 0.28–0.33 spread seen. **This is discreteness noise, not structure.**
- Over t = 3 → 60 t_J,neg the field remains visually indistinguishable from white noise. The only change is a slow widening of the colour-bar range: max 0.34 → 0.36 (t≈13) → 0.38 (t≈47) → 0.39 (t=60), i.e. the densest cell goes from ρ/⟨ρ⟩ ≈ 1.2 to ≈ 1.45; the minimum barely moves (0.25 → 0.24). Looking closely at t ≥ 30 there may be a slightly more mottled texture at 2–4-cell scale than at t = 0, but I would not assert this by eye.
- **Where the population ends up:** essentially where it started. No filaments, walls, voids, or clumps appear at any time. There is no cellular pattern because there is nothing for it to be cellular around.

## 2. Images vs. measured data

Agreement:
- `n_clumps_pos` = 0 at every output, `void_frac_pos` = 0 — matches the featureless images.
- `cells_form` and `stable_clumps` = NaN / "not measurable" — correct, no negative population.
- `sigma_pos_final` = 0.0015 vs initial 0.354: a factor 235 = a² (15.3²). In the comoving scheme x'' + 2Hx' = g/a³, free comoving velocities decay as a⁻², so the particles are nearly frozen in comoving coordinates by the end. That is why the noise pattern does not even rearrange itself.
- `dP_abs_rel` ~ 10⁻⁹: momentum conserved; integrator behaving.

Things to flag:
- **`persistence` = 0.906, verdict "met", is vacuous.** The persistence metric correlates the density field with its t = 5 reference; here it is measuring that a *shot-noise field frozen by Hubble drag* stays correlated with itself. A rough displacement estimate after t = 5 (velocity already decayed ~8×, further a⁻² decay) gives ≲ 1 structure cell of motion, consistent with ~0.9 correlation. This should not be read as "stable structure persists"; the criterion needs gating on the existence of structure (n_clumps > 0 or void_frac > 0).
- `dE_rel_max` ≈ −1 looks alarming, but E = KE + U is not a conserved quantity in a comoving integration (Layzer–Irvine is needed). Energy conservation therefore cannot be assessed from this run at all — a diagnostics gap rather than evidence of a bug. `virial` = 9.8 (KE ≫ |U|) just says the system is pressure/kinetically dominated, consistent with no collapse.
- `lambda_peak_pos` jumps from 0.333 to 0.0833 L at t = 11 and stays there. Both values are bin centres (L/3, L/12), so this is a quantised estimator switching bins, not a phase transition. Whether the shift to ~0.08 L (≈ 0.7 λ_J,pos) reflects weak real growth or estimator noise cannot be decided from one seed; the colour-bar growth hints at some small-scale amplification, but the structure finder (threshold ρ/⟨ρ⟩ > 5) is, correctly, nowhere near triggering.

## 3. Comparison with the original claims

- **Supports:** nothing directly about twin matter. What the run does establish is a null baseline: under these settings ordinary matter alone produces no clumps, voids, or cells in 60 t_J,neg. If a matching two-population run shows positive-matter walls or cells, this control would let one attribute them to the twin population rather than to positive self-gravity — which is the kind of separation Petit's narrative (twin clumps first, then confinement of ordinary matter) requires.
- **Contradicts:** nothing. Petit made no claim about a positive-only expanding box.
- **Cannot decide:** the cellular pattern, clump stability, and long-lived structure claims — all untestable here.
- **Important caveat on the baseline itself:** the expansion is kinematic, with H not tied to the mean density. By my estimate H(t₀) · t_J,pos ≈ 10 (1/H ≈ 1.5 t_J,neg vs t_J,pos ≈ 16 t_J,neg), and the ratio (growth rate)²/H² at λ_J,pos is only ~0.35, giving a comoving growth exponent δ ∝ t^{~0.3} at that scale — a factor ~3 over the whole run from ~1 % shot-noise seeds. So "positives don't clump" is to a large degree built into the chosen expansion rate, not a property of ordinary matter. The same arithmetic suggests the 64× heavier twin population would still be strongly unstable under this expansion (ratio ~ 90 at λ_J,neg), so a two-population run with these settings would be a lopsided test: twin matter collapses, ordinary matter is Hubble-suppressed regardless of twin matter. Treat these numbers as back-of-envelope.
- Also note that Petit's runs had 5,000 particles; their shot noise (tens of % on cell scales) was a far stronger seed than the ~1–5 % here. Part of what he saw may have been seeded by discreteness; this run, being much quieter, is not a like-for-like replication.

## 4. Visual / numerical artefact signs

- The scatter panel is an uninformative solid fill at this N — a rendering issue, not physics.
- The density panel is dominated by Poisson shot noise with an autoscaled colour bar that magnifies it; the apparent "texture" is discreteness.
- No visible grid imprint, box-scale mode, or periodic-boundary signature — but with no structure there is nothing to imprint on.
- The slow rise of the colour-bar maximum is either weak gravitational amplification of shot noise at small scales (plausible: in this scheme the comoving Jeans length shrinks ∝ 1/a, so small scales become unstable late) or PM discreteness/aliasing noise. Not distinguishable from one seed.
- Energy diagnostic not meaningful with expansion (above); momentum fine; float32 and dt = 0.02 t_J,neg are not stressed here.

## 5. Most informative next runs

1. **Same positive-only control with `expansion = false`** — the actual analogue of Petit's setup (he neglected expansion). This decides whether the null result is due to Hubble drag or intrinsic.
2. **The two-population run and the negative-only control** with identical settings and seed; without them the control has no partner.
3. **Expansion normalisation scan**: tie H to the mean density (self-consistent 2-D Friedmann analogue) or scan H(t₀)·t_J,pos ∈ {0.1, 1, 10}. The present value (~10) dominates the dynamics.
4. **Longer runs** (several t_J,pos, e.g. 200–300 t_J,neg) and a **seed ensemble** (≥ 5 seeds) to get the noise baseline for λ_peak and the colour-bar growth.
5. **Resolution/convergence**: grid 128/256/512, N 1.6 M/6.4 M, dt halved; plus a **5,000-particle direct-summation replica** to see how much Petit's pattern depends on shot-noise seeding.
6. **Diagnostics fixes**: Layzer–Irvine energy under expansion; gate `persistence` on structure existing; fixed colour-bar limits across snapshots with λ_J scales overlaid; drop or subsample the scatter panel; plot P(k, t) against linear theory so growth (or its absence) is quantified rather than eyeballed.

**Bottom line:** a clean, well-behaved but structure-free control. It shows that ordinary matter alone, under a strong kinematic expansion and quiet shot-noise initial conditions, stays uniform for 60 t_J,neg; the "persistence met" verdict is an artefact of a frozen noise field and should be disregarded. It neither supports nor undermines the twin-matter claims; it becomes useful only alongside the two-population run and a no-expansion counterpart.