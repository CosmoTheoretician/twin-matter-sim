# N-body simulations of "twin matter" and spiral galaxies on jp-petit.org

**Scope.** This review focuses on the numerical simulation work described on the site: the 1991–93 simulations with Pierre Midy and F. Landsheat (large-scale structure and a barred spiral), the 2001 conference paper that reports them, the 2004 "Epistémotron" citizen-science pages, and the reception documented by the author. Theoretical Janus pages are used only as context. Throughout, "Petit claims" marks statements I can only source to the site; "verified" marks what I can confirm from general knowledge; uncertainty is flagged explicitly.

---

## 1. Overview: what the work claims (author's terms)

Petit's starting point is that "galactic dynamics... based on the joint resolution of Vlasov and Poisson equations is now an empty box" and that dark matter is "sprayed" ad hoc into galaxies to fit rotation curves ([Colloque_2001_1.htm](https://www.jp-petit.org/science/colloque2001/Colloque_2001_1.htm)). His alternative is a universe containing two populations:

- positive masses attract each other (Newton);
- negative masses attract each other (Newton);
- opposite-sign masses repel ("anti-Newton").

The negative-mass population ("twin matter", *matière gémellaire*) is said to be invisible because electromagnetic interaction between opposite-energy particles is impossible, and is later re-interpreted as positive mass living on a second "fold" of a two-sheeted manifold, appearing negative from our side ([Colloque_2001_2.htm](https://www.jp-petit.org/science/colloque2001/Colloque_2001_2.htm), §10; [theorie_cinetique.htm](https://www.jp-petit.org/Cosmo_astro/theorie_cinetique.htm)).

The simulation claims are:

1. **Large-scale structure (LSS).** 2-D simulations with two populations of 5,000 mass-points each produce a stable "2d-cellular structure": the heavier negative-mass population clumps first and confines ordinary matter into cell walls, which Petit says resembles the observed "bubble" structure of the universe and suggests a galaxy-formation scenario (compression of walls, radiative cooling, fragmentation) ([Colloque_2001_1.htm](https://www.jp-petit.org/science/colloque2001/Colloque_2001_1.htm), §§3–4).
2. **Galaxy confinement and rotation curves.** Surrounding a Miyamoto–Nagai disk with ellipsoidal shells of repulsive twin matter yields "a good-looking galactic rotation curve" and a vertical (z) confining force offered as an alternative to Bahcall's disk dark matter (§5).
3. **Spiral structure.** A 1992 2-D simulation of a galaxy embedded in a twin-matter environment produced "a beautiful barred spiral, stable over an impressive number of turns, whereas others' lost their arms immediately"; galaxy morphology (spiral vs. ring) depends on the twin/normal mass ratio; the galaxy is braked while the spiral forms ([artefacts.htm](https://www.jp-petit.org/Cosmo_astro/artefacts.htm)).
4. **Negative lensing.** Twin-matter clumps act as concave lenses, making distant galaxies "smaller, fainter and more numerous"; a hole in a twin-matter distribution around a cluster reinforces positive lensing ([Colloque_2001_2.htm](https://www.jp-petit.org/science/colloque2001/Colloque_2001_2.htm), §7).

Petit presents this as a paradigm change that resolves "the missing mass effect, galaxy rotation curves, galaxy formation, the origin of large-scale structure" ([theorie_cinetique.htm](https://www.jp-petit.org/Cosmo_astro/theorie_cinetique.htm)).

---

## 2. Methods as described

**Numerical scheme.** The pages describe direct-summation N-body integration: for each particle, sum forces from the N−1 others, advance position and velocity by a Taylor expansion with a "frozen" field over Δt (cost ∝ N²) ([theorie_cinetique.htm](https://www.jp-petit.org/Cosmo_astro/theorie_cinetique.htm)). Timestep selection is heuristic: "small enough that reducing it further does not change the overall appearance." A 1991 run of 500 points with Midy on "the old Cray-1 at Orsay" used an adaptive step that backed up and divided Δt by ten when trajectories showed sharp kinks ([artefacts.htm](https://www.jp-petit.org/Cosmo_astro/artefacts.htm)). Early-1990s runs of 5,000 points were done on a machine at "DAISY" (presumably DESY) according to [theorie_cinetique.htm](https://www.jp-petit.org/Cosmo_astro/theorie_cinetique.htm). Petit says he has "always written everything in compiled BASIC."

**LSS run (Landsheat).** Two 2-D populations, 5,000 points each; |M̄| = 64 M; Maxwellian velocities with ⟨V̄⟩ = 4⟨V⟩; uniform initial distribution; expansion neglected ("we do not know how to describe gravitational force in an expanding universe"). The heavier population's Jeans time is 8× shorter, which is dimensionally consistent with √64. The paper reports standard deviations of clump number, mass and diameter across randomized runs, but the equations appear only as placeholders "(a)…(o)" in the HTML, and the figures are not reproduced ([Colloque_2001_1.htm](https://www.jp-petit.org/science/colloque2001/Colloque_2001_1.htm)).

**Spiral run (1992).** 2-D, periodic tiling ("closing the computational space on a flat torus"); initial conditions from "a very complex analytic solution... of a tensorial two-dimensional fluid mechanics that I had to create for the purpose"; a rotating central clump plus a confining ring of twin matter. Near-encounter softening ("cut in": force zeroed below a cutoff radius) is mentioned as a way to mimic a collisionless medium ([artefacts.htm](https://www.jp-petit.org/Cosmo_astro/artefacts.htm)).

**Rotation-curve model.** Not an N-body simulation: a semi-analytic superposition of a Miyamoto–Nagai potential and homothetic thick-ellipsoid shells, with the twin-matter distribution "shaped on empirical grounds, through numerous trials" ([Colloque_2001_1.htm](https://www.jp-petit.org/science/colloque2001/Colloque_2001_1.htm), §5).

**Theoretical backing.** Coupled Einstein-like field equations on a two-fold cover, Newtonian limit yielding the three-law interaction scheme, and coupled Friedmann-type equations whose numerical solution was tuned "in order to fit the initial condition for VLS numerical simulation" ([Colloque_2001_2.htm](https://www.jp-petit.org/science/colloque2001/Colloque_2001_2.htm), §§8–12). Later pages invoke coupled Vlasov equations with "elliptic solutions" Petit says he published in the 1980s *Comptes Rendus* ([2023-04-03 AI dialogue](https://www.jp-petit.org/nouv_f/2023-04-03-JPP%20face%20a%CC%80%20l'IA.html)).

**Artefact awareness.** The 2004 page explicitly discusses artefacts: the fragmentation of twin matter around the 1992 spiral is attributed to the periodic tiling; close encounters inflating velocity tails in a globular-cluster test are identified as collisional noise ([artefacts.htm](https://www.jp-petit.org/Cosmo_astro/artefacts.htm)).

**Citizen science.** The Epistémotron project asks readers to write their own C++ codes and eventually pool machines; Frédéric Baudemont's 2004 single-population 2-D spiral is shown as "encouraging, but is it significant?" ([theorie_cinetique.htm](https://www.jp-petit.org/Cosmo_astro/theorie_cinetique.htm), [artefacts.htm](https://www.jp-petit.org/Cosmo_astro/artefacts.htm)).

---

## 3. Relation to mainstream science and reception

**Where it agrees.** The N-body basics on the pages (N² cost, timestep convergence, softening, collisional relaxation, periodic boundaries) are standard and correctly described. The idea that a confining outer mass distribution stabilizes a disk is close to the mainstream Ostriker–Peebles (1973) result that a massive halo suppresses bar instability; Petit's "potential barrier" is functionally a repulsive analogue. Citing Zel'dovich pancakes, Melott & Shandarin, Peebles, Miyamoto–Nagai, Bahcall's K-star analysis and the Fort–Mellier weak-lensing maps shows familiarity with the pre-2000 literature (verified authors; Petit's spelling "Meillier" is wrong—the astronomer is Yannick Mellier).

**Where it departs.**

- *Negative mass.* In standard GR a negative mass attracts and repels in the Bondi (1957) runaway pattern; Petit's symmetric three-law scheme requires a bimetric framework, which is why Janus has two metrics. Bimetric negative-mass models exist in the literature (Hossenfelder 2008; Farnes 2018; Chardin's Dirac–Milne), and Petit disputes priority and quality with all of them ([nouveau.htm](https://www.jp-petit.org/nouveau.htm), 2020 entries). Thibault Damour's 2019 IHES note argued the Janus field equations are inconsistent; Petit replied in *Progress in Physics* and in his videos; the site quotes Luminet (page dated 2026) saying "the Bianchi identities are not satisfied: Damour was right" and that a 2024 *EPJ* paper "corrected nothing" ([Liens_et_videos.html](https://www.jp-petit.org/nouv_f/videos_liens/Liens_et_videos.html)). I cannot adjudicate this dispute from the pages; the page dates (2026) are also anomalous and may be script artefacts, so I treat them cautiously.
- *LSS.* Petit told the 2001 Marseille audience that CDM simulations "do not seem to have an obvious relation" to observed lacunar structure ([colloque_juin2001.htm](https://www.jp-petit.org/science/Presse_scient/colloque_juin2001.htm)). This contradicts the consensus then and since: CDM N-body simulations (Davis, Efstathiou, Frenk & White 1985 onward; Millennium 2005) reproduce filaments, walls and voids and match galaxy-survey statistics. His 2-D two-population run is a qualitative illustration, not a statistical comparison.
- *Spiral structure.* The claim that "others' spirals didn't last more than one turn" reflects a 1970s–80s problem (hot disks, bar instability) and ignores density-wave theory, swing amplification, and recurrent transient spirals (Sellwood & Carlberg 1984), as well as gas-cooling simulations. Petit calls Combes's remark that cold gas gives the same result an "intox" without evidence ([artefacts.htm](https://www.jp-petit.org/Cosmo_astro/artefacts.htm)).
- *Dark clumps.* The Abell 1942 "dark cluster" (Erben et al. 2000) that motivates the paper has not been confirmed as a mass concentration by later work, to my knowledge; the site itself reports Mellier saying "personally, I don't believe in it" ([colloque_juin2001.htm](https://www.jp-petit.org/science/Presse_scient/colloque_juin2001.htm)).
- *Pioneer anomaly.* The 2001 paper offers twin matter as an explanation ([Colloque_2001_2.htm](https://www.jp-petit.org/science/colloque2001/Colloque_2001_2.htm), §10); the anomaly was later explained by anisotropic thermal radiation (Turyshev et al. 2012).

**Publication and reception (as documented on the site).** The 1992 spiral paper "was refused by numerous journals and therefore never published" ([artefacts.htm](https://www.jp-petit.org/Cosmo_astro/artefacts.htm)). The LSS image was, Petit says, published in *Astrophysics and Space Science* 1995 (a 1995 Ap&SS paper by Petit does exist; I have not checked that it contains the figure). The 2001 contribution was a poster; Petit reports putting 70 copies of a 64-page version on the tables and receiving no response in six months. Dozens of seminar requests went unanswered ([janus.htm](https://www.jp-petit.org/janus.htm)). Alain Riazuelo's IAP critique page (2006/2008) is cited by Futura-Science moderators to exclude discussion. Petit lists later peer-reviewed Janus papers (Ap&SS 2014, MPLA 2014, 2015; verified to exist) and recent acceptances in *Journal of Modern Physics* and claimed *EPJ C* / *Rev. Math. Phys.* papers that I cannot verify, alongside repeated *Physical Review D* rejections as "not suitable." The 2023 "JPP face à l'IA" page shows a chatbot agreeing that Janus is "more mathematically coherent"; this is a leading dialogue with a language model, not scientific validation.

**Overall:** peer-reviewed in minor-to-mid-tier journals; essentially no uptake, replication, or citation by the galactic-dynamics or cosmology community; active published criticism (Damour, Riazuelo, Luminet). Note that the 9/11 and "antimatter weapons" material on neighbouring pages ([scenario_vol77.htm](https://www.jp-petit.org/Divers/PENTAGATE/docs_anticonspirationnistes/scenario_vol77.htm), [Liens_et_videos.html](https://www.jp-petit.org/nouv_f/videos_liens/Liens_et_videos.html)) does not bear on the physics but plausibly affects how the community engages with the author.

---

## 4. Strengths

- **A clear, testable dynamical hypothesis.** The three-law interaction scheme is simple to implement; any competent N-body coder can reproduce the qualitative results. Petit explicitly invites this ([theorie_cinetique.htm](https://www.jp-petit.org/Cosmo_astro/theorie_cinetique.htm)).
- **Genuine awareness of numerical artefacts.** The 2004 page on timestep convergence, periodic-tiling artefacts, collisional heating and softening is pedagogically sound and more candid than many popular accounts ([artefacts.htm](https://www.jp-petit.org/Cosmo_astro/artefacts.htm)).
- **Early priority on a now-explored idea.** Two-population gravitationally repulsive simulations in the early 1990s predate Farnes's 2018 negative-mass simulations; the structural analogy with halo-stabilization of disks is physically reasonable.
- **A plausible qualitative mechanism.** If one grants the interaction law, the result that the denser self-attracting population clumps first and confines the other into cells is dynamically expected and the Jeans-time ratio is consistent.
- **Transparency about limits.** The paper states that the runs are 2-D, that 3-D is "far beyond our computational capabilities," that the twin-matter distribution for rotation curves was empirical, and that results are "an illustration of an idea" ([Colloque_2001_1.htm](https://www.jp-petit.org/science/colloque2001/Colloque_2001_1.htm)).

---

## 5. Weaknesses and open questions

- **No quantitative comparison with data.** Rotation curves are "good-looking"; LSS is matched by eye to survey slices; no fits to measured rotation curves, correlation functions, power spectra, or lensing statistics are shown.
- **Ad hoc elements the author criticizes in others.** The confining ellipsoidal shells are tuned "through numerous trials"; the cosmological solution was tuned to the simulation's initial conditions; the 2-D spiral initial condition came from an unpublished analytic solution. This undercuts the Occam argument made in the AI dialogue.
- **Resolution and dimensionality.** 5,000 particles in 2-D, no expansion, direct summation, BASIC code: these are 1970s-scale experiments. 2-D gravity has different Jeans and swing-amplification behaviour; Petit himself concedes 3-D behaviour is unverified.
- **Unexamined alternative explanation of an artefact.** The fragmentation of twin matter around the 1992 spiral is attributed to tiling, but a self-attracting negative-mass population is Jeans-unstable by construction (as the LSS run shows). No convergence test (larger box, different tiling) is reported to separate the two.
- **Missing content.** The online paper shows placeholders for equations and figures; the 1992 article is unpublished; no code, parameters or energy-conservation diagnostics are provided. Reproducibility depends on volunteers.
- **Dated or incorrect characterizations of the field.** Claims that "no self-consistent model of a galaxy exists," that CDM simulations don't resemble observations, and that others' spirals cannot survive a rotation are not accurate for the 2001 state of the art, let alone today.
- **Theoretical foundation in dispute.** The bimetric field equations underlying the Newtonian limit have been challenged (Damour; Luminet) on consistency grounds; whether the 2024 revision resolves this is unresolved on the pages. The claimed QFT argument that opposite-energy particles cannot exchange photons is attributed to an unnamed "young brilliant researcher" without reference.
- **Observational tensions not addressed.** Negative lensing predicts fainter, more numerous distant galaxies and image demagnification; the cosmic shear and strong-lensing literature since 2000 constrain such effects, but no confrontation appears on the pages. The Pioneer-anomaly argument is now moot.
- **Self-reported reception.** All accounts of referee behaviour, Combes's "intox," and institutional silence come from the author; I cannot verify them.
- **Date anomalies.** Several pages carry 2026 dates; I cannot determine whether these are real or generated.

---

## 6. Conclusion

**Established.** The numerical-methods exposition (N² scaling, timestep control, softening, periodic boundaries, artefact hunting) is correct and competently explained. Petit did perform small 2-D two-population N-body runs in the early 1990s, reported a cellular LSS pattern and a long-lived barred spiral, presented them at a 2001 conference, and published related theory in minor peer-reviewed journals. Mainstream engagement has been minimal and the explicit critiques that exist are negative.

**Plausible but unproven.** *Given* the symmetric three-law interaction, two self-gravitating populations repelling each other would plausibly form cells with clumps at their centres, and an outer repulsive envelope would plausibly stabilize a disk—analogous to the known halo-stabilization effect. These are reasonable qualitative outcomes, but they have not been demonstrated in 3-D, at adequate resolution, with expansion, or against quantitative observational statistics, and the key result (the spiral) was never refereed.

**Not supported.** The claim that these simulations explain rotation curves, galaxy formation, lensing and large-scale structure better than ΛCDM; the assertion that CDM simulations fail to reproduce the cosmic web; the dismissal of competing spiral-structure results; and the inference that the model "resolves" the dark-matter problem. The underlying field equations remain contested, several motivating observations (Abell 1942 dark clump, Pioneer anomaly) have since been undermined, and the work's uptake has not progressed beyond the author's own circle and volunteer coders.

The fair verdict is that this is an unfinished, decades-old exploratory numerical experiment attached to a contested theory—interesting as a hypothesis-generator and easy to re-run, but not evidence for the cosmological claims built on it.