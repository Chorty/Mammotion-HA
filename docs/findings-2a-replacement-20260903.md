# Findings — candidate replacements for criterion 2a (2026-09-03)

**Predeclared at commit `88b7fddb`** (`docs/predeclared-2a-replacement-20260903.md`)
**before any verdict below was computed.** The git timestamp of that commit is
what makes these results trustworthy. Harness:
`scripts/rescore_2a_candidates.py`; full record with input SHA-256s:
`docs/rescore-2a-candidates-20260903.json`.

**Offline. No motion, no gate change, no scoring code changed by this document.**

---

## 0. Reproduction gate — PASSED

Rule A reproduced the shipped verdict every raw file carries to ±0.01 °/s, and
matches the five statistics pinned in
`tests/components/mammotion/test_step_response_vio_scoring.py`
(2.156 / 3.664 / 2.319 / 0.130 / 3.4049). Proceeding was permitted.

⚠️ Reported honestly by the harness: only **1 of 5** raw files carries a
`vio_analysis` block, since the four 2026-08-30 captures predate E-VIO. The
pinned test roster is the fuller anchor.

---

## 1. All rules, all runs

| run | step | A (status quo) | B (drop 1st) | C (2 s onset) | **D (plateau)** | E (slope) |
| --- | --- | --- | --- | --- | --- | --- |
| R1 +120 ⚓ | 5000 | FAIL 2.156 | 🚨 **PASS 0.816** | 🚨 **PASS 0.700** | FAIL 2.974 | FAIL 0.340 |
| R1r +120 ⚓ | 5000 | FAIL 3.664 | FAIL 2.084 | 🚨 **PASS 0.108** | FAIL 3.664 | FAIL 0.553 |
| R2 +180 | 7000 | PASS 0.130 | FAIL 1.625 | PASS 0.919 | **PASS 0.919** | FAIL 0.326 |
| R2r +180 | 7000 | FAIL 3.405 | PASS 0.107 | PASS 0.023 | **PASS 0.538** | PASS 0.298 |
| SX +120 | 7000 | FAIL 2.320 | PASS 0.897 | PASS 1.197 | **PASS 0.628** | FAIL 0.453 |

⚓ = predeclared anchor: these runs did NOT reach steady rotation, so any rule
passing them is rejected.

---

## 2. The cascade, applied in the predeclared order

### 2.1 🚨 Rule C — the rule I PREDICTED WOULD WIN — is REJECTED

§3 of the predeclaration said, in writing, *"This is the rule I expect to win,
stated now so the expectation is on the record and can be falsified."*

**It was falsified.** C passes **both** 5000 ms anchors — runs known to have been
still accelerating when the step ended. A rule that certifies those as steady is
broken, whatever else it gets right.

🔑 **Why, and it generalises:** excluding a fixed 2000 ms window from a 5000 ms
step removes 40% of the phase and leaves only the tail, where a ramp looks
locally flat. **An onset allowance that is a large fraction of the step destroys
the very ramp detection the criterion exists for.** Any future onset-exclusion
rule must scale with step length, not be a fixed constant — and even then it
would need re-anchoring.

**Rule B is rejected on the same test** (passes R1 at 0.816).
⚠️ Note B is the "obvious" fix whose outcomes were already published, and which
§2 disclosed could not be judged blind. It fails on its own merits regardless.

### 2.2 Rule E is INADMISSIBLE

Criterion 1 is absolute: a statistic whose 2σ noise exceeds its own bound is
inadmissible *whatever verdicts it produces*. From the measured σ_h = 1.002°:

| | 2σ | bound | |
| --- | --- | --- | --- |
| E slope, n = 4 over 3.9 s | **1.276 °/s²** | 0.30 | **4.3× over** |
| E slope, n = 5 over 4.9 s | **0.913 °/s²** | 0.30 | 3.0× over |

E's apparent verdicts are noise. Its 0.30 °/s² bound was declared in advance and
is not moved to rescue it.

### 2.3 Rule A survives admissibility and the anchors — and is demoted

A is admissible (2σ = 1.145 vs 1.5) and anchor-clean. But it **splits the two
+180/7000 runs** (PASS 0.130 / FAIL 3.405) whose plant agrees to 0.195 °/s, and
it **fails criterion 3**: it does not address the onset bias that §0.1 of the
predeclaration identified as the actual failure mode.

---

## 3. Recommendation — Rule D, the settle-anchored plateau

Compare the endpoint rate over the **final 3000 ms of the step** against the
**3000 ms preceding it**; pass within the unchanged 1.5 °/s bound.

| criterion | D |
| --- | --- |
| 1. admissible | ✅ 2σ = **1.336** vs 1.5 |
| 2. anchor-clean | ✅ FAILs both 5000 ms runs (2.974, 3.664) |
| 3. addresses the onset bias | ✅ **ignores the onset by construction**, not by exclusion |
| 4. minimal disruption | flips 2 published verdicts (see §4) |

🔑 **It also satisfies the new anchor: it scores the two +180 runs the SAME
way** (PASS 0.919 / PASS 0.538), where A splits them. And it asks what 2a
actually means — *had rotation stopped changing by the END of the step?* — rather
than whether two halves happen to be symmetric.

Window occupancy is adequate everywhere: n = 5, 6, 5 on the three 7000 ms runs,
so the ≥2-distinct-readings rejection criterion is not triggered.

---

## 4. The costs of the winner, stated plainly

- **It flips two published verdicts**: SX (+120/7000) FAIL → **PASS**, and
  R2r (+180/7000) FAIL → **PASS**. Under D, three of five banked runs pass 2a.
- ⚠️ **Its admissibility margin is THINNER than the status quo's** — 2σ 1.336
  against A's 1.145, because 3 s windows are shorter than 3.5 s halves. It clears
  the bound by 11%. **This is the weakest part of the recommendation.**
- 🔑 **It changes what ω means, and therefore τ.** ω is currently
  `half_rates[1]`, defined only when 2a passes. Under D the natural ω is the
  **late-window rate** (−11.336, −11.489, −7.447 on the three passing runs).
  That is arguably cleaner — it is the plateau by construction — but **τ computed
  under D is not comparable to any τ quoted before it**, and the pinned
  `tau = 0.7995 s` would have to be recomputed or withdrawn.
- On the 5000 ms runs D's windows span nearly the whole phase (its R1r statistic
  is numerically identical to A's), so at that step length D degenerates toward
  A. It rejects those runs correctly, but not by a mechanism independent of A.

---

## 5. What the banked data cannot settle

⚠️ **n = 5 scoreable runs, 4 configurations, 1 mower, 1 site.** This cannot
establish a false-positive rate. D was selected against five runs it can see.

🚨 **No rule selected here is validated.** The only test that would validate it is
a run it has never seen — which is the single on-mower trigger in the plan, and
it remains unauthorized and unscheduled.

Specifically unsettled: whether D's thin admissibility margin holds at other
angular commands; whether the 3000 ms window is right at step lengths other than
7000 ms; and whether the +180 pair agreeing under D is a property of D or a
coincidence of two runs.

**Nothing here changes shipped code.** Adopting D is a separate decision with its
own release, its own tests, and the ω/τ redefinition in §4 to resolve first.

---

## 6. DECISION — Rule D is DENIED (operator, 2026-09-03)

**Rule D is not adopted. Shipped scoring is unchanged. Criterion 2a continues to
use Rule A**, with its measured bias documented in `CLAUDE.md` so no verdict is
quoted as fact.

**The reasoning, recorded so it does not have to be re-derived:**

🔑 **The asymmetry decided it.** Adopting D would add a **more permissive**,
**noisier**, **unvalidated** rule in order to gain a number — τ — that nothing
currently needs, because 2a → τ → dead time → Phase 2, and Phase 2 is parked
(standing decision 5). Denying costs nothing today.

- **D is more permissive BY CONSTRUCTION.** Excluding the onset makes any run
  likelier to pass; here it turns two FAILs into PASSes, and it would do the same
  on future runs. ⚠️ **A criterion becoming more permissive should face a higher
  bar, not a lower one.**
- **Its noise margin is worse than the rule it would replace** — 2σ 1.336 against
  A's 1.145, because 3 s windows are shorter than 3.5 s halves.
- **It has had exactly one round of in-sample selection and no out-of-sample
  test.** This same study showed what that produces: Rule C was predeclared as
  the expected winner and passed BOTH known-ramping anchors.

**What denial does NOT mean.** The study stands, D remains the studied
replacement, and the diagnosis in §0.1 of the predeclaration — that 2a fails on
onset BIAS, not variance — is unaffected and is the durable result.

**The condition that would reverse this:** if Phase 2 resumes and τ is actually
needed, adopt D **together with** one out-of-sample validation run, and resolve
the ω/τ redefinition in §4 before quoting any τ.
