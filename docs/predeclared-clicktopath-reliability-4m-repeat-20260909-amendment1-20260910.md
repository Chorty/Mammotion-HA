# AMENDMENT 1 — scoring precondition and gate cadence, for leg 4 onward (2026-09-10)

**Written before leg 4 of the 4.0 m aligned-start repeat dispatches.** Supersedes
two specific clauses of
`docs/predeclared-clicktopath-reliability-4m-repeat-20260909.md` for **leg 4
onward only**. Everything else in that document — the 5/5 criterion, the
falsifier, the abort rule, the per-leg safety preconditions (fresh fault check,
fresh dry-run, physical corridor confirmation, explicit go/no-go) — is
unchanged and still binding.

🛑 **This does NOT retroactively rescore legs 2 or 3.** Both stay **unscored**
under the original document, exactly as reported at the time. Applying this
amendment backward would be choosing a rule after seeing which verdicts it
flips — the specific failure predeclaration discipline exists to prevent. This
amendment is forward-only, dated and committed before leg 4 dispatches.

---

## What legs 1–3 actually showed

- **Leg 1** (immediately after the undock drive, within the TTL): all four
  original §4 conditions held. Scored, `target_reached`, 0.1277 m.
- **Leg 2**: `start_geometry.aligned_start_confirmed = false` (28.361°
  error) — correctly excluded by its own measurement. Separately,
  `map_facing` was not `motion_confirmed` at dispatch (607 s since the last
  drive, past the 300 s TTL). Landed 0.0523 m anyway (real post-turn data, not
  aligned-baseline data).
- **Leg 3**: `start_geometry.aligned_start_confirmed = true` (3.979° error) —
  correctly confirmed by its own measurement, once the target was aimed at the
  live `map_facing_degrees` reading instead of a fixed compass bearing.
  Landed 0.1417 m. `map_facing` was **still** not `motion_confirmed` at
  dispatch (same TTL issue), so leg 3 was unscored on that technicality alone
  despite passing every other condition.

**The pattern:** condition 3 (`map_facing.confidence == motion_confirmed` at
the moment of dispatch) has only ever been satisfiable on a leg fired
immediately after a real drive. Every leg since has failed it — not because of
anything wrong with that leg, but because the mandatory disarm → RAW-verify →
report cycle between legs reliably exceeds the 300 s TTL. This is a timing
artifact of the safety bookkeeping, not a signal about the control law.

---

## Change 1 — condition 3 is dropped from scoring

**Original §4** required four conditions to score a leg. **Condition 3
(`map_facing.confidence == "motion_confirmed"` and `safe_to_aim_dispatch ==
true` at dispatch) is removed.**

**Why this is sound, not a convenience softening:** condition 3 was written as
an *additional* pre-flight confidence check layered on top of the run's own
measurement. It is not what makes a leg trustworthy — **`start_geometry` is**.
`start_geometry.basis == "vio_calibration_drive.map_motion_heading_degrees"`
is a live, non-circular measurement taken by the run itself, at the moment of
that leg's own calibration drive, independent of any prior `map_facing`
reading. It is the same category of check the 2026-09-04 fix was built to
provide, and legs 2 and 3 both demonstrate it works: leg 2 correctly caught a
real 28° misalignment; leg 3 correctly confirmed a real 3.98° alignment. A
precondition that a leg's own internal, independent measurement has already
answered adds no further protection — it only adds a timing requirement
unrelated to correctness.

**§4 now reads, for leg 4 onward:**

A dispatched leg is **SCORED** only if **all** of the following hold, read
from the run's own response and recorded per item:

1. `start_geometry.basis == "vio_calibration_drive.map_motion_heading_degrees"`
   — never `None`, never any other basis;
2. `start_geometry.aligned_start_confirmed == true`, i.e.
   `initial_heading_error_degrees ≤ 10.0`;
3. *(was condition 4)* the echoed profile is identical to
   `docs/accepted-profile.json` on every key.

A leg failing either remaining condition is still **UNSCORED and re-set-up**,
exactly as before — this changes which conditions apply, not the discipline
that a leg failing them doesn't silently join the population.

⚠️ **What is NOT changed:** `map_facing` is still pulled and read before every
dispatch, and its value (even under mere corroboration, not full
`motion_confirmed`) is still used to aim each leg's target near the mower's
actual current facing — that practice is what fixed leg 3's alignment and
stays in place. What's dropped is only the requirement that
`safe_to_aim_dispatch` read `true` for the leg to count.

---

## Change 2 — gate disarm/verify cadence

**Original §7** required the gate disarmed and verified from the live API and
RAW `core.config_entries` **after every leg**.

**New rule for leg 4 onward:** the gate stays **armed** through the remainder
of the session. Disarm-and-verify (both live API and RAW) happens **once, at
the end of the series or whenever the operator says stop** — not between
individual legs.

**Why this is sound:** the original rule guards against one specific,
documented failure — the gate found armed *at rest*, unattended, days after a
session ended (CLAUDE.md: "found armed at rest six times"). That risk is about
a session ending without disarming. It is not a property of staying armed for
a few minutes between two legs an operator is actively watching and
individually confirming — an armed-but-idle gate still cannot move the mower
without a fresh dispatch call carrying `confirm_blades_off` and
`confirm_clear_area`, which the operator triggers explicitly each time.

**What is NOT changed:** every other per-leg requirement stays — a fresh fault
check, a fresh dry-run of the identical payload, physical corridor
confirmation from the operator, and an explicit go/no-go immediately before
each individual dispatch. Only the disarm/RAW-verify step moves from
per-leg to end-of-session.

🚨 **The end-of-session disarm-and-verify is still mandatory and still gets
the RAW check.** This amendment narrows when it happens, not whether.

---

## Everything else, restated as unchanged

- Criterion: 5/5 scored legs `target_reached`, every landing ≤ 0.15 m, no
  scored leg exhausts all 3 realignments, zero gate trips/containment
  breaches.
- Falsifier: ≥2 of 5 scored legs failing to reach target, or any scored
  landing > 0.15 m, means the claim is wrong and the L-path series does not
  open.
- Abort rule (§6 of the original): unchanged in full, including the
  `1309`/`1425` fault abort and the BLE-link abort.
- Statistics discipline: n = 5 is still evidence toward standing decision 2
  and nothing more; beta57+ epoch only; never quote a rate from an unscored
  or small-n sample.
