# PREDECLARATION — Option D, auto `return_to_dock` after a confirmed-stationary comms abort (2026-09-11)

🛑 **NOT AUTHORIZED. NOT BUILT. This document exists to be decided on, and is
written before any implementation so the criteria cannot be chosen after seeing
what happens.** Options B and C are built and shipped; D is deliberately not
inheriting their approval, per the design doc's own §5 and this project's
standing practice for anything that moves the mower.

Parent: `docs/design-comms-loss-recovery-20260910.md` §4D.

---

## 1. What D would actually do

After a real run aborts on `command_failed` or `stop_failed_aborting`, and
**after Option C has returned `confirmed_stationary`**, and after a configurable
quiet period elapses with no operator response: issue `return_to_dock` — the
vendor's own navigation, with its own onboard obstacle handling, not this
integration's raw motion primitives.

## 2. 🚨 Why this is categorically different from B and C

B and C are read-only. **D sends a new motion command, unattended, to a mower
whose last command could not be confirmed delivered.** That sentence is the whole
risk, and it does not get smaller with careful implementation:

- The trigger condition *is* "comms just failed". D would dispatch into exactly
  the conditions that caused the abort — plausibly the same 2.0 s queue-start
  contention that refused legs 7 and 8 on 2026-09-10.
- It is the only option that touches `coordinator.py`'s job-control surface
  rather than staying in the diagnostic layer.
- **Every real run in this project's history has had explicit operator go/no-go
  immediately before dispatch** (see CLAUDE.md, "How this project works"). D is a
  deliberate, permanent exception to that rule. That is the decision, not the
  code.
- ⚠️ **`return_to_dock` is not a proven-safe primitive here.** It failed twice on
  2026-09-04 with `1309` before succeeding first-try from 8.7 m on 2026-09-05 —
  and that success is explicitly recorded as **n = 1, corroboration not proof**.

## 3. What would have to be true BEFORE any code is written

1. **Operator says yes explicitly**, in the moment, not inferred from approving
   B or C.
2. A decision on **whether D may run when nobody is home** — the scenario it
   exists for. If the answer is "only when someone is present", D collapses into
   a notification with a button, which is a much smaller change and probably the
   better one.
3. A decision on **what D does when it fails**. A `return_to_dock` that aborts
   halfway leaves the mower somewhere new, possibly worse, with the operator's
   mental model now stale. D must not retry silently.

## 4. Predeclared criteria — written before any run exists

If D is ever built, it ships only if **all** of these hold on a supervised trial:

| # | criterion | falsifier |
| --- | --- | --- |
| 1 | D never triggers while C's verdict is anything other than `confirmed_stationary` | one trigger on `cannot_confirm_*` or `still_moving` = FAIL |
| 2 | D never triggers inside the operator-response window | one early trigger = FAIL |
| 3 | A `return_to_dock` that fails leaves exactly one notification and **no retry** | any automatic second dispatch = FAIL |
| 4 | The operator can disable D without editing code | config-flow option absent = FAIL |
| 5 | 5 of 5 supervised trials reach the dock, or D does not ship | 4 of 5 = FAIL, not "mostly works" |

🔑 **Criterion 5 is deliberately strict.** D exists to be trusted unattended; a
rate of "usually docks" is worse than no D at all, because it replaces a known
unknown (mower is somewhere) with a false belief (mower is docked).

## 5. The alternative that may dominate D

**A notification with an actionable button** — C's notification gains "Send to
dock", which the operator taps from their phone. It solves the real "stranded
and nobody's coming" case whenever the operator is reachable, sends nothing
unattended, needs no new trust, and is a much smaller change.

D only beats it when the operator is **unreachable**, which is a narrower
scenario than it first appears. ⚠️ **Recommend deciding whether that scenario is
worth the exception before building D**, not after.

## 6. Status

**AWAITING OPERATOR DECISION.** Nothing in this document authorizes
implementation, and no code for D exists in the tree.
