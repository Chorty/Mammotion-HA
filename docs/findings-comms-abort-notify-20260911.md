# Comms aborts now reach the operator, and the queue instrumentation now covers the phases legs actually die in (2026-09-11)

Offline session — **no live access, no mower, no dispatch.** Worked from a cloud
container with no `.env` and no network path to the HA host, so nothing here is
a measurement. Two code changes, both additive, plus corrections to two docs.

Build: still **beta103** (`4f908b04`). ⚠️ **Nothing in this record is deployed.**
`docs/accepted-profile.json` untouched; no Gate 5 owed. No motion-control-law
value changed: `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS` is still **2.0**, per
issue 1's "measure before changing it".

---

## 1. What the plan asked for, and what the tree actually said

Issue 1 step 1 was reported complete on 2026-09-10: capture
`_ble_link_liveness` as `command_result["queue_diagnostics"]` at the moment of
a queue refusal, scoped to `_raw_pymammotion_execute_vector_segment`.

That much is true and verified — `services.py:17364` and `:17406`, exactly as
the plan says. **The scoping is what did not hold up.** Both sites are the
executor's *linear* phase. The same executor calls three helpers on the way:

| helper | called from the executor at | its abort sites |
| --- | --- | --- |
| `_vio_segment_calibration_drive` | 16812 | `command_failed`, `stop_failed_aborting` |
| `_raw_pymammotion_turn_to_heading` | 16917, 16938 | `command_failed` |
| `_vio_turn_to_heading` | 17108, 17763 | `command_failed`, `stop_failed_aborting` |

🚨 **These are not "other executors" — they are phases of the one service being
measured**, running on every leg. A leg dying in any of them produced the same
reason string with **no queue snapshot at all**, which is precisely the
blindness step 1 existed to remove. The plan's deferral reasoning ("the same
small edit, deferred rather than done in bulk at 1 AM") is sound for
`manual_velocity_pulse_test` and friends; it does not reach these three.

🔑 **This was not academic.** Leg 4 died 0.29 m into a 4.0 m leg. The VIO
calibration drive is the executor's first real motion, so it is a live
candidate for where that abort happened — **and the record cannot say which**,
because no per-leg evidence JSON was kept for the 09-10 series. Step 2
("reproduce and measure") could have burned another real session and come back
just as blind.

**Extended on operator approval** to all four in-scope functions: seven capture
sites now. The ~17 sites in genuinely other executors stay deferred as planned.

⚠️ **`_vio_segment_calibration_drive` reports `reason`, not `stop_reason`**, for
the identical condition. Anything consuming these must read both keys — the
notifier below does.

## 2. Option B, built — with the scope correction that makes it worth building

The operator chose **B (notify only)** from
`docs/design-comms-loss-recovery-20260910.md`, over B-as-written.

🚨 **B as written would have missed the point.** That document is scoped
end to end to `stop_failed_aborting` — its title, its §2 grep, its §4B trigger.
But the 09-10 table records **leg 4** as `stop_failed_aborting` and **legs 7 and
8** — the two that tripped the series' own abort rule — as `command_failed`.
Notifying on the former alone fires for one leg and stays silent for the pair
that actually stopped the series.

As built:

- **Trigger set** is `{command_failed, stop_failed_aborting}`, read from either
  `stop_reason` or `reason`.
- **One hook point**: `_wrap_exclusive_manual_motion`'s real-run return path.
  Every motion service already passes through it (24 registrations), dry runs
  return before it, and the result is returned unchanged.
- **Two outputs**: a `persistent_notification` naming the service, entity, UTC
  time, last known position and the BLE queue snapshot at refusal; and a
  `mammotion_motion_comms_abort` event on the HA bus.
- **No command of any kind is sent.** Not a stop, not a reconnect, not a dock.
  The abort being reported is defined by a command that could not be confirmed
  delivered; adding another would be strictly worse. C and D stay unbuilt, and
  D still needs its own separate decision.

Two deliberate implementation choices:

1. **Mobile push is not integration config.** The bus event lets an operator
   automation route the abort onward, so no notify-service name is stored here.
2. **`persistent_notification` is called by service name, not imported**, so the
   manifest needs no new dependency entry for hassfest to police.
3. **Every failure inside the notifier is swallowed and logged.** It runs on the
   return path of a *completed* run; an exception there would convert a finished
   result into a raised error, losing the telemetry the operator needs exactly
   when something has already gone wrong. There is a test for this.

## 3. Corrections to the 09-10 docs

Both the findings doc (§1.5) and the design doc (§1) claimed **five call sites,
all sharing an identical `# Never keep driving/turning when stops are not
deliverable` comment dated 2026-07-12**. Against the tree:

- that exact comment string matches **nothing** (three variants exist; two carry
  the 2026-07-12 date);
- there are **four** `stop_failed_aborting` assignment sites, not five — the
  fifth grep hit is a docstring;
- they sit in **four different functions**, and one
  (`_raw_pymammotion_execute_segment`) belongs to a service this series never
  ran — not in `manual_velocity_pulse_test` or a "final-approach loop" as the
  design doc says;
- §1.5 also said "`stop_failed_aborting` twice, `command_failed` twice" for
  three legs; §0's own table says once and twice.

🔑 **The §1.5 mechanism is unaffected — where a fix has to land is not.** That
is the whole reason it mattered: the site map is what determined the
instrumentation was under-scoped. Corrected in place, with the verified map now
at findings §1.5.1.

## 4. Verification

Toolchain built from scratch in the container (CPython **3.14.7** via `uv`; the
project needs ≥3.14.2 and 10 integration files do not parse below it).

| check | before | after |
| --- | --- | --- |
| `ruff check` | clean | clean |
| `ruff format --check` | clean | clean |
| `mypy --follow-imports=skip` | clean | clean |
| `pytest` | **1057 passed** | **1085 passed** |
| `npm run test:frontend` | — | 91 passed |
| JSON parse sweep | — | clean |

**28 new tests.** ✅ **Both new behaviours were falsifier-checked**: with the
calibration-drive capture reverted the instrumentation test fails on
`KeyError: 'queue_diagnostics'`; with the wrapper hook removed the notification
test fails on its assertion. A test that passes with the change reverted proves
nothing, so both were confirmed to fail without it.

⚠️ **One anchor in the instrumentation patch matched three sites, not one**, and
the patch refused to apply until disambiguated. The other two were
`_raw_pymammotion_execute_segment` and `_raw_pymammotion_angular_calibration` —
both out of scope. Had it applied blind it would have silently widened the
change into two executors the operator did not approve.

## 5. What this authorizes

**Nothing on hardware.** No run, no dispatch, no profile change, no deploy.

Issue 1 step 2 (reproduce and measure) is still owed and still needs a real
session on the dev machine. What changed is that a leg aborting in a turn or the
calibration drive will now carry a queue snapshot instead of nothing, so that
session has a chance of producing the measurement it is for.

The 4.0 m series stays **ABORTED at 3 of 5 scored**. Do not resume it on the
strength of this work.
