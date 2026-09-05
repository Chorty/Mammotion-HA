# Working prompt — 2026-09-05: the orientation problem

Read `CLAUDE.md` in full first (all live), then
`docs/findings-clicktopath-reliability-4m-20260904.md` in full. **That findings
doc is the input to this session** — everything below is scoped by it.

**Goal for today: fix the orientation problems found last night, and get real
testing going while there is daylight.** Most of the work is offline and can
start immediately; only §5 needs the mower.

---

## Live state (verify before acting — true 2026-09-05 ~02:30 UTC)

- Host runs **0.6.4-beta101**, backend `chorty-0.8.12.post4`. Deploy verified
  byte-identical; the card wording change is **still not browser-confirmed**.
- Mower **docked and charging** at (4.3188, 3.2862), ~33% and climbing. Gate
  **disarmed**, verified live API **and** RAW. VIO was dead at session end
  (`tracked_features: 0`) — that was nightfall, expect it back in daylight.
- `main` is pushed and clean. 🚨 The operator keeps an uncommitted edit in
  `docs/agora_outbound_audio_probe.md` — **stage by explicit path, never
  `git add -A`.**

---

## What last night actually found

The 4.0 m series is a determined FAIL at n = 4, but the verdict is the least
interesting output. Three findings underneath it matter more, and they are all
the same problem wearing different hats: **nobody — not the integration, not the
orchestrating session, not the operator through HA — could reliably answer
"which way is this mower pointing?"**

1. 🚨 **The heading model used to place every target was wrong by a mean 87°.**
   Measured on 43 real pulses against the executor's own settled-displacement
   heading: the compass mirror `90.13 - toward` predicts the driven direction to
   **1.000°** mean error; `toward + calibrated_forward_heading_offset_degrees`
   (102.4) is off by **87.478°** mean, 166.8° worst.
2. 🚨 **The "aligned start confirmed" check was circular.** The target was placed
   along `toward`, so the echoed `target_reported_heading_degrees` agreed with
   `toward` by construction. It measured nothing. Runs 3 and 4 actually opened
   with real ~120–135° turns — post-turn legs, which that series explicitly put
   out of scope.
3. 🚨 **The device emits `Robot orientation unavailable (1309)` and this
   integration cannot see it.** Five of them during the docking failures, while
   `sensor.*_last_error` read `"mcu: , "` with an hour-stale timestamp. The
   vendor app diagnosed in one line what cost this session hours.

Plus one process trap, now in CLAUDE.md: **heading telemetry is stale after a
manual reposition until the mower actually drives**, and
`current_orientation.trustworthy` publishes on *corroboration between two
sources*, not on *freshness* — both can be stale together and agree.

---

## 🔑 The key structural fact, already verified in the tree

**The correct transform already exists and the code already says why.**
`custom_components/mammotion/services.py:14769-14795`:

```python
#: The map frame is a math angle (CCW from +x) while ``toward`` is a compass
#: bearing (CW from north), so their relation is a reflection, not an offset.
_TOWARD_MIRROR_DEGREES = 90.13

def _map_heading_to_toward_degrees(...)   # (mirror - map_heading) % 360
def _toward_to_map_heading_degrees(...)   # (mirror - toward) % 360
```

⚠️ **This is a model-form bug, not a mistuned constant.** A reflection cannot be
emulated by adding a constant, so **no value of
`calibrated_forward_heading_offset_degrees` can ever be correct** for converting
`toward` into a map bearing. It happens to look right near the headings where the
two curves cross, which is how it survived.

🔑 **`_map_heading_to_toward_degrees` has exactly ONE call site** (~line 16021).
The additive offset is what every schema exposes and what callers reach for.
**Work out which paths genuinely need which, and be precise about it** — the
findings doc §4 establishes that in `turn_mode: vio` the executor steers on
`target_vision_heading` from its own calibration drive, so the additive offset
did **not** steer last night's runs; it only decided *where the targets were
placed*. Confirm that reading yourself before acting on it.

---

## The work

### 1. 🔴 Surface the device's own error codes — do this first

Highest value per hour, entirely offline, no hardware. An operator watching Home
Assistant had **strictly less information** than one watching the vendor app,
during a failure the app explained in one sentence.

- `1309` appears nowhere in `custom_components/` or the pinned pymammotion —
  confirm that, then find where device errors actually arrive (start from
  whatever feeds `sensor.*_last_error`; last night it produced `"mcu: , "`,
  which suggests a code/text pair where both were empty or unmapped).
- Getting the **raw numeric code plus whatever text the device sends** in front
  of the operator is the win. A full code→description mapping table is a bonus,
  not the requirement — do not let its absence block surfacing the number.
- ⚠️ Check whether the code was available to us all along and merely dropped. If
  it was, say so plainly in the commit message; that is the more useful finding.

### 2. 🔴 Fix how facing is derived, and make the fix impossible to get wrong

The remedy is **not** "change 102.4 to something else" (see above), and **not**
editing `docs/accepted-profile.json` — 🛑 **any change to the accepted profile
owes a new Gate 5**, per `docs/route-b-collinear-split-20260819.md:115-118` and
the findings doc §7. Prefer a route that leaves the profile untouched.

Think about what would actually have prevented last night:

- A single, obvious, correct way for a caller to ask **"what is this mower's
  current map-frame facing, and is that answer fresh?"** — one that returns
  *unknown* rather than a wrong number when it cannot tell.
- A freshness concept in `current_orientation`. Today `trustworthy: true` means
  two sources agree; it said `true` at 0.571° while (per §6.3) both may have been
  describing the pre-reposition facing. **Corroboration is not freshness.**
  Consider: time since the last real motion, whether `toward` has moved since the
  last commanded rotation, the ~1 pulse-cycle (~5 s) lag §4 quantifies.
- A **non-circular** alignment check. Any check whose inputs derive from the
  number being checked must be refused or renamed. The executor's calibration
  drive already measures true facing independently (`map_motion_heading_degrees`)
  — that, or the operator's eyes, is what "aligned" has to mean.

✅ **Validate against last night's banked data before touching hardware.** The
43-pulse dataset is in `docs/evidence-clicktopath-reliability-4m-20260904.json`
with the per-pulse `movement_vector_heading_degrees` the executor measured
itself. A corrected model must reproduce the ~1.0° mirror result and the ~87°
additive result. **That is a real regression test and it needs no mower.**

### 3. Write the tests that would have caught this

- The circular check (§2 above) should be impossible to reintroduce.
- The stale-after-reposition case: heading unchanged across a manual move, first
  real motion producing a ~166° jump.
- Both are unit-testable against banked numbers. `tests/components/mammotion/`
  was split into six files yesterday — put these where the name predicts them
  (`test_turn_primitives.py` or a new focused file, not the visibility grab bag).

### 4. Deploy, if §1–3 land

Use the `release-and-deploy` skill. ⚠️ **beta101's card wording is still not
browser-confirmed** — fold that check into whatever ships next: ask the operator
to confirm the rendered advisory text and the version in the card footer.

### 5. 🚨 Hardware validation — daylight, operator present, predeclared

**Only after §2 is in and deployed.** This needs its own short predeclaration —
it is a new measurement, not a resumption of the failed 4.0 m series.

What to actually measure: **does the corrected facing derivation predict the
driven direction on the ground?** Small, cheap, low-risk shapes are fine — this
does not need 4.0 m legs, and short moves make a wrong answer harmless. Fix the
criterion and the abort rule before dispatching, as always.

⚠️ **Every hard-won lesson from last night applies, and they are cheap:**
- Derive facing **two ways** and require agreement before any armed dispatch.
- **State the destination to the operator in compass/landmark terms** ("it will
  drive toward the fence, about 4 m") and have them confirm the ground before you
  send it. That one sentence would have caught the incident.
- **Ask for a tape measurement** on any corridor tighter than a couple of metres.
  The map polygon was wrong by 0.71 m in the unsafe direction last night; the
  operator's tape caught it and two other hazards no software check would have.
- A "small test move" is an armed dispatch. Treat it as one.
- Per-run operator go/no-go immediately before each dispatch; fresh corridor scan
  against the map; gate disarmed and verified from live API **and** RAW after.

---

## Boundaries

- 🛑 **Nothing here reopens a standing decision.** Phase 2 (5), OTA (6), accuracy
  (3) and night (4) all stay CLOSED. Reliability statistics remain beta57+ epoch
  only (7). None of last night's findings bear on any of them.
- 🛑 **Do not change `docs/accepted-profile.json`** without a predeclaration and a
  Gate 5. The §4 measurement is an observation, not an authorization.
- Do not quote "3 of 4" or "75%" as a reliability rate — partial n, aborted
  series, and the start-geometry precondition was violated on every run.
- Never push to `mikey0000/*`; pass `-R Chorty/Mammotion-HA` to every `gh`
  command. Stage by explicit path.
- Keep the discipline in CLAUDE.md → "How this project works". It caught the
  circular-alignment error, eventually — but only in the write-up, after four
  real runs had already been dispatched on it. **Earlier is the whole point.**
