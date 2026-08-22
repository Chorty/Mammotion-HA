# Next session prompt — Mammotion-HA

Copy everything below the line as the opening prompt.

---

Continue the Mammotion-HA work. Read `CLAUDE.md` "Start here" and
`docs/NEXT-SESSION.md` §0 first, then this.

## STATE (verified 2026-08-20 ~17:00 EDT — re-verify before acting)

- Host runs **`0.6.4-beta65`**, deployed and verified (46/46 byte-identical,
  card md5 `96a27a39` at both paths, resource `?v=0.6.4-beta65&build=96a27a39`).
- `main` clean and pushed at `845ddf3e`.
- Motion gate **DISARMED and verified** (`enabled: false`).
- Mower is **clear of the no-go zone**, roughly `(5.87, -5.14)`, `AREA_INSIDE`,
  RTK Fix, battery ~53%, VIO `tracked_features` 80.
- Gate baseline: **742 pytest, 76 frontend**, ruff check, ruff format, mypy,
  ten pre-commit hooks, `check_accepted_profile.py` **ACCEPTED**.

## TASK 1 — correct a wrong claim I committed. Do this first.

`docs/evidence-routeb-retry-overshoot-20260820.json`'s commit message
(`845ddf3e`) says the beta42 quadrature term "needs checking" and implies it may
not be applied. **That is wrong, and it was disproved before the session ended.**

`_projected_landing_after_next_pulse` uses `t = min(d, metres_per_pulse)`, so
with `d = 0.2994` and `mpp = 1.06` it uses `t = 0.2994`, giving
`perpendicular 0.1244`, `overshoot 0.0271`, `hypot 0.1273` — reproducing the
recorded `projected_landing_m: 0.1273` exactly. The term is present and correct.
`perpendicular` and `projected_landing` sitting close together is CORRECT at
that geometry, not evidence of a missing term.

Record the correction where a later session will see it (a short note in the
evidence file plus `docs/NEXT-SESSION.md`). This is exactly the stale-claim
class `CLAUDE.md` warns about.

**The real limitation, stated accurately:** the guard is a SINGLE-PULSE
lookahead. It correctly projected 0.1273 m for one more pulse; the segment then
fired TWO more (pulses 12 and 13) and landed 0.1673 m out with the target
130.656° behind it. The function's own docstring already notes measured
next-pulse travel runs **0.30x to 1.16x** of the remaining distance. Do NOT
change the guard on this single datapoint — two previous attempts to improve a
re-aim heuristic on thin evidence were both reviewed and reverted.

## TASK 2 — retry Route B at 3.0 m sub-legs

Send `split_leg_target_length_m: 3.0` instead of 3.85. A ~11.5 m click then
splits into 4 sub-legs of ~2.9 m.

⚠️ **Do not present this as likely to succeed.** The 3.85 m failure was not
primarily leg length: it was an 18° opening aim error, a mid-drive correction at
pulse 2, then an over-suppression during final approach. A 3.0 m leg starting
with the same opening error can reproduce all three. The measured-good regime is
~0.8 m; 3.0 m is still ~4x outside it. It is the right experiment, not a likely
win.

Procedure: dry run first (free, and confirms the keep-out check passes the
specific path) → explicit per-run authorization → arm → run → **disarm and
verify** → bank evidence immediately.

Pick the bearing with a scan that checks the area polygon **and**
`export_map.keep_out_polygons`. A scan against area geometry alone is how the
mower drove into a trampoline on 2026-08-20.

## WHAT ROUTE B HAS AND HAS NOT SHOWN

- ✅ **The collinear junction costs ZERO turn commands on hardware** —
  `turn_commands_sent: 0`, measured, not argued from code.
- ✅ Split geometry is exact: sub-legs equal to 6 dp, headings identical to 9 dp.
- ✅ A 3.6 m sub-leg landed **0.092 m** (n = 1).
- ❌ **End-to-end across a chain is 0 for 2.** Run 1 hit a trampoline on sub-leg
  2; run 2 failed on sub-leg **1** at 3.83 m, before any junction. **Neither
  failure was the splitter.**

## OTHER OPEN ITEMS (not scheduled — ask before starting)

1. ✅ **Keep-out segment containment shipped in beta69.**
   `test_a_leg_that_clips_a_corner_is_caught` pins the closed gap; live map and
   browser verification passed.
2. **`safety_overrides` is not wired into the primitives** — `MOVEMENT_SCHEMA`
   and `MANUAL_VELOCITY_PULSE_TEST_SCHEMA` cannot express an override. That gap
   is why the nudge buttons had to be ungated rather than override-gated.
3. **The card does not draw keep-outs yet.** `export_map.keep_out_polygons` is
   available; refusing at click time beats refusing at dispatch.
4. ✅ **SUPERSEDED — installed 2026-08-22.** *(Was: still NOT installed,
   verified absent from 69 automations.)* The gate was found armed at rest **four** times, once with zero
   blockers and the mower off its dock. YAML:
   `docs/automations/disarm-motion-gate.yaml`. Operator's call.
5. Ceiling `14 -> 22` still untested; needs a leg over ~5 m.

## DISCIPLINE THIS REPO ENFORCES

- **Verify against the tree; `CLAUDE.md` goes stale.** One grep beats it.
- **Check gate EXIT CODES.** Never `cmd | tail` — a pipeline's status is
  `tail`'s.
- **Tests must exercise the code, not mocks that accept anything.** On
  2026-08-20 an `AsyncMock` let a `TypeError` ship: the nudge handler called
  `async_move_*()` with no arguments against a required positional `speed`, and
  `assert_awaited_once_with()` passed anyway. Bind against real signatures.
- **Scripted edits: anchor on unique strings.** `src.index("BUTTON_SENSORS:
  tuple[")` matched `SPINO_BUTTON_SENSORS` and corrupted `button.py`.
- **NO MOTION without explicit per-run authorization.** Arm only for the run,
  then disarm and verify. Daylight only for VIO paths — check
  `sensor.*_vio_tracked_features` is non-zero rather than assuming.
- **Bank evidence into `docs/` immediately** — the card keeps only ~3-5 full
  runs, and a scratchpad file is not a record.

## THE NUDGE BUTTONS ARE DELIBERATELY UNGATED

`_nudge_available` returns `True` unconditionally and `_unguarded_nudge` calls
the coordinator primitive directly with `(0.4, use_wifi)`. This is the
operator's explicit decision (2026-08-20), taken after the trade-offs were
stated, because the mower stranded itself inside a no-go zone where every
guarded path refused `position_not_valid_for_motion`. **Do not "fix" this back
as a bug** — `tests/components/mammotion/test_nudge_buttons_ungated.py` pins it
and explains why.
