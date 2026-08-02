# Claude handoff: finish Mammotion-HA P0 beta

Updated 2026-08-02 (fourth pass) after a night session of linear calibration in
darkness. This is the current handoff;
`docs/archive/NEXT-SESSION-2026-07-28.md` and the chronological sections in
`docs/p0-beta-release.md` are evidence, not current instructions.

## 🚨 READ FIRST — the open finding that should shape the next run

**`calibrated_forward_heading_offset_degrees: 102.4` looks about 11 degrees
low.** Three straight-line runs in darkness on 2026-08-01/02 all travelled on a
bearing well off what the configured offset predicts — implied offsets of
111.43, 113.29 and 115.54 degrees, mean **113.42** against a configured
**102.40**. The Nudge missed by 0.312 m and the miss was almost entirely
**cross-track**, which is an aim error, not a distance error.

It also explains Gate 4 better than anything previously proposed: an 11 degree
aim error predicts ~5.7 cm on a 30 cm leg, and Gate 4 landed 4.70 cm out.

**Do not change the profile on this alone.** Two things block a derivation:
`toward` is course-over-ground and did not update at all across a 1.36 m drive
(so the baseline may be stale), and the implied offset trends upward run to run
(consistent with the mower rotating while `toward` fails to track it). Daylight
resolves both, because VIO gives a real heading rather than one inferred from
displacement.

👉 **Treat the next Gate 5 run as an offset re-derivation as well as an
acceptance run.** Also expect the card's heading arrow to point ~11 degrees off,
since it is drawn with 102.4.

Full data, the Nudge hardware result, the docking-readback lesson and the stale
`last_error` trap are in the "Night session 2026-08-01/02" section of
`docs/p0-beta-release.md`.

## Third pass — what changed on 2026-07-31

**The card drove the mower for the first time.** Gate 5 is now PARTIAL, not
untried. `0.6.4-beta12` is deployed to the host, and the run, its two measured
constants, and the darkness abort are recorded in the Gate 5 entries of
`docs/p0-beta-release.md`. Raw telemetry:
`docs/evidence-gate5-run1-20260731.jsonl`.

Read those before planning the next run. The short version:

- Gate 5 still needs a **clean pass**: both segments `target_reached` from the
  card. The first attempt used 2.1 m legs, which the accepted profile cannot
  complete (one linear command per segment, no loop-to-tolerance). Use ~35 cm
  legs. A pre-validated path from the mower's current resting position is
  `(5.048, -3.111)` then `(5.398, -3.111)`.
- Two constants were measured and are **hypotheses, not decisions**:
  ~~`final_approach_metres_per_pulse` looks ~25% low~~ — **REFUTED 2026-08-01**
  by two isolated night pulses (1.0785 m and 1.0449 m, mean 1.0617, within
  0.16% of the configured 1.06). The constant is correct; the original figure
  came from measuring across a phase boundary in a two-segment run. And
  `heading_tolerance_degrees: 18` is
  far too loose (a turn landed 2.11 deg from target). `LUBA_ACCEPTANCE_PROFILE`
  is deliberately unchanged — editing it un-accepts it.
- ⚠️ **The VIO dusk cliff is steep and the HA sensor entities lag it.** 80/80
  features and `light` at 20:40; 0/0 and `dark` at 20:47. The sensor entities
  are coordinator-tick cached and are **not** a readiness signal — the live
  `initial_vio_feed` returned by a dry run is. Dry-run immediately before any
  Real Go near dusk.
- `scripts/ha_set_experimental_motion.py on|off|status` now toggles the motion
  gate without hand-driving the options flow. It verifies the result through
  runtime state rather than the flow's reply, because the flow returns
  `create_entry` with empty `data` even on success. It preserves other options
  by reading the flow's own schema defaults — **not** from
  `/api/config/config_entries/entry`, which never exposes `options` and would
  silently reset them.
- `scripts/ha_set_card_resource.py <version> --apply` sets the Lovelace resource
  cache key. Deploying the card file alone is not enough: browsers key on the
  query string, so an unchanged `?v=` leaves every browser on the previous card
  while every server-side check reports the new one.

## Fourth pass — night of 2026-08-01/02

Host is on **`0.6.4-beta15`**. Everything below is deployed and pushed.

- **Nudge shipped and is hardware-proven.** A straight line along the current
  facing, for moving the mower when VIO is unavailable. First hardware run:
  1.3575 m, **2 linear commands, 0 turn commands**, clean stop. It works at
  night because RTK holds in darkness while VIO does not, and because the target
  sits on the heading ray so the turn phase has nothing to do.
  ⚠️ It sends `turn_mode: "legacy"` **only** to clear the `vio_active` gate,
  which blocks up-front on `turn_mode == "vio"` regardless of whether a turn is
  needed. Legacy steers by course-over-ground and is safe there **only because
  no turn occurs** — it is *not* a night-capable turn mode. A non-zero turn count
  on a Nudge means it tried to steer blind; investigate.
- **Real Go cannot run at night, by design.** It uses `turn_mode: "vio"` and is
  refused with `blockers: ["vio_active"]` before dispatch. Confirmed on the real
  armed path: `would_send: false`, 0 commands, `phases: []`, mower unmoved. Both
  VIO gates are deliberately *advisory* in a dry run (`passed: dry_run`), so a
  dry run reports `passed: true` while its own detail says VIO is unusable.
- **The card shows heading.** A green arrow on the mower marker plus a
  `facing (map bearing)` preflight row, computed the way the backend aims. It is
  drawn in **map space** (a point one metre ahead is transformed) because `toSY`
  flips the Y axis and a screen-space rotation would mirror the bearing.
- **`final_approach_metres_per_pulse: 1.06` is correct** — mean 1.0617 over two
  isolated pulses, within 0.16%. The earlier "25% low" claim was refuted; it came
  from measuring across a phase boundary in a two-segment run.
- Mower ended the session **docked and charging**, gate off, blades off.

## Gate 5 attempt 2026-08-02 morning — SET UP, NOT YET RUN

Session ended on token budget mid-setup. **No motion occurred.** Capture proved
it: 374 samples over 10 minutes, net travel **0.0000 m**, position and both
headings bit-identical.

⚠️ **The experimental-motion gate was left ARMED** (`real_motion_allowed: true`).
Disarm with `scripts/ha_set_experimental_motion.py off` unless resuming
immediately.

Live state at handoff:

- mower `MODE_READY` at **(4.795, −1.9502)**, `toward` 173.1006, Backyard Right,
  `AREA_INSIDE`, RTK Fix, `valid_for_motion: true`, blades off, BLE ~−60
- **VIO ALIVE** — `vio_state: 2`, 80/80 features, `"Light"`,
  `initial_vision_heading: −83.673`. This is the condition the offset
  re-derivation needs and could not get at night.
- route clear (`reason: "no_route"`), no session, host on `0.6.4-beta15`

### What blocked the run: leg length, not gates

Two dry runs passed **every** safety gate. The operator's clicked path was simply
too long for the accepted profile:

| leg | clicked | needed |
| --- | --- | --- |
| 1 | 1.180 m | ~0.4 m |
| 2 | 1.870 m | ~0.4 m |

With `max_linear_commands: 1` at ~1.06 m per command, leg 1 stops **0.12 m**
short (just outside the 0.08 m tolerance) and leg 2 stops **0.81 m** short.
Neither reports `target_reached`, so neither passes Gate 5.

🔑 **Usable leg band: 0.3–0.5 m.** Above ~1.0 m one linear command cannot finish
it; below 0.08 m it is inside `waypoint_tolerance` and may count as already
arrived.

🔑 **Coordinates cannot be typed into the card** — it is a click-to-go map. The
workflow is: click roughly → **Dry-run** → read the per-segment `distance` →
drag waypoints and repeat until both legs are in band → Real Go. Precision does
not matter; leg length does.

### Two things learned setting this up

- **`stale_route_while_ready` is a real, named, non-blocking case.** Undocking by
  starting a mow and pausing leaves `route_present: true`,
  `progress_is_active: true` with `blocks_motion: false`. Cancelling the mow
  clears it to `no_route`. Not a defect; know it exists.
- ⚠️ **`sensor.*_vio_heading` is coordinator-tick cached** and stayed
  bit-identical across 374 samples. It is **not** fine-grained enough to measure
  a turn. For the offset re-derivation use the **run result JSON**
  (`vio.initial_vision_heading`, and the turn phase's before/after), not the
  sensor entity. `scripts/motion_capture.py` is still the right tool for the RTK
  position track and for proving whether `toward` updates.

## Gate 5 setup retry 2026-08-02 afternoon — NO MOTION

The operator gave a fresh daylight `GO`, but the motion gate remained off while
the card geometry was checked. The shortest practical clicks produced legs of
**0.767 m** and **0.934 m**, still outside the 0.3–0.5 m acceptance band. The
dry-run was valid with live VIO (`Light`, 80 features) and no segment blockers,
but it was not suitable for Gate 5, so Real Go was not enabled and no motion
occurred. A 402-sample capture stayed bit-identical at (4.795, −1.9502).

This exposed a card usability blocker rather than a motion-profile defect: the
full-map click/drag surface cannot reliably place sub-metre legs. Beta16 adds a
guarded coordinate editor for existing waypoints at 0.001 m precision. Every
coordinate edit clears stale dry/real results and re-runs backend Preview; Real
Go still requires a valid final preview and dry-run. The accepted motion profile
is unchanged.

Evidence:

- `docs/evidence-gate5-setup2-dry-run-20260802.json`
- `docs/evidence-gate5-setup2-no-motion-20260802.jsonl`
- `docs/evidence-gate5-setup2-ble-report-20260802.txt`

## Start here

- Branch: `feat/vio-turn-to-heading`, pushed to `Chorty`. Working tree clean.
- Personal-fork PR: [Chorty/Mammotion-HA#10](https://github.com/Chorty/Mammotion-HA/pull/10),
  still a draft. Checks were green on the pushed beta12 work (python and
  hassfest pass; HACS `skipping` is expected on a fork).
- Safety state: experimental motion **off** (verified), no session, mower at
  approximately x 5.049, y -2.753 in `Backyard Right`, blades off.
- Do not push, comment, open issues, or otherwise write to a `mikey0000`
  repository. The upstream remotes have disabled push URLs. A later push, if
  authorized, goes only to `Chorty/Mammotion-HA`.
- Do not merge fork `main`, mark the PR ready, or dispatch a beta release until
  the remaining release checks below are resolved and current CI is green.

The exact safety model, deployment instructions, chronological hardware record,
and limitations are in `docs/p0-beta-release.md`. Do not reconstruct live-test
facts from chat history when that document has the structured result.

## What is complete

- P0 backend surfaces are implemented: fail-closed capability registry,
  lowercase HA enum migration, diagnostics redaction, internal-only camera
  credentials, task/map services, experimental-motion option, exclusive motion
  sessions, `stop_manual_motion`, runtime gate export, and the click-to-go card.
- Preview and dry-run allow seven destinations. Real click-to-go is BLE-only,
  opt-in, capability-probed, and limited to two segments.
- PyMammotion is pinned consistently to the Chorty `0.8.12.post1` wheel. It is
  upstream `v0.8.12` plus the rate-limit fix, BluFi reassembly reset, and BLE
  teardown fix. No official upstream release contains the teardown fix.
- The deployed LUBA uses PyMammotion `0.8.12.post1` and integration version
  `0.6.4-beta11`. `coordinator.py` and `__init__.py` match this tree. The local
  `services.py` differs from the deployed checksum only by the corrected schema
  comment about VIO refresh; functional code is the code that passed Gate 4.
- The sole tested proxy path is P1S with passive BLE scanning and active GATT
  proxying:

  ```yaml
  esp32_ble_tracker:
    scan_parameters:
      active: false

  bluetooth_proxy:
    active: true
  ```

- Supervised LUBA acceptance Gates 1-4 all passed:
  - three-write confirmed zero stop;
  - bounded straight segment (9.69 cm toward 10 cm, 5.6 mm final error);
  - active-session abort (owner returned in 673 ms; no post-abort nonzero
    dispatch or replay);
  - 176-degree VIO regression (4.44-degree residual, 10.48 cm turn drift);
  - corrected two-leg L path (both 30 cm segments `target_reached`, final error
    4.70 cm, no delayed replay).
- Gate 4's independent stop confirmed all three zero writes in 530.4 ms. The
  mower stayed bit-identical for 18 seconds, blades stayed off, no session
  reappeared, and experimental motion was disabled afterward.

## Local changes in the handoff commit

1. A cloud-backed mower now registers its late BLE advertisement callback even
   when no proxy is ready during entry setup.
2. A temporarily absent transport reports `none` instead of raising during HA
   entity setup.
3. Bluetooth option changes invalidate the five-second motion-gate cache and
   refresh entities immediately. Enabling survives a temporarily unavailable
   advertisement so the later callback can attach.
4. `ManualMotionCancelledError` propagates out of the refresh loop after a
   defensive stop, so an operator abort releases the exclusive owner promptly.
5. A translating VIO turn now receives the configured displacement limit,
   recalculates the bearing from fresh post-turn position, and fails before
   linear motion unless alignment is freshly proven. A bounded correction has
   at most two turn commands and shares the normal realignment budget.
6. Regression coverage was added for all of the above. The exact handoff tree
   passed all 456 Python tests with coverage after the documentation pass.

## Validation at handoff

Passed on the exact pre-commit tree, after the card-profile and pre-commit work:

- 456 Python tests with coverage;
- CI-scoped Ruff (`custom_components tests`) — and, separately, `ruff check .`
  is clean repo-wide including `scripts/`;
- CI-scoped Ruff format over 48 files;
- CI-scoped mypy over 28 source files;
- all **eleven** frontend card tests (six previous, four profile assertions, and
  a README drift guard);
- 15 integration JSON files parse;
- `git diff --check`;
- `pre-commit run --all-files` — **green, and modifies nothing.**

Re-run all of the above after any further card or default change; the frontend
tests are what stop the card silently drifting off the accepted profile again.

## Assumptions that changed during hardware testing

These corrections matter more than the original chat plan:

- The recurring app/HA BLE conflict was not evidence that the mower radio was
  defective. Disabling the integration let the official app connect
  immediately; isolating proxies then implicated the IRK proxy path. P1S
  restored immediate app BLE access and stable confirmed writes. Do not re-enable
  multiple proxies during acceptance without a new controlled comparison.
- Passive `esp32_ble_tracker` scanning does not make a proxy passive for GATT.
  `bluetooth_proxy.active: true` is required and is the configuration that
  passed. `ble_link_live` may not be bypassed; routing preference alone does not
  prove a writable link.
- A VIO pivot is not an in-place geometric turn. It translated 14.43 cm in the
  failed Gate 4 attempt and changed the bearing enough to miss. Always compute
  the forward bearing from post-turn position. The corrected retry translated
  8.80 cm but freshly proved a 0.285-degree aim error before driving.
- Delayed RTK reports can make bounded physical motion appear later. Treat
  replay as a session/dispatch fact, not merely as a later position change.
- Do not increase the four-second BLE write timeout to hide stalls. That also
  lengthens uncertain nonzero delivery. The confirmed-dispatch and emergency
  stop behavior is the safety boundary.
- There is still no firmware arbitrary-waypoint upload API. The accepted path
  is a guarded chain of raw manual-motion segments, not autonomous navigation.

## Card execution profile — RESOLVED (backend-equivalent, still not UI-accepted)

The backend Gate 4 call used a deliberately bounded acceptance profile. That
profile is now the card's built-in default, named and frozen as
`LUBA_ACCEPTANCE_PROFILE` at the top of
`custom_components/mammotion/www/mammotion-custom-path-card.js`:

- `max_real_segments: 2` (already `MAX_REAL_SEGMENTS`)
- `max_turn_commands: 4`, `vio_turn_max_commands: 4`
- `max_linear_commands: 1`, no loop-to-tolerance ceiling
- `waypoint_tolerance: 0.08`
- `min_progress_distance: 0.0025`
- `calibrated_forward_heading_offset_degrees: 102.4`
- `motion_refresh_interval_ms: 200`
- `ble_auto_recover: false`

Both option 2 and option 3 from the previous handoff were taken: the accepted
profile *is* the default **and** it is a named, exported, test-pinned object
rather than values scattered through `setConfig` and `_motionPayload`.

Implementation notes that matter:

- `max_linear_pulse_ceiling` is `null` in the profile and the key is **omitted**
  from the payload, because the backend schema is `vol.Optional` with no default
  and `Range(min=1)` — sending `0` would be a validation error, and sending any
  number would re-enable loop-to-tolerance, which Gate 4 did not use.
- Resolution is `??`-based (`_profileValue`), never `||`. The old code used
  `Number(this._config.x || default)`, which silently discarded a configured `0`
  — including `motion_refresh_interval_ms: 0`, the legacy single-shot mode.
- `_profileOverrides()`/`_profileLabel()` compute which profile keys the
  dashboard YAML overrode. The card renders an **execution profile** row that
  reads either `LUBA acceptance profile (Gates 1-4, 2026-07-31)` or
  `customised (not hardware-accepted): <keys>`. An operator can now see from the
  card whether the payload is the accepted one.
- `heading_tolerance_degrees` stays at 18. It is a known-loose value from the
  July 18 single-shot calibration, but it is what the accepted run used, so
  changing it here would have left the accepted profile in the name of tuning.

Frontend assertions and README YAML were updated in the same change: four new
DOM tests pin the profile values, the omitted ceiling, the override labelling,
and the falsy-value regression. A fifth test pins the README block itself —
README is a third copy of these numbers in paste-ready YAML, and nothing else
stopped it drifting from the values the hardware ran (11 frontend tests total).

Independently re-verified this session: the payload the card emits from an
unmodified dashboard config validates against the *real*
`RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA` and
`RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA`, with
`max_linear_pulse_ceiling` absent from the validated result. The frontend tests
alone could not have shown that — they never cross the JS/Python boundary.

**Still true, and the reason this is not a release sign-off:** the card has
never driven the mower end to end. Gates 1-4 exercised the *backend* with these
values. If release criteria require UI-to-mower acceptance, that needs a repeat
preview/dry-run and then one Real Go from the actual card, under a **new**
daylight operator `go`. No physical motion is authorized by this handoff.

✅ Versions were bumped together to `0.6.4-beta12` for the Gate 5 build:
`manifest.json`, `pyproject.toml`, `CARD_VERSION`, and `uv.lock` (PEP 440 form
`0.6.4b12`). The bump is **local only** — nothing was deployed, and the host
still runs beta11. The point of the bump is that a Gate 5 run must be able to
prove in the browser console banner that the new card loaded; the card is
served from two paths, so deploy to both or the stale card is served silently.

## Remaining P0 work — dispositions

1. **Card-default mismatch — DONE.** See the section above.
2. **The three deferred defects — two were already fixed, one is reframed:**
   - *Map edits not visible until HA restart:* **already fixed** in `6cf4d5fd`,
     before this handoff was written; the entry was stale. The per-tick block is
     reachable now because `_async_short_circuit_update()` returns `None` on the
     healthy path and every caller tests `is not None` rather than truthiness.
     Covered by `test_short_circuit_update_returns_none_on_the_healthy_path`,
     `test_every_coordinator_tests_short_circuit_with_is_not_none` (an AST check
     over all five coordinators), and
     `test_update_loop_only_starts_a_map_sync_through_the_gate`.
   - *False turn-phase `no_actuation_detected`:* **already fixed** in the same
     commit, and the suggested fix was wrong. `heading_went_fresh` cannot be the
     discriminator: it is True exactly when before/after differ by more than the
     epsilon, which is exactly when `_streak_shows_no_actuation` (bit-identical
     heading) is False. The two are perfectly correlated, so gating on it would
     delete the no-actuation branch instead of refining it. The real
     discriminator is `heading_poll_feed_alive` — "did *any* channel move at
     all" during the poll, since a live feed jitters ~2-4 mm in position and
     ~0.0018 deg in heading even when stationary. `_streak_shows_dead_telemetry`
     runs first and reports `vio_telemetry_stream_stale`; `no_actuation_detected`
     now only fires with a demonstrably live feed. A replay of the exact
     2026-07-25 run is pinned as a regression test.
   - *Task-2 constants:* **reframed, not release-blocking.** The original
     statement dates from when Step 5b had died twice on transport. Since then
     the 2026-07-27 3.0 m segment landed 1.0 cm along-track, and Gates 1-4
     executed the pulse-geometry ceilings, `min_progress_distance` and cadence
     on hardware. Those constants are no longer hypotheses — they are the values
     in `LUBA_ACCEPTANCE_PROFILE`, now pinned by tests. What is genuinely
     un-re-derived is narrower: `heading_tolerance_degrees` (18, derived from the
     single-shot rotation quantum that refresh made obsolete) and the refreshed
     turn-pulse floor. Both are beta tuning behind a new operator `go`, not
     release gates, because the release ships values hardware actually ran.
3. **All-files pre-commit — REPAIRED, now green.** See the next section.
4. **Version agreement — bumped to `0.6.4-beta12`.** `manifest.json`,
   `pyproject.toml`, `CARD_VERSION` and `uv.lock` all agree, and
   `manifest.json`, `pyproject.toml` and `requirements_test.txt` all declare the
   identical `chorty-0.8.12.post1` wheel URL. The next `Beta Release` dispatch
   will compute `0.6.4-beta13` (dry-run verified).
   **New:** `Beta Release` could not have run at all before this session —
   doubled backslashes in YAML block scalars made every sed capture group fail,
   so it proposed the already-existing `v0.6.4-beta1` on every dispatch and
   exited 1, and its `uv.lock` verify grepped a package name that is not in the
   file. Fixed in `.github/workflows/beta-release.yml`; both steps dry-run
   against this tree. This is also the mechanism behind the previously recorded
   version regression.
5. With explicit operator authorization, push only to the Chorty feature branch
   and wait for current PR checks. HACS `skipping` on a fork is expected; Python
   and hassfest must pass.
6. Only after the card profile decision and current CI pass: mark PR #10 ready,
   merge to Chorty `main` without force-pushing, and run `Beta Release` with the
   LUBA-acceptance confirmation. Never publish to Mikey's repositories.

## All-files pre-commit — repaired

`pre-commit run --all-files` now passes and modifies nothing. Each failure had a
distinct cause, and each was fixed rather than scoped away where fixing was
cheap:

- **Hook/CI version skew (the root of the mypy and Ruff noise).** The `ruff`
  hook pinned `v0.12.8` while CI pins `ruff==0.15.16`; the older ruff still
  enforced `UP038`, a rule later ruff removed, so the hook failed on two lines
  CI passes. Pinned to `v0.15.16`. Likewise `mirrors-mypy` was `v1.17.1` against
  a `mypy==2.1.0` pin; now `v2.1.0`. Both pins must move with
  `requirements_test.txt` or the gate lies again.
- **mypy scope.** The hook ran `--strict` over all of `custom_components` and
  reported 168 errors, essentially all `Class cannot subclass "SensorEntity"
  (has type "Any")` — an artifact of HA shipping untyped entity base classes to
  the checker, and none of it checked by CI. Now `--follow-imports=skip` over
  `custom_components/mammotion/`, matching CI exactly. Passes.
- **Ruff over `scripts/`.** 22 real findings, so they were fixed, not scoped
  out: an unused `sys` import, three missing docstrings and a
  `subprocess.run(check=...)` in `scripts/linear_duration_sweep.py`. The 17
  `T201` prints are legitimate — these are operator CLIs whose entire output
  contract is stdout — so `scripts/*.py` gets a documented `T201` per-file
  ignore. `scripts/` stays linted otherwise. `ruff-format` still skips
  `scripts/`, matching CI, so a live investigation cannot be reformatted
  mid-flight.
- **codespell.** `--skip="./.*,*.csv,..."` was written with literal quotes
  inside a YAML args list, so the whole skip list was inert. Fixed, and the
  APK-verbatim identifiers (`entitys`, `swtich`, `piar`, `buttom`, `unknow`) plus
  `unparseable` are now in `--ignore-words-list`, because correcting them would
  stop them matching the thing they document. The two standalone browser dev
  tools are skipped.
- **pyupgrade removed.** It crashed on every file under Python 3.14
  (`tokenize.cookie_re` is a bytes pattern there, so `_fix_tokens` raises
  `TypeError`) and was redundant: ruff already selects the `UP` ruleset.
- **prettier scoped to the JS this integration ships**
  (`custom_components/mammotion/www/*.js`, `tests/frontend/*.mjs`). Unscoped it
  rewrote ~20 Markdown evidence files, the APK feature catalogue,
  `services.yaml` and `manifest.json` on every run. Bringing `agora-client.js`
  into scope reformatted it once; that diff was verified non-semantic
  (quotes, trailing commas, wrapping, arrow parens only) and it is a
  redeploy-worthy but behaviourally identical file.
- **`*.patch` protected.** `trailing-whitespace` and `end-of-file-fixer` were
  stripping trailing whitespace from `docs/upstream-patches/*.patch`, where it
  is significant in unified-diff context lines — the hooks were silently
  corrupting patches so they would no longer apply. Both now exclude `\.patch$`.
- **yamllint config.** `args: -c .yamllint.yaml` sat one level out, as a key of
  the repo mapping rather than the hook, so `.yamllint.yaml` was never applied.

The config now carries a scope rule at the top: every hook must agree with
`.github/workflows/validate.yml`, and any deliberate narrowing states its
reason inline.

## Safety state at handoff

- `enable_experimental_motion: false`
- no active motion session
- last Gate 4 session completed normally
- mower last reported `MODE_READY`, blades off, fixed RTK, inside `Backyard Right`
- last settled map position: x 4.3911, y -2.8064
- no live test may reuse the previous operator confirmation

## Useful commands

```sh
git status --short --branch
git log -1 --stat
.venv/bin/python scripts/mammotion_preflight_gates.py --quick
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests
.venv/bin/python -m ruff check custom_components tests
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend
```

Never enable broad PyMammotion debug logging: cloud and raw BLE loggers can
expose credentials, network responses, device identifiers, and payloads. Use
only the scoped `bleak_esphome` and `habluetooth` loggers documented in
`docs/deploy-runbook-p0.md`.
