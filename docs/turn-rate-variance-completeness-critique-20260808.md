# Completeness critic - workflow wf_cec0c5f1-051 (2026-08-08)

## 1. What was missing — three things I opened

**(a) The turn phase records no stop timing at all, and its stop is not emergency-priority. This kills the "elapsed + stop latency" rate table.**

`_vio_turn_to_heading` sets `command_result["stop_ack"] = await _stop_manual_motion_confirmed(coordinator)` (`custom_components/mammotion/services.py:7982-7984`) — no `emergency=True`, and `_stop_manual_motion_confirmed` returns the literal `{"movement_ok": True}` with **no duration field whatsoever** (`services.py:3321-3333`). The only stop helper that times itself is `_manual_velocity_stop_attempt`, which records `result["duration_ms"]` in its `finally` block (`services.py:3280-3278`, timing at `:3286` and `:3276`) and dispatches with `emergency=True` (`services.py:3307`).

Its call sites are `services.py:5534, 8610, 9828, 10141, 11047`. Of those, the ones in this run's path are `_vio_segment_calibration_drive` (def `services.py:10061`, stop at `:10141`) and the linear branch of `_raw_pymammotion_execute_vector_segment` (def `:10193`, stop at `:11047`). Attempt 5 segment 1 recorded `calibration: 1, turn: 4, linear: 3` (`docs/evidence-gate5-PASSED-20260808.json:93-97`) — **1 + 3 = exactly the four stop-confirmation values quoted (1175/1819/402/628 ms)**, and the four turn pulses produced none.

Consequence: A-refresh-timing's recomputed rates of 9.47 / 6.62 / 26.66 deg/s, built by adding stop latency to turn-pulse elapsed time, rest on a pairing that the code makes impossible. Discard them.

Second consequence nobody flagged: turn-pulse stops enqueue at `Priority.NORMAL`, linear stops at `Priority.EMERGENCY` (`services.py:5719-5720`, `Priority.EMERGENCY if emergency_stop else Priority.NORMAL`). The code's own comment on why emergency priority exists is directly on point: *"Live 2026-08-02 a normal-priority stop took 1392.7 ms to confirm while the mower continued past its target"* (`services.py:3301-3306`). So the project has a committed live observation that the mower **keeps moving during a normal-priority stop's confirmation latency** — and the turn phase is the one path still using normal priority. That is an unmeasured contributor to per-pulse rotation, and it is an asymmetry, not a design statement: no comment anywhere justifies it.

**(b) `_vio_turn_probe` already implements the exact settled-rotation measurement Problem A needs, and its code comment already states the mechanism.**

`_vio_turn_probe` (def `services.py:6853`) samples during the drive (`services.py:6989-7003`, each sample carrying `elapsed_seconds`) **and** takes `post_stop_samples` after the mandatory stop (`services.py:7038-7052`), then judges rotation across `samples + post_stop` combined (`services.py:7065-7071`). The comment justifying that is the single most relevant thing in the file:

> `services.py:7059-7064` — *"VIO heading refreshes ~1.5s into the command and the position feed lags ~4s, so on a short pulse the ONLY sample taken during the command is the t=0 one (bit-identical to baseline) and every real change lands in post_stop. Judging the during-command samples alone therefore reports a real rotation as zero: live 2026-07-19 a taped 13.18 deg pivot came back `vision_heading_static_during_command` with `final_displacement_m: 0.0` while this function's own post_stop samples held the answer."*

That is a committed, live-measured statement that on a ~1.5 s turn pulse the VIO heading has **not** finished registering by the time the pulse ends. The turn executor's poll (`services.py:8020-8056`) breaks on the *first* sample exceeding `_VIO_HEADING_FRESH_EPSILON_DEGREES = 0.1` (`services.py:7158`, break at `:8049-8054`) with no settle requirement — unlike `_settle_linear_position_feed` (`services.py:8265-8334`), which requires two consecutive agreeing samples (`:8318-8321`). So the executor's per-pulse rotation number is, structurally, "the heading at the first instant it moved off baseline," not "the heading after this pulse finished registering." Measurement contamination is not a speculative hypothesis here; it is the documented behaviour of the neighbouring probe.

**(c) An arithmetic check nobody ran: degrees per delivered non-zero write.**

Non-zero writes per pulse = `refresh_commands_sent + 1` (the initial write at `services.py:7945-7950` is the caller's, outside the refresh window — see the docstring at `services.py:4875-4877`, and the same +1 convention codified for linear at `services.py:10029`). That gives 4 / 3 / 3 writes.

- deg per write: 30.46/4 = **7.615**, 22.17/3 = **7.390**, 57.63/3 = **19.210** → pulses 1 and 2 agree to **3%**; pulse 3 is a 2.6× outlier.
- ms of window per write: 2043/4 = **510.8**, 1530/3 = **510.0**, 1760/3 = **586.7** ms → near-constant, and consistent with the reported 540 ms median write latency. The pulse window was entirely write-latency-bound, exactly as A-refresh-timing's loop model predicts (`services.py:4909-4945`).

So the 2.6× "rate variance" is not three different rates — it is **two matching pulses and one anomaly**, and the matching pair matches on *writes delivered*, not on wall-clock time. That reframes the question: it is not "why is the rate noisy," it is "why did pulse 3 produce 2.6× the rotation per write." Delayed registration from pulses 1–2 arriving during pulse 3's poll is the leading candidate and is testable.

**Still unopened, and I did not open them:** the pymammotion-side update cadence of `report_data.vision_info.heading` (whether it is push-driven or polled, and at what interval) — this bounds how much truncation is possible and lives in `.venv/lib/python3.14/site-packages/pymammotion/`; whether any firmware deadman timer stops the motors between writes (A-refresh-timing established no such value exists anywhere in this repo or the vendored package — that remains genuinely unknown and it is the physical parameter that would explain a ~50% duty cycle at 510 ms write spacing); and whether the attempt-5 raw service response still exists in HA's service-call trace or the card's response pane.

## 2. Contradictions, adjudicated

**(i) A-refresh-timing's stop-latency rate table vs. the code.** Refuted above: `services.py:7982-7984` + `:3321-3333` mean no turn pulse carries a stop duration. The finding's own caveat ("this pairing is an ASSUMPTION") understates it — the pairing is not merely unverifiable, it is contradicted. The four values are calibration + linear stops.

**(ii) A-measurement-window's 110.26° vs. 4 pulses consumed — internal inconsistency in the data, not between findings.** `services.py:8107-8109` returns `target_heading_reached` the moment `abs(new_error) <= heading_tolerance_degrees` (18, `mammotion-custom-path-card.js:41`). If the three quoted rotations were same-signed and the VIO-frame initial error were the geometric 93.5° (`docs/evidence-gate5-PASSED-20260808.json:79`), the loop would have returned after pulse 3 with error −16.76°. It did not (`:96-99`, "4 of max 4 — FULLY CONSUMED"). At least one premise is false: the signs are not all the same, or the VIO-frame target error ≠ 93.5° (the target is derived at `services.py:10643-10646`, not from raw path bearing), or pulse 3's overshoot was larger than the quoted magnitude implies and pulse 4 was a reversal. `heading_error_after` is recorded per command (`services.py:7936`, set at `:8073`) — the uncommitted JSON settles this in one line. **Do not treat 30.46+22.17+57.63 as a coherent account of the turn until that field is read.**

**(iii) B-reach-arithmetic's "0.85–0.97 m single pulse."** Already refuted in-set by the verifier (max committed single pulse 0.7835 m, at 3500 ms). The refutation stands and I found nothing to disturb it; note the same finding block also mis-cites `path_m` (planned) as travel.

**(iv) A-refresh-timing "no max_refresh_commands is passed."** Confirmed correct — the turn call at `services.py:7966-7977` passes only `coordinator`, `resend`, `duration_seconds`, `refresh_interval_ms`. Also worth knowing: the computed ceiling is itself echoed as `report["max_refresh_commands"]` (`services.py:4907`), so the nominal-7 vs actual-2/3 gap is visible in the recorded JSON without any code change.

## 3. Problem A: rotation or measurement?

**Cannot be answered from committed data. Not "hard" — impossible.** `docs/evidence-gate5-PASSED-20260808.json` contains no `command_results` array; the 30.46/22.17/57.63 figures appear only as prose in `key_findings` at line 155. No per-attempt raw JSON exists for attempts 4 or 5 (only `docs/evidence-gate5-attempt3-20260808.json`, the failed attempt). `docs/evidence-turn-validation-20260808.json`, which I opened in full, is the other 2026-08-08 turn run and is also a hand-authored summary: it gives `turn_commands: 4`, two realignments with aim errors −25.313° / −32.36°, and no per-pulse rotation at all.

**Every field needed is already recorded — no code change is required.** Per turn command (`services.py:7920-7943` initialiser, populated at `:8057-8092`):
- `motion_refresh.elapsed_ms` — `services.py:4946`
- `motion_refresh.refresh_write_durations_ms` (per-write, list) — `services.py:4888`, appended `:4942-4944`
- `motion_refresh.refresh_commands_sent` / `max_refresh_commands` — `:4945` / `:4907`
- `heading_poll_seconds` — `services.py:7939`, set `:8057-8059`
- `heading_went_fresh` — `services.py:7940`, set `:8060`
- `heading_poll_count` — set `:8061`
- plus `measured_change_degrees`, `heading_error_after`, `progress_degrees`, `displacement_m`, `angular_speed`, `pulse_duration_ms`, `final_approach` (`services.py:7920-7943`).

**Step 0 (free, read-only): recover the attempt-5 response JSON before authorizing anything.** If it still exists, `heading_poll_count` + `heading_poll_seconds` for pulses 1–3 already discriminate partially: a pulse whose poll needed 2+ iterations (~4 s) before crossing the 0.1° epsilon was read *at the instant registration began*, i.e. truncated; a large rotation captured on poll #1 is the catch-up signature. And `heading_error_after` on pulse 4 resolves contradiction (ii). This costs nothing and may make a run unnecessary.

**Smallest read-only-plus-one-authorized-run experiment (daylight, blades off, VIO active, gate armed):**

Three to five *isolated single pulses*, each bracketed by a settled reading. All three services already exist; nothing is deployed.

For each repetition *i*:
1. `vio_turn_probe` with `dry_run: true` — returns `baseline.vision_heading` with zero motion (early return at `services.py:6954-6956`, baseline built at `:6665`). Record `H_before(i)`.
2. `vio_turn_to_heading`, `dry_run: false`, `max_commands: 1`, `pulse_duration_ms: 1500`, `angular_speed: 500`, `motion_refresh_interval_ms: 200`, `slow_threshold_degrees: 1.0` (keeps the base pulse full — `services.py:7898-7908`), `heading_tolerance_degrees: 1.0`, `target_vision_heading = H_before(i) + 60` (60° ≫ one pulse, and at the 37 deg/s default `_turn_final_approach_pulse_ms` computes 1621 ms → capped to 1500, `services.py:7458-7463`, so the pulse is not scaled). Record from `command_results[0]`: `measured_change_degrees`, `heading_poll_seconds`, `heading_poll_count`, `heading_went_fresh`, `displacement_m`, `angular_speed`, `pulse_duration_ms`, and the whole `motion_refresh` dict including `refresh_write_durations_ms`.
3. Wait ≥10 s, then `vio_turn_probe` with `dry_run: true` again. Record `H_after(i)`.

Verdict rules:
- `Δsettled(i) = |H_after(i) − H_before(i)|` is the true rotation. If `Δsettled` spread is ≲1.3× while `measured_change_degrees` spread is ~2.6× → **measurement artifact** (poll truncation + cross-pulse catch-up), and the fix is a heading settle loop mirroring `_settle_linear_position_feed` (`services.py:8265-8334`), not a control change.
- If `Δsettled` itself varies ~2.6× → **rotation genuinely varies**. Then divide: `Δsettled / (refresh_commands_sent + 1)`. If that is near-constant, rotation is set by *writes delivered*, i.e. BLE latency, and `refresh_write_durations_ms` gives the inter-write spacing to compute duty cycle. If it is not constant either, the cause is outside anything this codebase instruments (no terrain, slope, voltage or motor-current field exists anywhere — verified in the A-displacement pass).

Single-pulse isolation is the whole point: it removes cross-pulse attribution, which is the confound no multi-pulse run can resolve. If only one run is authorized, `vio_turn_probe` alone (`dry_run: false`, `angular_speed: 500`, `drive_seconds: 1.5`, `sample_interval_seconds: 1.5`, `post_stop_samples: 6`, `motion_refresh_interval_ms: 200`) repeated 3× is the cheaper substitute — it captures the full post-stop settling curve (`services.py:7038-7052`) — but it discards per-write timings, retaining only the summed `refresh_commands_sent` (`services.py:7024`), so it cannot test the duty-cycle sub-hypothesis.

## 4. Problem B: a reach-extending change touching no profile key

**Yes, one exists: raise the real-segment limit.** `max_real_segments` is not among the 19 `LUBA_ACCEPTANCE_PROFILE` keys (`mammotion-custom-path-card.js:31-61`; the card adds it only in the `points.length > 2` branch at `:960-970`, outside the profile spread). Raising it extends per-plan reach by letting a third planned waypoint actually drive, and the card already accepts up to `MAX_WAYPOINTS = 7` (`:1`).

Sites, all of which I verified:
- `custom_components/mammotion/manual_motion.py:24` — `REAL_CLICK_TO_GO_SEGMENT_LIMIT = 2`. The schema bound (`services.py:1052-1055`) and the runtime re-check (`services.py:11490`) both reference the constant, so they follow automatically.
- `custom_components/mammotion/www/mammotion-custom-path-card.js:2` — `const MAX_REAL_SEGMENTS = 2;`, used by the preflight blocker (`:418-419`) and the real payload (`:965-967`). The card never reads the backend's exposed `real_click_to_go_segment_limit` (`manual_motion.py:198`), so this is a genuinely independent second axis.
- `README.md:107` prose ("limited to two segments") — an edit, not a test.

Test churn: **zero.** The frontend test asserting the cap imports the constant rather than hard-coding it (`tests/frontend/mammotion-custom-path-card.test.mjs:40` imports `MAX_REAL_SEGMENTS`, `:128` compares against it), the README pin test iterates only `PROFILE_KEYS` (`:233`, and `max_real_segments` is not one), and `REAL_CLICK_TO_GO_SEGMENT_LIMIT` appears in no test file. Python tests pass `max_real_segments=1` explicitly. So none of the §4 acceptance obligations (`docs/gate4-repass-20260805.md:117-121`) are triggered; the only deploy obligation is the unconditional one — bump `CARD_VERSION` on every deploy (`mammotion-custom-path-card.js:6-7`) and serve both paths.

**The honest caveat:** this is untested reach. `docs/gate4-repass-20260805.md` flags that the VIO forward-heading offset is refreshed only from linear travel and never across a turn; segment 3+ behaviour is measured nowhere in this repo. And attempt 5's own segment 2 already produced the worst landing of the four (0.1449 m against a 0.15 tolerance — `docs/evidence-gate5-PASSED-20260808.json:143-148`), which is the wrong trend to extrapolate from.

**Also zero-code-change:** chain runs, which is literally what Gate 5 did — attempt 4's segment-2 landing `[4.8095, -2.0722]` is attempt 5's `points[0]` `[4.809, -2.072]` (`docs/evidence-gate5-PASSED-20260808.json`, attempt_4 landed / attempt_5 points).

**Explicitly not available:** `max_linear_pulse_ceiling` (a profile key, currently `null` — `mammotion-custom-path-card.js:39`; setting it un-accepts the profile), `max_linear_commands` (profile key, already at schema max 3 — `services.py:939-941`, `:1068-1070`), and `linear_pulse_duration_ms` (profile key). `linear_distance_ceiling_factor` is *not* a profile key, but it only bounds travel and only in loop-to-tolerance mode (`services.py:11306-11313`), so it extends nothing.