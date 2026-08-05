# P0 deploy and rollback runbook (host 192.168.1.106)

First deployed 2026-07-29 and updated through the 2026-08-02 Gate 5 characterization. For
any later deploy, work while the mower is stopped and experimental motion is
off. Restarting HA while BLE is unhealthy has previously left the integration
in `setup_error` with no auto-retry, needing a manual entry reload.

## What the host is running now

|                         | Host                                                           | Branch         |
| ----------------------- | -------------------------------------------------------------- | -------------- |
| Integration version     | `0.6.4-beta19` candidate                                       | `0.6.4-beta19` candidate |
| pymammotion pin         | `0.8.12.post1` fork wheel (container verified)                 | same           |
| Card `CARD_VERSION`     | `0.6.4-beta19`; integration and HACS copies checksum-identical | `0.6.4-beta19` candidate |
| `manual_motion.py`      | present                                                        | present        |
| `backend_capability.py` | present                                                        | present        |
| `capabilities.py`       | present                                                        | present        |

The live host ran backend Gates 1-4 and two failed-safe beta16 daylight
short-approach runs. The independent 0.450 m characterization proved that
motion is stepwise by confirmed refresh writes and that normal stop latency can
dominate nominal pulse duration. It also exposed useless realignment after the
last permitted forward command. The deployed but unaccepted beta18 candidate
retains beta17's fixes for those three behaviors and adds only the
device-tracker direction correction without changing the accepted profile.
Experimental motion is verified off. Gate 5 and release remain blocked; do not
merge or publish. Repeat affected backend Gates 2 and 4 before Gate 5.

### beta17 correction deploy — 2026-08-02 20:44-20:52 EDT

`0.6.4-beta17` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260802-2045.tgz`. All 46 integration files matched
the local tree by content hash, no AppleDouble files were present, and both
card paths matched (`170aca89b0cc5a514dc9d835aee7a3b8`). HA's API returned in
66 seconds and all 128 Mammotion entities returned in 151 seconds. The loaded
wheel is `pymammotion 0.8.12.post1`; backend capabilities and BLE are verified.
The mower remained ready, RTK Fix, blades off, session-free, and experimental
motion off.

Chrome exposed a cache-key collision: the updated card first loaded as beta17
through the prior `beta23` key, then the plain beta17 URL replayed an older
cached beta16 response. The live resource is therefore
`?v=0.6.4-beta17&build=a2b0d4bf`. After reload, both console and footer reported
beta17. The exact accepted-profile label rendered, Preview was valid, and the
card Dry-run returned `valid: true`, `would_send: false`, and
`stop_reason: dry_run`; Real Go stayed disabled. The card was reset afterward.
VIO was dark/0 features, so no reacceptance motion was attempted.

### beta18 map-marker correction deploy — 2026-08-02 21:29-21:39 EDT

Live UI inspection separated two arrows: the custom-path card's green
arrow correctly rendered upper-right at 72.8 degrees, while HA's standard map
card rendered the mower's black tracker picture upper-left. The tracker was
publishing raw Mammotion orientation `-29`; HA expects clockwise compass
degrees. Beta18 publishes `(-orientation) % 360` (29 degrees for that sample).
This is presentation-only: no executor or accepted-profile value changed. The
candidate passed 469 coverage-enabled Python tests, 19 frontend tests, Ruff,
format, scoped mypy, all-files pre-commit, and the GitHub validation workflow.

The integration was backed up to
`/config/mammotion-backup-20260802-2129.tgz` and deployed motion-disabled. All
46 files matched the tree aggregate hash
`f14c608a02203602f2463fd0e6a30f6b`; no AppleDouble files were present; both
card copies matched `694fd1b0b54ab336c9490c620bd4f8cb`. HA's API returned in
31 seconds and all 128 Mammotion entities returned in 115 seconds. The live
wheel remains `pymammotion 0.8.12.post1`, backend capabilities are verified,
the accepted-profile label is exact, console/footer report beta18, and the
Lovelace resource is `?v=0.6.4-beta18&build=6da6c3d3`. Experimental motion
remained off, no session existed, and no motion was commanded.

The installed `custom:map-card` 1.15.0 accepts marker CSS but does not itself
consume the tracker's `direction`. After backing up the dashboard to
`/config/.storage/lovelace.dashboard_yard.bak.codex-20260802-213848`, a
Jinja-backed `card-mod` rule was added to rotate its `.entity-picture` by the
live `direction` attribute. Readback matched the requested config. Browser
inspection found `direction: 29.0` and computed transform
`matrix(0.87462, 0.48481, -0.48481, 0.87462, 0, 0)`, confirming a 29-degree
clockwise upper-right marker that will update with the entity.

That last conclusion was **rejected by direct operator observation**. The mower
physically faced upper-left while the custom card's projected arrow and the
rotated third-party marker pointed upper-right. A zero-command dry-run snapshot
showed why: `toward: -29.589` and `location.orientation: -29` were frozen last
travel, while VIO was inactive with heading 0 and RTK yaw was 0. No available
field represented stationary body orientation. The `card-mod` rotation was
removed with verified Lovelace readback.

### beta20 turn-feasibility guard deploy — 2026-08-04 20:16-20:35 EDT

Ships the corrected turn-feasibility guard. `LUBA_ACCEPTANCE_PROFILE` is
byte-identical; the only card change is `CARD_VERSION`. No service schema and
no entity platform file changed (`git diff 9ef3d103..HEAD` touches exactly
`manifest.json`, `services.py`, and the card). No motion ran and no gate is
claimed.

Backup `/config/mammotion-backup-20260804-2016.tgz`. All **46** deployed files
were byte-identical to the local tree by per-file md5; both card copies matched
`2b1d37bb99069020d2c3eea54b512e9b`; zero AppleDouble entries in the tarball.
HA API returned after 31 s, Mammotion entities after 235 s, backend verified
`pymammotion 0.8.12.post1`. Lovelace resource bumped to
`?v=0.6.4-beta20&build=2b1d37bb`, re-read to verify, keeping the collision-proof
pattern with the build hash tied to the deployed card's md5.

Two dark-safe dry runs proved the guard is live on the host:

- standalone turn, 179.571° error against a 4-command budget → `feasible:
  false`, `turn_budget`, 7 commands needed, with the new
  `translation_bound_m_per_degree: 0.0026` and
  `translation_bound_source: conservative_observed_translation_per_degree`;
  `estimated_translation_m: 0.467` = 179.571 × 0.0026, i.e. angle-scaled rather
  than command-scaled. `would_send: false`, `commands_sent: 0`.
- multi-segment L path → `junction_turn_feasibility` for the −90° junction is
  **feasible**: 3 commands against 4, `estimated_translation_m: 0.234` against
  `max_displacement_m: 0.25`.

⚠️ That junction cap is **0.25 m, not the schema's 0.5 m default**, and it is
what bounds the per-degree constant from above. 90 × 0.0026 = 0.234 fits with
0.016 m to spare; the initially proposed 0.0028 would have given 0.252 and
**refused Gate 4's own junction geometry** on this build. Do not raise the
constant without re-checking this dry run.

Experimental motion was verified off with no session before and after.

### beta19 stale-orientation correction deploy — 2026-08-02 22:07-22:12 EDT

Beta19 draws only the mower position dot unless the backend explicitly supplies
a trustworthy map-aligned current orientation. It labels the old calculation
as a last-travel projection rather than mower orientation. Nudge now fails
closed on `current_orientation_unavailable`; Real Go behavior, service schemas
and the accepted profile are unchanged. Frontend tests pin the separation
between projected travel and trusted current orientation.

The host was backed up to `/config/mammotion-backup-20260802-2207.tgz`. All 46
deployed files matched the tree aggregate hash
`2c344e3234c175fd85be066259ffcd75`; both card copies matched
`9152496e514058948ad338103130519f`; no AppleDouble files were present. HA's API
returned in 76 seconds and all 128 Mammotion entities returned in 201 seconds.
The installed backend remains `pymammotion 0.8.12.post1`; experimental motion
was off with no session before and after deployment.

Chrome initially replayed beta16 despite the beta19 resource key, so the final
auditable URL is `?v=0.6.4-beta19&build=617337d3`. The reloaded card reported
beta19 and the exact accepted-profile label. DOM inspection found one green
mower-position dot, zero green heading lines, zero arrowheads, and the explicit
`current orientation unavailable ... not mower orientation` text. Nudge was
disabled. No physical motion was commanded.

### Gate 5 deploy — 2026-07-31 19:50-20:05 EDT

`0.6.4-beta12` is deployed. Done while the mower was docked (`CHARGE_ON`,
`MODE_READY`), with no active session and `enable_experimental_motion` off.
**No motion of any kind was commanded.** Backup taken first:
`/config/mammotion-backup-20260731-1950.tgz`.

Verified after the restart:

- all 46 integration files md5-identical to the tree;
- no AppleDouble `._*` files (tarball built with `COPYFILE_DISABLE=1` plus an
  explicit exclude);
- both card paths byte-identical to the tree
  (`a186a394ec17593c5dca8e86484a9983`) and both serving `CARD_VERSION`
  `0.6.4-beta12` over HTTP, containing `LUBA_ACCEPTANCE_PROFILE`;
- HA API back in 30 s, 128 Mammotion entities in 112 s — no `setup_error`;
- container `pymammotion` `0.8.12.post1`; `backend_verified: true` with both
  capabilities true;
- `enable_experimental_motion` still **false**, no session, `real_motion_allowed`
  false.

⚠️ **The registered Lovelace resource was the trap, not the files.** The
dashboard had `/hacsfiles/mammotion/mammotion-custom-path-card.js?v=12` — an
arbitrary cache key that does **not** track the version, so replacing the file
alone would have left every browser on the cached beta11 card while every
server-side check said beta12. It is now
`?v=0.6.4-beta12` (updated through the `lovelace/resources/update` websocket
call, not by editing `.storage`). Note the live dashboard registers the
**`/hacsfiles/` path**, not the documented `/mammotion/` one, which is exactly
why both copies must be written every time.

### beta13 deploy — 2026-08-01 23:02-23:12 EDT

`0.6.4-beta13` is deployed (heading arrow on the card, plus the VIO
fail-closed evidence). Mower was stopped in `Backyard Right`, off the dock,
experimental motion **off**, no session. **No motion commanded.** Backup:
`/config/mammotion-backup-20260801-2302.tgz`.

Verified: 46/46 files md5-identical to the tree; no AppleDouble entries; both
card paths and the tree byte-identical (`6bec2ca3a83186be9c4f7b410a0a2a3c`) and
both serving `CARD_VERSION 0.6.4-beta13` containing `_headingDegrees`; API back
in 35 s with 128 entities and no `setup_error`; container `pymammotion`
`0.8.12.post1`; `backend_verified: true` with both capabilities; experimental
motion still off.

The Lovelace resource was already at `?v=0.6.4-beta13` (updated by the operator),
so no change was needed — but **check it every deploy**, and use
`scripts/ha_set_card_resource.py` rather than editing `.storage`. Bumping the
card file without bumping that key leaves every browser on the previous card.

### beta16 deploy — 2026-08-02 15:33-15:36 EDT

`0.6.4-beta16` is deployed with the precise waypoint-coordinate editor. The
motion gate was verified off before backup and remained off throughout; no
motion was commanded. Backup:
`/config/mammotion-backup-20260802-1532.tgz`.

Verified: 46/46 integration files checksum-identical to the tree; no AppleDouble
entries; both card paths byte-identical (`ea6a84303293addba1d18a65738cdefa`);
API back in 51 s and 128 Mammotion entities in 165 s; container `pymammotion`
`0.8.12.post1`; backend capabilities verified; BLE live; RTK Fix; VIO Light with
80 features; blade RPM zero; no session. The Lovelace resource is verified at
`?v=0.6.4-beta16`.

### The override guard earned itself on first live use

With beta12 loaded, the operator's card reported
`customised (not hardware-accepted): linear_pulse_duration_ms`. The
`dashboard-yard` card config still carried **`linear_pulse_duration_ms: 2000`**
from an older calibration.

That is the known no-op. Forward motion needs **>= 3 s to trigger at all** —
2 s pulses taped as physical no-ops on 2026-07-18, which is why the accepted
profile uses 3500 ms. The backend schema is `Range(min=50.0, max=4000.0)`, so
`2000` is **accepted silently**: it would have been dispatched, not rejected.
Run as Gate 5, the mower would most likely have stood still while the session
reported dispatched pulses and confirmed stops — the exact failure the
execution-profile row exists to surface.

Removed through the `lovelace/config/save` websocket call (backup taken first;
the live config was then diffed against the backup to prove the dropped key was
the only change). Re-checked by running the real card module against the live
saved config: profile row `LUBA acceptance profile (Gates 1-4, 2026-07-31)`,
zero overrides, `linear_pulse_duration_ms` 3500, ceiling still omitted.

Lesson for any future deploy: a correct file deploy is **not** a correct
run configuration. Check the execution-profile row, not just the version banner.

Before any newly authorized Gate 5 motion: deploy and reaccept the beta19
correction recorded in `docs/p0-beta-release.md`; do not treat 0.3-0.5 m as a
proven usable band. Confirm the browser console banner reads
`v0.6.4-beta19` (a hard refresh may be needed), confirm the card's execution
profile row reads exactly
`LUBA acceptance profile (Gates 1-4, 2026-07-31)`, save the emitted payload and
dry-run result, and obtain a fresh daylight operator `go`. Use the same card
instance without editing waypoints between the final dry-run and Real Go. Run
`motion_capture.py` and `ble_session_report.py` across the whole window, save
the Real Go result, and always disarm and verify no session after success,
failure, or abort.

### beta19 backend Gate 2 acceptance — 2026-08-03

A daylight backend 0.100 m segment was accepted. The operator observed an
approximately 9 cm move, then a stop. The result reported `target_reached` and
a 0.0105 m final error; VIO calibration supplied the whole observed movement,
so no normal linear pulse was required. Teardown cleared the session, disabled
experimental motion, and telemetry remained stationary for over one minute.
The recorded evidence is `docs/evidence-gate2-beta19-*20260803*`. This is not
Gate 5: backend Gate 4 and then the unchanged-card UI acceptance remain
required before release.

### beta19 backend Gate 4 retry — failed, 2026-08-03

The guarded two-segment retry is not an acceptance. Its durable result records
segment 1 `turn_phase_incomplete`: VIO calibration passed, but the four-command
turn budget ended `max_commands_reached` 34.795° short of target after 0.185 m
of incidental turn translation. No linear command or segment 2 ran. The gate
was disarmed, session cleared, blades stayed off, and post-stop telemetry was
stationary. See `docs/evidence-gate4-beta19-retry-real-*20260803*` and the
offline analysis in `docs/evidence-gate4-beta19-retry-diagnosis-20260803.json`.
Do not retry or expand the turn budget without the implementation and daylight
re-characterization required by `docs/CLAUDE-FINAL-IMPLEMENTATION-PROMPT.md`.

### Turn-feasibility guard implemented on-branch — 2026-08-03, NOT deployed

The correction demanded by the Gate 4 retry is committed to
`feat/vio-turn-to-heading` and verified locally only; the host still runs the
`617337d3` beta19 build without it. A real VIO turn is now refused **before
its first command** (`turn_budget_infeasible`, `commands_sent: 0`) when the
evidence-bounded per-command progress (16.5°/s × pulse with refresh; the
8°/command single-shot quantum without) cannot reach tolerance within
`vio_turn_max_commands`, or when the refresh-regime translation estimate
(0.0026 m per degree swept, revised 2026-08-04) would breach the displacement
cap. The multi-segment
executor additionally refuses a real path whose fixed junction geometry is
infeasible (`path_turn_infeasible`) before any motion; dry runs report the
same math without refusing. `scripts/diagnose_motion_result.py` reports the
refusal as `vio_turn_refused_infeasible_preflight`. Tests:
`tests/components/mammotion/test_vio_turn_feasibility.py`; the full suite is
483 passing. The accepted profile, service schemas, and all four version
locations are unchanged, so this section adds **no new deploy step yet**:
shipping the guard requires a normal version-bumped deploy (both card paths +
backend + Lovelace key) after a daylight turn characterization validates the
constants, and any deployed build must re-run affected gates from scratch.

## Breaking enum migrations already applied

The host now reports lowercase states such as `mode_working`, `man`, `english`,
and `mowing`. Automations copied from an older release must be migrated from
uppercase matching. The original label remains available in
`raw_protocol_value`; see `p0-beta-release.md`.

## Steps

1. **Back up** the current integration and card:

   ```sh
   set -a && source .env && set +a
   scripts/ha_ssh.exp 'cd /config/custom_components && tar -czf /config/mammotion-backup-$(date +%Y%m%d-%H%M).tgz mammotion && ls -la /config/mammotion-backup-*.tgz'
   ```

2. **Ship the complete integration tarball.** Build it with
   `COPYFILE_DISABLE=1 tar …` — macOS BSD
   tar otherwise embeds AppleDouble metadata files, which extract as 46 junk
   `._*` entries **inside the integration**, including `translations/._en.json`
   next to the real translation files. They were removed by hand on 2026-07-29
   (`rm -f ._* translations/._* www/._*`); prevent them next time instead.

   ```sh
   scripts/ha_scp.exp <scratchpad>/mammotion_deploy.tgz /config/mammotion_deploy.tgz
   scripts/ha_ssh.exp 'cd /config/custom_components && tar -xzf /config/mammotion_deploy.tgz && echo extracted'
   ```

3. **Use the integration-served card resource.** Register
   `/mammotion/mammotion-custom-path-card.js?v=<installed-version>` as a
   JavaScript module. The old HACS copy currently matches, but it is no longer
   the documented source. If an existing dashboard still references
   `/hacsfiles/mammotion/`, update that copy during migration or change the
   resource URL:

   ```sh
   scripts/ha_ssh.exp 'cp /config/custom_components/mammotion/www/mammotion-custom-path-card.js /config/www/community/mammotion/mammotion-custom-path-card.js && md5sum /config/custom_components/mammotion/www/mammotion-custom-path-card.js /config/www/community/mammotion/mammotion-custom-path-card.js'
   ```

4. **Verify checksums match the tree** before restarting — compare against
   `scratchpad/local_md5.txt`:

   ```sh
   scripts/ha_ssh.exp 'cd /config/custom_components/mammotion && find . -type f \( -name "*.py" -o -name "*.json" -o -name "*.yaml" -o -name "*.js" \) ! -path "./__pycache__/*" -exec md5sum {} \; | sort -k2'
   ```

5. **Restart HA.** `scripts/ha_restart.sh`. Expect the API back in ~40-60 s and
   Mammotion entities in ~2-3 min.

   On restart HA will **pip-install the fork wheel from GitHub**, because the
   requirement is a URL and `is_installed()` always returns False for URLs. This
   needs working egress from the container. If it fails, the integration will not
   set up — go to Rollback.

6. **Bump the Lovelace resource cache key** to the installed release version.
   `CARD_VERSION` may change, but browsers key on the query string, so without
   this they can keep the cached card:

   ```sh
   scripts/ha_set_card_resource.py                 # show current
   scripts/ha_set_card_resource.py 0.6.4-betaN     # dry run
   scripts/ha_set_card_resource.py 0.6.4-betaN --apply
   ```

   It uses the `lovelace/resources` websocket API, keeps the registered path
   as-is (the live dashboard references the `/hacsfiles/` copy, not the
   integration-served `/mammotion/` one), and re-reads to verify. Do not edit
   `/config/.storage/lovelace_resources`; HA holds it in memory and overwrites.

7. **Verify** (all dark-safe, no motion):

   ```sh
   scripts/ha_ssh.exp 'docker exec homeassistant python -c "import importlib.metadata as m; print(m.version(\"pymammotion\"))"'
   # expect 0.8.12.post1
   ```

   Then `mammotion.export_runtime_state` should now contain an
   `experimental_motion` block with
   `backend_capabilities.capabilities` both true and `backend_verified: true`.
   Confirm entity count, maps and tasks, diagnostics, and card preview plus
   dry-run. Real Go is not part of a routine deployment check.

8. **Leave `enable_experimental_motion` off.** The gate now opens the moment it is
   toggled on. Toggle it with
   `scripts/ha_set_experimental_motion.py on|off|status` rather than by hand:

   ```sh
   set -a && source .env && set +a
   scripts/ha_set_experimental_motion.py status
   scripts/ha_set_experimental_motion.py on     # prompts for ARM unless --yes
   scripts/ha_set_experimental_motion.py off
   ```

   Two traps it exists to avoid. The options flow returns `create_entry` with an
   **empty `data`** payload even when it applied the change, so the reply proves
   nothing — the script re-reads `export_runtime_state` instead. And the flow
   replaces the whole options dict, so every other option must be resubmitted;
   those values come from the flow's **own schema defaults**, because
   `/api/config/config_entries/entry` never exposes `options` (it returns `{}`
   whatever is configured) and preserving from it silently resets them. The
   field is also `prefer_ble_over_wifi`, not the `prefer_ble` used elsewhere;
   the wrong name fails the flow with a bare HTTP 400 and no field detail.

9. **Enable the right logger before any BLE measurement.**
   `scripts/ble_session_report.py` matches `connected=… mtu=… error=…`, which is
   emitted by **`bleak_esphome.backend.client`** — not by pymammotion. Setting
   pymammotion to debug does nothing for it. Enable at runtime (no restart, and it
   does **not** survive one):
   ```
   service: logger.set_level
   data: {bleak_esphome: debug, habluetooth: debug}
   ```
   The lines only appear on connect/disconnect transitions, so a stable link
   produces none. A zero from the report means "no transitions or wrong logger",
   never "a healthy link" — verify the logger is on before trusting a zero.

## 🔑 Proxy coverage at the dock is already excellent (measured 2026-07-29)

`habluetooth.wrappers` logged the connection-path selection for the docked mower:

| proxy                   | RSSI    | slots free |
| ----------------------- | ------- | ---------- |
| `p1s-printer-a5774c`    | **−49** | **2/3**    |
| `esphomes3-irk`         | −65     | 3/3        |
| `bluetooth-proxy`       | −77     | 3/3        |
| `garage-m5stack-9bc1d4` | −85     | 3/3        |
| `atom-fireplace`        | −85     | 3/3        |

Five paths, best at −49, and the connection opened through P1S Printer. Two
standing beliefs need correcting: the p1s-printer proxy is **not** permanently
one-slot-busy (2/3 free here), and **the dock does not need a closer proxy**. The
coverage problem is out in the working area, not at the dock — so site the next
proxy where the mower mows, and start acceptance runs near the dock.

That table is historical discovery evidence, not the accepted topology. The
successful Gates 1-4 used P1S as the sole enabled mower proxy. The IRK proxy was
isolated after it reproduced the app/HA connection conflict; do not add it back
to a release acceptance run without a controlled stationary comparison.

## 🔬 How to attribute the improvement despite the confound

First 20-minute window after the deploy (docked, idle, much of it with the link
still down, so treat as indicative only):

| metric                     | 07-27 baseline (8 h, 0.8.8) | first 20 min (0.8.12.post1) |
| -------------------------- | --------------------------- | --------------------------- |
| sequence gaps              | 720 (**1.5/min**)           | 1 (~0.05/min)               |
| unparseable LubaMsg frames | 193                         | **0**                       |
| negotiated MTU             | 22x 517                     | 1x 517                      |

**The two metrics dissociate, and that is the attribution test.** Sequence gaps are
link-layer packet loss, which neither audited fix touches. Unparseable frames are
exactly what the BluFi reassembly reset prevents — a gap no longer poisons the
next message. So:

- unparseable frames near zero **while gaps still occur** = clean evidence the
  reassembly fix works, despite the bundled 0.8.8 -> 0.8.12.post1 jump.
- both dropping together = something environmental changed too, and no
  single-fix claim is safe.

Check both over the full overnight window before concluding anything.

## ⚠️ The overnight A/B is confounded — say so in the write-up

The 2026-07-27 baseline (median session 59 s, 42% `0x08`) was measured on
**0.8.8**, not 0.8.12. This deploy jumps 0.8.8 → 0.8.12.post1, which lands the
0.8.9-0.8.12 upstream changes, the #177 rate-limit fix, the BluFi reassembly
reset, **and** the teardown fix together. So an improvement cannot be attributed
to the teardown fix alone. The measurement still answers "is the link better
now?", which is the decision that matters for the acceptance run — but do not
record it as proof of the leak fix specifically.

## Rollback

```sh
scripts/ha_ssh.exp 'cd /config/custom_components && rm -rf mammotion && tar -xzf /config/mammotion-backup-<stamp>.tgz && echo restored'
scripts/ha_restart.sh
```

Then restore the prior Lovelace resource version. Reverting to any build without
the audited backend capabilities makes the probes re-lock real motion by
themselves, but disable experimental motion before rollback regardless.
