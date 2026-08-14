# P0 deploy and rollback runbook (host 192.168.1.106)

First deployed 2026-07-29 and updated through the 2026-08-14 corrected
working-tree deploy after the beta54 Night Go release. For
any later deploy, work while the mower is stopped and experimental motion is
off. Restarting HA while BLE is unhealthy has previously left the integration
in `setup_error` with no auto-retry, needing a manual entry reload.

## What the host is running now

### beta54 plus unreleased Night/Real Go corrections — 2026-08-14

`0.6.4-beta54` was published from merge commit `2573c29b` and version commit
`0bd35160`, then installed with experimental motion disabled. The release adds
separate Night dry-run and Night Go card controls; it does not change any value
in `LUBA_ACCEPTANCE_PROFILE`, and Real Go remains the accepted VIO path.

One later, separately authorized card-driven night run stopped safely at
0.117085 m on `no_target_progress`. The motion gate was disarmed afterward. See
`night-go-card-beta54-20260814.md`. The run exposed a night-only crossed-target
continuation defect and excessive diagnostic waits. The current working tree
contains a verified but uncommitted fix, and the Real Go path removes additive
feedback waits while preserving its payload and safety gates. The first
supervised Real Go check stopped before its second linear dispatch on
`command_queue_backlogged`. A bounded post-feedback queue-settle correction was
then verified and deployed motion-disabled. These working-tree behaviors are
**not** part of the beta54 release artifact. Full evidence and measured/inferred
separation: `real-go-throughput-hardware-20260814.md`.

Backup `/config/mammotion-backup-20260814-pre-queue-settle.tgz`; archive SHA-256
`ec46d8fb0fce6aefcf7b2032c88a17b0693cf14cdd5bce086fb9396da5674b5d`.
Host MD5s match local: services `d6ab89ff4e4286b33d4fa5755bba5b0d`, card
`4846aa9b6f9e0eefe67cb95f8326c3ba` at both serving paths, manifest
`ef8d5273c8e9730bc964579efb63116a`. API returned after 30 s and 132 Mammotion
entities after 116 s; container pymammotion is `0.8.12.post1`. Final gate
readback: disabled, `real_motion_allowed: false`, no active session,
`MODE_READY`.

One separately authorized corrected Real Go run then reached the 0.70 m target
in 19.2 s with 0.093100 m landing error. It used zero turns and three forward
pulses. The new per-pulse queue-settle records were all live at depth zero in
about 100–101 ms; all movement commands and stops succeeded. Final telemetry
was `(4.2954, -3.8079)`, RTK Fix, `MODE_READY`, BLE −64 dBm, blades off. The
gate was independently verified disabled with no active session. Evidence:
`real-go-throughput-hardware-20260814.md` and its linked raw JSON.

### beta52 — item-17 runtime diagnostics, 2026-08-13

`0.6.4-beta52` was deployed. It added runtime-export-only RapidState diagnostics
and the fixed backward-only item-17 harness; it did not add the diagnostics to
shared VIO telemetry. Backup:
`/config/mammotion-backup-20260813-beta52-predeploy.tgz`. The deployed archive
matched local md5 `efa589bc7eaa6bf72e065529f7d44369`. Card md5
`9512f504f4b861488e98f4d29ced6e4f` matched at both serving paths; services md5
was `7c94607698ce6d9f55a4fd4a1a30f85f`. Resource read back as
`?v=0.6.4-beta52&build=9512f504`; pymammotion is `0.8.12.post1`.

The motion-disabled preview sent no command. Item 17 then ran once under
separate explicit supervision; see `night-reverse-heading-20260813.md`.
Independent final readback: gate off, no active session, `MODE_READY`, BLE live,
RTK Fix, blades zero.

### beta51 — explicit fixed-budget null, 2026-08-13

`0.6.4-beta51` is deployed. It fixes the service schemas so the harness's
explicit JSON `max_linear_pulse_ceiling: null` reaches the fixed-budget night
executor instead of failing HTTP 400. Backup:
`/config/mammotion-backup-20260813-beta51-predeploy.tgz`. Local and host
checksums agreed at deployment: card `6645732a8e39eae7644bfe84b5be01de`
at both serving paths, services `1857e0ffe118bac5c556aa04c26f9c45`, manifest
`c85553cc7c6cc536658fe1ea1478e24c`. Resource read back as
`?v=0.6.4-beta51&build=6645732a`; pymammotion is `0.8.12.post1`.

The motion-disabled dry-run returned `valid: true`, `errors: []`. §7 item 15
then ran under separate explicit supervision; see
`night-segment-turn-quantum-20260813.md`. Final independent readback: gate off,
no active session, `MODE_READY`, BLE live, RTK Fix, blades zero.

### beta50 — night v1, 2026-08-13 ~17:02 local

`0.6.4-beta50` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260813-beta50.tgz`. Local and host checksums agree:
card `8510824e965f2dbf6b2403c822c54b39` at both serving paths, services
`d36789b2e622d066873cb396d82e5d76`, manifest
`78ce64d5138ca68a7283ba0d3d93248d`. Resource read back as
`?v=0.6.4-beta50&build=8510824e`; container pymammotion is `0.8.12.post1`.

Gate readback after restart: `enabled: false`, `real_motion_allowed: false`, no
active session. No movement service was called and no motion was commanded. The
mower independently reported `MODE_WORKING` with active mowing/route blockers
at final readback; the deploy session did not interfere. Night v1 hardware task
15 remains unrun and needs separate explicit supervised authorization.

|                         | Host                                                           | Branch         |
| ----------------------- | -------------------------------------------------------------- | -------------- |
| Integration version     | `0.6.4-beta32` staging candidate (deployed 2026-08-09, motion-disabled) | `0.6.4-beta32` candidate |
| pymammotion pin         | `0.8.12.post1` fork wheel (container verified)                 | same           |
| Card `CARD_VERSION`     | `0.6.4-beta32`; integration and HACS copies checksum-identical | `0.6.4-beta32` candidate |
| `manual_motion.py`      | present                                                        | present        |
| `backend_capability.py` | present                                                        | present        |
| `capabilities.py`       | present                                                        | present        |

The live host ran backend Gates 1-4, two failed-safe beta16 daylight
short-approach runs, the beta20 Gate 4 re-pass, and the beta21 second-geometry
reproduction. Beta22 is an **unaccepted, motion-disabled staging candidate**. It
carries the four-field Gate 4 re-pass profile, the off-mower executor/estimator
corrections recorded in `docs/gate4-repass-20260805.md`, and the reverse-recovery
containment guard; it does not establish mower acceptance.
Experimental motion is verified off. Gate 5 and release remain blocked; do not
merge or publish.

⚠️ **Beta22 is expected to make Gate 4 fail.** The guard refuses a correction
present in *both* recorded Gate 4 passes, so a Gate 4 run on this build will
most likely stop with `target_requires_reverse_recovery` where beta20/beta21
reported `target_reached`. That is containment, not regression: the earlier
passes were bought by driving past the waypoint and U-turning back. The next
motion decision is whether to fix control quality (stop-latency lead, pulse
sizing) or to accept overshoot-and-recovery — not a Gate 5 attempt.

### beta49 — beta48's live defects, 2026-08-13 ~05:20 UTC

Frontend-only again. `card md5 adaf0b71`, local == both host paths, resource
`?v=0.6.4-beta49&build=adaf0b71`, API back in 51 s, 132 entities,
`real_motion_allowed: false`.

🔑 **Four defects were found by looking at the deployed card, none by the test
suite.** The stubs supply blocker lists the real backend does not.

The duplicate-blocker one is confirmed directly in the live runtime state:

```
em blockers     ['experimental_motion_disabled', 'position_not_valid_for_motion']
safety blockers ['position_not_valid_for_motion']
```

`_preflight()` concatenated those two lists, so the banner printed
`position_not_valid_for_motion` twice (and `rtk_not_precise` twice, when RTK was
also down). Both call sites now dedupe. Two emitted codes
(`position_not_valid_for_motion`, `ble_client_not_connected`) had no help text
and silently dropped out of the explanation list; a test now pins the full set
observed on the host.

**Method note.** Render the card against the actual `export_runtime_state`
output, not the test fixtures, before calling a UI change done.

### beta48 — card usability and run export, 2026-08-13 ~04:45 UTC

**Frontend-only deploy.** Card + `manifest.json` only; no `services.py` change
and **no `LUBA_ACCEPTANCE_PROFILE` key touched**, so the profile stays accepted
and no §4 re-pinning is owed. Verified:

```
card md5      fbf4f621bd78eb5425f8beee4a2aa231   local == /config/custom_components/... == /config/www/community/...
served        /hacsfiles/... and /mammotion/...  both fbf4f621, CARD_VERSION 0.6.4-beta48
resource      ?v=0.6.4-beta48&build=fbf4f621
manifest      0.6.4-beta48
API back      35 s      132 mammotion entities at 153 s
gate          real_motion_allowed: false   (blockers: experimental_motion_disabled,
                                            ble_client_not_connected, position_not_valid_for_motion)
```

⚠️ `ha_set_card_resource.py` takes only a version string and does **not** append
the `build=` suffix the runbook requires. Pass it as part of the argument —
`ha_set_card_resource.py "0.6.4-beta48&build=fbf4f621" --apply` — or the
registered key loses the checksum half of its cache-busting.

What changed: Real Go / dry-run / history JSON **downloads**, a per-segment
**landing table** (leg, landing, tolerance, verdict, pulses, mean and worst), a
colour-coded **readiness banner** that keeps blocker codes verbatim and adds
plain English for every one of them, the toolbar grouped Path / Run / Export, and
the 14-row diagnostics panel collapsed behind its verdict line.

### beta43/44 — Gate 5 re-pass and the profile-echo fix, 2026-08-12

**`0.6.4-beta43` deployed motion-disabled 2026-08-12 ~20:35 UTC** and is the
build that **passed Gate 5 on the reach-enabled profile** — see
`docs/gate5-repass-PASSED-20260812.md`. 46/46 byte-identical, both card paths
`f92f2e71`, resource `?v=0.6.4-beta43&build=f92f2e71`, API back in 51 s, 132
entities in 134 s, `pymammotion 0.8.12.post1`, `real_motion_allowed: false`
verified before arming.

beta43's one motion change: the post-turn correction gets the same command
budget as any other turn (`vio_turn_max_commands`, was `min(2, ...)`). ⚠️ **It
was not exercised by the gate** — the only correction was −10.477°, inside the
old 21.50° envelope.

**`0.6.4-beta44` supersedes it and changes no motion behaviour.** The card's
execution-profile label drops "Gate 5 re-pass PENDING", and both executors now
echo every accepted-profile key. Two came back `null` during the gate
(`max_no_progress_pulses`, `motion_refresh_interval_ms`) and the pass had to be
argued around the hole — unacceptable in a gate whose entire purpose is proving
the card sent the accepted profile.

✅ **The profile is ACCEPTED again.** `max_linear_pulse_ceiling: 14` no longer
owes a Gate 5; it has one.

### beta42 reach-profile deploy — 2026-08-12 07:08-07:20 UTC

`0.6.4-beta42` is deployed **motion-disabled**. Backup:
`/config/mammotion-backup-20260812-0708-beta42.tgz`. All **46** files
byte-identical to the tree, zero AppleDouble; both card copies
`09a1d05ebbd79889a01a334dd9e3ef4b`. API back in **51 s**, 128 Mammotion entities
in 165 s. Resource read back as `?v=0.6.4-beta42&build=09a1d05e` (was
`beta41&build=174f317d`). Backend verified `pymammotion 0.8.12.post1`. Host
readback: `CARD_VERSION = "0.6.4-beta42"`, `max_linear_pulse_ceiling: 14`,
`manifest.version 0.6.4-beta42`. Gate off, `real_motion_allowed: false`, mower
docked at (4.3188, 3.2862) `CHARGE_ON`, `MODE_READY`.

🚨 **THIS BUILD CHANGES THE ACCEPTED PROFILE AND IS THEREFORE UNACCEPTED.**
`max_linear_pulse_ceiling` moved `null` → **14**, adopting loop-to-tolerance on
the operator's decision and 2026-08-11 hardware evidence (2/3/4 m legs landing
0.0690 / 0.0928 / 0.1023 m). Per §4 of `docs/gate4-repass-20260805.md` that owes
a **fresh Gate 5, card-driven, which has NOT been run.** The card's own
execution-profile label now says so out loud: "Gate 5 re-pass PENDING". Do not
describe any run on this build as profile-accepted until that gate passes.

Also in this build: the mid-drive re-aim guard projects to the end of the next
pulse rather than to the closest approach (no profile key).

### beta32 preflight-model correction deploy — 2026-08-09 01:16-01:22 EDT

`0.6.4-beta32` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260809-0116-beta32.tgz`. All **46** files
byte-identical to the tree (aggregate `634abef928fa5dcd4c53f5dfafe4cbae`), zero
AppleDouble; both card copies `16d883faa32fc8aa5d399038245474cf`. API back in
**40 s**, 128 entities in 133 s. Resource read back as
`?v=0.6.4-beta32&build=16d883fa` (was `beta30&build=8ec0fb01` — the host skipped
beta31 entirely). Backend verified `pymammotion 0.8.12.post1`. Gate off,
`real_motion_allowed: false`, no session, `MODE_READY`, RTK **Fix**, BLE −48.

**This is the first deploy of the beta31 reach work, and it carries a fix beta31
needed.** beta31 was adversarially reviewed before deployment and did not clear;
beta32 is beta31 plus one refusal-side correction — the turn feasibility preflight
now models the overshoot ceiling instead of assuming full-length pulses. See
`docs/HANDOVER-beta31-20260809.md` §2.6 for the four findings, three of which
remain **open**.

⚠️ **Four open findings ride along on this build.** The ceiling costs ~18° of turn
capability (a 90° junction completes on beta30 and exhausts the 4-command budget
on beta31/32 at the rates Gate 5 actually measured); its guarantee is written in
commanded rather than delivered milliseconds; `16.5 °/s` is no longer a rate
floor; and the ceiling biases turn landings into a tighter post-turn gate. **The
validation run therefore keeps every junction in the 45–70° band.**

Verified live after deploy, zero motion: `real_click_to_go_segment_limit: 4`, and
a dry-run 4-segment path with 60° junctions inside `Backyard Right` returns
`command_count_model: "executor_pulse_policy_replay"` with the ceiling-shortened
ladder `[1300.0, 942.5, 683.3]` — **pulse 1 is already ceiling-bound at 1300 ms
rather than the configured 1500**, which is the ceiling binding below 72° exactly
as designed, and 3 of 4 commands needed leaves one of margin. Evidence:
`docs/evidence-beta32-deploy-dryrun-20260809.json`.

Motion preconditions were **not** met at deploy time and no motion was attempted:
night (`initial_vio_feed.live: false`, `tracked_features: 0`, brightness `Dark`),
mower docked at `CHARGE_ON`, `position_not_valid_for_motion`. The 4-segment
validation run waits for daylight, a charged battery and fresh authorization.

### beta26 RTK design inversion deploy — 2026-08-07 20:46-20:50 EDT

`0.6.4-beta26` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260807-2046-beta26.tgz`. All **46** files
byte-identical (aggregate `18078de314c7c0b277dc121b504adf9c`), zero AppleDouble;
both card copies `62bbaaddfbce341e9bc7172f5adb1bb6`. API back in **31 s**,
entities in 120 s. Resource `?v=0.6.4-beta26&build=62bbaadd`. Backend verified
`pymammotion 0.8.12.post1`. Gate off, no session, `MODE_READY`, blades 0.

**Removes a live false-block and inverts the RTK design.** beta25 refused motion
with `rtk_telemetry_stale` past 1800 s, and a healthy Fix-locked *stationary*
mower was then measured reaching **3573 s** of unchanged RTK payload — so beta25
would have refused legitimate runs after ~30 idle minutes.

⚠️ **No age threshold can work, and an active probe cannot substitute.** The RTK
payload changes ~hourly at rest while the one observed fault lasted ~3 h. Forcing
a report burst on *healthy* RTK produced 49 messages and **zero** RTK channel
updates with the age still climbing — indistinguishable from latched. So the
earlier claim that a forced burst "caught" the latch is withdrawn.

Now: `rtk_report_age_seconds` / `rtk_report_quiet` are **reported for auditing
and never block**. The real guard is the **quality gate** — non-Fix refuses with
`rtk_not_precise` unless `allow_degraded_rtk` is passed — which catches the fault
actually observed (a latched *Float*). **Do not re-add an age blocker;**
`_RTK_REPORT_QUIET_SECONDS` records both failed attempts. Reasoning:
`docs/rtk-hardening-plan-20260807.md`.

Verified live after deploy: `rtk_report_age_seconds: 30.663`,
`rtk_report_quiet_threshold_seconds: 1800.0`, `rtk_report_quiet: false`,
`rtk_telemetry_stale` **absent from the payload entirely**, `blockers: []`,
`rtk_status_label: Fix`, `rtk_degraded: false`.

### beta25 RTK threshold correction deploy — 2026-08-07 19:22-19:26 EDT

`0.6.4-beta25` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260807-1922-beta25.tgz`. All **46** files
byte-identical (aggregate `f1bbea08235697eb75b618282b0df98c`), zero AppleDouble;
both card copies `3431ab4413eff37b3766f80937d17676`. API back in **30 s**,
entities in 114 s. Resource `?v=0.6.4-beta25&build=3431ab44`. Backend verified
`pymammotion 0.8.12.post1`. Gate off, no session, `MODE_READY`, blades 0.

**Why this shipped the same evening as beta24:** beta24's RTK staleness
threshold of 300 s was measured wrong within the hour. A healthy Fix-locked
*stationary* mower went 582 s without an RTK payload change, so beta24 would
have refused legitimate motion with `rtk_telemetry_stale`. Raised to 1800 s —
~3x the longest legitimate quiet observed, 6x under the three-hour latch it
catches. Verified live after deploy: `rtk_report_age_seconds` reporting,
`rtk_telemetry_stale: false`, `rtk_report_stale_threshold_seconds: 1800.0`, no
blockers.

⚠️ The bound is **under-characterised**: the upper limit on legitimate quiet is
unknown, only that it exceeds ten minutes. If a run is refused with
`rtk_telemetry_stale` while RTK is demonstrably healthy, raise the constant
rather than assuming a latch.

### beta24 RTK freshness guard + per-channel probe deploy — 2026-08-07 19:03-19:07 EDT

`0.6.4-beta24` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260807-1903-beta24.tgz`. All **46** integration
files byte-identical to the tree (aggregate
`2bf75c1312257fbcf58d1e75ab4a7cb8`), zero AppleDouble; both card copies
`73b1774077957755ef26eed846f4c186`. HA API returned in **41 s**, entities in
125 s, no `setup_error`. Lovelace resource read back as
`?v=0.6.4-beta24&build=73b17740`. Backend verified `pymammotion 0.8.12.post1`.
Gate verified off before and after; no session; no motion commanded.

Two changes, both off-mower:

- **RTK freshness guard.** The coordinator fingerprints the RTK payload on every
  report refresh; the safety summary reports `rtk_report_age_seconds` and blocks
  with `rtk_telemetry_stale` past 300 s. Wired into the three authorization
  boundaries only, not the fifteen diagnostic call sites. An unmeasurable age is
  `None` and does not block. **No Fix requirement was added** — that threshold
  remains an open operator decision.
- **Per-channel report attribution** in `report_stream_probe`.

⚠️ **First per-channel result changes the plan.** Stationary with RTK Fix, 40 s:
75 messages arrived at ~1.9 Hz and **none** carried a changed position, RTK or
VIO payload. The position-report cadence therefore cannot be measured from a
stationary mower — the measurement needs a moving one, in daylight, under fresh
authorization. See `docs/evidence-per-channel-report-probe-20260807.json`.

### beta23 report-rate probe deploy — 2026-08-07 00:17-00:21 EDT

`0.6.4-beta23` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260807-0017-beta23.tgz`. All **46** integration files
byte-identical to the local tree (aggregate `71ef5ccddb3dd04b33732cd4871b555a`),
zero AppleDouble entries; both card copies `885630808fafb80c7b38c39ea1bad628`.
HA API returned in **51 s**, no `setup_error`. Lovelace resource read back as
`?v=0.6.4-beta23&build=88563080`. Container backend verified
`pymammotion 0.8.12.post1`. The new `report_stream_probe` service is registered
on the host with all four fields (60 mammotion services total). Motion gate
verified off before and after; no session; no motion commanded.

Adds exactly one thing: a read-only `report_stream_probe` diagnostic. No motion
path, service schema default, `LUBA_ACCEPTANCE_PROFILE` value, or entity
platform changed.

⚠️ **Entity-count note — read before treating a mismatch as a regression.**
`scripts/ha_restart.sh` prints an entity count but **exits as soon as it reaches
100**, so the printed number depends on when the poll lands mid-load. It read
**130** here versus **128** on earlier deploys; the *settled* count is **131**
(18 unavailable/unknown for an idle mower). The historical "128 Mammotion
entities" in the sections below is therefore a readiness sample, not a stable
invariant. Compare settled counts, or ignore the number.

### beta22 reverse-recovery containment deploy — 2026-08-06 19:51-19:56 EDT

`0.6.4-beta22` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260806-1951-beta22.tgz`. All **46** integration files
were byte-identical to the local tree by per-file md5 (aggregate
`dbab51a64ff86032fec28b130d2d0605`); the tarball was built with
`COPYFILE_DISABLE=1` plus explicit `._*`/`__pycache__` excludes and the host
carries **zero** AppleDouble entries. Both card copies matched
`49dd1df816162f523285d485e4a8cb6e`. HA's API returned in **41 s** and all **128**
Mammotion entities in **108 s**, with no `setup_error`.

The Lovelace resource was updated and read back as
`/hacsfiles/mammotion/mammotion-custom-path-card.js?v=0.6.4-beta22&build=49dd1df8`.
Container backend verified `pymammotion 0.8.12.post1`;
`export_runtime_state` reported `backend_verified: true` with both capabilities
true, `real_motion_allowed: false`, `enabled: false`, no active session, no
route (`reason: no_route`), `MODE_READY`, blade `OFF` at 0 rpm with
`blade_safe_for_motion: true`, and position `(5.6444, -4.4875)` RTK `Fix` /
`AREA_INSIDE` — the unchanged 2026-08-06 second-geometry resting point.

Both card paths were additionally fetched over HTTP and returned `200` with the
same md5, so the new card is proven served, not merely present on disk. The
deployed `services.py` was grepped on the host and carries
`_MAX_FORWARD_REALIGNMENT_DEGREES = 90.0`; the deployed card carries
`max_linear_commands: 3`, `linear_pulse_duration_ms: 1300`, and
`max_turn_translation_distance: 0.3`.

What changed on the host relative to beta21: `services.py` (the ≥90°
reverse-recovery guard and the fail-closed realignment budget) and the card's
`CARD_VERSION`. `LUBA_ACCEPTANCE_PROFILE` is byte-identical, the
execution-profile label still reads exactly
`LUBA acceptance profile (Gate 4 re-pass, 2026-08-05)`, and no service schema or
entity platform file changed. No motion service was called and no motion was
commanded.

**BLE was initially unverifiable, then verified at 20:29.** At the 19:56 readback
the transport had not re-registered (`ble_transport_not_registered`,
`active_transport: none`, `online: false`) across an 8-minute poll, because the
mower battery was at **2%** — a flat mower never advertises, so the transport
cannot register. The `ble_rssi: -62` in that readback was the mower's own last
self-report, a **cached value and not a liveness signal** (the same trap recorded
on 2026-07-25); `online: false` corroborated. The read-only
`scripts/mammotion_preflight_gates.py` correctly reported `BLE link live` as a
FAIL throughout that window.

The mower docked itself at 20:10 EDT — **no motion was commanded by this
session** — and charged to 26%. The 20:29 re-check is clean: `active_transport:
ble`, `online: true`, `ble_rssi -46`, preflight `BLE link live PASS (entity=on
transport=ble rssi=-48)`, which matches the excellent dock proxy coverage
measured on 2026-07-29. Motion remained disabled with no session throughout.

⚠️ **The mower is now on the dock at `(4.3188, 3.2862)`**, so the standing
blockers are `experimental_motion_disabled` + `position_not_valid_for_motion`,
and `zone_hash: 0` with `pos_type: CHARGE_ON` are the correct docked readings.
Undock into a mapped area before any future authorized run.

Durable record: `docs/evidence-beta22-containment-deploy-20260806.json`.

### beta21 Gate 4 profile staging deploy — 2026-08-05 21:59-22:07 EDT

`0.6.4-beta21` is deployed motion-disabled from the candidate tree. Backup:
`/config/mammotion-backup-20260805-beta21-staging.tgz`. All **46** integration
files matched the local tree aggregate hash
`ee4a94bfb540bbcd05311cc7754047ba`; the tarball contained no AppleDouble or
`__pycache__` entries. Both card copies matched
`59a9a7dd4b7451ffce13cd0494df4646`. HA's API returned in 55 seconds and all
128 Mammotion entities returned in 142 seconds.

The registered Lovelace resource was updated and read back as
`/hacsfiles/mammotion/mammotion-custom-path-card.js?v=0.6.4-beta21&build=59a9a7dd`.
Browser readback showed the beta21 footer and console banner, exact
`LUBA acceptance profile (Gate 4 re-pass, 2026-08-05)` label, and
`PyMammotion backend 0.8.12.post1 (verified)`. Real Go, Nudge, and dry-run were
disabled with no path set. A transient BLE/queue blocker appeared immediately
after restart and cleared during the reconnect window; the final runtime-state
readback contained only `experimental_motion_disabled`, with
`real_motion_allowed: false`, no active session, and `MODE_READY`. No motion
service was called and no motion was commanded.

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
