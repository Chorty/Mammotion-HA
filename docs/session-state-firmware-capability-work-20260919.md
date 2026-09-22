# Session state — firmware capability work — 2026-09-19

Original snapshot for the next session. Updated 2026-09-20: the local host now
has the merged #17 code and a private PyMammotion validation wheel deployed.
The only motion-related action after that deployment was the single,
operator-authorized speed-only validation call recorded below; no mower start,
stop, schedule, OTA, or BMS-write command was sent by this work.

**Updated 2026-09-22 — both items in the "next-session queue" below are now
closed:**
- **#19's route overlay is browser-confirmed.** During an operator-started
  mow ("Backyard Hill"), `export_map`'s `mow_path` returned 12 lines of
  device-frame `{x, y}` points; the operator's screenshot shows the dashed
  blue serpentine route drawn correctly inside the area boundary, under the
  live position marker, boundary still legible underneath — correct z-order,
  nothing obscured.
- **PyMammotion #2 (`deviceOtherInfo` retention) + Mammotion-HA #18
  (diagnostics)** are live for the first time: the `beta114`/`chorty-0.8.12.post7`
  deploy (2026-09-22, see `docs/deploy-runbook-p0.md`) was the first deployed
  wheel to carry `device_other_info`, so `#18`'s privacy-whitelisted `health`
  block in config-entry diagnostics is now populated with real data
  (`soc_tmp`, coredump counters, `process_restart_count`, subsystem version
  strings, etc.) instead of being absent.
- 🔑 **Both new findings explained, same session, via static firmware
  inspection — no code or deploy change needed.**
  - `process_restart_count`'s +49 burst traces to
    `startup/agl_monitor_process.sh`, the on-device watchdog: it monitors
    ~17 processes (`agilex_navigation`, `mower_vslam_vio`,
    `mower_perception_node`, `ins_fusion`, lidar mapping, `multimedia`,
    `embed_service`, `sensorpub`, etc.) and calls its `restart_count_add()`
    — the exact counter HA surfaces — every time it detects and relaunches a
    missing one. Most of those processes only run while actively mowing, so
    starting a job spins up the whole vision/nav/lidar stack at once and the
    watchdog's polling can catch several mid-startup and "restart" them. A
    third reading confirmed the counter held flat at 14180 once the mow
    settled — burst-then-flat, consistent with normal work-start churn, not
    ongoing instability.
  - `vision_proxy: "fopen error!"` traces to the `system_io` binary's string
    table: it `fopen()`s `/app/pkgs/vision/mower_vision_proxy/version` to
    report that component's version. Neither the path nor the
    `mower_vision_proxy` package exist anywhere in this firmware image
    (`1.30.29.24`), confirmed by a full filesystem search — `system_io` is
    checking for a component this build never shipped. The real vision
    pipeline is independently confirmed healthy: `perception` and
    `vslam_vio` in the same payload report real version strings. A firmware
    packaging gap, not a fault.

## Five PRs opened this session

| PR | Repo | State | What |
| --- | --- | --- | --- |
| [#1](https://github.com/Chorty/PyMammotion/pull/1) | Chorty/PyMammotion | OPEN | Partial direct-MQTT property post zeroed battery/state/blade. Backport of upstream mikey0000#184 (`505a7e8`). **Not for upstream** — they have it; Chorty/main is 15 behind. |
| [#2](https://github.com/Chorty/PyMammotion/pull/2) | Chorty/PyMammotion | **MERGED** 2026-09-19 | Retain `deviceOtherInfo` (was decoded + discarded). All fields optional, +29 real keys, merge-on-presence. A released wheel still needs to include it. |
| [#17](https://github.com/Chorty/Mammotion-HA/pull/17) | Chorty/Mammotion-HA | **MERGED** 2026-09-19 | `start_mow(modify=true)` now uses fail-closed read-merge-write instead of pushing HA's cached plan. Hardware validation on 2026-09-20 sent one speed-only call: the Mammotion app confirmed the requested 1.0 ft/s, while HA retained its pre-change snapshot. The outstanding item is UI readback after a successful service change; see `docs/pr17-test-deployment-staging-20260919.md`. |
| [#18](https://github.com/Chorty/Mammotion-HA/pull/18) | Chorty/Mammotion-HA | **MERGED** `aaf797bc` | `deviceOtherInfo` → privacy-whitelisted diagnostics. Forward-compatible: no-op on the current wheel, lights up once #2's wheel lands. |
| [#19](https://github.com/Chorty/Mammotion-HA/pull/19) | Chorty/Mammotion-HA | **MERGED** `c90a3c33` | Draw the running job's planned route on the click-to-go card, from `export_map.mow_path` (device x/y, same frame as areas). Owed: browser-confirm it renders. |

`main` tip at this snapshot was `c90a3c33` (beta113 + #18 + #19). Later main
includes #17 (`466a439f`); the local HA instance has that code deployed.

## Dependency chain for #18 to actually show data
PyMammotion #2 merge → new wheel published → bump pin in **both**
`custom_components/mammotion/manifest.json` and `requirements_test.txt`
(currently `chorty-0.8.12.post4`, which lacks `device_other_info`). Until then
the diagnostics `health` section is correctly absent.

## Firmware capability map (findings, not code)
- Full protocol scan: **no firmware protobuf message is missing** from
  PyMammotion — that line of work is closed, not open.
- Coverage of 215 command builders: ~78 wired, ~105 buildable-and-applicable to
  a Luba 2, 45 gated off by model. 🚨 The "unused" count **over-reports** —
  saga-driven builders (`get_line_info`, `get_dynamic_route`) are reached via
  client methods, not by name, so they scan as unused but are implemented.
  Verify before building.
- Device resolves `Luba-VSPLV397` → `DeviceType.LUBA_2`. "AWD 5000" is
  marketing; the protocol sees only `Luba-VS`.
- Interactive map artifact: https://claude.ai/artifact/E5NmK43DxBGthWmW2f13ic

## Firmware modification assessment (separate OTA repo)
`docs/findings-firmware-modification-barriers-20260919.md`: app/rootfs layer has
**no** barrier — no image signature (CRC32 only), writable ext4, no dm-verity,
advisory-only sha256sums. Only unknown is **SoC secure boot (Sunrise X3)** on
the boot chain, not in the captured rootfs. Handoff bundle staged (gitignored)
at `ota_work/handoff/` for `/Users/mattjoslin/Documents/Luba 2 OTA`; move
commands in `ota_work/handoff/HANDOFF-2026-09-19.md`. Nothing OTA committed here.

## Next-session queue

1. ✅ **Resolved 2026-09-21**, merged upstream via Chorty#20: #17's
   service-path UI sync now does a post-modification read-merge-write
   readback. See `docs/TODO.md`.
2. ✅ **Resolved 2026-09-22**: beta114 deployed motion-disabled,
   browser-confirmed — see the update at the top of this file.
3. ✅ **Resolved 2026-09-22**: `chorty-0.8.12.post7` (built from a branch
   reconciliation that also restored two fixes missing from `main` since a
   `post4`/`main` divergence — see `docs/deploy-runbook-p0.md`) is pinned and
   deployed; #18's health data is live.
4. **Closed as safe probes:** `set_mtu_value`, `set_net_rtk_link_mode`, and
   `get_device_log_info` are writers, not read-only probes. Static evidence:
   `docs/findings-candidate-capabilities-static-triage-20260919.md`. Revisit
   only with an explicitly approved, reversible device-test plan.
5. ✅ **Resolved 2026-09-22, same session as items 2-3:** `process_restart_count`'s
   burst and `vision_proxy: "fopen error!"` are both explained by static
   firmware inspection — see the 2026-09-22 update at the top of this file.
   No code or deploy change needed.
