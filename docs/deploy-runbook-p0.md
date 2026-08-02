# P0 deploy and rollback runbook (host 192.168.1.106)

First deployed 2026-07-29 and updated through the successful 2026-07-31 Gate 4
run. For any later deploy, work while the mower is stopped and experimental
motion is off. Restarting HA while BLE is unhealthy has previously left the
integration in `setup_error` with no auto-retry, needing a manual entry reload.

## What the host is running now

|                         | Host                                                           | Branch         |
| ----------------------- | -------------------------------------------------------------- | -------------- |
| Integration version     | `0.6.4-beta12`                                                 | `0.6.4-beta12` |
| pymammotion pin         | `0.8.12.post1` fork wheel (container verified)                 | same           |
| Card `CARD_VERSION`     | `0.6.4-beta12`; integration and HACS copies checksum-identical | `0.6.4-beta12` |
| `manual_motion.py`      | present                                                        | present        |
| `backend_capability.py` | present                                                        | present        |
| `capabilities.py`       | present                                                        | present        |

The live host already ran the complete supervised acceptance sequence. Its
`coordinator.py` and `__init__.py` match this tree. Its functional `services.py`
is the Gate 4-passing build; the handoff tree differs only by a corrected schema
comment. Experimental motion was disabled after the run.

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

Still required before Gate 5 motion: confirm the browser console banner reads
`v0.6.4-beta12` (a hard refresh may be needed), confirm the card's execution
profile row reads the accepted profile, and obtain a fresh daylight operator
`go`.

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
