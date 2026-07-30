# P0 deploy runbook (host 192.168.1.106)

Staged 2026-07-29. Run **after the mower docks**, not mid-job: restarting HA while
BLE is unhealthy has previously left the integration in `setup_error` with no
auto-retry, needing a manual entry reload.

## What the host is running now

| | Host | Branch |
| --- | --- | --- |
| Integration version | `0.6.4-beta11` | `0.6.4-beta11` |
| pymammotion pin | **`==0.8.8`** (container confirmed 0.8.8) | `0.8.12.post1` fork wheel |
| Card `CARD_VERSION` | `2026.07.18b2` (42,055 B) | `0.6.4-beta11` (47,061 B) |
| `manual_motion.py` | **absent** | present |
| `backend_capability.py` | **absent** | present |
| `capabilities.py` | **absent** | present |

31 files differ: 3 new, 28 changed (including the whole entity layer —
`sensor.py`, `select.py`, `number.py`, `button.py`, `strings.json`, and all 12
translations). This is the first time the P0 branch goes live, not a surgical
patch.

The host card `2026.07.18b2` is an **ancestor** commit (`c56766b0`); the branch
moved past it in `a044d3e8`, so deploying is an upgrade, not a regression.

## ⚠️ Read before restarting: this applies the breaking enum migrations

The host still reports uppercase states — `activity_mode: MODE_WORKING`,
`voice_gender: MAN`, `voice_language: ENGLISH`, `task_area_path: MOWING`. After
this deploy they become lowercase (`mode_working`, `man`, `english`, `mowing`).
**Any automation, template, or dashboard condition matching the uppercase strings
will stop matching.** The original label remains available in
`raw_protocol_value`. See the migration table in `p0-beta-release.md`.

## Steps

1. **Back up** the current integration and card:
   ```sh
   set -a && source .env && set +a
   scripts/ha_ssh.exp 'cd /config/custom_components && tar -czf /config/mammotion-backup-$(date +%Y%m%d-%H%M).tgz mammotion && ls -la /config/mammotion-backup-*.tgz'
   ```

2. **Ship the tarball.** ⚠️ Build it with `COPYFILE_DISABLE=1 tar …` — macOS BSD
   tar otherwise embeds AppleDouble metadata files, which extract as 46 junk
   `._*` entries **inside the integration**, including `translations/._en.json`
   next to the real translation files. They were removed by hand on 2026-07-29
   (`rm -f ._* translations/._* www/._*`); prevent them next time instead.
   ```sh
   scripts/ha_scp.exp <scratchpad>/mammotion_deploy.tgz /config/mammotion_deploy.tgz
   scripts/ha_ssh.exp 'cd /config/custom_components && tar -xzf /config/mammotion_deploy.tgz && echo extracted'
   ```

3. **Copy the card to the HACS path as well.** The dashboard resource is
   `/hacsfiles/mammotion/mammotion-custom-path-card.js?v=11`, which serves from
   `/config/www/community/mammotion/` — *not* from the integration's own `www/`.
   Both copies must be updated or the dashboard silently serves the stale card:
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

6. **Bump the Lovelace resource cache key** from `?v=11` to `?v=12` in
   Settings → Dashboards → Resources. `CARD_VERSION` changed, but browsers key on
   the query string, so without this they keep the cached card.

7. **Verify** (all dark-safe, no motion):
   ```sh
   scripts/ha_ssh.exp 'docker exec homeassistant python -c "import importlib.metadata as m; print(m.version(\"pymammotion\"))"'
   # expect 0.8.12.post1
   ```
   Then `mammotion.export_runtime_state` should now contain an
   `experimental_motion` block with
   `backend_capabilities.capabilities` both true and `backend_verified: true`
   (the host currently has no such block at all). Confirm entity count, maps and
   tasks, diagnostics, and card preview plus dry-run.

8. **Leave `enable_experimental_motion` off.** The gate now opens the moment it is
   toggled on.

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

| proxy | RSSI | slots free |
| --- | --- | --- |
| `p1s-printer-a5774c` | **−49** | **2/3** |
| `esphomes3-irk` | −65 | 3/3 |
| `bluetooth-proxy` | −77 | 3/3 |
| `garage-m5stack-9bc1d4` | −85 | 3/3 |
| `atom-fireplace` | −85 | 3/3 |

Five paths, best at −49, and the connection opened through P1S Printer. Two
standing beliefs need correcting: the p1s-printer proxy is **not** permanently
one-slot-busy (2/3 free here), and **the dock does not need a closer proxy**. The
coverage problem is out in the working area, not at the dock — so site the next
proxy where the mower mows, and start acceptance runs near the dock.

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

Then set the Lovelace resource back to `?v=11`. Reverting also restores the
`pymammotion==0.8.8` pin, and the capability probes re-lock real motion by
themselves — no separate safety step is required.
