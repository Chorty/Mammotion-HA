# OTA firmware capture investigation — PAUSED 2026-08-16

**Goal:** capture a readable copy of the mower's OTA firmware binary (first
`1.30.29.8`, then `1.30.29.20`) before/without installing it, for research —
not to block updates forever, just to get one copy. **Status: paused at the
operator's request** to wait a day or two and watch for any issues before
actually running the next real attempt. Everything below is either a proven
finding, a genuine new permanent capability, or an armed-but-unused tool
ready to resume from. Nothing here was committed to
`custom_components/mammotion/` except the one deliberate, permanent,
read-only service (`ota_info_probe`) — everything else is operator-side
tooling in `scripts/`, which never ships to end users.

## Bottom line, if you read nothing else

- **We do not have the firmware file.** We have proof of the CDN domain it
  comes from and 43MB of encrypted traffic to it, nothing decryptable.
- **The wall is cryptographic, not procedural.** The mower needs its own
  Aliyun IoT device credentials to request a signed download URL; our
  account session doesn't have them, and there is no known software-only way
  to obtain them. No amount of better timing or interception position
  changes that.
- **One genuine, permanent new capability came out of this:** `ota_info_probe`,
  a read-only HA service (deployed `0.6.4-beta56`) that asks the mower
  directly, over its own BLE connection, for its OTA status. First time this
  request/response path has ever been exercised in this project's or
  pymammotion's history. It cannot return the download URL by design of the
  protocol (see below), but it works, and it's a real diagnostic tool now.
- **Resuming is a few commands away.** See "Resume checklist" at the bottom.

## What was actually tried, and what happened

### 1. Cloud API — direct query with the account's own session

`scripts/fetch_ota_firmware.py` calls the same Aliyun "Breeze" OTA endpoint
the app uses (`/thing/ota/info/queryByUser`). Result: `"non-existent job
record"`. `scripts/probe_ota_endpoints.py` tried nine more candidate paths
(Aliyun "Living" OTA family, generic `/thing/ota/*` guesses) — all either
don't exist or return the same "no record" story.

**Root cause, confirmed later:** the OTA job (and the signed download URL
it carries) is scoped to the *mower's own* Aliyun IoT device identity, not
the account's cloud session. Our account can check whether an update
*exists*; it cannot fetch the job a specific device was issued.

### 2. Phone-side MITM (mitmproxy)

`scripts/mitm_ota_capture.py`, with a CA cert installed on the operator's
phone, successfully decrypts and captures the app's normal traffic. But the
actual OTA trigger and the firmware download **never appear** in phone-side
capture, across the entire investigation.

**Root cause:** the mower downloads the firmware itself, directly, over its
own WiFi connection. The phone is never in the data path for the download —
only for the initial "check for update" UI flow.

### 3. Live capture of a real update (1.30.29.8) via MQTT

`scripts/listen_ota_mqtt.py` subscribes to the Aliyun IoT MQTT bridge (the
same channel HA's `MammotionRTKCoordinator`-style plumbing uses) under the
account's own AEP app-device identity. During a real, successful update to
`1.30.29.8`, this captured the **complete** `thing.properties` →
`otaProgress` telemetry stream: three modules (`deviceRTK` 2.38MB,
`mcu` 415KB, `SocMidware` 213MB), flash target `/dev/mmcblk0p15`, full
resize→write→verify sequence, `upgrade success` at the end. Real, valuable
telemetry — genuinely useful — but **the payload never carries a URL**, only
progress/status fields. Confirmed later this is structural: see finding 6
below.

This *is* a reliable, proven trigger-detection channel: the first
`otaProgress` message arrives right at the true start of a download. Both
`scripts/ota_safety_net.py` (block-on-timeout) and
`scripts/ota_url_probe_burst.py` (BLE probe burst) are built on watching
this exact signal.

### 4. Network-level MITM of the mower itself (ARP spoof)

The mower and RTK base were moved onto the same flat LAN as the operator's
Mac (VLAN removed), then `bettercap` ARP-spoofed the mower's traffic through
a local `mitmproxy` in transparent mode (`pf` NAT redirect on port 443).
Traffic genuinely routed through the Mac — confirmed via `tcpdump`. But:

- The mower's TLS client doesn't trust our CA (no way to install a
  certificate on an embedded device — no filesystem/settings UI access at
  all), so nothing ever decrypted.
- More importantly: **while the ARP-spoof/intercept was active, the app's
  own "check for firmware update" request failed to complete over WiFi**,
  and only succeeded once the operator switched to cellular. This reproduced
  even with a genuinely idle mower doing nothing but its own background
  telemetry. Root cause was **not** anti-tampering detection in the app —
  stopping `bettercap` entirely (clean untampered network) did not fix it,
  which rules that theory out. The leading remaining explanation is a
  conflict between the gateway's own hardware NAT/flow-offload path and
  ARP-spoof-based redirection (see finding 5), but this was never fully
  root-caused. **Do not re-arm ARP spoofing casually** — it has a
  demonstrated, unexplained side effect on the exact flow we're trying to
  observe.

### 5. Passive capture at the UniFi gateway

Realized the gateway is a natural, zero-tampering choke point: all of the
mower's internet-bound traffic passes through it regardless of anything we
do locally. SSH access already existed (`UNIF_SSH_PASSWORD` /
`UNIFI_SSH_USERNAME` / `UNIFI_SSH_PORT` in `.env`; gateway is a UniFi Cloud
Gateway Max, `192.168.1.1`, Linux `5.4.213-ui-ipq5322`, bridge `br0` =
`192.168.1.0/24`).

**First attempt returned nothing** — literally zero data packets, only ARP,
even during known-continuous MQTT chatter from the mower. Root cause: this
platform's Qualcomm **PPE (Packet Processing Engine)** hardware NAT/flow
offload (`ppe.ppe_drv.eth2eth_offload_if_bitmap` /
`ppe.ppe_drv.if_bm_to_offload` sysctls) accelerates established flows
entirely in the switch ASIC, bypassing the Linux/CPU path `tcpdump` reads
from.

**Fix:** disable **Hardware Acceleration** in the UniFi Network app, under
the gateway device's Services settings. This is the supported UI toggle for
the same PPE offload — confirmed the correct, sanctioned mechanism rather
than poking undocumented sysctls directly. After disabling it, real data
traffic became visible immediately and stayed visible.

⚠️ **Hardware Acceleration is currently OFF on the gateway.** This is a
real, standing router performance setting affecting the whole network, not
scoped to the mower. Re-enable it if network performance matters more than
this investigation before the next real attempt, or leave it off — it's a
deliberate tradeoff, not an accident. Check via the UniFi Network app,
gateway device → Services → Hardware Acceleration.

### 6. Live capture of the real 1.30.29.20 attempt — the actual finding

With hardware acceleration off, a live tap of "start update" on the mower
produced a genuine new TLS handshake — the first ever captured for this
investigation:

```
192.168.1.66 -> 155.102.176.87:443   SNI: mds.mammotion.com
```

43MB transferred (42MB down / 465KB up) before the operator manually
interrupted it via a UniFi client block (the *automated* block was already
known broken at this point — see finding 7). `robots.txt` on that host
returns an Aliyun OSS "NoSuchKey" XML error naming the real backing bucket:
`mammotion-us.oss-us-east-1.aliyuncs.com`. Root path returns `403`, no open
listing.

**This is genuinely new, real information** — the actual firmware CDN — and
it's the closest this investigation got to the file. It is also the ceiling:
TLS SNI reveals the hostname, never the object path or any signed-URL
parameters, and there's no way to reconstruct those without either (a) the
mower's own device credentials, or (b) probing the OSS bucket for
guessable/misconfigured paths, which was deliberately **not** attempted —
that crosses from "inspecting our own capture" into testing a third party's
cloud infrastructure, and the 403 already suggests it isn't trivially
misconfigured anyway.

Evidence: `scripts/ota_captures/gateway_capture_1.30.29.20_20260816.pcap`
(gitignored — 46MB raw pcap, not committed; regenerate by re-running the
capture, see resume checklist).

### 7. The UniFi automated block is broken

`{"cmd":"block-sta","mac":...}` against
`/proxy/network/api/s/default/cmd/stamgr` started returning
`{"meta":{"rc":"error","msg":"api.err.Invalid"}}` for **both** block and
unblock, against **both** the mower and an unrelated decoy device, some time
after the network topology changes above (VLAN flattening / hardware
accel toggle). It worked earlier the same session (measured ~640ms
issue-to-cutoff against a decoy device). Never re-diagnosed — read-only
endpoints (`stat/sta`, `self`) still work fine, so it's specifically the
`cmd/stamgr` write path. **Manual block via the UniFi app is the only
confirmed-working interrupt mechanism right now.**
`scripts/ota_safety_net.py` still attempts the automated call first (in
case it starts working again) but assume it will fail and be ready to block
manually.

### 8. The new capability: `ota_info_probe`

`pymammotion`'s `MessageOta.get_device_ota_info` (`EMBED_OTA` / `SubOtaMsg`,
`MctlOta.todev_get_info_req(type=IT_OTA)`) is a real, fully-defined request
the device's own protocol supports — but was never called anywhere in
`pymammotion` or this integration before now. The broker
(`messaging/broker.py`) already recognises the `ota` field group for
request/response correlation, so nothing needed inventing — just a new
read-only probe service following the exact `basestation_info_probe`
pattern. Added to `custom_components/mammotion/services.py`,
`services.yaml`, `strings.json`, `translations/en.json`; deployed as
`0.6.4-beta56` via the normal `Beta Release` workflow (see
`docs/deploy-runbook-p0.md` for that deploy's hash/version verification —
all 46 files byte-identical, card md5 `58acc956...`, backend
`pymammotion==0.8.12.post1` unchanged).

**Confirmed live, twice** (once idle, once via a full 6-call burst test):
the mower answers correctly (`command_sent: true, answered: true`), but the
response is always empty:

```json
{"ota": {"otaid": "", "version": "", "progress": 0, "result": -1, "message": ""}}
```

**Why it can never carry the URL, by protocol design, not a bug:** the
top-level `MctlOta` message is a `oneof` with five alternatives —
`todev_get_info_req` / `toapp_get_info_rsp` / `fw_download_ctrl` /
`fota_info` / `fota_sub_info`. `IT_OTA` info requests resolve to
`toapp_get_info_rsp` → `OtaInfo`, which only ever carries
`otaid`/`version`/`progress`/`result`/`message` — the same progress-only
shape already seen over MQTT. The URL lives in the sibling
`fota_sub_info.sub_img_url` field, and how `pymammotion` itself uses that
field elsewhere (`send_swimming_pool_device_ota_second`, for the Spino
product line) is to **construct and send** it — an app→device push of a URL
the app already obtained via its own cloud fetch — not something the Luba
mower reports back. There is no evidence this ever flows device→app for
this product.

**Worth trying once more, cheaply:** firing the same probe burst *during* a
real active download, in case internal state differs from idle — the burst
tool (`scripts/ota_url_probe_burst.py`) exists for exactly this, already
wired to the proven MQTT trigger. Genuinely low odds given the schema
evidence above, but zero cost/risk to check empirically rather than assume.

## Files, and where they live

All in `scripts/` (operator tooling, never shipped as part of the
integration) unless noted:

- `fetch_ota_firmware.py`, `listen_ota_mqtt.py`, `mitm_ota_capture.py` —
  pre-existing, already committed before this document.
- `probe_ota_endpoints.py` — the nine-candidate cloud API prober (§1).
- `ota_safety_net.py` — MQTT-trigger-watching block-on-timeout (§3, §7).
  3-second countdown (operator-requested; was 2s in earlier testing).
- `ota_url_probe_burst.py` — MQTT-trigger-watching BLE probe burst (§8),
  companion to the above, independent transport (BLE vs UniFi/WiFi), safe
  to run simultaneously.
- `pf_mammotion_ota.conf` — the `pf` NAT redirect rule from the ARP-spoof
  attempt (§4). Kept for reference; **not currently loaded**, and re-arming
  it means re-doing the whole ARP-spoof setup this doc recommends against.
- `ota_captures/` — gitignored. Raw `.pcap` evidence:
  `gateway_capture_1.30.29.20_20260816.pcap` (the 43MB `mds.mammotion.com`
  capture, §6), `mower_capture_arpspoof_20260815.pcap`,
  `rtk_capture_arpspoof_20260815.pcap` (§4, superseded by the gateway
  capture but kept).
- `custom_components/mammotion/services.py` (+ `.yaml`, `strings.json`,
  `translations/en.json`) — the one permanent addition, `ota_info_probe`
  (§8). This is real, shipped, deployed code — not research scaffolding.

## Resume checklist

Nothing is currently running — everything was stopped cleanly before
pausing (mower confirmed `blocked: false` and reachable at time of pause).

1. Confirm current firmware-check state hasn't changed:
   `curl` the mower's `update.back_yard_clip_skywalker_firmware` HA entity,
   or just open the app.
2. Re-arm, in order:
   ```sh
   set -a && source .env && set +a
   .venv/bin/python scripts/listen_ota_mqtt.py --timeout 3600 &
   python3 scripts/ota_safety_net.py &          # 3s countdown, block likely still broken -- have manual block ready
   python3 scripts/ota_url_probe_burst.py &     # independent BLE probe burst
   scripts/ha_ssh.exp 'tcpdump -i br0 -U -w -' > scripts/ota_captures/gateway_capture_<version>_<date>.pcap &   # if hardware accel still off
   ```
3. Verify hardware acceleration state on the gateway before relying on the
   passive capture (§5) — check the UniFi app if unsure.
4. Re-check the UniFi block API (§7) — try a real block/unblock against a
   non-critical device first. If still `api.err.Invalid`, the manual block
   is still the plan.
5. Tap the update. Tell the agent when you tap it (needed for real-time
   monitoring/narration, not for the automated pieces, which react on their
   own). Manually block the mower's network ~3 seconds after you see the
   download actually start, since the automated block is unconfirmed.
6. Afterward: check `scripts/ota_url_probe_burst.log` for anything other
   than the empty baseline shape, and re-run the SNI extraction
   (`tshark -r <pcap> -Y "tls.handshake.type==1" -T fields -e ip.dst -e
   tls.handshake.extensions_server_name`) on the fresh capture regardless of
   probe results.
