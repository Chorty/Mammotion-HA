# Vendor tool analysis — LubaLogTools and RTKReferenceStationUpgradeTools

Static analysis of two Mammotion-supplied Windows tools the operator has for
their own hardware. Interop/diagnostics only: **nothing was executed, no firmware
was flashed, and neither binary is committed** (both are in `.gitignore`).

The headline is in §3: the base station is a **Unicore UM980**, and pymammotion
already carries a `basestation.proto` this integration never uses that would
expose whether the base thinks it has moved.

## 1. What the tools are

| | LubaLogTools.exe | BaseStationUpgradeTools.exe |
| --- | --- | --- |
| Build | PyInstaller, Python 3.10, PyQt6 | PyInstaller, Python |
| Dated | 2023-07-11 (Luba 1 era) | — |
| Transport | `msgbus.serialbus` → **USB serial** | `pyserial` → **USB serial** |
| Purpose | pull logs off the mower | flash base-station firmware |

Extraction method, if it needs repeating: the PyInstaller CArchive cookie is
`MEI\014\013\012\013\016` near EOF; parse the TOC, then unpack `PYZ-00.pyz`
(magic `PYZ\0`, marshalled TOC) to get per-module `.pyc`. Python 3.10 bytecode
does not unmarshal on 3.14, so identifiers were read from the string tables.

## 2. LubaLogTools — mostly NOT new information

It speaks the **same protobuf families** we already use, over serial rather than
BLE: `Luba_msg`, `mctrl_nav`, `mctrl_sys`, `mctrl_driver`, plus `esp_driver`.

The log-retrieval protocol it uses:

```
FILE_TYPE_RTKLOG / FILE_TYPE_NAVLOG / FILE_TYPE_SYSLOG / FILE_TYPE_ALL
DrvUploadFileToAppReq / DrvUploadFileToAppRsp
DrvUploadFileReq / DrvUploadFileCancel / DrvListUpload
DEV_BASESTATION  (a MsgDevice enum value — the protocol can address the base)
```

⚠️ **An initial reading of this as a discovery was wrong.** Every one of those
symbols is **already in pymammotion** (`proto/dev_net.proto`,
`proto/luba_msg_pb2.pyi`). The tool confirms the protocol is real and used in
practice; it does not hand us anything the library lacks.

**The one genuine gap:** `esp_driver` has no counterpart in pymammotion
(0 hits). Unassessed whether it matters.

**Residual value:** `luba_logutils.comps.log_catch.worker` (17 KB) holds the
actual call *sequence* for a file pull — request, chunking, cancel. If we ever
want mower-side `RTKLOG`/`NAVLOG` files, that is the reference implementation.
Caveat: 2023-era, Luba 1. This mower is a `Luba-VSPLV397`; the protocol may have
moved.

## 3. The base station — the actually useful find

`RTKReferenceStationUpgradeTools/` contains:

- `luba_basestation_v1.0.0.58.bin` — base station MCU firmware (39 KB)
- **`UM980_7923.pkg`** (2.3 MB) — GNSS receiver firmware
- `res/bs_trace.json` → `{"mode": "MCU"}`

**The receiver is a Unicore UM980.** Its firmware image carries the Unicore
command set: `MODE ROVER`, `CONFIG`, `RTCM`, `LOG` / `UNLOG` / `LOGLIST`,
`VERSION`, `CONFIG AUTHCODE`, and the string `Fixed positio[n]`.

That matters because a UM980 base runs in one of two modes:

- **survey-in** — averages its own position over time, then transmits
  corrections referenced to that average;
- **fixed position** — transmits against a manually configured coordinate.

**If the survey never converges, or converges to a bad position, the base still
transmits perfectly well-formed corrections referenced to the wrong place.** A
rover receiving them sees a healthy link, plenty of co-viewed satellites, and
cannot resolve to Fix. That is exactly the 2026-08-07 signature: corrections
flowing, rover healthy, `sync_rtk_and_dock` useless, and only a base power cycle
— which restarts the survey — clearing it.

## 4. ⚠️ This refutes a claim in the RTK hardening plan

`docs/rtk-hardening-plan-20260807.md` states the base station's internal state is
unreachable from this integration. **That is wrong twice over.**

`last_error` already carries reference-station events (recorded there), and more
importantly pymammotion ships `proto/basestation.proto`, which defines:

```proto
message request_basestation_info_t { uint32 request_type = 1; }

message base_score {
    uint32 base_score = 1;    uint32 base_leve  = 2;
    uint32 base_moved = 3;    uint32 base_moving = 4;
}

message response_basestation_info_t {
    uint64 system_status = 1;   uint64 sats_num   = 5;
    uint64 rtk_status   = 10;   int32  mqtt_rtk_status = 12;
    int32  rtk_channel  = 13;   int32  rtk_switch = 14;
    base_score score_info = 15;
    // + ble_rssi, wifi_rssi, lora_scan/channel/locid/netid, lowpower_status
}
```

**`base_moved` and `base_moving` are precisely the survey-hypothesis
discriminator** — does the base believe its position changed? `base_score` and
`base_leve` plausibly express survey/solution quality.

This integration never requests any of it. `basestation_info` is only ever
*read* for display (`sensor.py` `position_mode`); there is a
`MammotionRTKCoordinator` for a separate `RTKBaseStationDevice`, but nothing
queries `request_basestation_info_t` for the dock-integrated base.

## 5. What follows — proposed, not done

1. **Query the base station.** Send `request_basestation_info_t` and record the
   response, especially `base_score`. Read-only, no motion. If `base_moved` or a
   low `base_score` shows up during a Float episode, the root cause is settled.
2. **Log it alongside RTK state** so the next Float episode captures base status
   at the same time — turning a recurrence log into a diagnosis.
3. Only then consider whether mower-side `FILE_TYPE_RTKLOG` retrieval adds
   anything.

**Explicitly NOT proposed:** flashing base-station or UM980 firmware. The images
are present and the procedure is documented, but a bricked base station ends all
RTK work, and nothing so far suggests the firmware is at fault rather than its
survey state.

## 6. `LubaBaseStationUpgrade.app` v1.0.6 — a much newer tool, and the best
   evidence yet for the survey hypothesis

Added 2026-08-07 later, from a third vendor tool the operator supplied: a
**macOS** app bundle (arm64), unlike the two Windows tools above. PyInstaller
onefile, Python 3.10 + PyQt6, `client.version = '1.0.6'`. Nothing was executed;
the bundle was parsed and its embedded firmware read. It is `.gitignore`d.

Extraction is easier to reproduce than the Windows tools: the CArchive cookie
parses the same way, and because a matching **Python 3.10.18** interpreter exists
on this machine, the PYZ code objects `marshal.loads` and disassemble properly
rather than being read as string tables.

### 6.1 The tool itself

App-specific modules: `client` (`handler`, `worker`, `ui`, `version`,
`reportbanken`), plus `msgbus` (shared with LubaLogTools), `esptool`,
`intelhex`, and `pythonping`. It flashes over **USB serial** — the UI text
instructs the operator to connect the base station to the computer with a
Type-A-to-Type-C data cable, and it bundles a CH341 USB-serial driver it installs
via `pnputil` on Windows.

`BaseMcuUpgradeWorker` implements the flash: `upgrade_prepare` → `transmit_file`
(chunked, Ack/Nak with retry, CRC16) → `upgrade_exit`, plus `get_mcu_version`,
`get_conf` (reads `lora:` config) and `bridging_now`, whose log line is
**`bridge e22 receive data:`**.

⚠️ **Two things I initially got wrong here.** I first read the bundled
`aiohttp`/`certifi`/`cryptography` as evidence the tool *downloads* firmware. It
does not: the firmware is embedded as a Qt resource at
`:/fw/res/luba_v087_base.bin`, and the only module that touches the network is
`reportbanken`, an outbound error-telemetry uploader (it carries an app key and a
`generate_sign` helper). There is **no Mammotion endpoint anywhere in the
binary** — the only URLs present are certificate-authority URLs from the bundled
driver's code-signing chain.

### 6.2 The base station's architecture, now confirmed from its own firmware

The embedded `luba_v087_base.bin` (note: **v087**, versus `v1.0.0.58` in the
2023-era `RTKReferenceStationUpgradeTools`) carries its own symbol and format
strings:

- **GNSS receiver: Unicore, confirmed from the device side.** The firmware
  probes it (`versionb`, `get version success!`) and matches a model table
  containing `UM980`, `UM4B0`, `UM482`, `UB482`, `UT4B0` and others, logging
  `version info, type:%u, model:%s, version:%s`. The earlier UM980 identification
  came only from a package filename; this is independent corroboration of the
  family, though the *specific* model this unit reports is still unread.
- **Correction datalink: an Ebyte E22 LoRa radio.** `lfh_send_rtcm_data`,
  `lfh_switch_channel`, `lfh_read_rssi_cmd`, `lfh_lbt_can_send`. This is what the
  `lora_channel` / `lora_scan` / `lora_locid` / `lora_netid` fields in
  `basestation.proto` describe.
- Configured RTCM output includes **`RTCM1005`** (base antenna reference
  position) alongside `1074/1084/1094/1124` MSM4 observations and `1033`, plus
  `CONFIG PPS ENABLE2 GPS POSITIVE 500000 4000 0 0` and
  `CONFIG RTCMB1CB2A ENABLE`.
- An init state machine: `unicore_init_stage: RTK_INIT_STAGE_PREPRARE` →
  `RTK_INIT_STAGE_SUCCESS`, with `rtk cur stage failed` on the error path.

### 6.3 Why this matters for the 2026-08-07 Float episode

The firmware contains, adjacent in its symbol table:

```
rtk_cfg_handler
setRTKBaseLocation
MODE BASE
rtk_position_reset
```

and, in its format strings, both `MODE BASE` (bare) and `MODE BASE ` (trailing
space — a prefix built up with arguments), together with:

```
pos_mgr_guard, basestation pos status is %d
the basestation pos status is: %d
lat: %.12lf, lon: %.12lf, alt: %.12lf
SocUptime: %f, lon_num: %lf, lat_num: %lf, baseStationQual: %d
```

In the Unicore command set, bare `MODE BASE` is self-positioning, while
`MODE BASE <args>` fixes the receiver to a supplied coordinate. The presence of
**both forms**, an explicit `setRTKBaseLocation`, a `rtk_position_reset`, and a
`pos_mgr_guard` tracking a "basestation pos status" says the base **stores and
guards a reference position** rather than simply surveying afresh each boot.

That supplies the missing *mechanism* for the leading hypothesis. A stored or
guarded reference position that is wrong — or a `pos_mgr` state that never
reaches valid — produces exactly the observed signature: well-formed corrections
transmitted continuously, a rover with healthy reception that cannot resolve,
`sync_rtk_and_dock` useless because the fault is not rover-side, and recovery
only on a base power cycle that re-runs positioning.

It also connects directly to the protobuf: **`base_moved` / `base_moving` are
plausibly this same position-guard state surfaced to the app**, which is what
`basestation_info_probe` will read.

⚠️ Still inference, not proof. These are strings and symbol names in a firmware
image; the control flow that uses them has not been disassembled, and nothing has
been read from the live device.

### 6.4 A workaround that exists but is not proposed

`basestation.proto` also defines `app_to_base_mqtt_rtk_t` with `rtk_switch`,
`rtk_url`, `rtk_port`, `rtk_username`, `rtk_password` — i.e. the base can be
pointed at an **NTRIP caster** instead of relying on its own reference position.
If the local base proves to be the fault, a network correction source is an
escape hatch. **Not proposed and not attempted:** it is a write to base-station
configuration, it depends on a subscription or public caster within baseline
range, and it is a decision for the operator, not a diagnostic step.

### 6.5 Does any of this help pull the *mower's* firmware? No.

Asked directly, and the answer is no. This tool never downloads firmware — it
carries one image for one target. That target is the **base station MCU**, not
the mower, and delivery is **USB serial to the base**, not OTA. There is no
firmware-distribution endpoint, no OTA manifest, and no download logic anywhere
in it to reuse. The mower-side path remains the one in §2: `FILE_TYPE_RTKLOG` /
`DrvUploadFileToAppReq`, which pulls *logs off* the mower and is unrelated to
fetching firmware onto it.

## 7. Caveats

- Static analysis only; nothing executed, no firmware flashed. Two tools are
  Windows binaries; the third is a macOS app bundle.
- LubaLogTools is 2023 / Luba 1 era; protocol drift versus `Luba-VSPLV397` is
  unassessed.
- It is **unverified** that this mower's base responds to
  `request_basestation_info_t` — the message exists in the library, which is not
  the same as the hardware answering it.
- `UM980` specifically is identified from a firmware package filename and a model
  table the firmware can match against; the model this unit actually reports has
  not been read from the device.
- §6.3 reasons from strings and symbol names, not from disassembled control flow.
