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

## 6. Caveats

- Static analysis only; nothing executed. Both tools are Windows binaries and
  this is macOS.
- LubaLogTools is 2023 / Luba 1 era; protocol drift versus `Luba-VSPLV397` is
  unassessed.
- It is **unverified** that this mower's base responds to
  `request_basestation_info_t` — the message exists in the library, which is not
  the same as the hardware answering it.
- `UM980` is identified from a firmware package filename and embedded strings,
  not from querying the device.
