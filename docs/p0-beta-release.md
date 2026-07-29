# P0 beta release status

## Safety model

- Experimental manual motion defaults off and is BLE-only.
- PyMammotion 0.8.12 is installed for its released notification cleanup. Real
  motion stays locked because the loaded backend is *probed* and demonstrably
  lacks both audited BLE fixes -- not because of its version number. See
  `custom_components/mammotion/backend_capability.py`: authorization requires
  `ble_teardown_failure_atomic` and `blufi_reassembly_reset` to be observed in
  the installed code, plus a release at or above the audited base 0.8.12. A
  version label alone never authorizes motion, so a fork, a rebuild, or a future
  upstream release carrying the same number cannot self-certify.
- Every real service run requires positive LUBA capability evidence, fresh BLE
  queue/liveness evidence, safe runtime state, both operator confirmations, and
  an exclusive backend session.
- `mammotion.stop_manual_motion` marks the active session cancelled before it
  queues three emergency-priority confirmed zero-velocity writes. Cancelled
  sessions cannot issue another nonzero confirmed write.
- Preview and dry-run accept seven destinations. Real click-to-go is limited to
  two segments.
- YUKA, RTK, SPINO, accessories, and unknown products are fixture-characterized
  and fail closed for hazardous actions until hardware acceptance exists.

## Deployment

1. Install the beta and restart Home Assistant with experimental motion off.
2. Confirm integration setup, maps/tasks, diagnostics, native camera entities,
   and card preview/dry-run.
3. Register
   `/mammotion/mammotion-custom-path-card.js?v=<installed-version>` as a
   JavaScript module dashboard resource.
4. Do not enable real motion while
   `export_runtime_state.experimental_motion.backend_verified` is false.
5. After a fixed PyMammotion release is pinned, follow the supervised daylight
   LUBA acceptance sequence in `docs/NEXT-SESSION.md`.

## Breaking migrations

| Previous HA enum state/option | New state/option | Compatibility |
| --- | --- | --- |
| `MODE_READY` and other uppercase mower enum labels | `mode_ready` and lowercase equivalents | Original label is in `raw_protocol_value`. |
| `AUTO`, `FLOOR`, `WALL`, etc. | `auto`, `floor`, `wall`, etc. | Select entity methods normalize legacy case during migration. |
| `MAN`, `WOMAN`, language labels | `man`, `woman`, lowercase language | Wire commands are converted back to vendor enum names. |
| Uppercase RTK, task-area, and SPINO sensor states | Lowercase equivalent | Update automations, templates, and dashboard conditions. |
| `mammotion.get_tokens` | Removed | Use the native camera/WebRTC entity; credentials stay server-side. |

## Verified limitations

- Only LUBA is eligible for supervised live acceptance in this release.
- Real motion remains unavailable on the current PyMammotion 0.8.12 pin: both
  capability probes report absent against it.
- No P1/P2 feature additions are included.
- RTK and SPINO firmware installation remains blocked pending hardware-derived
  prerequisite acceptance.

## Rollback

Disable experimental motion first, restore the previous HACS version, restart
Home Assistant, and remove or update the click-to-go resource cache key.
