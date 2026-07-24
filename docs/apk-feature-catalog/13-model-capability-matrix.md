# Mammotion Android APK device/model/capability matrix

## Scope and interpretation

This catalog is a static analysis of the decompiled Android application under
`/Users/mattjoslin/mammotion-apk-decompile/src`. It records what this APK
actually tests. It does **not** treat product labels, similar icons, adjacent
enum values, or shared service codes as proof of equivalent hardware.

Evidence references below are relative to the decompile root. Confidence means:

- **High** — direct enum, product-key, item-code, or boolean return expression.
- **High, conditional** — direct expression, but the result also depends on
  firmware, device-reported state, a subtype/internal-model record, or another
  runtime value.
- **Medium** — the direct rule is clear but the decompiler lost part of the
  surrounding control flow, or a type can only be reached through a secondary
  route.
- **Unknown** — this APK contains no exact mapping. No equivalence is inferred.

`Y` and `N` in the matrices mean the named helper returns true or false for the
enum. They are not general claims about physical hardware. `C` means runtime
conditional. `—` means the helper is not a meaningful capability for that
accessory/type.

## Canonical device identities

The enum is authoritative for the app's type IDs, discovery prefixes, and
code names (`sources/com/agilexrobotics/device/source/device/enums/DeviceType.java:12-43`).
Product-key arrays are literal IoT identifiers
(`sources/com/agilexrobotics/device/source/device/utils/DeviceProductKey.java:13-83`).
The service item code is a server/catalog grouping, not a unique hardware ID.

| Family / app enum | ID | discovery product name | code name | IoT product key(s) | service item | Confidence / exact caveat |
|---|---:|---|---|---|---:|---|
| `LUBA` | 1 | `Luba` | `Luba` | `a1UBFdq6nNz`, `a1x0zHD3Xop`, `a1pvCnb3PPu`, `a1kweSOPylG`, `a1JFpmAV5Ur`, `a1BmXWlsdbA`, `a1jOhAYOIG8`, `a1K4Ki2L5rK`, `a1ae1QnXZGf`, `a1nf9kRBWoH`, `a1ZU6bdGjaM` | 1301 | High. Enum: `DeviceType.java:15`; keys: `DeviceProductKey.java:13-14`; item: `ServiceDeviceItemCode.java:17-18`. |
| `LUBA_2` | 2 | `Luba-VS` | `Luba2` | `a1iMygIwxFC`, `a1LLmy1zc0j` (duplicate appears in array) | 1302 | High. `DeviceType.java:16`; `DeviceProductKey.java:16-17`; `ServiceDeviceItemCode.java:44-45`. |
| `LUBA_VP` | 6 | `Luba-VP` | `HM441` | `a1mb8v6tnAa`, `a1pHsTqyoPR` | 1306 | High. `DeviceType.java:20`; `DeviceProductKey.java:19-20`; `ServiceDeviceItemCode.java:20-21`. |
| `LUBA_MN` (LUBA mini) | 7 | `Luba-MN` | `HM430` | `a1L5ZfJIxGl`, `a1dCWYFLROK` | 1307 | High. `DeviceType.java:21`; `DeviceProductKey.java:22-23`; `ServiceDeviceItemCode.java:38-39`. |
| `LUBA_LD` | 11 | `Luba-LD` | `HM431` | `a1jDMfG2Fgj`, `a1vtZq9LUFS` | 1310 | High. `DeviceType.java:25`; `DeviceProductKey.java:43-44`; `ServiceDeviceItemCode.java:29-30`. |
| `LUBA_VA` | 15 | `Luba-VA` | `HM442` | `a1Ce85210Be`, `a1BBOJnnjb9` | 1313 | High identity, but item code collides with `SPINO`. `DeviceType.java:29`; `DeviceProductKey.java:46-47`; `ServiceDeviceItemCode.java:41-42,59-60`. |
| `LUBA_MD` | 17 | `Luba-MD` | `HM433` | `a1T6VTFTc0C`, `a14iRDqMepW` | 1314 | High. `DeviceType.java:31`; `DeviceProductKey.java:55-56`; `ServiceDeviceItemCode.java:35-36`. |
| `LUBA_LA` | 18 | `Luba-LA` | `HM432` | `CDYuKXTYrSP`, `a1YbcqQYFv2` | 1315 | High. `DeviceType.java:32`; `DeviceProductKey.java:64-65`; `ServiceDeviceItemCode.java:26-27`. |
| `LUBA_MB` | 23 | `Luba-MB` | `HM434` | `a1pb9toor70` | 1318 | High. `DeviceType.java:37`; `DeviceProductKey.java:58-59`; `ServiceDeviceItemCode.java:32-33`. |
| `LUBA_HM` | 28 | `Luba-HM` | `HM610` | `tBnCA8u2Aps`, `jvEDnj42DRK` | 1324 | High. `DeviceType.java:42`; `DeviceProductKey.java:49-50`; `ServiceDeviceItemCode.java:23-24`. |
| `LUBA_ME` | 29 | `Luba-ME` | `HM620` | `HK8snDC8Kxh` | **not present** | High for enum/key; unknown service item. `DeviceType.java:43`; `DeviceProductKey.java:82-83`; item-code list ends at `ServiceDeviceItemCode.java:87`. |
| `LUBA_YUKA` (YUKA) | 3 | `Yuka-` | `Yuka` | `a1kT0TlYEza`, `a1IQV0BrnXb` | 1304 | High. `DeviceType.java:18`; `DeviceProductKey.java:25-26`; `ServiceDeviceItemCode.java:68-69`. |
| `YUKA_MINI` | 4 | `Yuka-MN` | `MN230` | `a1BqmEWMRbX`, `a1biqVGvxrE` | 1305 | High. `DeviceType.java:17`; `DeviceProductKey.java:31-32`; `ServiceDeviceItemCode.java:71-72`. |
| `YUKA_MINI2` | 5 | `Yuka-YM` | `MN230` | no separate key array; parsed with YUKA mini name logic | 1305 (no separate code) | Medium. It is a distinct enum but shares code name and helper membership; product-key parsing returns `YUKA_MINI`, not `YUKA_MINI2` (`DeviceType.java:128-130,256-257,821-823`). Do not collapse it in HA identity. |
| `YUKA_VP` | 8 | `Yuka-VP` | `MN241` | `a1lNESu9VST`, `a1zAEzmvWDa` | 1309 | High. `DeviceType.java:22`; `DeviceProductKey.java:28-29`; `ServiceDeviceItemCode.java:86-87`. |
| `YUKA_MINIV` | 14 | `Yuka-MV` | `MN231` | `a1jFe8HzcDb`, `a16cz0iXgUJ`, `USpE46bNTC7`, `pdA6uJrBfjz` | 1311 | High. `DeviceType.java:28`; `DeviceProductKey.java:40-41`; `ServiceDeviceItemCode.java:83-84`. |
| `YUKA_ML` | 16 | `Yuka-ML` | `MN232` | `a1OWGO8WXbh`, `a1s6znKxGvI` | 1312 | High. `DeviceType.java:30`; `DeviceProductKey.java:52-53`; `ServiceDeviceItemCode.java:74-75`. |
| `YUKA_MN100` | 21 | `Ezy-VT` | `MN100` | `NnbeYtaEUGE` | 1317 | High. `DeviceType.java:35`; `DeviceProductKey.java:67-68`; `ServiceDeviceItemCode.java:77-78`. |
| `YUKA_MN101` | 25 | `Ezy-LD` | `MN101` | `rBGTwYhfhyY` | 1321 | High. `DeviceType.java:39`; `DeviceProductKey.java:70-71`; `ServiceDeviceItemCode.java:80-81`. |
| `CM900` | 24 | `Kumar-MK` | `KM01` | `zkRuTK9KsXG`, `6DbgVh2Qs5m`, `a1tyIkI4q0G`, `1SCa3mAX6G` | 1320 | High. `DeviceType.java:38`; `DeviceProductKey.java:73-74`; `ServiceDeviceItemCode.java:10-11`. |
| `RTK` | 0 | `RTK` | `RTK` | `a1qXkZ5P39W`, `a1Nc68bGZzX` | 1303 | High. `DeviceType.java:14`; `DeviceProductKey.java:34-35`; `ServiceDeviceItemCode.java:47-48`. |
| `RTK3A0` | 12 | `RBSA0` | `RBS03A0` | shared RTK3 keys | 1308 | High type/name; product key alone only identifies the RTK3 family, then the first seven device-name characters select A0/A1/A2 (`DeviceType.java:77-87`). |
| `RTK3A1` | 10 | `RBSA1` | `RBS03A1` | shared RTK3 keys | 1308 | High with same RTK3 caveat. `DeviceType.java:24,77-87`; `DeviceProductKey.java:37-38`. |
| `RTK3A2` | 13 | `RBSA2` | `RBS03A2` | shared RTK3 keys | 1308 | High with same RTK3 caveat. `DeviceType.java:27,77-87`; `DeviceProductKey.java:37-38`. |
| `RTKNB` | 22 | `NB` | `NB` | `a1NfZqdSREf`, `a1ZuQVL7UiN` | 1319 | High. `DeviceType.java:36`; `DeviceProductKey.java:61-62`; `ServiceDeviceItemCode.java:53-54`. |
| `SWIMMINGPOOL` (generic Spino) | 9 | `Spino` | `Spino` | no dedicated array | 1313 (`SPINO`) | Medium. Device-name fallback exists (`DeviceType.java:155-157`), but item 1313 collides with LUBA VA. |
| `SWIMMINGPOOL_S1` | 19 | `Spino-S1` | `PC200` | no dedicated array | 1316 | High enum/item; **name parser caveat:** `valueOfStr()` returns `SWIMMINGPOOL_SP` when the name contains `Spino-S1` (`DeviceType.java:147-149`). |
| `SWIMMINGPOOL_E1` | 20 | `Spino-E1` | `PC100` | no dedicated array | 1323 | High. `DeviceType.java:34,143-146`; `ServiceDeviceItemCode.java:62-63`. |
| `SWIMMINGPOOL_SP` | 26 | `Spino-SP,Spino-S1` | `PC210` | `FCtXbVnmd2C`, `YBRDhT2YTvY` | no distinct constant; server selection uses 1316 for S1, while generic SPINO is 1313 | High enum/key, unknown unique item code. The product-name field deliberately contains two aliases (`DeviceType.java:40,150-165`). |
| `SD_PX` (PC210 pile/dock identity) | 27 | `SDPX` | `SDPX` | `GJzsmaVk5za`, `fEaKVY28tNz` | 1322 | High. `DeviceType.java:41,159-163`; `DeviceProductKey.java:79-80`; `ServiceDeviceItemCode.java:56-57`. Runtime `pileIsPile` further distinguishes a pile representation from the paired SP representation. |

### Identity hazards that matter to HA

1. `has4G()` is implemented as `enum value >= LUBA_2.value`, so it returns true
   for almost every later enum, including RTK3, Spino, and `SD_PX`
   (`DeviceType.java:484-486`). It is not safe as an HA modem-capability test.
2. Service item `1313` is declared for both `LUBAVA` and `SPINO`
   (`ServiceDeviceItemCode.java:41-42,59-60`). Item code must never be the sole
   device discriminator.
3. `YUKA_MINI` and `YUKA_MINI2` share `MN230`, and product-key parsing maps the
   mini keys to `YUKA_MINI` (`DeviceType.java:128-130`). Preserve the raw device
   name/type if the transport provides it.
4. RTK3 product keys identify a family, not A0/A1/A2. The app uses the name
   prefix after product-key recognition (`DeviceType.java:77-87`).
5. `SD_PX` is not sufficient to decide whether an object is the pile. The app
   repeatedly requires both `type().isSD_PX()` and `getPileIsPile()`, e.g.
   `DeviceTypeExtensionsKt.java:201-204,276-278` and
   `device/info/SwimmingPoolUpgradeAndUpLogHelper.java:261-294`.
6. The S1/SP name parser is asymmetric: a literal `Spino-S1` can resolve to
   `SWIMMINGPOOL_SP` (`DeviceType.java:147-165`). Preserve product key, raw name,
   enum, and code name independently.

## Core mower capability matrix

These columns are direct `DeviceType` helpers:

- `Vis` = `isSupportVision()` (`DeviceType.java:765-767`)
- `Radar` = `isSupportRadar()` (`DeviceType.java:737-739`)
- `PureV` / `PureR` = `isPureVisual()` / `isPureRadar()`
  (`DeviceType.java:626-632`)
- `Pos` = `isSupportPositioning()` (`DeviceType.java:733-735`)
- `PC` = `isSupportPointCloud()` (`DeviceType.java:729-731`)
- `NRTK` = `isSupportNRTK()` (`DeviceType.java:721-723`)
- `RTK svc` = `isSupportRtkService()` (`DeviceType.java:753-755`)
- `2WD` / `4WD` = the literal drivetrain helper results
  (`DeviceType.java:662-668`)

| Model | Vis | Radar | PureV | PureR | Pos | PC | NRTK | RTK svc | 2WD | 4WD | Confidence / notes |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| LUBA | N | N | N | N | N | N | N | N | N | Y | High; literal helper results. |
| LUBA 2 (`LUBA_2`) | Y | N | N | N | N | N | N | Y | N | Y | High. |
| LUBA VP / HM441 | Y | N | N | N | N | N | Y | Y | N | Y | High. |
| LUBA mini / HM430 | Y | N | N | N | N | N | Y | Y | N | Y | High. |
| LUBA LD / HM431 | N | Y | N | Y | Y | Y | Y | N | N | Y | High. `Pos` is explicitly true for LD despite pure-radar classification. |
| LUBA VA / HM442 | N | Y | N | Y | N | N | Y | Y | N | Y | High. |
| LUBA MD / HM433 | N | Y | N | Y | N | N | Y | N | N | Y | High. |
| LUBA LA / HM432 | N | Y | N | Y | N | N | Y | N | N | Y | High. |
| LUBA MB / HM434 | Y | N | Y | N | Y | N | Y | N | N | Y | High. |
| LUBA HM / HM610 | N | Y | N | Y | N | N | Y | Y | N | Y | High. |
| LUBA ME / HM620 | N | Y | N | Y | N | N | Y | Y | N | Y | High. |
| YUKA | Y | N | N | N | N | N | N | Y | Y | N | High. |
| YUKA mini / MN230 | Y | N | N | N | N | N | Y | Y | Y | N | High. Applies to both mini enum values. |
| YUKA VP / MN241 | Y | N | N | N | N | N | Y | Y | Y | N | High. |
| YUKA MV / MN231 | N | N | Y | N | Y | N | Y | N | Y | N | High for type helpers; instance RTK is subtype/test-switch conditional (see below). |
| YUKA ML / MN232 | N | Y | N | Y | N | N | Y | N | Y | N | High. |
| YUKA MN100 | N | N | Y | N | Y | N | Y | N | Y | N | High. |
| YUKA MN101 | N | Y | N | Y | N | N | Y | N | Y | N | High. |
| CM900 / KM01 | Y | N | N | N | N | N | Y | Y | N | Y | High. |

The drivetrain helpers are unusually broad: `isSupport4wd()` returns true for
every non-YUKA type except no special accessory exclusion
(`DeviceType.java:666-668`). HA should not expose a drivetrain control merely
because this helper is true.

## Mapping, work, lighting, and maintenance matrix

- `All` = model side of all-area work (`DeviceType.java:670-672`); actual use
  additionally requires firmware `>= 1.14.5`
  (`DeviceUtils.java:1056-1059`).
- `Cross` = device-type cross/obstacle-point membership
  (`DeviceType.java:690-695`); the narrower operational gate uses selected
  types plus firmware `>= 1.15.0` (`DeviceUtils.java:1067-1079`).
- `Dyn` = dynamic-line support (`DeviceType.java:697-699`).
- `NoArea` = no-area work model (`DeviceType.java:725-727`).
- `Upd` = update-map model (`DeviceType.java:757-759`).
- `Blade`, `Light`, `Battery` are the corresponding helpers
  (`DeviceType.java:674-680,701-703`).
- `NoDraw`, `NoLoc`, `NoAuto`, `NoBackup` are negative helpers
  (`DeviceType.java:584-607`).

| Model | All | Cross | Dyn | NoArea | Upd | Blade | Light | Battery | NoDraw | NoLoc | NoAuto | NoBackup |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LUBA | N | N | N | N | N | N | N | Y | N | N | N | Y |
| LUBA 2 | N | N | N | N | N | N | N | Y | N | N | N | N |
| LUBA VP | Y | Y | N | N | N | Y | Y | Y | N | N | N | N |
| LUBA mini | Y | Y | N | N | N | Y | Y | N | N | N | N | N |
| LUBA LD | Y | Y | N | N | N | Y | Y | N | Y | Y | N | N |
| LUBA VA | N | Y | C | Y | N | Y | Y | Y | Y | Y | Y | N |
| LUBA MD | N | N | N | N | N | Y | Y | Y | Y | Y | Y | N |
| LUBA LA | N | Y | Y | Y | N | Y | Y | N | Y | Y | Y | N |
| LUBA MB | N | N | Y | Y | Y | Y | Y | N | Y | Y | Y | N |
| LUBA HM | N | Y | Y | Y | N | Y | Y | Y | Y | Y | Y | N |
| LUBA ME | N | Y | Y | Y | N | Y | Y | Y | Y | Y | Y | N |
| YUKA | N | N | N | N | N | N | N | Y | N | N | N | N |
| YUKA mini | Y | Y | N | N | N | Y | Y | N | N | N | N | N |
| YUKA VP | Y | Y | N | N | N | Y | N | Y | N | N | N | N |
| YUKA MV | N | N | Y | Y | Y | Y | Y | N | Y | Y | Y | N |
| YUKA ML | N | Y | Y | Y | N | Y | Y | N | Y | Y | Y | N |
| YUKA MN100 | N | N | Y | Y | Y | Y | Y | N | Y | Y | Y | Y |
| YUKA MN101 | N | N | N | N | N | Y | Y | N | Y | Y | Y | N |
| CM900 | Y | N | Y | N* | Y | Y | Y | Y | N | Y | Y | N |

`*` `CM900` is not in `isSupportNoAreaWorkDeviceModel()`, but major call sites
explicitly OR it into the same behavior, for example
`home/fragment/DeviceFragment.java:5100-5117`. This is call-site evidence, not
permission to rewrite the helper result.

Dynamic-line support for LUBA VA is firmware conditional: it requires
`>= 1.15.3.4422`; all other `Dyn=Y` rows are unconditional in that helper
(`DeviceType.java:697-699`). Dynamic-line data parsing is guarded in command
handling, not only UI, e.g. `command/app/HashDataManager.java:305,365,717`.

## Camera/video and user-facing controls

| Model group | Video helper | Grass-collection/cutting config helper | Camera wiper helper says unsupported | Vertical remote UI | Box/router device | Radar/RTK switch | Radar self-check | Evidence |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| LUBA | N | N | N | N | N | N | N | `DeviceType.java:614-618,682-683,705-706,741-746,761-762,789-790` |
| LUBA 2 | N | N | Y | N | N | N | N | same |
| LUBA VP | N | N | Y | N | N | N | N | same |
| LUBA mini | N | N | Y | N | N | N | N | same |
| LUBA LD | N | N | N | N | N | Y | N | same |
| LUBA VA | N | N | Y | Y | Y | N | Y | same |
| LUBA MD | N | N | N | N | N | Y | N | same |
| LUBA LA | N | N | N | Y | N | Y | Y | same |
| LUBA MB | N | N | Y | N | N | N | N | same |
| LUBA HM | N | N | Y | Y | Y | N | N | same |
| LUBA ME | N | N | N | Y | Y | N | N | same |
| YUKA | Y | Y | N | N | N | N | N | same |
| YUKA mini | N | N | Y | N | N | N | N | same |
| YUKA VP | Y | Y | N | N | N | N | N | same |
| YUKA MV | N | N | Y | Y | N | N | N | same |
| YUKA ML | N | N | Y | N | N | N | Y | same |
| YUKA MN100 | N | N | Y | N | N | N | N | same |
| YUKA MN101 | N | N | Y | N | N | N | N | same |
| CM900 | N | N | Y | N | N | N | N | same |

`isSupportVideo()` is not equivalent to generic vision. It returns true only
for original YUKA and YUKA VP (`DeviceType.java:761-767`). A major map call site
then explicitly excludes YUKA VP from one video path
(`map/activity/MapManualVideoActivity.java:945`), so HA must gate each stream or
command by observed protocol support, not the broad vision helper.

`isSupportGrassCutting()` is also narrowly named: it is true only for YUKA and
YUKA VP (`DeviceType.java:705-706`) and controls YUKA work-setting payload bytes
(`work/setting/api/WorkingSettingManage.java:869` and
`rn/module/WorkSettingModule.java:424`). It should not be interpreted as “can
mow grass.”

## RTK, dock, pile, and pool-device matrix

| Type | `isRTK()` | `isRTK3()` | PC210 family | Pool family | Local add | Local upgrade | Charge-station deploy | Instance `isSupportRtk()` | Evidence / caveat |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `RTK` | Y | N | N | N | N | N | N | Y | Family predicates: `DeviceType.java:634-655`; instance: `device/source/device/bean/RtkCarDevice.java:91-93`. |
| `RTK3A0` | Y | Y | N | N | N | N | N | Y | `DeviceType.java:634-647`; `Rtk3A0CarDevice.java:91-93`. |
| `RTK3A1` | Y | Y | N | N | N | N | N | Y | same; `Rtk3A1CarDevice.java:91-93`. |
| `RTK3A2` | Y | Y | N | N | N | N | N | Y | same; `Rtk3A2CarDevice.java:91-93`. |
| `RTKNB` | Y | Y | N | N | N | N | N | Y | `DeviceType.java:642-651`; `RtkNBCarDevice.java:91-93`. |
| Generic Spino | N | N | N | Y | Y | Y | N | N | `DeviceType.java:713-718,769-774`; `SwimmingPoolDevice.java:97-99`. |
| Spino S1 | N | N | N | Y | Y | Y | N | N | same; `SwimmingPoolDeviceS1.java:97-99`. |
| Spino E1 | N | N | N | Y | Y | Y | N | N | same; `SwimmingPoolDeviceE1.java:97-99`. |
| Spino SP / PC210 | N | N | Y | Y | N | N | Y | N | `DeviceType.java:618-624,686-688,713-718`; `SwimmingPoolDeviceSP.java:97-99`. |
| `SD_PX` | N | N | Y | Y | N | Y | N | N | `DeviceType.java:622-624,717-718,769-774`; `SDPXDeviceBean.java:97-99`. `isSwimmingPoolChargingPile()` is true for every `SD_PX`, while the UI additionally checks `pileIsPile`. |

`isSupportChargeStationDeploy()` is true for YUKA MV, YUKA MN100, LUBA MB,
LUBA VA, YUKA ML, Spino SP, and LUBA LA
(`DeviceType.java:686-688`). `isOnlySupportChargeStationDeploy()` excludes
LUBA MB and LUBA VA from its narrower set
(`DeviceType.java:618-620`). Home/device-network call sites use the broad
helper to enter deployment flows
(`home/fragment/HomeFragmentNew.java:1803` and
`device/setting/activity/DeviceNetworkActivity.java:522`).

### Instance RTK is a separate capability plane

Do not substitute enum `isSupportRtkService()`, enum `isSupportNRTK()`, or
family `isRTK()` for the device instance's `isSupportRtk()`:

- LUBA, LUBA 2, YUKA, YUKA mini, and YUKA VP instances return true directly
  (for example `LubaCarDevice.java:91-93`, `Luba2CarDevice.java:97-99`,
  `LubaYukaCarDevice.java:97-99`, `LubaYukaMiniCarDevice.java:97-99`,
  `LubaYukaPlusCarDevice.java:97-99`).
- YUKA ML and LUBA MB return false
  (`LubaYukaMLCarDevice.java:97-99`, `Luba2MBCarDevice.java:97-99`).
- LUBA LD, VA, and LA return `stateMachine.newRtkMode == 1`
  (`Luba2LDCarDevice.java:97-99`, `Luba2VACarDevice.java:97-99`,
  `Luba2LACarDevice.java:97-99`). This is device-state conditional.
- YUKA MV checks a stored test/subtype switch rather than returning a fixed
  enum value (`LubaYukaMVCarDevice.java:98-107`). Its subtype helper recognizes
  only specific internal model IDs as RTK-capable
  (`DeviceMultimodelHelper.java:438-446`).

## Internal model/sub-product distinctions

`DeviceMultimodelHelper` loads server/local model metadata and stores a
`currentDeviceMode`; defaults are selected by `DeviceType`
(`DeviceMultimodelHelper.java:448-543`). Therefore enum identity alone does not
fully determine cutting ranges, speeds, spacing, color, or RTK mode.

| Enum / subtype | Exact internal-model behavior | HA consequence | Confidence / evidence |
|---|---|---|---|
| YUKA mini | White shell for `HM020080YKMINI05`, `HM050090YKMINI05H`, `HM020080YKMINI06`, `HM050090YKMINI06H` | Cosmetic only unless a protocol difference is separately observed. | High, `DeviceMultimodelHelper.java:311-319`. |
| YUKA mini | “Need set cutting” for `HM050090YKMINI08H`, `...05H`, `...06H`, `...07H` | Cutting setup must be subtype-aware. | High, `DeviceMultimodelHelper.java:302-309`. |
| YUKA MV / MN231 | 500 subtype for `HM020080YKMINIV07` or `49`; 700/800 subtype for `48`, `50`, `HM020080YKMINI09`, `HM020080YKMINIV09` | Do not infer work-area/range or RTK support from `MN231` label. | High, `DeviceMultimodelHelper.java:292-300,405-420`. |
| YUKA MV / MN231 | RTK-capable subtype IDs: `HM020080YKMINI09`, `48`, `135`, `HM020080YKMINIV09`, `50` | Enable RTK entities only after subtype/state confirmation. | High, `DeviceMultimodelHelper.java:438-446`. |
| LUBA VA / HM442 family | Internal IDs map to capacity/mode classes 1500, 3000, 5000, 10000 and RTK-guide variants 3001, 5001, 10001 | Capacity labels are not interchangeable; RTK mode exists only when class is not 1500. | High, `DeviceMultimodelHelper.java:330-355,422-435`. |
| LUBA VA / HM442 family | RTK deploy guide only for 3001/5001/10001 | Deployment UI/entity should be conditional, not model-wide. | High, `DeviceMultimodelHelper.java:426-431`. |
| Model parameter ranges | Cutter height, work speed, and path spacing come from current model metadata, with per-device overrides | HA number entities should use reported/model bounds rather than family defaults. | High, `DeviceMultimodelHelper.java:165-191,357-403`. |
| YUKA MV test display | When `AppConstants.isOpenMN231RTKDisplay`, max path spacing is forced to 255 | This is a test/feature switch, not production hardware evidence. | High, `DeviceMultimodelHelper.java:388-391`; flag default false at `utils/constants/AppConstants.java:295-297`. |

Default external model labels in `getExtMod()` are useful only as fallback
display strings, not equivalence evidence. The helper uses separate defaults
for LUBA 2, LUBA mini, LUBA VP, LUBA LD/LA/VA/HM/ME/MD/MB, YUKA mini/MV/ML/VP,
MN100/MN101, and CM900 (`DeviceMultimodelHelper.java:216-276`).

## Firmware gates

| Feature/gate | Model scope in code | Minimum / behavior | Confidence / exact evidence |
|---|---|---|---|
| “new device version” behavior for legacy LUBA 2 and YUKA | `LUBA_2`, `LUBA_YUKA` | firmware `>= 1.12.0`; firmware code 0 is treated as supported | High conditional, `DeviceUtils.java:1037-1049`; consumed by `DeviceType.isNewDeviceVersionType()` at `DeviceType.java:564-566`. |
| All-area work | LD, VP, CM900, YUKA VP, LUBA mini, YUKA mini types | firmware `>= 1.14.5` **and** model helper true | High conditional, `DeviceUtils.java:1056-1059`. |
| Auto-upgrade | non-pool, non-RTK devices | firmware `>= 1.14.0` | High conditional, `DeviceUtils.java:1061-1065`. |
| Cross point (older numeric gate) | LUBA mini/VP/LD/LA, YUKA mini/VP | integer firmware `>= 114000000` | High conditional, `DeviceUtils.java:1067-1075`. |
| Crossing obstacle point | broader type helper (adds YUKA ML and LUBA VA/HM/ME) | firmware `>= 1.15.0` | High conditional. Type set: `DeviceType.java:690-695`; firmware: `DeviceUtils.java:1077-1080`. Note the two cross-point APIs are not equivalent. |
| Dynamic line on LUBA VA | LUBA VA only | firmware `>= 1.15.3.4422`; other listed dynamic-line models have no threshold in this helper | High conditional, `DeviceType.java:697-699`. |
| FPV down-conversion | new-device subsets | generally `>= 1.13.0`, but zero-version fail-open differs by model; LD/MD/MB/LA require a nonzero version before comparison | High conditional, `DeviceUtils.java:1082-1108`. |
| iNavi firmware | model-specific | YUKA mini/ML/MN101/MV/MN100/MB `>=1.12.950`; YUKA VP `>=1.12.930`; LUBA mini `>=1.12.960`; LUBA VP/VA/HM `>=1.12.940`; LD/MD/MB/LA `>=1.12.970`; version 0 returns true | High conditional, `DeviceUtils.java:1119-1151`. |
| Manual mapping area | X5 type group | firmware `>= 1.15.0` | High conditional, `DeviceUtils.java:1154-1157`; X5 membership: `DeviceType.java:793-795`. |
| New network single-switch | all `isNewDeviceType()` types, otherwise firmware check | new types return true directly; others require `>=1.13.0` | High conditional, `DeviceUtils.java:1159-1171`. |
| Point-cloud version | caller-selected device (model helper separately says LD only) | firmware `>=1.14.0`; version 0 fail-open | High conditional, `DeviceUtils.java:1173-1180`; type helper: `DeviceType.java:729-731`. |
| Router box | caller-selected device | firmware `>=1.14.0` | High conditional, `DeviceUtils.java:1182-1185`. |
| Update map version | caller-selected device | firmware `>=1.15.18.2319` | High conditional, `DeviceUtils.java:1187-1190`; model helper separately limits to MV, MN100, MB, CM900 (`DeviceType.java:757-759`). |
| Swimming low-power | any passed device; method does not check pool type | returns supported unless parsed version is below 1.13; empty/odd version behavior can return true | High for literal implementation, `DeviceVersionUtils.java:854-900`. |
| Upload log | any passed device | returns supported unless parsed version is below 1.14 | High for literal implementation, `DeviceVersionUtils.java:902-947`. |
| Job-history time-unit behavior | X5 group except firmware `<1.15.19.2315`, plus PC210 unconditionally | changes minute conversion/report behavior | High conditional call-site evidence, `home/fragment/HomeFragmentNew.java:4615,4653,5149`. |

`DeviceVersionUtils.isLessThanInputVersion(IDevice, String)` is central but
partially decompiled (`DeviceVersionUtils.java:209-217`). The visible code has
special handling for PC210 (`DeviceVersionUtils.java:211`) and obtains versions
from device state/preferences. HA should compare normalized firmware components
only after confirming that its parser matches the app's model-specific format;
do not compare version strings lexicographically.

## Server and runtime feature gates

The server-facing “dynamic settings” data is catalog content grouped by item
code: each `DataBean` contains an `itemCode` and a list of feature records
(`services/entitys/DataBean.java:14-23,68-73`), while each feature record carries
display text/media and an `interactionIdentifier`
(`services/entitys/FeaturesBean.java:17-59,97-124`). This proves that server
responses can change the app's advertised/help/deployment feature set. It does
**not** prove protocol capability.

The service item to app title conversion is explicit at
`bind/device/select_type/SelectDeviceTypeViewModel.java:662-712`. That switch
maps item codes 1301–1324 to product groups but has no LUBA ME entry and shares
the 1313 ambiguity described above. Deployment guidance separately branches on
those item codes at
`device/deploy/device/ui/step1/GuidelineInfoViewModel.java:271-354`.

Runtime state adds further gates:

- `ICarDevice.isSupportRtk()` can depend on `newRtkMode`, subtype preferences,
  or hardcoded per-class behavior (see the instance RTK section).
- `pileIsPile` is required to interpret `SD_PX` as a pile/dock in major UI,
  upgrade, feedback, and pairing flows
  (`device/source/device/interfaces/ICarDevice.java:75-77`;
  `device/info/SwimmingPoolUpgradeAndUpLogHelper.java:261-294`).
- Some UI gates add state-machine conditions after the model helper. For
  example, fill light is hidden in working/returning/drawing states
  (`map/fragment/PlanMapLandFragment.java:1103`), and radar/RTK switching also
  depends on map/hash and signal state
  (`signal/newstatus/SignalConnectionHomepageActivity.java:1199,2396`).
- Firmware version 0 is deliberately fail-open in several helpers, but not all
  (`DeviceUtils.java:1043-1048,1089-1105,1119-1135,1173-1180`). HA should expose
  “unknown” rather than silently copying a model-wide true when firmware is
  absent.

## Major call-site validation

The enum helpers are not dead metadata; representative behavior-changing uses
include:

| Capability | Major call-site behavior | Evidence |
|---|---|---|
| Blade speed | Adds/settings UI and commands only for supported types | `device/setting/fragment/appsetting/DrawerSettingsViewModel.java:405`; `CarSettingDrawerFragment.java:2216`; `CarSettingDrawerActivity.java:1950` |
| Fill light | Controls map and settings visibility with additional state/type exclusions | `map/fragment/PlanMapLandFragment.java:1103`; `CarSettingDrawerFragment.java:2208`; `CarSettingDrawerActivity.java:1932` |
| Radar | Selects radar settings and localization behavior across many screens | `work/setting/view/SettingOptionsView.java:1908`; `CarSettingDrawerFragment.java:954,2843` |
| Point cloud | Selects point-cloud mapping/activity behavior | `map/activity/MapManualVideoActivity.java:458,1447`; `MapManualActivityNew.java:1300` |
| Positioning | Selects map positioning flow | `map/fragment/BaseMapFragment.java:1244` |
| No auto-map | Changes create-map guide/action availability | `map/fragment/PlanMapLandFragment.java:1715`; `map/activity/CreateMapGuideNoBgActivity.java:166,217` |
| No draw-line / no location | Removes draw-line and localization actions | `map/fragment/BaseMapFragment.java:10909,8152`; `map/activity/MapManualActivityNew.java:3643` |
| NRTK | Controls signal/status sections and connection actions | `signal/newstatus/SignalConnectionHomepageActivity.java:1107,1132,2612` |
| Radar self-check | Adds deployment/create-map self-check | `device/deploy/device/ui/creatmapprepare/CreateMapCheckSelfActivity.java:1072,1502,1572` |
| RTK service | Selects RTK setup/deployment guidance | `device/deploy/device/ui/step1/GuideMapInfoActivity.java:65,145`; `DeployGuideLinesHelper.java:154` |
| Camera wiper | Selects alternate settings UI | `CarSettingDrawerFragment.java:948`; `CarSettingDrawerActivity.java:736` |
| Charge-station deploy | Opens deployment after adding a supported device | `home/fragment/HomeFragmentNew.java:1803`; `device/setting/activity/DeviceNetworkActivity.java:522` |
| PC210 | Changes command parsing, work settings, history units, and settings/navigation broadly | `command/app/MACarDataManager.java:2867,3399,3833`; `work/setting/api/WorkingSettingManage.java:198`; `CarSettingDrawerActivity.java:1864-1885` |

## HA capability-gating recommendations

1. **Use a layered identity key.** Retain raw product key, raw device name,
   parsed `DeviceType`, code name, service item, subtype/internal model, and
   `pileIsPile`. Never key capabilities from label or service item alone.
2. **Separate type eligibility from live availability.** Model matrices should
   provide an eligibility state (`yes`/`no`/`conditional`), then firmware,
   subtype, device-reported mode, connectivity, and server state should resolve
   current availability.
3. **Represent unknown explicitly.** Firmware 0, missing subtype metadata, and
   missing server catalog entries should not become false or true by model
   inference. This is especially important for FPV, point cloud, iNavi, and RTK.
4. **Gate commands more strictly than entities.** An entity may be visible for
   diagnostics, but command services should require the exact model helper plus
   firmware/state gate used by the app. Dynamic-line parsing demonstrates that
   these checks protect protocol shape, not just UI.
5. **Keep capability namespaces distinct.** At minimum, distinguish:
   `rtk_accessory_family`, `rtk_service_ui`, `nrtk`, `device_supports_rtk`,
   `radar_rtk_switch`, and `rtk_deploy_guide`. The APK does.
6. **Treat negative helpers literally.** `isNoSupportMapBackup(boolean)` has a
   misleading implementation: when the no-support base helper is true it
   returns false, then may return true for the supplied boolean/X5 condition
   (`DeviceType.java:596-607`). Do not expose it under the same semantic name
   without tests and a renamed HA predicate.
7. **Do not use enum ordering as capability inheritance.** `has4G()` and
   `isLubaPro()` are ordinal/range-style checks
   (`DeviceType.java:484-486,548-550`) and include later unrelated types. Use
   explicit allowlists or reported modem properties.
8. **Model PC210 mower and pile separately.** `SWIMMINGPOOL_SP` is the PC210
   mower family; `SD_PX` is the pile/product-key family, but `pileIsPile` still
   decides presentation and paired-device handling. Expose separate HA devices
   and link them through the pairing relation when available.
9. **Do not infer S1 = SP.** The parser aliases `Spino-S1` into SP in one path,
   while the enum and service catalog keep S1 distinct. Preserve source identity
   and report parser ambiguity.
10. **Derive numeric bounds from model metadata.** Cutter height, work speed,
    path spacing, and area/capacity variants are subtype-specific. Populate HA
    number entity min/max/step from reported values, with defaults only as
    clearly marked fallbacks.

## Confidence summary

- **High:** enum identities; explicit product-key arrays; most type-level
  matrices; pool/RTK family predicates; service item constants; direct major
  call sites.
- **High, conditional:** firmware-gated features, LUBA VA dynamic lines,
  LUBA LD/VA/LA instance RTK, YUKA MV RTK/subtype behavior, SD_PX pile
  interpretation, numeric parameter bounds.
- **Medium:** generic Spino reachability, `YUKA_MINI2` product-key identity, and
  behavior relying on the partially decompiled
  `isLessThanInputVersion(IDevice, String)`.
- **Unknown / intentionally not inferred:** LUBA ME service item code, a unique
  Spino SP service item code, physical modem presence from `has4G()`, and
  hardware equivalence between any models that merely share labels, icons,
  code names, product-key families, or service items.
