# Account, sharing, cloud user services, notifications, voice, and support

## Scope and confidence

This catalog covers the decompiled Mammotion Android application at
`/Users/mattjoslin/mammotion-apk-decompile/src` (manifest version `2.3.8.19`,
version code `247`). It focuses on `login/`, `me/`, `device/share/`, `message/`,
`feedback/`, `hybrid/`, `rn/`, `services/school/`, and the account/cloud-facing
parts of `home/`, plus relevant resources and manifest declarations.

Evidence is static and decompiler-derived. A Retrofit annotation is strong
evidence for an HTTP method and relative route, but the final host is sometimes
selected dynamically through `HttpConstants` and `RxHttpUtils`; routes without
a leading slash may also be mounted below a base path. Old and new service
generations coexist. Server-side authorization, entitlement, expiry, and
regional policy cannot be proven from the client alone.

No credentials, embedded signing material, access tokens, OAuth client values,
or push-provider secrets are reproduced here. Symbolic constant names are
retained where useful.

## Executive catalog

| Area | Findings | Gates / uncertainty | Home Assistant relevance |
|---|---|---|---|
| Authentication | Email/password registration and login, email verification, password recovery, password change, Google/third-party sign-in, third-account linking/unlinking, regional selection, agreements, and token-backed user state. | Region and environment choose service hosts; third-party availability depends on Google services/provider support. Phone UI strings exist, but reviewed current APIs are email-centric. | **Foundational.** A cloud HA integration needs a supported token flow, refresh/logout behavior, regional endpoint discovery, and careful secret storage. |
| Account lifecycle/security | Profile retrieval/editing, avatar/name/country, email change, password change, logout, account cancellation/deletion with email verification, third-party unbinding, and account-disabled/abnormal-region handling. | Multiple old/new logout and deletion routes coexist. Server preconditions are not fully visible. | Medium. Useful for diagnostics and reauthentication; destructive account operations should not be exposed as HA services. |
| Preferences | App language, follow-system mode, length/area unit, account country/region, dynamic resource language, and app version. | Language and measurement preferences are primarily app-local; device/cloud propagation varies. | Low–medium. Unit preference can inform display but HA should use HA’s unit system. |
| Device sharing | Share by Mammotion account/email or QR, owner/shared-recipient lists, accept/reject, confirmation, cancellation, QR scan bind, record cleanup, owner lookup, pagination, and legacy Ali IoT compatibility. | Owner versus authorized recipient is explicit; no fine-grained per-feature permission matrix was found. Model/backend split (`isMaIot`) affects flow. | **High.** Device-list ownership flags must be honored; shared devices may have reduced capabilities and must never be treated as owned. |
| Notifications | FCM token synchronization, Firebase Messaging service, Alibaba/Huawei/Honor push components, Android notification permission/channel, in-app banners, persisted device messages, and authenticated SSE with reconnect/backoff. | Push provider varies by platform/region. SSE URL is dynamically composed. Message-center UI is largely React Native. | **High.** SSE could reduce polling and expose cloud events, but push tokens are app-instance-specific and unsuitable for reuse. |
| Feedback/support | Fault/feedback tickets, device selection, photos/videos, optional and mandatory logs, device log-path discovery, progress polling, upload-server discovery, archive/upload tasks, cancellation, and 4G upload support. Customer-service navigation and Zoho web support are present. | Upload URLs and required fields are server-provided; log collection is device/model/network dependent. | Medium for diagnostics. HA should provide redacted diagnostics, not imitate opaque app log uploads without explicit user action. |
| Voice assistants | Alexa skill status/link/enable/disable and app-to-app/deep-link flow; Google Home sign-in/OAuth authorization and supported-voice content; MA Voice linking route. | Login, account, region, product compatibility, and external app availability gates apply. | Low for direct implementation. The cloud’s voice-account linkage confirms a server-side command surface but does not document it. |
| Hybrid/web | AgentWeb-based web surfaces, configurable academy/support URLs, privacy/agreement/update-history pages, Zoho web support, and a native JS bridge for navigation, language, device status, settings, and experimental features. | Remote content can change independently of the APK. Some constants are test/legacy residue. | Low for control; useful for support/documentation links and for identifying remote feature flags. |
| React Native | Message center/details/share acceptance/customer service, mowing reports, work settings, battery manager, pool plans, guides, native map views, localization/user info, analytics/log bridge, and hotfix bundle/version checks. | RN bundle/version is cloud-selected and may contain behavior absent from decompiled Java. | Medium. Native bridge names reveal feature boundaries, but not all JS-side routes or API schemas. |
| Academy/help/content | Product-filtered academy (“school”) web pages, full-screen and floating web view, orientation preservation, and JS events for menu visibility/product key. | Base URL can be remotely overridden; product keys observed are `luba1`, `luba2`, and `yuka`. | Low; documentation links only. |
| Subscriptions/commerce | No native Play Billing client or in-app payment processor was found. The app does request server-generated Shopify goods links for SIM/4G and iNavi purchase or renewal and opens the returned commerce destination. | Product/service eligibility, account, region, and backend response gate the flow. Remote web/RN content can add further commerce independently of the APK. | Low for HA control. Paid entitlement state may matter diagnostically, but HA should not reproduce checkout. |
| Server-driven tips and campaigns | Fetches product-key-gated tip resources, reports display, and marks pushed tips read. Runtime routing selects which campaign/tip to show. | Backend campaign targeting, product key, account, locale, and app version. | Low for HA; useful evidence that shipped UX and feature promotion can change server-side. |

## 1. Login, authentication, agreements, and regions

### User-visible flow

The manifest declares dedicated activities for regional selection, login method,
email login, registration authorization code, password setup, account linking,
third-party authentication, bind-email, forgot-password verification, password
reset, old-password reset, and deletion
(`resources/AndroidManifest.xml:294-355`). The UI therefore supports:

| Capability | Static evidence | Notes / gates |
|---|---|---|
| Region selection | `RegionalSelectionActivity`; `GET area`; country/region resources and “Popular Country/Region” strings. | Region is selected before/around authentication and is used by dynamic host configuration. Account-region mismatch has a dedicated support error. |
| Email registration | Send registration code, verify code, create user, then set password. | Current routes are email-specific. |
| Email/password login | Encrypted `POST /oauth/token` query flow returning `LoginByEmailResponse`; older signed `POST token` also remains. | Encryption/signature headers are symbolic here; values are intentionally omitted. |
| Third-party login | `AccountThirdActivity`, `LoginThirdManager`, `GoogleLoginManager`, `POST oauth/token/third`. | Google services and external identity-provider availability apply. |
| Link account | `LinkingAccountsActivity`, `POST /v1/third/user/link`, bind-email activity. | Requires an authenticated access token. |
| Unlink identity | `DELETE user/type`. | Authenticated; provider/type is supplied as query data. |
| Agreement | `GET/POST v2/user/agreement`, status and record submission APIs, privacy/user-agreement web pages. | Agreement version/content is server-controlled and likely region/language dependent. |
| User state | `UserStateImpl` exposes `isLogin`, token, user ID, account, and email from persisted state. | This is app-local state backed by shared preferences/database, not proof of server token validity. |

### Login and lifecycle routes

| Method and route | Client method | Purpose | Auth / confidence |
|---|---|---|---|
| `POST /oauth/token` | `loginByEmail` | Current email/password token exchange. | Special encryption/version headers; high confidence in relative route. |
| `POST token` | `login` | Legacy signed token exchange. | Timestamp/signature/app-key headers; likely legacy service host. |
| `POST oauth/token/third` | `thirdLogin` | Third-party identity token exchange. | Signed headers; legacy/dynamic host. |
| `POST v2/email/register/code` | `registerAuthByEmail` | Send registration email code. | No bearer shown. |
| `POST register/verify` | `registerVerify` | Legacy code verification. | Legacy host/path. |
| `POST v2/email/register` | `registerUser` | Create email account. | Registration body. |
| `POST /v2/email/forgot-pwd/code` | `forgotPWDByEmail` | Send recovery code. | No bearer shown. |
| `POST /v2/email/forgot-pwd/verify` | `verifyCodeAndEmail` | Verify recovery code/email. | No bearer shown. |
| `PUT /v2/email/password` | `getFindPwSetPwConstract` | Set recovered password. | Recovery contract/body. |
| `PUT /v1/user/password` | `resetPwdByOld` | Change password using old password. | Bearer required. |
| `POST /v2/email/reset/code` | `resetEmail` | Send/verify email-change code stage. | Bearer required. |
| `PUT /v2/email` | `modifyEmail` | Change account email. | Bearer required. |
| `POST /v1/third/user/link` | `linkThirdAccount` | Link a third-party account. | Bearer required. |
| `DELETE user/type` | `unbind` | Unbind a third-party identity type. | Bearer plus query map. |
| `POST /v3/user/logout` | `logout`; `loginOut2` | Current logout. | Bearer required. |
| `POST v2/user/logout` | `loginOut` | Older logout. | Bearer required. |
| `GET /v2/email/logout/mail` | `sendLogoutEmailCode` | Send account-cancellation verification email. | Bearer required. |
| `POST /v2/user/verify/logout` | `confirmLogoutWithCode` | Confirm cancellation/deletion with code. | Bearer required. |
| `POST v1/user/quit` | `quit` | Older account-quit/delete operation. | Bearer required. |
| `GET/POST logout/mail`, `verify/logout` | `logoutMail`, `verifyLogout` | Legacy cancellation verification. | Bearer required; host supplies versioning. |
| `GET area` | `getAre` | Region/area metadata. | Dynamic Mammotion host. |
| `POST v2/user/agreement` | `getAgreement` | Agreement content/state despite method name. | No explicit bearer in interface. |

Evidence: `sources/com/agilexrobotics/login/api/LoginApiService.java:25-100`,
`login/api/LoginApiUtils.java:16-44`, and
`sources/com/agilexrobotics/me/api/MeApiService.java:25-92`.

### Region and account gates

- `RegionalSelectionActivity` and the `area` endpoint establish explicit region
  selection. `HttpConstants` contains multiple base URLs and runtime-derived
  URLs, so a cloud client cannot safely assume one global host.
- Resource strings distinguish a disabled account from an “abnormal region”
  account requiring after-sales verification
  (`resources/res/values/strings.xml:954-956`).
- Agreements and dynamic resources are fetched separately. Region, locale, app
  version, and resource type are therefore likely inputs to server-selected
  content, though the exact server policy is not visible.
- Phone registration/recovery strings remain in resources, but the reviewed
  active `LoginApiService` exposes email registration and recovery. Phone auth
  should be cataloged as legacy/possible, not confirmed current behavior.

### Authentication implications for HA

A legitimate HA cloud login should implement region discovery/selection,
email/password or a documented supported authorization grant, token expiration
and reauthentication, and account-disabled errors. It should not duplicate the
APK’s proprietary request signing/encryption by extracting embedded client
material. Third-party interactive sign-in is a poor fit for unattended HA.

## 2. Account profile, security, preferences, and lifecycle

| Feature | Evidence and behavior | Gate / uncertainty | HA relevance |
|---|---|---|---|
| User profile | `GET v1/user/user-info`; `AccountCenterActivity` binds avatar, account identity, country, and third-party sets. | Login required. Response fields beyond client DTOs may evolve. | Diagnostic only. |
| Avatar | Local image chooser/crop flow and profile update view model. | Media permission/provider behavior varies by Android version. Upload route is obscured outside the small Retrofit interface. | None. |
| Display name | `InputNameDialogCommon` and account-center profile update. | Server validation/length rules are resource-driven. | None. |
| Country/region | Profile displays country name; region metadata comes from `area`. | Changing display country may not migrate account region. | Important diagnostic distinction. |
| Email | Display, bind email, change email with verification. | Third-party-only accounts may require binding email first. | Used as account identifier; redact in diagnostics. |
| Password | Forgot/reset and authenticated old-password change. | Third-party-only accounts may not initially have a password. | Reauthentication only. |
| Linked identities | Third-party list, Google-login detection, link/unlink. | Provider and region gated. | None. |
| Logout | Multiple v2/v3 logout routes clear cloud and local state. | Route generation depends on client/service generation. | Integration must clear stored tokens on logout/401. |
| Account deletion | Separate cancellation/close/delete screens; email code; relinquishes assets/virtual rights, requires no active services/requests, unbinds robot and third parties. | Server enforces conditions; strings are policy UI, not proof of immediate deletion. | Never expose as a routine HA action. |
| Language | `UserLanguageActivity`, `LanguageSettingsAdapter`, follow-system key `sys`; dynamic resources and RN navigator carry language. | Primarily app presentation state. | Map to HA locale only for text/content requests. |
| Units | Shared preference `"length unit"` uses `1` for metric and `2` for imperial; area renders `㎡` or `ft²`. | App-local display preference; device payload units can differ. | Convert values through HA’s unit system. |
| App version | `AccountAppVersionActivity`; app-version adaptation and RN version APIs. | Update availability is region/version controlled. | Useful diagnostic metadata. |
| Forum/service email | `POST user-server/v1/email/forum`; `POST email/requirement`. | Purpose and server-side consent workflow are not fully decoded. | None. |

Account deletion policy text is unusually explicit:
`resources/res/values/strings.xml:1141-1144`. Profile and security activity
declarations are at `resources/AndroidManifest.xml:359-438`.

## 3. Device sharing, ownership, roles, and permissions

### Sharing model

The client has two user-facing creation paths: **share via account** and
**share via QR code** (`resources/res/values/strings.xml:506,732`). The owner can
open a manager list, create shares, inspect recipient email/account, and cancel
records. A recipient can accept or reject an incoming share. QR records carry a
`qrKey`, device name/type, record ID, IoT ID, and an `isMaIot` backend flag
(`device/share/api/ShareDeviceBean.java:11-121`).

The observed role model is:

| Role/state | Evidence | Effective meaning |
|---|---|---|
| Owner | Device list uses `isOwned`; QR dialog displays owner; owner-info API exists. | Bound account with management authority, including creating/canceling shares. |
| Authorized/shared user | Device manager labels non-owned devices “authorized”; receiver-shared collections and accept/reject flows exist. | Can see/use a device after accepted share, subject to client/server restrictions. |
| Pending recipient | Share records and message-center `handleShare(isAgree, data)` support invitation response. | Not active until confirmation/acceptance. |
| Rejected/canceled | Accept/reject and cancel APIs, plus record cleanup. | Share ceases or never becomes active. |

No explicit named roles such as administrator/operator/viewer, nor a per-command
permission bitset, were found in the reviewed share entities or UI. Ownership is
the primary authorization distinction. Actual command restrictions may still be
enforced by cloud/firmware and must be tested.

### Current share APIs

| Method and route | Client operation | Main inputs / result |
|---|---|---|
| `POST /user-server/v1/share/device/specify/user` | `accountSharing` | Share specified devices with account/user. |
| `POST /user-server/v1/share/device/qr/create` | `createQRCode` | Create a QR share key/record. |
| `POST /user-server/v1/share/device/qr/scanQrBind` | `qrCodeBinding` | Recipient scans QR and binds/accepts. |
| `POST /user-server/v1/share/device/confirm` | `confirmShare` | Confirm/accept or finalize a share. |
| `POST /user-server/v1/share/device/cancel` | `cancelShareDevice` | Cancel one or more device shares. |
| `POST /user-server/v1/share/device/page` | `getShareDevicePage` | Paginated share records. |
| `POST /device-server/v1/device/bind/owned-info` | `bindOwnedinfo` | Query owner/bind information. |

### Legacy/auxiliary share APIs

| Route | Purpose / caveat |
|---|---|
| `POST share-record`, `share-record/list`, `share-record/clean` | Legacy record save/list/cleanup. |
| `GET account-email` | Resolve account/email for sharing. |
| `POST ali-identify` | Resolve Ali identity IDs; evidence of older Ali IoT integration. |
| `POST notice` | Send legacy share notice. |

Evidence:
`sources/com/agilexrobotics/device/share/api/DeviceShareApiService.java:27-83`,
`DeviceShareApiUtils.java:17-30`,
`DeviceShareOperator.java:15-55`, and
`viewmodel/DeviceSharedModel.java`.

### Model, region, and account gates

- All practical share operations require an authenticated account and a cloud
  device binding.
- Creation/management is owner-facing. `DeviceMangementAdapter` explicitly
  distinguishes owned from authorized devices
  (`sources/com/agilexrobotics/me/DeviceMangementAdapter.java:26-39`).
- `isMaIot` branches separate older Ali IoT devices from the newer Mammotion
  user/device-server flow. Legacy records use IoT IDs and provider identity
  mapping.
- Shares can contain several device records and are paginated. Static code does
  not prove maximum recipients, QR expiry, same-region requirements, or whether
  cross-region account lookup is allowed.

### Sharing implications for HA

The HA integration should retain ownership/share metadata on each device and
default shared devices to the least privilege observed. It should not infer that
successful list access grants owner-only operations such as unbinding, sharing,
firmware changes, anti-theft administration, or cloud-account management.
Invitation acceptance is a user/account action and should remain outside normal
entity control.

## 4. Notifications, messages, push, and SSE

### Transport catalog

| Transport/UI | Static evidence | Behavior |
|---|---|---|
| Firebase Cloud Messaging | `MyFirebaseMessagingService`; FCM receiver and metadata; `POST_NOTIFICATIONS`; `com.google.android.c2dm.permission.RECEIVE`. | Receives remote pushes; Android 13+ runtime permission applies. |
| Token registration | Home API `POST client/fb` with `SyncFMCTokenBean`. | Associates an app-instance FCM token with the authenticated account/cloud. |
| Alibaba push | `AgooFirebaseMessagingService` and Alibaba ACCS/Agoo resources/components. | Likely China/Alibaba delivery path or FCM-to-Agoo bridge. |
| Huawei/Honor push | Manifest services with Huawei and Honor messaging actions. | Vendor-specific delivery when Google services are unavailable. |
| SSE | `SseClient`, `AppSseBootstrapKt`, `SseConfig`, reconnect policy, application scope. URL constant ends in `channel-server/v1/sse/connect`. | Authenticated foreground/application event stream. |
| In-app banner | `GlobalNotificationManager` inflates `layout_global_notification` for device error events and routes clicks. | Immediate device error/status banner with auto-dismiss. |
| Message center | Native shell/resources plus RN `MessageCenterModule`; progress/error/maintenance item types and history/activity screens. | Paginated per-device messages, details, interaction buttons, share invitations, support navigation. |

Manifest evidence:
`resources/AndroidManifest.xml:42,108,224-250,895-900,1171-1219,1480-1488`.
Token API evidence: `sources/com/agilexrobotics/home/api/HomeApiService.java`
(`syncFMCToken`, `POST client/fb`).

### SSE protocol behavior

`HttpConstants` composes the SSE target as the selected base plus
`channel-server/v1/sse/connect`
(`sources/com/agilexrobotics/utils/HttpConstants.java:206`). The client:

- creates a **POST** request with an empty JSON body;
- sends `Authorization`, `Accept: text/event-stream`, and
  `Cache-Control: no-cache`;
- exposes connection state as `StateFlow` and events as `SharedFlow`;
- emits event ID/type/data in `SseEvent`;
- avoids starting a second connection while connected/connecting;
- stops/cancels the active event source on termination;
- reconnects with exponential delay from 1 second up to 60 seconds, 20% jitter,
  and effectively unlimited retries by default;
- treats a configured set of HTTP codes as terminal; exact default terminal
  codes should be verified from the complete constructor/runtime;
- contains special handling for invalid content type, indicating non-SSE server
  responses are recognized as failures.

Evidence:
`message/sse/SseClient.java:188-305`,
`AppSseBootstrapKt.java:1-225`,
`ReconnectPolicy.java:17-130`, and `SseConfig.java`.

The event consumer is supplied by the application, so the generic SSE package
does not itself define all event schemas. Message details and database mapping
are split across RN/native/common modules. Event payload JSON should therefore
be treated as versioned and opaque until runtime samples or the receiving
application callback are traced.

### Message-center behavior

`MessageCenterModule` exposes React Native calls to:

- fetch a paginated list by device name;
- update/read message status;
- open detail/error/detail variants;
- process activity-detail interaction identifiers and parameters;
- accept/reject device sharing;
- navigate to customer service;
- close the message center.

Native resources distinguish progress, error, and maintenance rows
(`message/R.java:44-48`). `GlobalNotificationManager` uses stored
`DeviceMessageDB` content and an `ErrorCodeEvent`, meaning at least some device
events are persisted and rendered independently of Android system pushes
(`message/GlobalNotificationManager.java:41-290`).

### Notification implications for HA

SSE is the most promising cloud-event transport for HA, provided login/region
and event schemas are resolved. An HA implementation should maintain one stream
per account/region, use bounded exponential backoff, avoid logging payloads that
contain account/device identifiers, and fall back to polling. FCM/Huawei/Honor
registration should not be emulated: those tokens belong to mobile push
providers and app instances, not general API clients.

## 5. Feedback, support, and log upload

### User workflow

The feedback flow supports selecting a device, selecting a feedback/fault type,
entering details, adding photos and videos, including app/device logs, choosing
optional logs, monitoring upload progress, and canceling an upload. A background
`DeviceLogService` polls progress. Separate tasks handle images, video, logs,
archive creation, and upload callbacks
(`feedback/fragment/Feedback1Fragment.java`,
`FeedbackAPPLogFragment.java`, `FeedbackDeviceSelectFragment.java`,
`feedback/task/UploadImgTask.java`, `UploadVideoTask.java`,
`UploadLogTask.java`, and `feedback/DeviceLogService.java:1-250`).

| Route | Purpose | Notes |
|---|---|---|
| `POST fault/report` | Submit fault report. | Legacy task host. |
| `POST feedback` | Submit feedback. | Appears against more than one legacy base URL. |
| `POST device-server/v2/feedback` | Current optional feedback submission. | Authenticated Mammotion device server. |
| `POST device-server/v2/feedback/log-path` | Discover device report log paths. | Takes device/log selection request. |
| `GET log-server` | Obtain upload server/location metadata. | Query map; response is `LogServerUrlBean`. |
| `POST issue-instruction` | Request/coordinate device log upload. | Returns upload response metadata. |
| `POST logProgress` | Poll device log collection/upload progress. | Used by service/timer. |
| `POST videoInfo` | Save uploaded video metadata. | Legacy flow. |
| `POST support-4G-upload` | Check/activate supported cellular upload flow. | Name/response types are inconsistent due to decompilation; server semantics uncertain. |
| Upload helper paths | Upload one or multiple files/images. | Final URL comes from `HttpConstants` or server response and is intentionally not expanded here. |

Evidence:
`sources/com/agilexrobotics/feedback/api/FeedbackApiService.java:25-63` and
`FeedbackApiUtils.java:17-71`.

### Data and privacy considerations

Feedback DTOs include device/log-path metadata, report content, upload request
information, and attached media. Device logs can contain serials, IoT IDs,
network state, positions, task history, and cloud errors. The app also offers
“all logs” and mandatory/optional log selections. Any HA diagnostic export
inspired by this flow should:

- require an explicit user action;
- redact access/refresh tokens, account/email, precise location, Wi-Fi
  credentials, push tokens, and signed upload URLs;
- show exactly which files/fields will be uploaded;
- use bounded retention and cancellation;
- avoid uploading to undocumented Mammotion endpoints unless the user is
  actively opening a support case.

### Support surfaces

`MessageCenterModule.navigateToCustomerService()` provides a native-to-support
route. Hybrid activities include `ZohoWebActivity`; school/support pages use
AgentWeb. `WebConstant` includes privacy, agreement, warranty/after-sales, update
history, and academy URLs
(`hybrid/agentweb/WebConstant.java:8-29`). The warranty constant points to a
US after-sales page, but runtime dynamic resources may provide region-specific
content.

## 6. Alexa, Google Home, and MA Voice

| Assistant | Feature evidence | Cloud/native operations | Gates and uncertainty |
|---|---|---|---|
| Amazon Alexa | `AlexaActivity`, account-link DTO, presenter/model, “Link Alexa”/unbind/status UI, supported voice list, Amazon app package queries. | `POST voice-server/v1/alexa/skill/enablement/status`, `/account/link`, `/skill/disable`; legacy `POST enablement`; app-to-app URL with web fallback. | Requires Mammotion login; product name/compatibility shown; Amazon app or browser fallback; region/skill availability applies. |
| Google Home | `GoogleAgreeActivity`/fragment, Google Sign-In, `GoogleViewModel`, OAuth/link model, “Link Google Home,” supported voice content. | `GET supported/voices`, `GET id`, `POST authorization/code`, `POST oauth`. | Requires Google Play services/sign-in and Mammotion account authorization. Exact OAuth host/path mounting is dynamic. |
| MA Voice | `MAVoiceLinkActivity`, manifest deep link from Alexa app, shared OAuth/link infrastructure. | Deep-link/app-link completion and account association. | “MA Voice” semantics are not fully decoded; likely Mammotion’s common voice-link landing flow rather than a separate speech engine. |

### Deep links

The manifest exposes:

- `https://application.mammotion.com/voice/alexa` (and an HTTP variant) to
  `AlexaActivity`;
- `https://application.mammotion.com/voice/linkfromalexaapp` (and HTTP) to
  `MAVoiceLinkActivity`.

Evidence: `resources/AndroidManifest.xml:380-426`. These are callback/navigation
links, not credentials and not proof that arbitrary commands can be issued
through them.

### Alexa behavior

`AlexaActivity` checks account-link status, requests an app-to-app Alexa URL and
fallback URL, enables/finalizes linking from callback query parameters, and
disables/unbinds the skill after confirmation. It displays supported utterances
returned with status/content and model compatibility text
(`me/activity/AlexaActivity.java:65-240`).

### Google behavior

`GoogleAgreeActivity` launches Google sign-in, receives a
`GoogleSignInAccount`, requests an authorization code through the Mammotion
cloud, and provides Google Home, Alexa, and privacy navigation
(`me/activity/GoogleAgreeActivity.java:80-258`). `MeApiService`’s generic
`id`, `authorization/code`, and `oauth` routes appear to implement the account
link grant (`me/api/MeApiService.java:47-68`).

### Voice-assistant implications for HA

Voice integration confirms Mammotion has a server-side account-link and command
adapter, but it does not expose the assistant command schema or authorize HA to
reuse assistant OAuth clients. It is better treated as architectural evidence
than an integration API. HA should use the normal Mammotion account/device APIs.

## 7. Hybrid/web features and native bridge

### Web surfaces

| Surface | Evidence / purpose |
|---|---|
| Generic `WebActivity` / `CommonActivity` | Remote pages with toolbar/network/error handling. |
| `RenewWebActivity` | Update/renewal-style web content. |
| `ZohoWebActivity` | Zoho-hosted support/academy content. |
| Experimental features | Release/test H5 page under Mammotion’s H5 domain. |
| Privacy and agreements | Mammotion privacy/application agreement and user agreement pages. |
| Device update history | Release/test app-H5 history page selected by environment. |
| Academy | Configurable base, defaulting to a Zoho Sites portal. |
| Warranty/after-sales | Region-specific web constant (US page observed). |

`AgentWebFragment` applies special handling to
`https://application.mammotion.com` URLs
(`hybrid/agentweb/AgentWebFragment.java:130`). `WebActivity` dispatches a
JavaScript `MessageEvent('message', {type:'ready'})` after setup
(`WebActivity.java:127`).

### Native JavaScript bridge

The bridge uses request/response objects with action, callback ID, status code,
message, and JSON data. Registered handlers include:

| Handler | Exposed native capability |
|---|---|
| `AppLangHandler` | Return app language/locale to web content. |
| `DeviceStatusHandler` | Return selected device/status context. |
| `RouterHandler` | Route from H5 to native destinations. |
| `NavigationBarHandler` | Change native navigation-bar presentation. |
| `CloseWebPageHandler` | Close current web surface. |
| `JumpHomePageHandler` | Return to app home. |
| `JumpSyStemSettingHandler` | Open Android system settings. |
| `DropMowHandler` | Enter/coordinate the experimental drop-mow feature. |

Evidence: `sources/com/agilexrobotics/hybrid/bridge/*.java` and
`hybrid/handler/*.java`.

Because web content is remote and mutable, the handlers are a more reliable
catalog of native authority than any one page snapshot. Static review does not
prove origin allowlisting for every activity; any reused bridge should strictly
limit trusted origins and validate action parameters.

## 8. React Native feature container

The manifest declares portrait, landscape, and transform RN containers
(`resources/AndroidManifest.xml:630-641`). `HomeApiService` checks
`POST rn/version/check`; `RNHotfixWorkManager` manages downloaded/hotfix bundle
work. A `NavigatorBean` passes Android platform, app version, language, target
page, and a `DeviceModule`
(`rn/api/bean/NavigatorBean.java:10-81`).

### RN routes/features

`ReactNativeNavigator` defines native entry points for:

- mowing report list and detail;
- work settings;
- message center, message detail, error detail, and landscape message center;
- battery manager;
- pool work plan and pool work settings;
- before-mapping, after-mapping, and welcome guides.

Evidence: `rn/api/ReactNativeNavigator.java:9-35`.

### Native modules exposed to JavaScript

| Module | Confirmed bridge methods / capabilities | HA relevance |
|---|---|---|
| `CommonModule` | Close, localized string lookup (single/all/sync/async), user info, logging/upload flag, vibration, analytics trace. | Reveals RN can access account context; do not expose tokens to frontend code. |
| `MessageCenterModule` | Paginated message list, detail interactions, share accept/reject, support navigation, status changes. | High for understanding cloud events. |
| `MowingReportModule` | Report list/detail/summary, all work-record elements, upgrade popup. | Useful cloud-history evidence. |
| `WorkSettingModule` | Query/send/set work parameters and open native pattern/boundary-distance screens. | Device control is delegated back to native protocol helpers. |
| `BatteryManager` | Base/charge-max/continue-work/recharge data, button state, settings, smart sleep, listeners. | Candidate HA sensors/settings, cataloged more fully elsewhere. |
| `PoolWorkPlanModule` | Add/edit/delete plans and select pool mode. | Pool-model gated. |
| `UserGuideModule` | Route from RN guide pages to native views with IoT ID/page parameters. | Documentation/navigation only. |
| Native map managers | Lawn/pool map rendering and map-element conversion. | Visualization, not a cloud account API. |

Evidence: `sources/com/agilexrobotics/rn/module/*.java`, especially
`CommonModule.java:163-347`, `MessageCenterModule.java:258-550`,
`MowingReportDataModule.java:70-230`, `WorkSettingModule.java:313-575`,
`BatteryManagerModule.java:245-470`, `PoolWorkPlanModule.java:189-232`, and
`UserGuideModule.java:51-70`.

The downloaded JS bundle can change independently from the decompiled classes.
Therefore Java bridge signatures are confirmed, while JS page composition,
remote API calls made directly from JS, and feature rollout flags remain
uncertain.

## 9. School/academy/help/content

`SchoolPageUtils` constructs the academy URL from a remotely persisted
`ACADEMYURL` override or the default Zoho portal. It appends
`?product=<device school type>` and optional `SchoolType` query data, or opens a
specific relative academy URL
(`services/school/SchoolPageUtils.java:25-69`).

`SchoolIndexActivity` is the full-screen entry. `SchoolFloatWindowsUtils` can
retain the academy as a draggable floating AgentWeb window, preserves current
URL across orientation changes, and opens full-screen Zoho web content. Its
JavaScript interface accepts:

- `showOpMenuView` with `1`/`0` to show/hide the operation menu and enable/disable
  dragging;
- `pageCurrentProductKey` with observed values `luba1`, `luba2`, or `yuka`, used
  to preserve product filtering.

Evidence:
`services/school/SchoolFloatWindowsUtils.java:1-260` and
`SchoolToolImpl.java`. The manifest entry is
`resources/AndroidManifest.xml:619`.

This is remote help/training content, not a device-control API. Availability and
content are model, language, region, and remote-configuration dependent.

## 10. Relevant home/cloud APIs

These home APIs directly intersect account/cloud services:

| Method and route | Purpose | Gate |
|---|---|---|
| `GET /device-server/v1/device/list` | Current account device list. | Auth injected by common client; region/base URL selected dynamically. |
| `GET /device-server/v1/product/product/list` | Product metadata/model catalog. | Cloud/region. |
| `POST /device-server/v1/device/function` | Query function support for a device. | Model/account access. |
| `POST /device-server/v1/device/nickname` | Rename device. | Ownership/authorization likely server-enforced. |
| `POST /device-server/v1/device/setting` | Update cloud device setting (including auto-upgrade usage in view model). | Device/model/role. |
| `POST client/fb` | Synchronize FCM token. | Logged-in mobile app instance. |
| `POST item/resource` | Dynamic localized/configured resources. | Locale/region/version/request type. |
| `GET user/group` | Developer/user group lookup. | Authenticated; likely rollout/test cohort. |
| `POST rn/version/check` | Select/check RN bundle version. | App/device/version/region. |
| `GET/POST v2/user/agreement/status`, `v1/user/agreement/record` | Agreement state and acceptance record. | Account/app version. |
| `POST iot/sim/detail` | SIM/4G detail. | Cellular-capable model and account. |
| `POST app/version/adaptation` | App-update/adaptation check. | App version/region. |
| `POST device/wakeup` | Cloud wake-up request. | Device/model/role/network. |

Evidence: `sources/com/agilexrobotics/home/api/HomeApiService.java:25-134`.

## 11. Subscriptions and commerce

No Mammotion-owned Play Billing client, native payment processor, purchase
acknowledgement, or Play-managed subscription flow was found in the scoped
packages. The app does, however, expose server-directed commerce for connected
services: `GET /device-server/v1/shopify/goods/link` returns a goods link used by
the 4G/SIM and iNavi detail screens for purchase or renewal
(`sources/com/agilexrobotics/device/info/api/DeviceInfoApiService.java:76`;
`sources/com/agilexrobotics/signal/newstatus/StatusBarExtend4gActivity.java:403`;
`sources/com/agilexrobotics/signal/activity/INaviDetailActivity.java:138`).

Other searches for subscription/commerce/payment terms mostly matched:

- reactive-programming `subscribe(...)` calls;
- pagination `Order` DTOs in share records;
- generic third-party/library resources;
- device-service allowance concepts outside an actual purchase path.

The defensible conclusion is **no native Play Billing/payment implementation**,
not “no commerce.” Shopify-backed service purchase/renewal is confirmed, and
remote H5/RN content can add more commercial behavior without a native billing
bridge.

## 11.1 Server-driven tips and campaign tracking

The service tab fetches remotely selected product resources and separately
reports that a pushed tip was shown or read:

- `/user-server/v2/tips/resource`
- `/tips/push/show`
- `/tips/push/read`

The runtime service router matches and displays the returned material rather
than relying only on static academy/help content
(`sources/com/agilexrobotics/services/apis/ServiceTabApi.java:43,52,57`;
`sources/com/agilexrobotics/services/RouterSMServiceImp.java:413`).

## 11.2 First-party behavioral telemetry

In addition to Firebase telemetry, Mammotion queues and uploads its own
“buried-point”/behavioral events to `/user-server/v1/user/collection`. The
payload includes app, device, product and phone identifiers, event ID/value,
time, and area
(`sources/com/agilexrobotics/base_module/trace/ma/TraceApiService.java:14`;
`sources/com/agilexrobotics/base_module/trace/ma/TraceHelper.java:73`;
`sources/com/agilexrobotics/base_module/trace/ma/DataCollectBeans.java:5`).
This is privacy-relevant evidence, not an HA feature to reproduce.

## 12. Consolidated gates and HA design guidance

| Gate | Where observed | Design consequence |
|---|---|---|
| Account login/token | Nearly all user/device/share/support APIs. | Treat 401/disabled/region mismatch as reauth or support errors, not device offline. |
| Region/base URL | Regional selection, `area`, dynamic `HttpConstants`. | Store region with config entry; do not hard-code one global endpoint. |
| Ownership vs sharing | `isOwned`, authorized label, owner-info/share APIs. | Surface role and suppress owner-only operations for shared devices. |
| Backend generation | New `/user-server` and `/device-server` routes coexist with Ali/legacy routes; `isMaIot`. | Detect device/account backend instead of mixing routes opportunistically. |
| Model/product | Voice compatibility, academy product query, pool/mower RN routes, function-support API. | Build entities from reported capabilities, not model-name guesses alone. |
| Firmware/version | Dynamic resources, app adaptation, RN version, product function endpoint. | Feature discovery must be dynamic and cached with version metadata. |
| Locale/units | App language, dynamic resources, RN navigator, length unit. | Use locale for content; normalize telemetry to HA units. |
| Mobile platform/provider | Google sign-in, FCM, Alibaba/Huawei/Honor push, external Alexa app. | Do not make mobile push or interactive assistant apps prerequisites for HA. |
| Network/service state | SSE, cloud APIs, 4G log upload, web content. | Distinguish cloud unavailable, device offline, and permission denied. |

Priority for HA:

1. region-aware authentication and device listing;
2. explicit owner/shared role handling;
3. SSE event decoding with polling fallback;
4. redacted diagnostics and support metadata;
5. optional cloud history/message entities after schemas are validated.

Avoid exposing account deletion, third-party unlinking, share creation, push
token registration, or support log upload as ordinary HA services.

## 13. Uncertainties and negative findings

- Final hosts for many relative routes are runtime-selected. This catalog does
  not concatenate or publish sensitive/configuration-bearing constants.
- Several methods are partially decompiled or have generic `Map` bodies, so
  field names and enum meanings are not always recoverable.
- The client contains concurrent legacy and current APIs. Presence does not
  prove every route is reachable in this build for every region.
- Phone-auth UI resources exist, but the reviewed current API is email-centric.
- No fine-grained share permission matrix was found; server-side command
  authorization may be stricter than the client UI.
- SSE payload schemas and application-level event routing are not fully defined
  in the generic message package.
- RN Java bridge methods are visible, but downloaded JavaScript can add or
  remove page-level behavior.
- Remote H5/Zoho content can change without an APK release.
- No native Mammotion Play Billing or payment-processing flow was found;
  server-generated Shopify purchase/renewal links are present for SIM/iNavi
  in scope.
- Alexa/Google account linking is confirmed; assistant command vocabulary and
  server adapter implementation are not.

## 14. Files reviewed

The review used package-wide searches and then inspected implementation/API
files. Generated binding and `R.java` files were used only to corroborate UI
presence.

### Primary sources

- `resources/AndroidManifest.xml`
- `resources/res/values/strings.xml`, `arrays.xml`, and relevant layout/resource
  names
- `sources/com/agilexrobotics/utils/HttpConstants.java`
- `sources/com/agilexrobotics/login/api/LoginApiService.java`,
  `LoginApiUtils.java`, `RequestApi.java`, `UserManager.java`,
  `UserInfoResp.java`, `ThirdPartSet.java`, and `DynamicSettingsReq.java`
- `sources/com/agilexrobotics/login/activity/*.java`,
  `login/ui/activity/*.java`, `login/manage/*.java`,
  `login/viewmodule/*.java`, and login request/response DTOs
- `sources/com/agilexrobotics/me/api/MeApiService.java`,
  `me/api/model/*.java`, `me/api/bean/*.java`
- `sources/com/agilexrobotics/me/activity/AccountCenterActivity.java`,
  `AccountSecurityActivity.java`, `AccountCancellationActivity.java`,
  `CloseAccountActivity.java`, `UserInfoActivity.java`,
  `UserLanguageActivity.java`, `AlexaActivity.java`,
  `GoogleAgreeActivity.java`, `MAVoiceLinkActivity.java`, and `AuthActivity.java`
- `sources/com/agilexrobotics/me/viewmodel/*.java`,
  `me/presenter/AlexaPresenter.java`, `me/fragment/AccountFragment.java`,
  `me/LanguageSettingsAdapter.java`, and `me/DeviceMangementAdapter.java`
- `sources/com/agilexrobotics/device/share/api/*.java`,
  `device/share/entity/*.java`, `device/share/viewmodel/DeviceSharedModel.java`,
  and `device/share/activity/*.java`
- `sources/com/agilexrobotics/message/GlobalNotificationManager.java`,
  `message/adapter/DeviceMessageAdapter.java`, `message/api/*.java`, and
  `message/sse/*.java`
- `sources/com/agilexrobotics/feedback/api/*.java`,
  `feedback/api/entity/*.java`, `feedback/model/*.java`,
  `feedback/fragment/*.java`, `feedback/task/*.java`,
  `feedback/DeviceLogService.java`, and `feedback/activity/FeedBackActivity.java`
- `sources/com/agilexrobotics/hybrid/agentweb/*.java`,
  `hybrid/bridge/*.java`, and `hybrid/handler/*.java`
- `sources/com/agilexrobotics/rn/ReactNativeContainer*.java`,
  `RNHotfixWorkManager.java`, `rn/api/*.java`, `rn/api/bean/*.java`,
  `rn/module/*.java`, and `rn/view/manager/*.java`
- `sources/com/agilexrobotics/services/school/*.java`
- `sources/com/agilexrobotics/home/api/HomeApiService.java`,
  `home/api/HomeNavigator.java`, `home/DefaultHomeRepository.java`,
  `home/viewmodel/HomeViewModel.java`, `home/viewmodel/DeviceListViewModel.java`,
  and relevant home fragments/adapters.

### Corroborating cross-package sources

- `sources/com/agilexrobotics/mvp/fieldmower/service/firebase/MyFirebaseMessagingService.java`
- `sources/com/agilexrobotics/base_module/event/SyncFMCTokenBean.java`
- user/device/share/message database and event DTOs referenced by the primary
  implementations
- common navigation, dynamic-resource, locale, shared-preference, and network
  framework helpers called by the scoped packages.
