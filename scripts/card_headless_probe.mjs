// Exercise the real card without a browser.
//
// WHY. Several checks that matter before a supervised run -- what the card
// would DISPLAY, which blockers it would show, and above all the exact payload
// it would SEND -- are properties of the card's own code, not of the browser
// rendering it. Those can be answered headlessly, against live state.
//
// This matters for profile identity. Driving the mower from a hand-built
// payload proves nothing about the card; the whole point of Gate 5 is that the
// CARD demonstrably sent the accepted values. This probe calls the card's own
// `_motionPayload()`, so what it prints is card-generated, not transcribed.
//
// ⚠️ WHAT IT CANNOT COVER, and do not let it stand in for these:
//   * which build a BROWSER actually loaded (the console banner). Server-side
//     the equivalent risk is covered by md5-matching both serving paths and the
//     Lovelace cache key, which the deploy already verifies.
//   * rendering, layout, touch handling -- and therefore the iPhone lockup.
//   * the card's own preview -> dry-run -> Real Go ordering, which is an
//     operator procedure, not a code path.
//
// Usage:
//   node scripts/card_headless_probe.mjs
//   node scripts/card_headless_probe.mjs 8.568,-6.636 8.002,-7.202 7.214,-7.063
import { readFileSync } from "node:fs";

// Same DOM shim the frontend suite uses.
globalThis.HTMLElement = class {
  attachShadow() {
    return { appendChild() {}, querySelector: () => null };
  }
};
globalThis.customElements = {
  _items: new Map(),
  define(n, v) { this._items.set(n, v); },
  get(n) { return this._items.get(n); },
};
globalThis.window = {};
globalThis.localStorage = {
  _i: new Map(),
  getItem(k) { return this._i.get(k) ?? null; },
  setItem(k, v) { this._i.set(k, String(v)); },
  removeItem(k) { this._i.delete(k); },
};

const { MammotionCustomPathCard, LUBA_ACCEPTANCE_PROFILE, CARD_VERSION } =
  await import("../custom_components/mammotion/www/mammotion-custom-path-card.js");

function env(name) {
  const v = process.env[name];
  if (v) return v;
  // .env is the project's convention; read it rather than demand a shell export.
  for (const line of readFileSync(new URL("../.env", import.meta.url), "utf8").split("\n")) {
    const m = line.match(new RegExp(`^${name}=(.*)$`));
    if (m) return m[1].trim();
  }
  throw new Error(`${name} not set`);
}

const HA = env("HA_URL").replace(/\/$/, "");
const TOKEN = env("HA_TOKEN");
const ENTITY = "lawn_mower.back_yard_clip_skywalker";

async function svc(name, body) {
  const r = await fetch(`${HA}/api/services/mammotion/${name}?return_response`, {
    method: "POST",
    headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  const j = await r.json();
  return j.service_response ?? j;
}

const waypoints = process.argv.slice(2).map((s) => {
  const [x, y] = s.split(",").map(Number);
  return { x, y };
});

const runtime = await svc("export_runtime_state", { entity_id: ENTITY });

const card = new MammotionCustomPathCard();
// The dashboard's real options. Kept minimal and explicit: any extra key here
// would be an override the card must report, which is the point of the check.
card._config = { entity: ENTITY, card_height: 520, speed: 0.2, sample_delays: [0, 3] };
card._runtimeState = runtime;
card._validation = { valid: true };
card._waypoints = waypoints;
card._render = () => {};
card._confirmBladesOff = true;
card._confirmClearArea = true;

console.log(`card version (module)   : ${CARD_VERSION}`);
console.log(`profile overrides       : ${JSON.stringify(card._profileOverrides())}`);
console.log(`execution profile row   : ${card._profileLabel()}`);

const pre = card._preflight();
console.log(`\npreflight safe          : ${pre.safe}`);
console.log(`preflight blockers      : ${JSON.stringify(pre.blockers)}`);
const readiness = card._readiness();
console.log(`readiness               : ${readiness.summary ?? "(no summary)"}`);
for (const d of readiness.details ?? []) console.log(`   - ${d}`);

if (waypoints.length) {
  const payload = card._motionPayload(false);
  console.log(`\n=== payload the CARD would send (${waypoints.length} waypoint(s)) ===`);
  console.log(JSON.stringify(payload, null, 1));
  const drift = Object.keys(LUBA_ACCEPTANCE_PROFILE).filter(
    (k) => JSON.stringify(payload?.payload?.[k] ?? payload?.[k]) !==
           JSON.stringify(LUBA_ACCEPTANCE_PROFILE[k]));
  console.log(`\nkeys in payload differing from LUBA_ACCEPTANCE_PROFILE: ${JSON.stringify(drift)}`);
} else {
  console.log("\n(pass waypoints as x,y args to see the emitted payload)");
}
