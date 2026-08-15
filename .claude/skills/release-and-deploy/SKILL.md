---
name: release-and-deploy
description: Cut the next beta and install it on the Home Assistant host motion-disabled. Use when asked to release a beta, deploy to HA, ship a build, or install the current tree on the mower host. Covers the Beta Release workflow, the version quartet, the two card-serving paths, the Lovelace cache key, and the mandatory verification tail.
---

# Release a beta and deploy it motion-disabled

The prose source of truth is `docs/deploy-runbook-p0.md`. This skill is the
executable order plus the traps that are **not** obvious from reading it — every
one below has actually bitten a session.

## Safety boundary — read before step 1

- **A deploy never commands motion.** Do not enable experimental motion, do not
  arm the gate, do not call a `mammotion.*` movement service. A dry run is the
  only executor call that belongs here.
- Deploy while the mower is **stopped** and the gate is **off**. Confirm before
  and after.
- ⚠️ **Always pass `-R Chorty/Mammotion-HA` to every `gh` command.** No default
  repo is configured, so bare `gh` can resolve to `mikey0000/Mammotion-HA`,
  which is **read-only for this work**. `gh pr view 14` failed exactly this way
  on 2026-08-14.
- ⚠️ **Run every `scripts/*.py` under `.venv/bin/python`.** Their shebang
  resolves to system python, which has no `aiohttp`;
  `scripts/ha_set_card_resource.py` dies with `ModuleNotFoundError` if invoked
  directly.

## 0. Preconditions

```sh
set -a && source .env && set +a
.venv/bin/python scripts/ha_set_experimental_motion.py status   # expect enabled: False
git status --short                                              # expect clean
```

Run the full gate suite and record the ACTUAL counts — never carry forward
numbers from a handoff document:

```sh
.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests
.venv/bin/python -m ruff check custom_components tests
.venv/bin/python -m ruff format --check custom_components tests
.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion
npm run test:frontend
.venv/bin/python -m pre_commit run --all-files
```

## 1. Release

The `Beta Release` workflow computes the next number, bumps all four version
sites, commits, tags, and creates a prerelease. Do not bump versions by hand.

```sh
gh workflow run "Beta Release" -R Chorty/Mammotion-HA --ref main \
  -f confirmed_luba_acceptance=true
gh run watch <run-id> -R Chorty/Mammotion-HA --exit-status
git fetch origin main && git merge --ff-only origin/main
```

Verify the **version quartet** agrees (the workflow checks this too, but confirm
locally — the lock file uses the PEP 440 form `0.6.4bNN`, not `0.6.4-betaNN`):

```sh
jq -r .version custom_components/mammotion/manifest.json
sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1
grep -o 'const CARD_VERSION = "[^"]*"' custom_components/mammotion/www/mammotion-custom-path-card.js
grep -A2 'name = "mammotion-ha"' uv.lock | grep version
git tag --points-at HEAD
```

⚠️ `git fetch --tags` may fail with "would clobber existing tag" — the fork and
upstream disagree on old tags. That is pre-existing and harmless; fetch `main`
alone.

## 2. Back up, then ship

```sh
scripts/ha_ssh.exp 'cd /config/custom_components && tar -czf /config/mammotion-backup-$(date +%Y%m%d-%H%M)-pre-betaNN.tgz mammotion && ls -la /config/mammotion-backup-*.tgz'
```

⚠️ **`COPYFILE_DISABLE=1` is mandatory.** macOS BSD tar otherwise embeds
AppleDouble metadata that extracts as 46 junk `._*` files *inside* the
integration, including next to real translation files.

```sh
COPYFILE_DISABLE=1 tar -czf <scratchpad>/mammotion_deploy.tgz -C custom_components mammotion
shasum -a 256 <scratchpad>/mammotion_deploy.tgz
tar -tzf <scratchpad>/mammotion_deploy.tgz | grep -c '\._'   # expect 0

scripts/ha_scp.exp <scratchpad>/mammotion_deploy.tgz /config/mammotion_deploy.tgz
scripts/ha_ssh.exp 'sha256sum /config/mammotion_deploy.tgz'   # must equal local
```

## 3. Extract and sync BOTH card paths

⚠️ **The card is served from two locations.** Deploy to only one and the browser
silently loads the stale card while every server-side check reports the new one.

```sh
scripts/ha_ssh.exp 'cd /config/custom_components && tar -xzf /config/mammotion_deploy.tgz && echo extracted && rm -f /config/mammotion_deploy.tgz && cp /config/custom_components/mammotion/www/mammotion-custom-path-card.js /config/www/community/mammotion/mammotion-custom-path-card.js && md5sum /config/custom_components/mammotion/www/mammotion-custom-path-card.js /config/www/community/mammotion/mammotion-custom-path-card.js && find /config/custom_components/mammotion -name "._*" | wc -l'
```

Both md5s must match each other **and** the local card. Keep the first 8 hex
characters — that is the `build=` suffix for step 5.

## 4. Verify all 46 files before restarting

```sh
# local
cd custom_components/mammotion && find . -type f \( -name "*.py" -o -name "*.json" \
  -o -name "*.yaml" -o -name "*.js" \) ! -path "./__pycache__/*" -exec md5 -r {} \; | sort -k2
# host
scripts/ha_ssh.exp 'cd /config/custom_components/mammotion && find . -type f \( -name "*.py" -o -name "*.json" -o -name "*.yaml" -o -name "*.js" \) ! -path "./__pycache__/*" -exec md5sum {} \; | sort -k2'
```

⚠️ **Normalise before diffing.** The expect wrapper returns CRLF, so a raw
`diff` reports *all 46 files as differing* when every hash is identical. Pipe
both through `tr -d '\r'` and `awk '{print $1, $2}'` first.

## 5. Restart, then bump the Lovelace cache key

```sh
scripts/ha_restart.sh          # API ~30-40 s, entities ~2 min
```

⚠️ **`ha_set_card_resource.py` does not append the `build=` suffix.** Pass it
*inside* the version argument or browsers keep the cached card:

```sh
.venv/bin/python scripts/ha_set_card_resource.py                          # show current
.venv/bin/python scripts/ha_set_card_resource.py "0.6.4-betaNN&build=<md5prefix>"          # dry run
.venv/bin/python scripts/ha_set_card_resource.py "0.6.4-betaNN&build=<md5prefix>" --apply  # verifies by re-reading
```

## 6. Verification tail — a deploy is not done without this

Report measured values, not expectations:

- **Hashes** — all 46 files byte-identical; card md5 equal at both paths;
  archive SHA-256 identical local and host.
- **Versions** — the quartet on the host (`manifest.json`, `CARD_VERSION`) plus
  the Lovelace resource read back as `?v=0.6.4-betaNN&build=<md5prefix>`.
- **Backend** — `scripts/ha_ssh.exp 'docker exec homeassistant python -c "import importlib.metadata as m; print(m.version(\"pymammotion\"))"'`
- **Entity recovery** — API return time, Mammotion entity count, and the config
  entry state (`loaded`, not `setup_error`).
- **Gate** — `real_motion_allowed: false`, `enabled: false`, no active session.
- **A dark-safe dry run** proves the deployed executor loads and runs. It sends
  no movement (`would_send: false`).

Known-benign readings: five entities read `unavailable` — the four
`emergency_nudge_*` buttons (`_nudge_available` returns `False` by design) and
`start_camera_on_mower`. On the dock, `position_not_valid_for_motion` is an
expected blocker (`CHARGE_ON`, `zone_hash 0`).

**The one step you cannot do:** confirming a *browser* loaded the new card. Ask
the operator to check the console banner and card footer both read the new
version. A correct backend deploy with a stale card cache is still a failed
deployment.

## 7. Record it

Add a dated section to the top of "What the host is running now" in
`docs/deploy-runbook-p0.md` with exact hashes, timings and the gate readback,
and update the live state in `CLAUDE.md` and `docs/NEXT-SESSION.md` §0. Then
re-run the gate suite (docs are covered by pre-commit) and push.

## Rollback

Backups are `/config/mammotion-backup-*.tgz`. Extract over
`/config/custom_components/`, restart, and re-verify the same tail.
