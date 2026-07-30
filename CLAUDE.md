# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

There is no global `uv` on the dev machine — `uv sync`/`uv run` fail with
`command not found`. Use the project venv directly. (`.venv/bin/uv` exists only
because it was pip-installed into the venv to regenerate `uv.lock`; it cannot
bootstrap the venv it lives in.)

Run the same commands CI runs, so a green local run means a green CI run
(`.github/workflows/validate.yml` is the source of truth):

- Tests: `.venv/bin/python -m pytest --cov=custom_components.mammotion --cov-report=term-missing tests`
- Lint: `.venv/bin/python -m ruff check custom_components tests`
- Format check: `.venv/bin/python -m ruff format --check custom_components tests`
- Type checking: `.venv/bin/python -m mypy --follow-imports=skip custom_components/mammotion`
- Frontend card tests: `npm run test:frontend`
- Pre-commit: `.venv/bin/python -m pre_commit run --all-files`
- Install/refresh deps: `.venv/bin/python -m pip install -r requirements_test.txt`
  (CI's install step; keep the `pymammotion` pin here identical to
  `manifest.json` or CI tests a different backend than the one shipped)

⚠️ Use mypy's `--follow-imports=skip custom_components/mammotion` form above, not
a whole-tree `mypy custom_components/`. The broader form reports 4 pre-existing
errors in `select.py` and `device_tracker.py` that CI never checks — they are not
regressions from your change.

Note: `ruff format` rewrites `except (A, B):` (no `as` binding) into
`except A, B:`. That is **correct and intentional** — PEP 758 allows
unparenthesized exception tuples from Python 3.14, which this project targets
(`requires-python = ">=3.14.2"`). It looks like the Python 2 form and is not;
verified parsing and catching on 3.14.6. Do not "fix" it back.

## Code Style Guidelines

- Follow Home Assistant integration patterns
- Use async/await patterns (prefix functions with `async_`)
- Line ending format: LF (not enforced by any hook or editor config)
- Prefer specific exception types over broad ones

When making changes, follow existing patterns in similar files and follow Home Assistant best practices.

## Home Assistant Integration Rules

- All imports within the integration must be relative (e.g. `from . import Foo`, `from .services import bar`). Never use `from custom_components.mammotion import ...` — HA loads integrations in a way that makes absolute imports from `custom_components` fail at runtime.

## Subagent Model Routing

Keep expensive reasoning on the session model; route delegated work to cheaper models automatically:

- **Search/scan fan-out** (code-review finder angles, diff scans, symbol/caller hunts, convention audits): use the `finder` agent, or pass `model: sonnet` when spawning a general-purpose agent for this kind of work.
- **Verification and adjudication** (confirming/refuting review candidates, checking that a library API actually exists, call-site impact analysis): use the `verifier` agent, or pass `model: opus`.
- **Fix authoring in a subagent**: `model: opus`.
- Only leave a subagent on the session (inherited) model when the task genuinely needs top-tier long-horizon reasoning — orchestration itself already runs there.

## Translations

- When adding or renaming any entity (sensor, switch, button, number, select, etc.) or an ENUM entity state, you MUST update the translations in **every** language file, not just English.
- The files to keep in sync: `custom_components/mammotion/strings.json` (the source) **and** every file under `custom_components/mammotion/translations/`. Treat that directory listing as the source of truth for which languages exist.
- Translate the entity `name` and every ENUM `state` value into each language's own language — do not copy the English text into the other locales as a placeholder.
- Also add an icon entry in `custom_components/mammotion/icons.json` for the new entity where appropriate.
- After editing, confirm every JSON file still parses and that the new key (with all its `state` values) is present in each file before considering the change complete.
