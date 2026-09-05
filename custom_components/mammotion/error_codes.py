"""Offline decoding for the numeric fault codes the mower emits.

🚨 Why this module exists. On 2026-09-04 the mower failed to dock twice and the
vendor app said why in one line -- ``Robot orientation unavailable (1309)``,
five times, with the manufacturer's own remedy. Home Assistant showed
``sensor.<mower>_last_error`` reading ``"mcu: , "`` with an hour-stale
timestamp. An operator watching this integration had strictly less information
than one watching the app, during a failure the app diagnosed in a sentence.
Full record: ``docs/findings-clicktopath-reliability-4m-20260904.md`` section 6.6.

Two separate defects produced that empty string, and this module addresses the
second one:

1. The live push that carries the code was parsed and thrown away. Fixed in
   :class:`~.coordinator.MammotionDeviceErrorUpdateCoordinator`.
2. ``get_error_message`` formatted ``f"{module}: {implication}, {solution}"``
   from the cloud CSV. When that row's English text is blank -- which it was --
   the operator gets punctuation and **no code at all**, because the numeric
   code never appears in the message. The code was available the whole time in
   ``sensor.<mower>_last_error_code``; nothing surfaced it where anyone looked.

``error_code_table`` is an offline fallback so a blank or missing cloud row can
never again hide a fault. It is not the primary source: the cloud CSV wins when
it actually carries text, because it is localised and the device fleet can
change under us. A full mapping table was explicitly a bonus rather than the
requirement -- surfacing the raw number is the win, and
:func:`describe_error_code` degrades to exactly that when a code is unknown.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, Final, TypedDict

_ERROR_CODES_PATH: Final = Path(__file__).parent / "error_codes.json"


class ErrorCodeText(TypedDict):
    """English implication/solution pair for one numeric fault code."""

    level: int | None
    implication: str
    solution: str


@cache
def error_code_table() -> dict[str, ErrorCodeText]:
    """Return the bundled code -> English text table, loaded once.

    Extracted from ``assets/errorcodejson.txt`` in the decompiled Android app
    (449 codes, every one carrying English text). Cached because it is a static
    resource read on a coordinator update path.
    """
    try:
        with _ERROR_CODES_PATH.open(encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)
    except OSError, json.JSONDecodeError:
        # A missing or corrupt bundle must never break error reporting -- the
        # numeric code alone is the deliverable.
        return {}
    codes = payload.get("codes")
    return codes if isinstance(codes, dict) else {}


def lookup_error_code(code: int | str) -> ErrorCodeText | None:
    """Return bundled English text for a numeric code, or None if unknown."""
    return error_code_table().get(str(code))


def describe_error_code(
    code: int | str,
    *,
    implication: str = "",
    solution: str = "",
    module: str = "",
) -> str:
    """Return an operator-facing description that ALWAYS carries the number.

    ``implication``/``solution``/``module`` are whatever the cloud table
    supplied. They win when non-empty; the bundled table fills in when they are
    blank; and when neither has text the result is still the code plus a plain
    statement that no description is known -- never ``"mcu: , "``.
    """
    normalized = str(code).strip()
    text_implication = implication.strip()
    text_solution = solution.strip()
    if not text_implication or not text_solution:
        if (bundled := lookup_error_code(normalized)) is not None:
            text_implication = text_implication or bundled["implication"].strip()
            text_solution = text_solution or bundled["solution"].strip()

    prefix = f"{normalized} ({module.strip()})" if module.strip() else normalized
    if not text_implication and not text_solution:
        return f"{prefix}: no description available for this code"
    if not text_solution:
        return f"{prefix}: {text_implication}"
    if not text_implication:
        return f"{prefix}: {text_solution}"
    return f"{prefix}: {text_implication} — {text_solution}"
