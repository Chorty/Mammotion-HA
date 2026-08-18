#!/usr/bin/env python3
"""Report whether the card's execution profile is the hardware-accepted one.

WHY THIS EXISTS. `LUBA_ACCEPTANCE_PROFILE` is the exact set of values that
passed supervised LUBA acceptance. Changing any of them un-accepts the profile
and owes the section 4 re-pinning in `docs/gate4-repass-20260805.md` plus
another Gate 5 -- but nothing enforced that, and nothing said so anywhere a
person would look.

The `Beta Release` workflow has a `confirmed_luba_acceptance` boolean, and it is
NOT that enforcement. It is a job-level `if:` guard and nothing else: it is
never written into the release body, the tag or any file, it verifies nothing,
and because it gates the job rather than failing a step, setting it false
silently skips the release and still reports success. It also cannot be
informative in the release body, because the job only runs when it is true --
echoing it would print "true" on every release by construction.

So the question a release page should answer is not "did someone tick a box"
but "is the profile in this build the accepted one", which is derivable from the
repository. This script derives it: it parses the profile out of the card and
diffs it against the snapshot in `docs/accepted-profile.json`.

USAGE
    check_accepted_profile.py                 human-readable verdict
    check_accepted_profile.py --markdown      the block the release body embeds
    check_accepted_profile.py --strict        exit 1 when the profile diverges
    check_accepted_profile.py --write-accepted  re-snapshot AFTER a Gate 5 pass

⚠️ `--write-accepted` RECORDS A CLAIM ABOUT HARDWARE. Run it only after a Gate 5
has actually passed on the current profile, and say which evidence file proves
it. It is not a way to silence this check.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
CARD = REPO / "custom_components/mammotion/www/mammotion-custom-path-card.js"
ACCEPTED = REPO / "docs/accepted-profile.json"


def _strip_line_comments(src: str) -> str:
    """Remove `//` comments, ignoring `//` that appears inside a string."""
    out = []
    for line in src.splitlines():
        quote = None
        cut = None
        i = 0
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "\"'`":
                quote = ch
            elif ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                cut = i
                break
            i += 1
        out.append(line[:cut] if cut is not None else line)
    return "\n".join(out)


def extract_profile(card_source: str) -> dict:
    """Parse `LUBA_ACCEPTANCE_PROFILE` out of the card into a dict.

    The literal is a flat `Object.freeze({...})` of primitives plus one array,
    so converting it to JSON is a bounded job: strip comments, quote the bare
    keys, drop trailing commas. Anything more exotic appearing in the profile
    should fail loudly here rather than be silently mis-parsed.
    """
    marker = "const LUBA_ACCEPTANCE_PROFILE = Object.freeze({"
    start = card_source.index(marker) + len(marker)
    depth = 1
    i = start
    while i < len(card_source) and depth:
        if card_source[i] == "{":
            depth += 1
        elif card_source[i] == "}":
            depth -= 1
        i += 1
    body = _strip_line_comments(card_source[start : i - 1])
    body = re.sub(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'"\1":', body, flags=re.M)
    body = re.sub(r",(\s*[}\]])", r"\1", body)
    # The slice excludes the closing brace, so the final entry's trailing comma
    # has nothing after it for the rule above to match.
    body = re.sub(r",\s*$", "", body.strip())
    return json.loads("{" + body + "}")


def compare(current: dict, accepted: dict) -> list[dict]:
    """Return one row per key that differs, in a stable order."""
    rows = []
    for key in sorted(set(current) | set(accepted)):
        was, now = accepted.get(key, "<absent>"), current.get(key, "<absent>")
        if was != now:
            rows.append({"key": key, "accepted": was, "current": now})
    return rows


def main() -> int:
    """Compare the card profile against the accepted snapshot."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--write-accepted", metavar="EVIDENCE")
    ap.add_argument("--accepted-on", default=None)
    args = ap.parse_args()

    current = extract_profile(CARD.read_text())

    if args.write_accepted:
        ACCEPTED.write_text(
            json.dumps(
                {
                    "_comment": (
                        "The exact profile that last passed supervised LUBA "
                        "acceptance. Regenerate ONLY after a Gate 5 pass, with "
                        "scripts/check_accepted_profile.py --write-accepted "
                        "<evidence file>."
                    ),
                    "accepted_on": args.accepted_on or "unknown",
                    "evidence": args.write_accepted,
                    "profile": current,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"wrote {ACCEPTED.relative_to(REPO)} from the current card profile")
        return 0

    doc = json.loads(ACCEPTED.read_text())
    diffs = compare(current, doc["profile"])

    if args.markdown:
        if not diffs:
            print("## Execution profile\n")
            print(
                f"✅ **Hardware-accepted.** Byte-identical to the profile that "
                f"passed supervised LUBA acceptance on {doc['accepted_on']} "
                f"(`{doc['evidence']}`)."
            )
            return 0
        print("## Execution profile\n")
        print(
            "🚨 **NOT hardware-accepted.** This build changes "
            "`LUBA_ACCEPTANCE_PROFILE`, so it owes the section 4 re-pinning in "
            "`docs/gate4-repass-20260805.md` and **another Gate 5**. No "
            "supervised run on this profile may be described as accepted.\n"
        )
        print(f"Last accepted: {doc['accepted_on']} (`{doc['evidence']}`)\n")
        print("| key | accepted | this build |")
        print("| --- | --- | --- |")
        for d in diffs:
            print(f"| `{d['key']}` | `{d['accepted']}` | `{d['current']}` |")
        return 0

    if not diffs:
        print(f"ACCEPTED: profile matches {doc['accepted_on']} ({doc['evidence']})")
        return 0
    print(f"NOT ACCEPTED: {len(diffs)} key(s) differ from the accepted profile")
    print(f"  last accepted: {doc['accepted_on']} ({doc['evidence']})")
    for d in diffs:
        print(f"  {d['key']}: accepted={d['accepted']!r} current={d['current']!r}")
    print("\nOwes: section 4 re-pinning (docs/gate4-repass-20260805.md) + a Gate 5.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
