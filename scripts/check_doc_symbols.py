#!/usr/bin/env python3
"""Fail when a session-entry doc cites code that no longer exists.

Every session is told to read `CLAUDE.md`, `docs/NEXT-SESSION.md` and
`docs/CODEX-HANDOFF.md` first. On 2026-08-14 two of their claims were found to
be stale by 13 and 18 betas respectively, and one of them caused a session to
recommend re-doing work that had already shipped. This makes that class of rot
mechanical instead of a discipline nobody sustains.

Three claim classes are checked, all extracted from inline backtick spans:

  symbol    a code identifier (contains `_`), must appear somewhere in the tree
  path      a repo-relative file path, must exist
  citation  `file.py:1234` is REFUSED for in-repo files -- cite the symbol,
            which does not move and can actually be validated. Out-of-repo
            paths (pymammotion) are left alone.

⚠️ **A clean run is necessary, not sufficient.** This proves a name still
exists; it cannot prove the surrounding prose still describes it correctly. The
worse defect found on 2026-08-14 -- a paragraph calling an implemented fix "NOT
implemented" -- named no symbol at all and is invisible to this check. Do not
read green as "the docs are accurate".

Deliberate historical references go in `docs/doc-symbol-allowlist.txt` with a
reason, which is the point: removing a symbol from the code then forces an
explicit, reviewable note rather than leaving a false claim in place.

Usage:  .venv/bin/python scripts/check_doc_symbols.py [--list]
"""

from __future__ import annotations

import contextlib
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ["CLAUDE.md", "docs/NEXT-SESSION.md", "docs/CODEX-HANDOFF.md"]
ALLOWLIST = ROOT / "docs" / "doc-symbol-allowlist.txt"

CORPUS_GLOBS = [
    "custom_components/**/*.py",
    "custom_components/**/*.json",
    "custom_components/**/*.yaml",
    "custom_components/**/*.js",
    "scripts/**/*.py",
    "scripts/**/*.sh",
    "tests/**/*.py",
    "tests/**/*.mjs",
    ".github/workflows/*.yml",
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
    "hacs.json",
]

IDENT = re.compile(r"^_?[A-Za-z][A-Za-z0-9_]*$")
CITATION = re.compile(r"^([\w./-]+\.(?:py|js|json|yaml|md|toml)):(\d+)(?:-(\d+))?$")
PATHLIKE = re.compile(r"^[\w./-]+\.(?:py|js|json|yaml|md|toml|mjs|tgz|jsonl)$")
NON_SYMBOL = re.compile(r"^(\d[\d.]*|v?\d+\.\d+\.\d+.*|[A-Z]{2,5})$")


def read_allowlist() -> dict[str, str]:
    """Return {name: reason} for deliberately historical references."""
    allowed: dict[str, str] = {}
    if not ALLOWLIST.exists():
        return allowed
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, reason = line.partition("#")
        allowed[name.strip()] = reason.strip() or "no reason given"
    return allowed


def load_corpus() -> str:
    """Return every searchable source file, plus the installed backend."""
    parts = []
    for pattern in CORPUS_GLOBS:
        for path in ROOT.glob(pattern):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            with contextlib.suppress(OSError):
                parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    # Claims about the shipped backend are legitimate and checkable too.
    for path in ROOT.glob(".venv/lib/python*/site-packages/pymammotion/**/*.py"):
        with contextlib.suppress(OSError):
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def basename_index() -> dict[str, list[pathlib.Path]]:
    """Map bare filenames to real paths so `services.py:1234` resolves."""
    index: dict[str, list[pathlib.Path]] = defaultdict(list)
    for pattern in [*CORPUS_GLOBS, "docs/**/*.md", "*.md"]:
        for path in ROOT.glob(pattern):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            index[path.name].append(path)
    return index


def spans(text: str):
    """Yield (lineno, span) for inline code, ignoring fenced blocks.

    Fences must be stripped first or their backticks mis-pair with the next
    inline span and silently halve the claims seen.
    """
    text = re.sub(
        r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.DOTALL
    )
    # Allow one wrapped line: markdown reflow splits `NAME =\n60.0`.
    for match in re.finditer(r"`([^`\n]+(?:\n[^`\n]+)?)`", text):
        yield text.count("\n", 0, match.start()) + 1, " ".join(match.group(1).split())


def _check_citation(
    cite: re.Match[str], span: str, index: dict[str, list[pathlib.Path]]
) -> tuple[int, list[tuple[str, str, str]]]:
    """Validate one `file:line` citation."""
    target, end = cite.group(1), cite.group(3)
    high = int(end) if end else int(cite.group(2))
    found = (
        [ROOT / target]
        if (ROOT / target).exists()
        else index.get(pathlib.PurePath(target).name, [])
    )
    if not found:
        return 0, []  # out-of-repo reference, not ours to police
    total = len(found[0].read_text(errors="ignore").splitlines())
    if high > total:
        why = f"{found[0].relative_to(ROOT)} has {total} lines"
        return 1, [("citation", span, why)]
    # 🚨 An in-repo line number is REFUSED even when it is inside the file.
    #
    # Being within EOF proves nothing about whether the line still holds
    # what the prose says. Audited 2026-08-20: of 10 sampled citations, 8
    # pointed at unrelated code -- `services.py:10939` was cited for
    # `_SEGMENT_TURN_MODES`, which had moved to 11613 and left a
    # `dry_run=dry_run,` argument behind. Two carried the annotation "line
    # numbers verified 2026-08-17", so the verification itself had rotted.
    #
    # No checker can validate a bare number: nothing states what SHOULD be
    # there. A symbol can be validated, and does not move. So cite the
    # symbol -- `_SEGMENT_TURN_MODES`, not `services.py:10939` -- and this
    # check becomes real instead of decorative.
    # Markdown-to-markdown citations get the EOF check only: a prose
    # section has no symbol to cite instead, so refusing them would leave
    # no correct way to reference one.
    if found[0].suffix.lower() not in {".py", ".js", ".pyi", ".mjs"}:
        return 1, []
    why = (
        "in-repo line numbers rot silently; cite the symbol instead "
        "(8 of 10 sampled citations were stale on 2026-08-20)"
    )
    return 1, [("citation", span, why)]


def check_span(
    span: str,
    *,
    allowed: dict[str, str],
    corpus: str,
    index: dict[str, list[pathlib.Path]],
) -> tuple[int, list[tuple[str, str, str]]]:
    """Return (claims checked, [(kind, name, why)]) for one inline span."""
    cite = CITATION.match(span)
    if cite:
        return _check_citation(cite, span, index)

    if PATHLIKE.match(span) and "/" in span:
        if span.startswith("/"):
            return 0, []  # a path on the HA host, not in this repo
        if not (ROOT / span).exists() and span not in allowed:
            return 1, [("path", span, "does not exist")]
        return 1, []

    candidates = (
        [span] if IDENT.match(span) else re.findall(r"_?[A-Za-z][A-Za-z0-9_]*", span)
    )
    checked = 0
    bad: list[tuple[str, str, str]] = []
    for token in dict.fromkeys(candidates):
        if "_" not in token or NON_SYMBOL.match(token):
            continue
        checked += 1
        if token not in corpus and token not in allowed:
            bad.append(("symbol", token, "not found in tree"))
    return checked, bad


def main() -> int:
    """Check every session-entry doc and report unresolved claims."""
    allowed = read_allowlist()
    corpus = load_corpus()
    index = basename_index()
    unresolved: list[tuple[str, int, str, str, str]] = []
    checked = 0

    for doc in DOCS:
        for lineno, span in spans((ROOT / doc).read_text(encoding="utf-8")):
            count, bad = check_span(span, allowed=allowed, corpus=corpus, index=index)
            checked += count
            unresolved.extend((doc, lineno, *item) for item in bad)

    if "--list" in sys.argv:
        for name, reason in sorted(allowed.items()):
            print(f"{name:<52} {reason}")
        return 0

    if not unresolved:
        print(
            f"doc symbols: {checked} claims checked, all resolve "
            f"({len(allowed)} allowlisted)"
        )
        return 0

    print(f"doc symbols: {len(unresolved)} unresolved claim(s) of {checked}\n")
    for doc, lineno, kind, span, why in unresolved:
        print(f"  {doc}:{lineno}  [{kind}] {span} -- {why}")
    print(
        "\nFix the doc, or add the name to docs/doc-symbol-allowlist.txt with a\n"
        "reason if the reference is deliberately historical."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
