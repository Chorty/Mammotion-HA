"""One pymammotion pin, everywhere, and the tests run against it.

🚨 The pin lives in four files. On 2026-10-03 `pyproject.toml` (and so the
`uv.lock` every Beta Release regenerates) was still on 0.8.12.post4, two bumps
behind the 0.9.6.post1 the integration ships. And the shared local venv still had
0.8.12.post7, so a green local run was a green run against a backend the host
does not run. These tests make both mistakes fail loudly instead.
"""

import importlib.metadata
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PIN = re.compile(r"pymammotion@(https://\S+?/pymammotion-([^-]+)-py3-none-any\.whl)")


def _pin(text: str, source: str) -> str:
    matches = {match.group(1) for match in _PIN.finditer(text)}
    assert len(matches) == 1, f"{source}: expected one pymammotion pin, got {matches}"
    return matches.pop()


def _pins() -> dict[str, str]:
    manifest = json.loads(
        (ROOT / "custom_components/mammotion/manifest.json").read_text()
    )
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {
        "manifest.json": _pin("\n".join(manifest["requirements"]), "manifest.json"),
        "requirements_test.txt": _pin(
            (ROOT / "requirements_test.txt").read_text(), "requirements_test.txt"
        ),
        "pyproject.toml": _pin(
            "\n".join(pyproject["project"]["dependencies"]), "pyproject.toml"
        ),
    }


def test_every_file_pins_the_same_pymammotion_wheel() -> None:
    """manifest.json (what ships), requirements_test.txt and pyproject.toml agree."""
    pins = _pins()
    assert len(set(pins.values())) == 1, pins


def test_uv_lock_resolves_the_pinned_wheel() -> None:
    """The lock the release workflow commits names the same wheel."""
    shipped = _pins()["manifest.json"]
    lock = (ROOT / "uv.lock").read_text()
    sources = set(re.findall(r'source = \{ url = "(\S*/pymammotion-[^"]+)" \}', lock))
    assert sources == {shipped}, sources


def test_installed_pymammotion_is_the_pinned_version() -> None:
    """A stale venv must fail here, not pass 1,000 tests against the wrong backend."""
    wanted = _PIN.fullmatch(f"pymammotion@{_pins()['manifest.json']}")
    assert wanted is not None
    pinned = wanted.group(2)
    installed = importlib.metadata.version("pymammotion")
    assert installed == pinned, (
        f"installed pymammotion {installed}, but the pin is {pinned}. "
        "Refresh the venv: .venv/bin/python -m pip install -r requirements_test.txt"
    )
