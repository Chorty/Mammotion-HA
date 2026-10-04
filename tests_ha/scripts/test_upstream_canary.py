"""Version selection and report accounting must not hide backend failures."""

import json
from pathlib import Path

import pytest

from scripts import upstream_canary as canary


def test_latest_ignores_previews_yanked_and_empty_releases() -> None:
    """The advisory candidate must be a real stable release."""
    metadata = {
        "releases": {
            "0.9.6": [{"yanked": False}],
            "0.10.7": [{"yanked": False}],
            "0.10.8": [{"yanked": True}],
            "0.11.0rc1": [{"yanked": False}],
            "0.11.0.dev1": [{"yanked": False}],
            "99.0": [],
            "invalid": [{"yanked": False}],
        }
    }
    assert canary.latest_stable(metadata) == "0.10.7"


def test_latest_accepts_a_release_with_one_non_yanked_file() -> None:
    """A yanked file does not invalidate other available files in that release."""
    assert (
        canary.latest_stable(
            {"releases": {"0.10.7": [{"yanked": True}, {"yanked": False}]}}
        )
        == "0.10.7"
    )


def test_no_stable_release_fails_instead_of_installing_a_preview() -> None:
    """Empty stable populations produce an explicit resolver failure."""
    with pytest.raises(ValueError, match="no non-yanked stable"):
        canary.latest_stable({"releases": {"0.11rc1": [{}]}})


@pytest.mark.parametrize(
    ("requirement", "wanted"),
    [
        ("pymammotion==0.10.7", "0.10.7"),
        (
            "pymammotion@https://github.com/Chorty/PyMammotion/releases/download/chorty-0.9.6.post1/pymammotion-0.9.6.post1-py3-none-any.whl",
            "0.9.6.post1",
        ),
    ],
)
def test_pin_reads_stock_and_fork_requirements(requirement: str, wanted: str) -> None:
    """Compare the versions represented by each supported manifest pin form."""
    assert (
        canary.backend_pin({"requirements": ["websockets==16.0", requirement]})
        == wanted
    )


@pytest.mark.parametrize(
    "requirements",
    [
        [],
        ["pymammotion>=0.10"],
        ["pymammotion==0.10.*"],
        ["pymammotion==0.10.7", "pymammotion==0.10.4"],
    ],
)
def test_ambiguous_or_unpinned_backend_is_an_error(requirements: list[str]) -> None:
    """A range or duplicate entry must not produce a misleading drift report."""
    with pytest.raises(ValueError):
        canary.backend_pin({"requirements": requirements})


def test_snapshot_compares_base_not_the_fork_postrelease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fork suffix alone must not create a spurious upstream drift issue."""
    manifest = tmp_path / "custom_components/mammotion/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "requirements": [
                    "pymammotion@https://example.org/pymammotion-0.9.6.post1-py3-none-any.whl"
                ]
            }
        )
    )
    values = iter(
        [
            {"releases": {"0.9.6": [{}]}},
            {
                "tag_name": "v0.6.16",
                "html_url": "https://github.com/mikey0000/Mammotion-HA/releases/tag/v0.6.16",
            },
            {"requirements": ["pymammotion==0.9.6"]},
        ]
    )
    monkeypatch.setattr(canary, "fetch_json", lambda _: next(values))
    result = canary.snapshot(tmp_path)
    assert result["shipping_version"] == "0.9.6.post1"
    assert result["fork_base_version"] == "0.9.6"
    assert result["drift"] is False


def test_report_retains_the_pin_failure_but_separates_it_from_compatibility(
    tmp_path: Path,
) -> None:
    """Keep raw totals, collection errors, and functional failures visible."""
    path = tmp_path / "suite.xml"
    path.write_text("""<testsuites><testsuite tests="5" failures="2" errors="1" skipped="1">
      <testcase classname="tests_ha.test_pymammotion_pin" name="test_installed_pymammotion_is_the_pinned_version"><failure message="pin mismatch" /></testcase>
      <testcase classname="settings" name="test_preserve_blade"><failure message="rain_tactics absent" /></testcase>
      <testcase classname="collection" name="module"><error message="missing dependency" /></testcase>
      <testcase classname="evidence" name="private"><skipped /></testcase>
      <testcase classname="other" name="pass" />
    </testsuite></testsuites>""")
    result = canary.suite_summary(path)
    assert result["passed"] == 1
    assert result["failures"] == 2
    assert result["errors"] == 1
    assert len(result["expected_pin_failures"]) == 1
    assert len(result["compatibility_failures"]) == 1
    assert result["failure_groups"] == {"settings": 1}


def test_missing_xml_is_reported_as_incomplete(tmp_path: Path) -> None:
    """An aborted install or collection cannot appear as an all-green suite."""
    assert canary.suite_summary(tmp_path / "absent.xml")["available"] is False


def test_report_identifies_results_from_the_wrong_backend() -> None:
    """An unsuccessful candidate install cannot make baseline passes look like a canary pass."""
    text = canary.render_summary(
        {
            "shipping_version": "0.9.6.post1",
            "fork_base_version": "0.9.6",
            "latest_upstream_version": "0.10.7",
            "upstream_ha_release": "v0.6.16",
            "upstream_ha_backend_version": "0.10.7",
        },
        {
            "available": False,
            "reason": "missing XML",
            "installed_pymammotion": "0.9.6.post1",
        },
    )
    assert "Backend actually tested: `0.9.6.post1`" in text
    assert "Candidate installation mismatch" in text
