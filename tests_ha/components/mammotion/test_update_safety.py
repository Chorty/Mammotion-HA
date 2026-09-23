"""Firmware update prerequisite tests."""

from types import SimpleNamespace

from custom_components.mammotion.update import _firmware_install_readiness


def _coordinator() -> SimpleNamespace:
    return SimpleNamespace(
        last_update_success=True,
        device_name="LUBA-TEST",
        has_cloud_account=True,
        is_online=lambda: True,
        data=SimpleNamespace(
            report_data=SimpleNamespace(
                dev=SimpleNamespace(
                    sys_status=SimpleNamespace(name="MODE_READY"),
                    charge_state=SimpleNamespace(name="charging"),
                    battery_val=80,
                ),
                connect=SimpleNamespace(wifi_rssi=-55),
            )
        ),
    )


def test_mower_firmware_update_requires_all_positive_evidence() -> None:
    """A known, idle, charging, powered, online mower can pass."""
    result = _firmware_install_readiness(
        _coordinator(),
        installed_version="1.0.0",
        target_version="1.1.0",
    )

    assert result["allowed"] is True
    assert result["blockers"] == []


def test_unknown_prerequisite_blocks_firmware_install() -> None:
    """Unknown is never treated as a safe firmware prerequisite."""
    coordinator = _coordinator()
    coordinator.data.report_data.dev.charge_state = None
    coordinator.data.report_data.connect.wifi_rssi = 0

    result = _firmware_install_readiness(
        coordinator,
        installed_version="1.0.0",
        target_version="1.1.0",
    )

    assert result["allowed"] is False
    assert "device_not_confirmed_charging" in result["blockers"]
    assert "wifi_link_unknown" in result["blockers"]


def test_unaccepted_hardware_update_paths_fail_closed() -> None:
    """RTK and SPINO fixtures cannot start firmware installs."""
    for kind in ("rtk", "spino"):
        result = _firmware_install_readiness(
            _coordinator(),
            installed_version="1.0.0",
            target_version="1.1.0",
            hardware_kind=kind,
        )
        assert result["allowed"] is False
        assert result["blockers"] == [f"{kind}_firmware_acceptance_missing"]
