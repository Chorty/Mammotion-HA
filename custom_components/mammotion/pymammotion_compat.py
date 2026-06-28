"""Compatibility patches for pinned pymammotion versions."""

from __future__ import annotations

from typing import Any

from pymammotion.aliyun.model.dev_by_account_response import ShareNoticeListResponse

_PATCHED_ATTR = "_mammotion_ha_share_notice_parser_patched"


def _normalize_share_notice_response(data: Any) -> Any:
    """Normalize Mammotion/Aliyun share-notice fields before pymammotion parsing.

    pymammotion 0.8.8 requires ``ShareNotification.initiator_alias`` but the
    cloud API can return accepted/expired share records without ``initiatorAlias``.
    That makes cloud login fail before device setup. The app treats the field as
    display metadata, so an empty string is a safe fallback.
    """
    if not isinstance(data, dict):
        return data

    notice_data = data.get("data")
    if not isinstance(notice_data, dict):
        return data

    records = notice_data.get("data")
    if not isinstance(records, list):
        return data

    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("initiatorAlias") is None and record.get("initiator_alias") is None:
            record["initiatorAlias"] = ""

    return data


def apply_pymammotion_compat_patches() -> None:
    """Apply idempotent compatibility patches for the pinned pymammotion version."""
    if getattr(ShareNoticeListResponse, _PATCHED_ATTR, False):
        return

    original_from_dict = ShareNoticeListResponse.from_dict

    @classmethod
    def from_dict_with_share_notice_defaults(cls, data: Any) -> ShareNoticeListResponse:
        return original_from_dict(_normalize_share_notice_response(data))

    ShareNoticeListResponse.from_dict = from_dict_with_share_notice_defaults
    setattr(ShareNoticeListResponse, _PATCHED_ATTR, True)
