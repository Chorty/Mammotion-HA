"""Tests for Mammotion pymammotion compatibility patches."""

from custom_components.mammotion.pymammotion_compat import (
    apply_pymammotion_compat_patches,
)
from pymammotion.aliyun.model.dev_by_account_response import ShareNoticeListResponse


def test_share_notice_parser_accepts_missing_initiator_alias() -> None:
    """Share notices without initiatorAlias should not break cloud login."""
    apply_pymammotion_compat_patches()

    response = ShareNoticeListResponse.from_dict(
        {
            "code": 200,
            "data": {
                "total": 1,
                "data": [
                    {
                        "gmtModified": 1774309213000,
                        "targetId": "H3vkT2oB2edD5xmEjO4p000000",
                        "receiverIdentityId": "receiver",
                        "description": "The sharing of the device has timed out",
                        "targetType": "DEVICE",
                        "gmtCreate": 1774309055000,
                        "batchId": "ACCOUNT_DEV_SHARE_test",
                        "nodeType": "DEVICE",
                        "deviceName": "RTKBNA235279309",
                        "productName": "ReferenceStation",
                        "recordId": "record",
                        "initiatorIdentityId": "initiator",
                        "isReceiver": 1,
                        "receiverAlias": "UNKNOW",
                        "status": 3,
                    }
                ],
                "pageNo": 1,
                "pageSize": 100,
            },
        }
    )

    assert response.data is not None
    assert response.data.data[0].initiator_alias == ""
