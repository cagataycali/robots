# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A forwarding header that arrives twice is read as the one hop chain it is.

Two proxies spell their chain either as one joined ``x-forwarded-for: client,
proxy1`` or as that header twice, appended rather than merged (RFC 7239 permits
both). The joined form is pinned elsewhere; the repeated one is the shape a
stand-in holding a ``dict`` of headers cannot express at all, since a second
value under one key is the first one overwritten. Both readers must be
indifferent to which arrived: the first-enrollment gate, whose evidence is the
NAME and not how many instances of it are on the wire (F-007), and the per-ip
challenge cap, whose key is the client at the far end rather than the proxy that
appended the last entry.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth
from tests._dashboard_connection import STRANGER, connection, repeating


@pytest.mark.parametrize("header", auth._PROXY_EVIDENCE_HEADERS)
def test_a_header_sent_twice_is_the_same_evidence_of_a_hop(header: str) -> None:
    request = repeating(header, "first", "second")
    assert auth._arrived_through_a_proxy(request) == header
    with pytest.raises(HTTPException) as raised:
        auth.begin_registration(request, label="stranger-through-two-hops")
    assert raised.value.status_code == 403
    assert header in raised.value.detail


def test_the_cap_key_is_the_client_at_the_far_end_of_the_chain() -> None:
    """The nearest proxy appends itself, so the last entry is never the client."""
    assert auth._client_ip(repeating("x-forwarded-for", STRANGER, "10.0.0.1")) == STRANGER
    # The joined spelling of the same chain has to answer identically, or the cap
    # counts one flooding client under two keys depending on its proxy's habits.
    assert auth._client_ip(connection(x_forwarded_for=f"{STRANGER}, 10.0.0.1")) == STRANGER
