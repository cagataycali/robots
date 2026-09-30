# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bootstrap proof a fresh-install dashboard test presents to be admitted.

Since f002 the open posture admits a caller on ``STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN``
(or the ``0600`` file the server mints) presented as the bearer, never on a
loopback peer alone. A test that stands in for the operator at the machine on a
fresh install configures the env token and sends it on every request; a test
about a refusal sends something else, or nothing, on the one request it is about.
"""

from __future__ import annotations

import pytest

#: A value no real deployment has: the test fixture configures it, so it is the expectation too.
BOOTSTRAP = "test-fixture-bootstrap-proof"


def bootstrap_headers() -> dict[str, str]:
    """The header the operator's own page sends before a passkey exists."""
    return {"authorization": f"Bearer {BOOTSTRAP}"}


def configure_bootstrap(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Make :data:`BOOTSTRAP` the first-enrollment proof; returns the header to present it."""
    monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", BOOTSTRAP)
    return bootstrap_headers()
