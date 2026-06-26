"""Shared fixtures for lerobot_local policy tests.

The lerobot_local policy keeps a process-level model cache (see
``strands_robots.policies.lerobot_local.policy._MODEL_CACHE``) keyed by
``(checkpoint, type, device)``. Many unit tests reuse the placeholder path
``"test/model"`` with differently shaped mock models, so the cache must be
cleared around every test to keep them hermetic.
"""

from __future__ import annotations

import pytest

from strands_robots.policies.lerobot_local.policy import clear_model_cache


@pytest.fixture(autouse=True)
def _clear_lerobot_local_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()
