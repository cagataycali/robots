"""A ``sim_call`` ``repo_id`` is a Hub id ``owner/name`` on the wire, never a path on the peer host.

``repo_id`` is not on :data:`strands_robots.mesh.security.SIM_CALL_DENIED_PARAMS`:
a recording needs a name. But :func:`strands_robots.dataset_source.local_dataset_dir`
reads a ``/`` or ``./`` prefixed id as a verbatim directory, and
``_prepare_dataset_target`` removes an existing target when ``overwrite`` is set,
so an unbounded ``repo_id`` let a peer name a directory on the recording host to
delete. The wire admits the Hub shape only; every refused shape below is one the
reader would have taken as a directory.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.dataset_source import local_dataset_dir
from strands_robots.mesh import security

_HOST_PATHS: tuple[str, ...] = ("/tmp/datasets/one", "./relative/one", "/", "./")
_NOT_A_HUB_ID: tuple[Any, ...] = (
    "",
    "owner",
    "owner/",
    "/name",
    "a/b/c",
    "..",
    "owner/..",
    "owner/../../etc",
    ".hidden/x",
    "owner/x ",
    7,
    None,
    ["a/b"],
)
_HUB_IDS: tuple[str, ...] = ("lerobot/pusht", "cagataycali/so101_pick.v2", "a/b", "org-name/data_set-1.0")


def _start_recording(repo_id: Any) -> dict[str, Any]:
    return security.validate_command(
        {"action": "sim_call", "sim_action": "start_recording", "params": {"repo_id": repo_id}}
    )


@pytest.mark.parametrize("repo_id", _HUB_IDS)
def test_a_hub_id_rides_the_wire_unchanged(repo_id: str) -> None:
    assert _start_recording(repo_id)["params"]["repo_id"] == repo_id


@pytest.mark.parametrize("repo_id", _HOST_PATHS + _NOT_A_HUB_ID)
def test_anything_else_is_refused_with_the_rule(repo_id: Any) -> None:
    with pytest.raises(security.ValidationError, match="Hub dataset id `owner/name`"):
        _start_recording(repo_id)


@pytest.mark.parametrize("repo_id", _HOST_PATHS)
def test_every_refused_path_is_one_the_reader_would_open_as_a_directory(repo_id: str) -> None:
    assert local_dataset_dir(repo_id) is not None, f"{repo_id!r} is not a local path to the reader; widen the wire rule"


@pytest.mark.parametrize("repo_id", _HUB_IDS)
def test_every_admitted_id_is_one_the_reader_leaves_to_the_hub(repo_id: str) -> None:
    assert local_dataset_dir(repo_id) is None


def test_the_value_bound_names_only_published_params() -> None:
    assert security.SIM_CALL_HUB_ID_PARAMS <= security.sim_call_published_params()
    assert not security.SIM_CALL_HUB_ID_PARAMS & security.SIM_CALL_DENIED_PARAMS
