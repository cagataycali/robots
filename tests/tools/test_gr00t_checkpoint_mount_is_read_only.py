"""Pin: the container mounts the checkpoint directory read-only.

``_download_checkpoint`` writes the checkpoint **on the host** through
``snapshot_download`` -- the docstring says so, "no docker mediation" -- and the
inference server only reads it back, so the container has no reason to hold that
directory read-write. It held it read-write anyway: every default mount was
emitted as a bare ``-v host:container``, so a checkpoint that executes on load
(a torch pickle is arbitrary code) could rewrite the checkpoint corpus the
operator trusts, or drop a new file into a directory the host reads.

This is the mount half of the narrowing in #3755, which decided *which*
directories may be mounted; this decides *how*. Both halves come from the same
review: "admit only the checkpoint's own directory ... mount it ``:ro``".

The one directory that stays read-write is the TensorRT engine cache when the
operator put it inside the checkpoint mount: ``--trt-engine-path`` is written by
the server on first compile so that "subsequent runs load from
``trt_engine_path``", and the checkpoint mount is the only writable place in the
default layout where an engine can persist across container recreation. An
unconditional ``:ro`` would break that with a permission error minutes later in
the container log rather than as the call's result.

Pre-fix, every cell asserting ``:ro`` fails: no mount carried a mode at all.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

# See tests/tools/test_gr00t_container_hardening.py: the package's lazy
# __getattr__ resolves the from-import to the tool function, not the module, and
# these cells need the module for its private helpers and ``gi.subprocess``.
gi = importlib.import_module("strands_robots.tools.gr00t_inference")

CHECKPOINT = gi._CHECKPOINT_CONTAINER_PATH
HF_CACHE_CONTAINER_PATH = "/root/.cache/huggingface"


def _mount_specs(**overrides: object) -> list[str]:
    """The ``-v`` arguments of the ``docker run`` argv ``_start_container`` emits."""
    runs: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        runs.append(list(cmd))
        return MagicMock(stdout="", stderr="", returncode=0)

    kwargs: dict[str, object] = {
        "image_name": "gr00t:latest",
        "container_name": "gr00t",
        "port": 5555,
        "volumes": None,
        "hf_token": None,
        "container_command": "tail -f /dev/null",
        "hf_local_dir": "/tmp/ckpt",
        "force": True,
    }
    kwargs.update(overrides)
    with (
        patch.object(gi, "_container_state", return_value="absent"),
        patch.object(gi.subprocess, "run", side_effect=fake_run),
    ):
        result = gi._start_container(**kwargs)  # type: ignore[arg-type]
    assert result["status"] == "success", result
    argv = next(cmd for cmd in runs if cmd[:2] == ["docker", "run"])
    return [argv[i + 1] for i, token in enumerate(argv) if token == "-v"]


def _spec_for(container_path: str, specs: list[str]) -> str:
    """The one mount spec whose container path is ``container_path``."""
    matches = [s for s in specs if s.split(":")[1] == container_path]
    assert len(matches) == 1, f"expected one {container_path} mount, got {matches} in {specs}"
    return matches[0]


# --- the default: the checkpoint is read-only, the HF cache is not -------


def test_the_default_checkpoint_mount_is_read_only():
    """The host wrote the checkpoint and the server only reads it."""
    assert _spec_for(CHECKPOINT, _mount_specs()).endswith(":ro")


def test_the_hugging_face_cache_mount_stays_writable():
    """The container's ``huggingface_hub`` writes into the cache it reuses.

    Reusing already-downloaded snapshots means adding to them, so a ``:ro``
    here would refuse the download the mount exists to make cheap.
    """
    assert not _spec_for(HF_CACHE_CONTAINER_PATH, _mount_specs()).endswith(":ro")


def test_the_checkpoint_mount_names_the_directory_the_caller_asked_for():
    """Read-only changes the mode, not which directory is mounted."""
    assert _spec_for(CHECKPOINT, _mount_specs(hf_local_dir="/tmp/ckpt")).startswith("/tmp/ckpt:")


# --- the TensorRT engine cache carve-out ---------------------------------


@pytest.mark.parametrize(
    ("use_tensorrt", "trt_engine_path", "read_only", "why"),
    [
        pytest.param(False, gi._DEFAULT_TRT_ENGINE_PATH, True, "no engine is written at all", id="trt-off"),
        pytest.param(
            True,
            gi._DEFAULT_TRT_ENGINE_PATH,
            True,
            "the default is relative, so it resolves against the server's working dir",
            id="trt-on-default-relative",
        ),
        pytest.param(
            True, "engines/v1", True, "any relative path resolves outside the mount", id="trt-on-relative-nested"
        ),
        pytest.param(
            True, "/opt/engines/v1", True, "an absolute path elsewhere does not touch the mount", id="trt-on-elsewhere"
        ),
        pytest.param(
            True,
            f"{CHECKPOINT}-other/v1",
            True,
            "a sibling directory that merely shares the prefix is not inside the mount",
            id="trt-on-lookalike-sibling",
        ),
        pytest.param(
            False,
            f"{CHECKPOINT}/v1",
            True,
            "the flag is off, so the path is inert however it is spelled",
            id="trt-off-path-inside",
        ),
        pytest.param(
            True, f"{CHECKPOINT}/v1", False, "the server compiles the engine into the mount", id="trt-on-inside"
        ),
        pytest.param(True, CHECKPOINT, False, "the engine cache is the mount itself", id="trt-on-is-the-mount"),
        pytest.param(
            True,
            "/data/other/../checkpoints/v1",
            False,
            "a spelling that leaves and re-enters normalises to a path inside the mount",
            id="trt-on-traversal-re-enters",
        ),
        pytest.param(
            True,
            "/data/./checkpoints/v1",
            False,
            "a redundant separator does not put the engine cache outside the mount",
            id="trt-on-dot-segment",
        ),
    ],
)
def test_the_engine_cache_decides_the_mode_only_when_it_lands_in_the_mount(
    use_tensorrt: bool, trt_engine_path: str, read_only: bool, why: str
) -> None:
    """A TensorRT engine written into the checkpoint mount keeps it writable."""
    assert gi._checkpoint_mount_is_read_only(use_tensorrt=use_tensorrt, trt_engine_path=trt_engine_path) is read_only, (
        why
    )
    spec = _spec_for(CHECKPOINT, _mount_specs(use_tensorrt=use_tensorrt, trt_engine_path=trt_engine_path))
    assert spec.endswith(":ro") is read_only, f"{spec}: {why}"


# --- what the mode must not touch ----------------------------------------


def test_an_operator_supplied_volume_map_is_emitted_exactly_as_written():
    """Only the operator knows what their own container needs to write.

    A caller who passes ``volumes`` replaces the default layout, so this
    function adds no mode to it - including for the checkpoint path, which is
    only known to be host-written under the default layout.
    """
    specs = _mount_specs(volumes={"/tmp/cp": CHECKPOINT})
    assert specs == [f"/tmp/cp:{CHECKPOINT}"]


def test_the_determinism_wrapper_is_still_mounted_read_only():
    """The wrapper mount had a mode before this change and keeps it."""
    specs = _mount_specs(deterministic=True)
    wrapper = [s for s in specs if s.endswith(f"{gi._DETERMINISTIC_WRAPPER_CONTAINER_PATH}:ro")]
    assert len(wrapper) == 1, specs


def test_the_default_engine_path_is_relative():
    """The reason the default mount can be read-only at all.

    An absolute default under the checkpoint mount would make every TensorRT
    caller keep the mount writable, so this is load-bearing rather than
    incidental.
    """
    assert not gi._DEFAULT_TRT_ENGINE_PATH.startswith("/")


# --- the two producers both forward the facts the mode is decided from ----


_LIFECYCLE_KWARGS: dict[str, object] = {
    "repo_url": gi._DEFAULT_REPO_URL,
    "repo_tag": "n1.7-release",
    "image_name": "gr00t:latest",
    "hf_repo": "nvidia/foo",
    "hf_subfolder": None,
    "hf_local_dir": None,
    "hf_token": None,
    "container_name": None,
    "volumes": None,
    "container_command": "tail -f /dev/null",
    "remove_volumes": False,
    "force": False,
    "checkpoint_path": f"{CHECKPOINT}/m",
    "policy_name": None,
    "port": 8000,
    "data_config": "libero_panda",
    "embodiment_tag": "libero_sim",
    "denoising_steps": 4,
    "host": "0.0.0.0",
    "timeout": 1,
    "vit_dtype": "fp8",
    "llm_dtype": "nvfp4",
    "dit_dtype": "fp8",
    "http_server": False,
    "api_token": None,
    "protocol": "n1.7",
    "use_sim_policy_wrapper": True,
}


def _start_container_kwargs_from(entry_point: str, engine_path: str) -> dict[str, object]:
    """The kwargs the named entry point hands to ``_start_container``."""
    seen: dict[str, object] = {}

    def spy(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "success", "container_name": "gr00t", "skipped": False}

    with patch.object(gi, "_start_container", side_effect=spy):
        if entry_point == "start_container":
            result = gi.gr00t_inference(
                action="start_container", port=8000, use_tensorrt=True, trt_engine_path=engine_path
            )
        else:
            with (
                patch.object(gi, "_build_image", return_value={"status": "success", "skipped": True}),
                patch.object(
                    gi,
                    "_download_checkpoint",
                    return_value={"status": "success", "local_dir": "/tmp/cp", "skipped": True},
                ),
                patch.object(gi, "_start_service", return_value={"status": "success", "message": "up"}),
            ):
                result = gi._lifecycle(
                    phase="full", **{**_LIFECYCLE_KWARGS, "use_tensorrt": True, "trt_engine_path": engine_path}
                )
    assert result["status"] == "success", result
    assert seen, f"{entry_point} did not reach _start_container"
    return seen


@pytest.mark.parametrize("entry_point", ["start_container", "lifecycle"])
def test_both_entry_points_forward_the_engine_cache_facts(entry_point: str) -> None:
    """The mode is decided inside ``_start_container``, from arguments it is given.

    A producer that dropped either keyword would silently mount an operator's
    engine-cache directory read-only and fail their compile in the container
    log, so the forwarding is pinned per call site rather than only through the
    helper's own default.
    """
    engine_path = f"{CHECKPOINT}/v1"
    seen = _start_container_kwargs_from(entry_point, engine_path)
    assert seen.get("use_tensorrt") is True
    assert seen.get("trt_engine_path") == engine_path
