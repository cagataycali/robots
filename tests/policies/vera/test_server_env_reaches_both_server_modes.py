"""Every server-environment value the config declares reaches either server mode.

``VeraConfig.server_env()`` is the one place that says what the policy server
must be configured with - checkpoint roots, the IDM run id, the point-tracker
backend. Two runner classes consume that intent, and only one of them calls the
method: ``VeraServerRunner`` hands the overlay to ``Popen(env=...)``, while
``DockerServerRunner`` re-enumerates the same vocabulary by hand as ``docker
run -e`` flags. A value present in one enumeration and absent from the other is
silently dropped - ``docker run`` does not reject an environment variable nobody
passed it, so the server simply starts with a default the caller overrode.

``make_server_runner`` promises both runners drive the same server, and
``docs/policies/vera.md`` tabulates each kwarg against the server input it maps
to without qualifying either mode, so the vocabularies have to agree.

The headline check derives what must be carried from ``server_env()`` itself
rather than from a copied list, so a fifth value added to the overlay is graded
here without touching this file.

A value reaches the container by one of three routes, all of which put it in
front of the server process:

* directly, as ``-e VAR=value`` (a scalar, e.g. the tracker backend);
* translated, as ``-e VAR=<container path>`` beside ``-v <host path>:...`` (a
  host path that must be renamed to where it was mounted);
* by mount alone, where the entrypoint defaults the variable to the mount point
  (``VERA_CKPT_ROOT`` -> ``/ckpts``).

So the rule below counts a value as carried when the container command either
names the variable in an ``-e`` flag or bind-mounts the value it holds.

The overlay is one of two vocabularies the config drives the server with. The
other is the server's own **launch flags**: ``VeraServerRunner._build_command``
composes ``--embodiment``, ``--port``, ``--sample-steps``, ``--teacache-thresh``
and friends directly, while the container path has to get each of those values
in front of the same flags indirectly - as ``-e`` variables the entrypoint turns
back into argv. That indirection is where a flag goes missing without a word
from anyone: ``docker run`` accepts any environment it is handed, and an ``-e``
the entrypoint does not read is inert, so both halves have to line up before a
value actually reaches the server.

``teacache_thresh`` was the flag neither half carried. The subprocess argv passed
``--teacache-thresh 0.25``; the container command named no variable for it and
the entrypoint had no branch to emit it, so the same config ran the DiT cache at
the caller's threshold under ``server_mode="subprocess"`` and at the server's own
default under ``server_mode="docker"`` - reporting success either way. It is a
bare float, needing none of the host->container path translation that keeps
``algo_config`` out of the sweep, so nothing about it was hard to carry.
"""

from __future__ import annotations

import inspect
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from strands_robots.policies.vera import server_runner as server_runner_module
from strands_robots.policies.vera.config import VeraConfig
from strands_robots.policies.vera.server_runner import (
    DockerServerRunner,
    VeraServerRunner,
    make_server_runner,
)

# A config that exercises every branch of the overlay at once: two host paths,
# one wandb run id and one backend name.
FULLY_CONFIGURED: dict[str, Any] = {
    "embodiment": "mimicgen",
    "server_port": 8800,
    "ckpt_root": "/data/vera-ckpts",
    "wan_ckpt_root": "/data/wan",
    "dynamics_run_id": "37oa162u",
    "tracker_backend": "cotracker",
    "text_prompt": "stack the cube",
    "sample_steps": 10,
    # Deliberately not the 0.10 default: a forwarded value has to be the one the
    # caller asked for, and a default would pass even if the server chose it.
    "teacache_thresh": 0.25,
}

# The shipped container entrypoint, located from the module that launches it so
# a move is a failure here rather than a silently skipped check.
ENTRYPOINT = Path(server_runner_module.__file__).with_name("docker") / "entrypoint.sh"

# Launch flags the container path deliberately does NOT carry verbatim, each with
# the reason it differs. Anything else the subprocess argv can compose must reach
# the container command, or the two modes launch different servers.
FLAGS_NOT_CARRIED_VERBATIM: dict[str, str] = {
    "--host": (
        "the container binds 0.0.0.0 and publishes the port, so the host-side "
        "address is the client's business and not the server's"
    ),
    "--algo-config": (
        "a host filesystem path, which a container can only be given under a "
        "bind mount; picking that mount point (and how it interacts with the "
        "entrypoint's own per-embodiment default) is a separate decision"
    ),
}

# Non-vacuity floor for the headline sweep: the overlay must really be
# populated, or "every value is carried" would be a statement about nothing.
MINIMUM_OVERLAY_VALUES = 4


def _config(mode: str, **overrides: Any) -> VeraConfig:
    return VeraConfig(server_mode=mode, **{**FULLY_CONFIGURED, **overrides})


def _docker_argv(cfg: VeraConfig) -> list[str]:
    """The real ``docker run`` argv, with only the ``docker`` lookup stubbed.

    Nothing is started. ``_docker`` shells out to ``which`` purely to find the
    binary, which a test host need not have; the command under inspection is
    composed by shipped code.
    """
    runner = make_server_runner(cfg)
    assert isinstance(runner, DockerServerRunner), f"server_mode={cfg.server_mode!r} chose {type(runner).__name__}"
    runner._docker = lambda: "/usr/bin/docker"  # type: ignore[method-assign]
    return list(runner._build_run_command())


def _env_flags(argv: list[str]) -> dict[str, str]:
    """The ``-e NAME=value`` pairs of a ``docker run`` argv, keyed by name."""
    flags: dict[str, str] = {}
    for flag, payload in zip(argv, argv[1:], strict=False):
        if flag == "-e" and "=" in payload:
            name, _, value = payload.partition("=")
            flags[name] = value
    return flags


def _mounted_host_paths(argv: list[str]) -> set[str]:
    """The host side of every ``-v <host>:<container>[:opts]`` bind mount."""
    return {payload.split(":", 1)[0] for flag, payload in zip(argv, argv[1:], strict=False) if flag == "-v"}


def _subprocess_flags(cfg: VeraConfig) -> dict[str, str]:
    """The ``--flag value`` pairs of the subprocess launch argv, keyed by flag.

    A bare switch (``--no-teacache``) maps to the empty string, so presence and
    payload stay distinguishable.
    """
    runner = make_server_runner(cfg)
    assert isinstance(runner, VeraServerRunner), f"server_mode={cfg.server_mode!r} chose {type(runner).__name__}"
    argv = runner._build_command()
    flags: dict[str, str] = {}
    for token, following in zip(argv, [*argv[1:], ""], strict=True):
        if token.startswith("--"):
            flags[token] = "" if following.startswith("--") else following
    return flags


def _entrypoint_server_argv(tmp_path: Path, **overrides: str) -> list[str]:
    """Run the shipped entrypoint under bash and return the server argv it execs.

    Nothing model-sized runs: a stub ``python`` earlier on ``PATH`` echoes the
    arguments it was handed and exits, so what comes back is the argv the
    container would really have started the policy server with. ``pusht`` is
    used because it is the fully-local embodiment - no WAN base to find.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text('#!/bin/sh\nfor a in "$@"; do echo "ARG:$a"; done\n')
    stub.chmod(0o755)
    ckpts = tmp_path / "ckpts"
    ckpts.mkdir()
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "VERA_EMBODIMENT": "pusht",
        "VERA_CKPT_ROOT": str(ckpts),
        **overrides,
    }
    done = subprocess.run(  # noqa: S603 - list args, no shell
        ["bash", str(ENTRYPOINT)], capture_output=True, text=True, env=env, timeout=120
    )
    assert done.returncode == 0, f"entrypoint exited {done.returncode}: {done.stderr}"
    return [line[len("ARG:") :] for line in done.stdout.splitlines() if line.startswith("ARG:")]


def _unreachable_in_docker(cfg: VeraConfig) -> dict[str, str]:
    """Overlay values the container command carries by no route at all."""
    argv = _docker_argv(cfg)
    named = _env_flags(argv)
    mounted = _mounted_host_paths(argv)
    return {name: value for name, value in cfg.server_env().items() if name not in named and value not in mounted}


class TestTheOverlayIsCarriedByBothModes:
    def test_the_overlay_is_populated(self):
        """Premise: a clean result below has to be about a real vocabulary."""
        overlay = _config("subprocess").server_env()
        assert len(overlay) >= MINIMUM_OVERLAY_VALUES, overlay

    def test_the_subprocess_mode_carries_the_overlay_wholesale(self):
        """The mode that calls ``server_env()`` is the reference to grade against.

        It carries the overlay by merging it into the child environment rather
        than by naming values one at a time, which is why it cannot omit one.
        """
        launch = inspect.getsource(VeraServerRunner.start)
        assert "cfg.server_env()" in launch, "the subprocess mode no longer reads the overlay"
        assert "env=env" in launch, "the overlay no longer reaches the child process"

    def test_the_container_command_carries_every_overlay_value(self):
        """The headline: no overlay value is dropped on the container path.

        Pre-fix this reports ``VERA_TRACKER_BACKEND``, the one value the hand-
        written ``-e`` list omitted.
        """
        missing = _unreachable_in_docker(_config("docker"))
        assert not missing, (
            "server_env() declares these for the server, and the docker run command "
            f"neither names them in an -e flag nor mounts their value: {missing}. "
            "A dropped value means the server starts with a default the caller overrode."
        )

    @pytest.mark.parametrize("backend", ["cotracker", "alltracker", "vggt"])
    def test_the_tracker_backend_reaches_the_container(self, backend):
        """A backend name is forwarded verbatim - no path translation applies."""
        cfg = _config("docker", tracker_backend=backend)
        assert cfg.server_env()["VERA_TRACKER_BACKEND"] == backend, "premise: the overlay carries it"
        assert _env_flags(_docker_argv(cfg))["VERA_TRACKER_BACKEND"] == backend

    def test_both_modes_agree_on_the_tracker_backend(self):
        """The same config configures the same tracker whichever mode runs it."""
        sub = _config("subprocess")
        dock = _config("docker")
        assert _env_flags(_docker_argv(dock))["VERA_TRACKER_BACKEND"] == sub.server_env()["VERA_TRACKER_BACKEND"]


class TestNothingElseMoved:
    """Controls: these hold before and after, and fail for the shortcut fixes."""

    def test_an_unset_tracker_backend_is_not_forwarded(self):
        """No empty override is injected when the caller expressed no preference.

        The entrypoint treats an empty value as unset, but emitting one states a
        choice the config never made.
        """
        cfg = _config("docker", tracker_backend=None)
        assert cfg.tracker_backend is None, "premise: nothing to forward"
        assert "VERA_TRACKER_BACKEND" not in " ".join(_docker_argv(cfg))

    def test_a_host_path_is_forwarded_as_its_container_path(self):
        """``wan_ckpt_root`` is mounted, so the container sees ``/wan``.

        Forwarding the host spelling verbatim - the shortcut a scalar fix invites
        - would name a directory that does not exist inside the container.
        """
        cfg = _config("docker")
        argv = _docker_argv(cfg)
        assert _env_flags(argv)["VERA_WAN_CKPT_ROOT"] == "/wan"
        assert "/data/wan:/wan:ro" in argv

    def test_the_checkpoint_root_is_carried_by_its_mount(self):
        """``ckpt_root`` needs no ``-e``: the entrypoint defaults it to the mount."""
        argv = _docker_argv(_config("docker"))
        assert "/data/vera-ckpts:/ckpts:ro" in argv
        assert "VERA_CKPT_ROOT" not in _env_flags(argv)

    def test_the_run_id_is_still_forwarded_directly(self):
        """The one scalar that was already carried keeps its route."""
        assert _env_flags(_docker_argv(_config("docker")))["VERA_DYNAMICS_RUN_ID"] == "37oa162u"

    def test_the_subprocess_launch_argv_is_unchanged(self):
        """The mode that already worked composes exactly the same command.

        ``tracker_backend`` has never been a server flag - it travels by
        environment - so it must not appear in the subprocess argv either.
        """
        cfg = _config("subprocess")
        runner = make_server_runner(cfg)
        assert isinstance(runner, VeraServerRunner)
        argv = runner._build_command()
        assert "--tracker-backend" not in argv
        assert "cotracker" not in argv

    def test_the_command_stays_list_args(self):
        """Every token is a plain string: no shell string is ever assembled."""
        argv = _docker_argv(_config("docker"))
        assert all(isinstance(token, str) for token in argv)
        assert argv[-1] == VeraConfig(server_mode="docker", embodiment="mimicgen").docker_image


requires_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="the container entrypoint is a bash script")


class TestTheLaunchFlagsAreCarriedByBothModes:
    """The server's own flags, the second vocabulary the config drives it with."""

    def test_the_flag_sweep_grades_more_than_one_flag(self):
        """Premise: a clean sweep below has to be about a real set of flags."""
        graded = {
            f: v for f, v in _subprocess_flags(_config("subprocess")).items() if f not in FLAGS_NOT_CARRIED_VERBATIM
        }
        assert len(graded) >= 5, graded

    def test_the_container_command_carries_every_launch_flag_value(self):
        """The headline: no launch flag's value is dropped on the container path.

        Derived from the subprocess argv rather than from a copied list, so a
        seventh flag added to ``_build_command`` is graded here on arrival.
        Pre-fix this reports ``--teacache-thresh``.
        """
        argv = _docker_argv(_config("docker"))
        carried = set(_env_flags(argv).values()) | _mounted_host_paths(argv)
        dropped = {
            flag: value
            for flag, value in _subprocess_flags(_config("subprocess")).items()
            if value and flag not in FLAGS_NOT_CARRIED_VERBATIM and value not in carried
        }
        assert not dropped, (
            "the subprocess mode launches the server with these flags, and the docker run "
            f"command carries their values by no route: {dropped}. The container would start "
            "on the server's own default for each one, under a success."
        )

    @pytest.mark.parametrize("thresh", [0.05, 0.15, 0.25])
    def test_the_threshold_reaches_the_container(self, thresh):
        """A tuned threshold is named for the container, not left behind."""
        cfg = _config("docker", teacache_thresh=thresh)
        assert _env_flags(_docker_argv(cfg))["VERA_TEACACHE_THRESH"] == str(thresh)

    @requires_bash
    def test_the_entrypoint_turns_the_variable_into_the_flag(self, tmp_path):
        """An ``-e`` nothing in the container reads would be inert.

        The other half of the fix: the shipped entrypoint has to translate the
        variable back into the flag the server takes, so this runs it.
        """
        argv = _entrypoint_server_argv(tmp_path, VERA_TEACACHE_THRESH="0.25")
        assert "--teacache-thresh" in argv, argv
        assert argv[argv.index("--teacache-thresh") + 1] == "0.25"

    @requires_bash
    def test_both_modes_launch_the_server_on_the_same_threshold(self, tmp_path):
        """End to end: one config, two launch modes, one server argv value."""
        subprocess_value = _subprocess_flags(_config("subprocess"))["--teacache-thresh"]
        container_env = _env_flags(_docker_argv(_config("docker")))
        argv = _entrypoint_server_argv(tmp_path, VERA_TEACACHE_THRESH=container_env["VERA_TEACACHE_THRESH"])
        assert argv[argv.index("--teacache-thresh") + 1] == subprocess_value == "0.25"


class TestTheTeacachePairStaysOneEitherOr:
    """Controls: the off-switch keeps its meaning, and the exclusions are stated."""

    def test_disabling_teacache_names_no_threshold_in_the_container(self):
        """A threshold means nothing once the cache is off, so none is sent."""
        argv = _docker_argv(_config("docker", teacache=False))
        named = _env_flags(argv)
        assert named["VERA_NO_TEACACHE"] == "1"
        assert "VERA_TEACACHE_THRESH" not in named

    def test_disabling_teacache_names_no_threshold_in_the_subprocess_argv(self):
        """The mode that already worked keeps the same either/or."""
        flags = _subprocess_flags(_config("subprocess", teacache=False))
        assert "--no-teacache" in flags
        assert "--teacache-thresh" not in flags

    @requires_bash
    def test_the_off_switch_wins_in_the_container(self, tmp_path):
        """Both variables set is still one request: the cache is off."""
        argv = _entrypoint_server_argv(tmp_path, VERA_NO_TEACACHE="1", VERA_TEACACHE_THRESH="0.25")
        assert "--no-teacache" in argv
        assert "--teacache-thresh" not in argv

    def test_each_excluded_flag_states_why_it_differs(self):
        """An exclusion without a reason is how the dropped flag stayed dropped."""
        assert all(reason.strip() for reason in FLAGS_NOT_CARRIED_VERBATIM.values())
        assert set(FLAGS_NOT_CARRIED_VERBATIM) <= set(
            _subprocess_flags(_config("subprocess", algo_config="/data/algo.yaml"))
        )

    def test_the_container_binds_every_interface_by_design(self):
        """``--host`` is the one flag the container is meant to disagree on."""
        cfg = _config("docker", host="127.0.0.1")
        assert _env_flags(_docker_argv(cfg))["VERA_HOST"] == "0.0.0.0"
        assert f"{cfg.server_port}:{cfg.server_port}" in _docker_argv(cfg)
