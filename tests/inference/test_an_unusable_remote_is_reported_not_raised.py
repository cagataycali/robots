"""A remote policy that cannot be used is reported, never raised as a bare error.

Two ways ``run_policy(policy_provider="remote")`` left its ``status`` envelope:

* ``websockets`` is not a base dependency, so a base install can resolve a
  release older than 17.1 through another package. The client passes
  ``connect(..., legacy=True)``, which such a release forwards to
  ``socket.create_connection``: ``TypeError: ... unexpected keyword argument
  'legacy'`` on the first rollout step, naming neither the package nor the fix.
  Both WebSocket clients now refuse at construction with the missing-dependency
  ``ImportError`` every optional provider gives, naming the extra.
* the client connects lazily, and the first contact is the rollout's
  ``requires_images`` read, made before the loop whose handler reports a policy
  failure. A server nobody listens on raised ``ConnectionError`` out of every
  rollout surface; it is now the envelope the same failure one step later gets.
"""

from __future__ import annotations

import socket
from importlib import metadata
from typing import Any

import pytest

from strands_robots.inference.client import RemotePolicy
from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient

CLIENTS = [
    (RemotePolicy, {}, "inference"),
    (Cosmos3WebsocketClient, {"host": "127.0.0.1", "port": 1}, "cosmos3-service"),
]


def _installed(release: str | None):
    """``importlib.metadata.version`` as it answers with ``release`` installed (``None``: absent)."""

    def version(name: str) -> str:
        if release is None:
            raise metadata.PackageNotFoundError(name)
        return release

    return version


@pytest.mark.parametrize(("client", "kwargs", "extra"), CLIENTS, ids=[c.__name__ for c, _, _ in CLIENTS])
class TestAnOldWebsocketsIsAMissingDependency:
    @pytest.mark.parametrize("release", ["16.0", "17.0.1", "13.1"])
    def test_a_release_before_17_1_is_refused_naming_the_extra(
        self, monkeypatch, client, kwargs, extra, release
    ) -> None:
        monkeypatch.setattr(metadata, "version", _installed(release))
        with pytest.raises(
            ImportError, match=rf"needs websockets>=17\.1.*{release} is installed.*strands-robots\[{extra}\]"
        ):
            client(**kwargs)

    def test_an_absent_websockets_is_refused_naming_the_extra(self, monkeypatch, client, kwargs, extra) -> None:
        monkeypatch.setattr(metadata, "version", _installed(None))
        with pytest.raises(ImportError, match=rf"not installed: pip install 'strands-robots\[{extra}\]'"):
            client(**kwargs)

    def test_the_installed_release_builds(self, client, kwargs, extra) -> None:
        client(**kwargs)


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


SURFACES = ["run_policy", "eval_policy", "evaluate_benchmark", "run_multi_policy"]


@pytest.mark.parametrize("surface", SURFACES)
def test_a_server_nobody_listens_on_is_an_envelope(surface: str) -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation.benchmark import register_benchmark, unregister_benchmark
    from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    port = _closed_port()
    config: dict[str, Any] = {"port": port, "connect_timeout": 2.0}
    sim = MuJoCoSimEngine()
    sim.create_world()
    sim.add_robot("so101")
    bench = DeclarativeBenchmark.from_dict(
        {"name": "unreachable_remote", "max_steps": 2, "supported_robots": ["so101"], "default_robot": "so101"}
    )
    register_benchmark(bench.name, bench)
    try:
        calls = {
            "run_policy": lambda: sim.run_policy("so101", policy_provider="remote", policy_config=config, duration=0.1),
            "eval_policy": lambda: sim.eval_policy(
                robot_name="so101", policy_provider="remote", policy_config=config, n_episodes=1, max_steps=2
            ),
            "evaluate_benchmark": lambda: sim.evaluate_benchmark(
                bench.name, robot_name="so101", policy_provider="remote", policy_config=config, n_episodes=1
            ),
            "run_multi_policy": lambda: sim.run_multi_policy(
                {"so101": RemotePolicy(port=port, connect_timeout=2.0)}, duration=0.1
            ),
        }
        result = calls[surface]()
    finally:
        unregister_benchmark(bench.name)
        sim.cleanup()
    assert result["status"] == "error"
    assert "RemotePolicy could not reach a PolicyServer" in result["content"][0]["text"]
