"""GR00T N1.7 is a ``lerobot_local`` policy type, resolved through the plain path.

The hand-rolled ``groot`` provider carried three release-specific loaders, a ZMQ
service mode and its own data-config table so that N1.5, N1.6 and N1.7 could
all be driven from one class. lerobot 0.6 ships the N1.7 port itself
(``lerobot.policies.groot``: ``GrootConfig`` registered as ``"groot"``,
``GrootPolicy`` with ``from_pretrained``), self-contained - no Isaac-GR00T
checkout, no ``gr00t`` import. So the provider is gone and GR00T N1.7 is
``policy_type="groot"`` on ``lerobot_local``, the same generic path ``act`` and
``smolvla`` take. This module pins the two halves of that claim without a GPU
or a download: the type resolves through the shared resolver (no molmoact2-style
side loader is needed), and the registry routes NVIDIA's Hub ids and the extra
to the provider that now owns them. The live half - loading
``nvidia/GR00T-N1.7-3B`` and rolling it out - is the CUDA-gated cell at the
bottom.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

import strands_robots
from strands_robots.registry.policies import get_policy_provider, resolve_policy

_REPO = Path(strands_robots.__file__).resolve().parent.parent
_HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None


class TestTheTypeResolvesThroughThePlainPath:
    """``resolve_policy_class_by_name("groot")`` is lerobot's own class, found like any other."""

    @pytest.mark.skipif(not _HAS_LEROBOT, reason="needs lerobot")
    def test_the_class_is_lerobot_groot_policy(self) -> None:
        from strands_robots.policies.lerobot_local.resolution import resolve_policy_class_by_name

        cls = resolve_policy_class_by_name("groot")
        assert cls.__module__ == "lerobot.policies.groot.modeling_groot"
        assert cls.__name__ == "GrootPolicy"
        assert hasattr(cls, "from_pretrained")

    @pytest.mark.skipif(not _HAS_LEROBOT, reason="needs lerobot")
    def test_the_config_registers_as_the_groot_type(self) -> None:
        from lerobot.policies.groot.configuration_groot import GrootConfig

        cfg = GrootConfig()
        assert cfg.type == "groot"
        # The knobs the trainer forwards exist on the config itself.
        for field in ("embodiment_tag", "tune_llm", "tune_visual", "tune_projector", "tune_diffusion_model"):
            assert hasattr(cfg, field), field

    @pytest.mark.skipif(not _HAS_LEROBOT, reason="needs lerobot")
    def test_the_type_is_discoverable(self) -> None:
        from strands_robots.policies.lerobot_local.resolution import list_policy_types

        assert "groot" in list_policy_types()

    @pytest.mark.skipif(not _HAS_LEROBOT, reason="needs lerobot")
    def test_no_side_loader_is_needed(self) -> None:
        """The molmoact2 wrapper exists for a checkpoint lerobot's factory cannot read; GR00T is not one."""
        from strands_robots.policies.lerobot_local import molmoact2

        assert molmoact2.is_molmoact2("nvidia/GR00T-N1.7-3B", "groot") is False

    @pytest.mark.skipif(not _HAS_LEROBOT, reason="needs lerobot")
    def test_lerobot_groot_imports_no_isaac_groot(self) -> None:
        """The port is self-contained: the removed provider's ``gr00t`` dependency is not back through lerobot."""
        import sys

        import lerobot.policies.groot.modeling_groot  # noqa: F401

        assert not any(name == "gr00t" or name.startswith("gr00t.") for name in sys.modules)


class TestTheRegistryRoutesGrootToLerobotLocal:
    def test_nvidia_hub_ids_resolve_to_lerobot_local(self) -> None:
        provider, kwargs = resolve_policy("nvidia/GR00T-N1.7-3B")
        assert provider == "lerobot_local"
        assert kwargs == {"pretrained_name_or_path": "nvidia/GR00T-N1.7-3B"}

    def test_policy_type_is_a_declared_keyword(self) -> None:
        spec = get_policy_provider("lerobot_local")
        assert spec is not None
        assert "policy_type" in spec["config_keys"]
        assert "nvidia" in spec["hf_orgs"]

    def test_the_groot_extra_layers_lerobot_groot_on_the_lerobot_extra(self) -> None:
        extras = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]
        assert "groot-service" not in extras
        assert extras["groot"] == ["strands-robots[lerobot]", "lerobot[groot]>=0.6.1,<0.7.0"]
        assert "strands-robots[groot]" in extras["all"]


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_LEROBOT, reason="needs lerobot")
def test_n17_loads_and_emits_a_chunk_on_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live: the 3B checkpoint loads through the plain path and answers one observation."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    pytest.importorskip("lerobot.policies.groot.modeling_groot", reason="needs the [groot] extra")
    monkeypatch.setenv("STRANDS_TRUST_REMOTE_CODE", "1")
    from strands_robots import create_policy

    policy = create_policy(
        "lerobot_local",
        pretrained_name_or_path="nvidia/GR00T-N1.7-3B",
        policy_type="groot",
        embodiment="so101",
        device="cuda",
    )
    from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

    assert isinstance(policy, LerobotLocalPolicy)
    assert policy.policy_type == "groot"
    assert type(policy._policy).__name__ == "GrootPolicy"
