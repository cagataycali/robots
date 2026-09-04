"""Render dimension + payload-size safety caps.

``render()`` and ``render_depth()`` are LLM-callable tools, so the width/height
and the ``STRANDS_ROBOTS_RENDER_MAX_BYTES`` size cap are attacker-influenced.
These tests pin the out-of-memory / bad-input guardrails: non-integer
dimensions, the absolute 4096x4096 framebuffer ceiling, the per-model offscreen
framebuffer cap, and a non-positive byte budget. The dimension guards
short-circuit before any GL context is created, so they are deterministic even
on a headless host.

``render_depth`` and ``get_camera_params`` share the same
``_validate_render_dims`` guard as ``render`` but through independent call
sites, so each is pinned with its own suite plus a cross-call-site parity
suite: a refactor that drops one of them (letting a 8000x8000 depth request
reach the offscreen framebuffer, or building a pinhole ``K`` for a 0-pixel
image) would slip past the RGB tests alone. The guard takes its caller's name,
so the parity those suites assert is of the domain and the reason - each entry
point names itself as the subject; see
``tests/simulation/test_render_dimension_refusal_names_the_caller.py``.

``get_camera_params`` reports failure by exception rather than by an
agent-tool dict, and its dimensions are not merely a buffer size: ``fx``,
``fy``, ``cx`` and ``cy`` are all linear in the image size, so a 0 height
yields a singular ``K`` (``fy == 0``, no unprojection possible) and a negative
height yields an axis-flipped ``K`` that silently mirrors every unprojected
point. A size past the framebuffer cap describes a frame ``render`` and
``get_frame`` refuse to draw, breaking the symmetry those three APIs promise.
"""

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.rendering import _max_render_bytes  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim_with_world():
    """A minimal simulation with an empty world (no robot, no GL needed)."""
    sim = Simulation()
    sim.create_world()
    yield sim
    sim.destroy()


class TestRenderDimensionCaps:
    def test_non_integer_dimensions_rejected(self, sim_with_world):
        """A string width is refused with a type-explicit error, not a crash."""
        res = sim_with_world.render(camera_name="default", width="640", height=480)
        assert res["status"] == "error"
        assert "must be int" in res["content"][0]["text"]

    def test_dimensions_over_absolute_ceiling_rejected(self, sim_with_world):
        """Beyond the hard 4096 ceiling is refused regardless of model config."""
        res = sim_with_world.render(camera_name="default", width=8000, height=480)
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        assert "absolute maximum" in text
        assert "4096x4096" in text

    def test_dimensions_over_model_offscreen_cap_rejected(self, sim_with_world):
        """Within the absolute ceiling but past the model's offscreen framebuffer
        cap (default 1280x960) is refused with the actual cap surfaced."""
        cap_w = int(sim_with_world._world._model.vis.global_.offwidth)
        assert cap_w < 4096  # precondition: model cap is below the hard ceiling
        res = sim_with_world.render(camera_name="default", width=cap_w + 1, height=48)
        assert res["status"] == "error"
        assert "offscreen framebuffer cap" in res["content"][0]["text"]


class TestRenderDepthDimensionCaps:
    """``render_depth`` must enforce the same dimension safety caps as ``render``.

    The depth path has its own ``_validate_render_dims`` call site, so these
    mirror the RGB caps to guarantee the two paths cannot drift: a bad-dimension
    depth request is rejected before any offscreen framebuffer is allocated.
    """

    def test_non_integer_dimensions_rejected(self, sim_with_world):
        res = sim_with_world.render_depth(camera_name="default", width="640", height=480)
        assert res["status"] == "error"
        assert "must be int" in res["content"][0]["text"]

    def test_dimensions_over_absolute_ceiling_rejected(self, sim_with_world):
        res = sim_with_world.render_depth(camera_name="default", width=8000, height=480)
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        assert "absolute maximum" in text
        assert "4096x4096" in text

    def test_dimensions_over_model_offscreen_cap_rejected(self, sim_with_world):
        cap_w = int(sim_with_world._world._model.vis.global_.offwidth)
        assert cap_w < 4096  # precondition: model cap is below the hard ceiling
        res = sim_with_world.render_depth(camera_name="default", width=cap_w + 1, height=48)
        assert res["status"] == "error"
        assert "offscreen framebuffer cap" in res["content"][0]["text"]

    def test_depth_and_rgb_agree_on_bad_dimensions(self, sim_with_world):
        """The two render paths reject identical bad dimensions for the same
        reason, each naming itself - the parity contract that keeps the shared
        guard from drifting.

        Byte-identity of the two texts is what this asserted before, and it is
        a stronger claim than the drift it exists to catch: it also required
        ``render_depth`` to report ``render``, so the caller of one method was
        pointed at the other. Comparing the reason with each subject stripped
        keeps the anti-drift guarantee and makes the misattribution a failure
        rather than the pinned behaviour.
        """
        for width, height in (("640", 480), (8000, 480), (0, 480), (640, -1)):
            rgb = sim_with_world.render(camera_name="default", width=width, height=height)
            depth = sim_with_world.render_depth(camera_name="default", width=width, height=height)
            assert rgb["status"] == "error"
            assert depth["status"] == "error"
            rgb_text = rgb["content"][0]["text"]
            depth_text = depth["content"][0]["text"]
            assert rgb_text.startswith("render: "), rgb_text
            assert depth_text.startswith("render_depth: "), depth_text
            assert rgb_text.removeprefix("render: ") == depth_text.removeprefix("render_depth: ")


class TestRenderMaxBytesCap:
    def test_non_positive_byte_budget_rejected(self, monkeypatch):
        """A non-positive size cap surfaces an error rather than disabling the cap."""
        monkeypatch.setenv("STRANDS_ROBOTS_RENDER_MAX_BYTES", "-5")
        with pytest.raises(ValueError, match="must be positive"):
            _max_render_bytes()

    def test_zero_byte_budget_rejected(self, monkeypatch):
        """Zero is treated as non-positive (would otherwise reject every render)."""
        monkeypatch.setenv("STRANDS_ROBOTS_RENDER_MAX_BYTES", "0")
        with pytest.raises(ValueError, match="must be positive"):
            _max_render_bytes()


class TestCameraParamsDimensionCaps:
    """``get_camera_params`` must honor the same dimension contract as ``render``.

    It builds intrinsics for a caller-supplied image size, so a dimension the
    renderer cannot produce yields a ``K`` describing a frame that does not
    exist. Failure is reported by ``ValueError`` (this API returns a
    ``CameraParams``, not an agent-tool dict).
    """

    @pytest.mark.parametrize(
        ("width", "height"),
        [(0, 240), (320, 0), (-64, 240), (320, -48)],
    )
    def test_non_positive_dimensions_rejected(self, sim_with_world, width, height):
        """A 0 height would give a singular K; a negative one an axis-flipped K."""
        with pytest.raises(ValueError, match="must be > 0"):
            sim_with_world.get_camera_params(camera_name="default", width=width, height=height)

    @pytest.mark.parametrize("width", [2.7, "640", [320]])
    def test_non_integer_dimensions_rejected(self, sim_with_world, width):
        """A fractional/string/sequence width is refused, not silently truncated."""
        with pytest.raises(ValueError, match="must be int"):
            sim_with_world.get_camera_params(camera_name="default", width=width, height=240)

    def test_bool_dimension_rejected_by_type(self, sim_with_world):
        """``True`` is an int subclass but never a pixel count."""
        with pytest.raises(ValueError, match="got bool/int"):
            sim_with_world.get_camera_params(camera_name="default", width=True, height=240)

    def test_dimensions_over_absolute_ceiling_rejected(self, sim_with_world):
        """Params for a frame past the hard 4096 ceiling are refused."""
        with pytest.raises(ValueError, match="absolute maximum"):
            sim_with_world.get_camera_params(camera_name="default", width=8000, height=480)

    def test_dimensions_over_model_offscreen_cap_rejected(self, sim_with_world):
        """Past the model's offscreen framebuffer cap is refused with the cap named."""
        cap_w = int(sim_with_world._world._model.vis.global_.offwidth)
        with pytest.raises(ValueError, match="offscreen framebuffer cap"):
            sim_with_world.get_camera_params(camera_name="default", width=cap_w + 1, height=48)

    def test_valid_dimensions_scale_the_intrinsics(self, sim_with_world):
        """Accepted dimensions still produce K linear in the image size."""
        sim_with_world.add_camera("look", position=[0.6, 0.0, 0.4], target=[0.0, 0.0, 0.1], width=320, height=240)

        base = sim_with_world.get_camera_params(camera_name="look")
        assert (base.width, base.height) == (320, 240)

        doubled = sim_with_world.get_camera_params(camera_name="look", width=640, height=480)
        assert (doubled.width, doubled.height) == (640, 480)
        assert doubled.K[1][1] == pytest.approx(2.0 * base.K[1][1])
        assert doubled.K[0][2] == pytest.approx(2.0 * base.K[0][2])
        assert doubled.K[1][2] == pytest.approx(2.0 * base.K[1][2])


class TestRenderDimensionGuardParity:
    """Every dimension ``render`` rejects, ``get_camera_params`` rejects too.

    The three render-dimension call sites (``render``, ``get_frame`` /
    ``render_depth``, ``get_camera_params``) describe the same image. An
    accepted domain that diverges between them lets a caller hold intrinsics
    for a frame no call can render. What must match is the verdict and the
    reason - each entry point names itself as the subject, so a caller is
    pointed at the call it made.
    """

    @pytest.mark.parametrize(
        ("width", "height"),
        [(0, 240), (320, 0), (-64, 240), (320, -48), (2.7, 240), ("640", 240), (True, 240), (8000, 480)],
    )
    def test_render_and_camera_params_reject_identically(self, sim_with_world, width, height):
        """Same rejection, same reason, each naming the call it came through.

        Byte-identity of the two texts is what this compared before, which also
        required ``get_camera_params`` to report ``render`` - so the divergence
        it guards against was pinned together with a subject naming the wrong
        method. The reason is what has to agree; the subject is what has to
        differ.
        """
        rendered = sim_with_world.render(camera_name="default", width=width, height=height)
        assert rendered["status"] == "error"
        rendered_text = rendered["content"][0]["text"]
        assert rendered_text.startswith("render: "), rendered_text

        with pytest.raises(ValueError) as excinfo:
            sim_with_world.get_camera_params(camera_name="default", width=width, height=height)
        params_text = str(excinfo.value)
        assert params_text.startswith("get_camera_params: "), params_text
        assert params_text.removeprefix("get_camera_params: ") == rendered_text.removeprefix("render: ")
