"""``Cosmos3DiffusersBackend`` refuses a checkpoint the installed diffusers cannot build.

``Cosmos3OmniPipeline.from_pretrained`` does not raise when the installed
``diffusers`` builds a different architecture than the checkpoint holds: it logs
"newly initialized" / "not used when initializing" warnings and leaves every
unmatched parameter on the ``meta`` device. Measured against the released wheels
with ``nvidia/Cosmos3-Edge`` (built against diffusers 0.40.0.dev0):

* diffusers 0.39.0 - ``from_pretrained`` returns a pipeline with 112 of 633
  transformer parameters still on ``meta``; the only symptom was
  ``NotImplementedError: Cannot copy out of meta tensor`` out of the *next*
  statement, naming neither diffusers, nor its version, nor the checkpoint.
* diffusers 0.40.0.dev0 - 0 of 549 on ``meta``, loads and runs.

``nvidia/Cosmos3-Nano`` loads cleanly on both, so the required diffusers is a
property of the checkpoint rather than of the library, which is why this is
reported at load time instead of pinned as a version range.

The module's ``_install_hint`` covers only the ImportError path; on a diffusers
that *has* the symbol but is too old for the checkpoint the import succeeds, so
that remedy was unreachable on exactly the state it exists for. These pin the
refusal, its placement (before the device copy, so the failure costs no 9 GB
transfer and the silent-random-weights case is caught too), and that a healthy
load is untouched.

No GPU and no weights: the pipeline is injected through the ``sys.modules``
seam the safety-checker tests use, and a parameter stand-in only needs a
``.device.type``, so the matrix runs with neither torch nor diffusers installed.
Running torch-less costs one explicit argument, ``dtype=_SERVED_DTYPE`` -- see
that constant.
"""

import sys
import types

import pytest

from strands_robots.policies.cosmos3 import Cosmos3DiffusersBackend
from strands_robots.policies.cosmos3.embodiments import get_embodiment
from strands_robots.policies.cosmos3.policy_diffusers import (
    _checkpoint_mismatch_hint,
    _unloaded_checkpoint_tensors,
)

# ``_load_pipeline`` resolves its dtype string against the imported torch module.
# The backend defaults to ``"bfloat16"``, which the numpy-backed torch stand-in
# ``conftest`` installs when real torch is absent deliberately does *not* serve -
# ``float16`` and ``bfloat16`` are both pinned as reaches past its subset in
# ``tests/test_torch_stand_in_serves_or_skips.py``. That stand-in's serve-or-skip
# contract is defeated here by the three-argument ``getattr(torch_mod, dtype,
# None)`` the resolver uses: the default swallows the ``AttributeError`` half,
# and with it the skip, so an unserved dtype arrives as ``None`` and is reported
# as ``ValueError: Unknown torch dtype 'bfloat16'`` -- a message about the
# caller's dtype string, on a run whose actual shortfall is the absent extra.
#
# So these cases name a dtype the stand-in serves. They pin the checkpoint scan,
# not dtype resolution, and the dtype reaches only the injected fake pipeline's
# ``from_pretrained`` kwargs. Dropping the argument passes under real torch and
# fails in the torch-less environment the module docstring above promises.
_SERVED_DTYPE = "float32"


def _param(device_type):
    """A stand-in for a torch parameter: the scan reads only ``.device.type``."""
    return types.SimpleNamespace(device=types.SimpleNamespace(type=device_type))


class FakeModule:
    """A pipeline component exposing ``named_parameters`` like ``torch.nn.Module``."""

    def __init__(self, **params):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())


class FakePipe:
    """A ``Cosmos3OmniPipeline`` stand-in: a ``components`` dict plus ``to``."""

    def __init__(self, components):
        self.components = components
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


def _loaded_pipe():
    """Every parameter on a real device: the checkpoint was fully consumed."""
    return FakePipe(
        {
            "transformer": FakeModule(**{"layers.0.norm_q.weight": _param("cpu")}),
            "vae": FakeModule(**{"encoder.conv.weight": _param("cpu")}),
        }
    )


def _mismatched_pipe():
    """Two transformer parameters left on ``meta`` - the 0.39.0-with-Edge shape."""
    return FakePipe(
        {
            "transformer": FakeModule(
                **{
                    "layers.0.self_attn.norm_q.weight": _param("meta"),
                    "layers.0.mlp.gate_proj.weight": _param("meta"),
                    "layers.0.self_attn.q_proj.weight": _param("cpu"),
                }
            ),
            "vae": FakeModule(**{"encoder.conv.weight": _param("cpu")}),
        }
    )


def _install_fake_diffusers(monkeypatch, pipe):
    """Point ``from diffusers import Cosmos3OmniPipeline`` at ``pipe``."""

    class FakeOmniPipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            return pipe

    fake = types.ModuleType("diffusers")
    fake.Cosmos3OmniPipeline = FakeOmniPipeline
    fake.CosmosActionCondition = object
    monkeypatch.setitem(sys.modules, "diffusers", fake)


class TestUnloadedCheckpointTensors:
    """The scan is an exact test of "did the load consume the checkpoint"."""

    def test_names_every_parameter_left_on_meta(self):
        assert _unloaded_checkpoint_tensors(_mismatched_pipe()) == [
            "transformer.layers.0.self_attn.norm_q.weight",
            "transformer.layers.0.mlp.gate_proj.weight",
        ]

    def test_a_fully_loaded_pipeline_reports_nothing(self):
        assert _unloaded_checkpoint_tensors(_loaded_pipe()) == []

    def test_non_module_components_are_skipped_not_crashed(self):
        """A real pipeline's ``components`` holds a tokenizer, a scheduler and
        ``None`` slots alongside the modules; none has ``named_parameters``."""
        pipe = FakePipe(
            {
                "text_tokenizer": object(),
                "scheduler": object(),
                "sound_tokenizer": None,
                "safety_checker": None,
                "transformer": FakeModule(**{"w": _param("meta")}),
            }
        )
        assert _unloaded_checkpoint_tensors(pipe) == ["transformer.w"]

    def test_a_pipeline_without_a_components_mapping_reports_nothing(self):
        assert _unloaded_checkpoint_tensors(types.SimpleNamespace()) == []


class TestTheRefusalIsActionable:
    """The message must name what diffusers' own error does not."""

    def test_names_the_checkpoint_the_version_the_count_and_the_remedy(self):
        text = _checkpoint_mismatch_hint("nvidia/Cosmos3-Edge", ["transformer.a", "transformer.b"])
        assert "nvidia/Cosmos3-Edge" in text
        assert "2 tensor(s)" in text
        assert "transformer.a" in text
        assert "diffusers" in text
        assert "git+https://github.com/huggingface/diffusers" in text
        assert "backend='service'" in text

    def test_says_why_it_matters_rather_than_only_that_it_failed(self):
        text = _checkpoint_mismatch_hint("m", ["t.a"])
        assert "randomly initialized" in text


class TestLoadPipelineRefusesAMismatchedCheckpoint:
    """The refusal must precede the device copy, and spare a healthy load."""

    def test_refuses_and_names_the_unfilled_tensors(self, monkeypatch):
        pipe = _mismatched_pipe()
        _install_fake_diffusers(monkeypatch, pipe)
        with pytest.raises(RuntimeError, match="was not fully loaded"):
            Cosmos3DiffusersBackend(
                embodiment=get_embodiment("umi"),
                model="nvidia/Cosmos3-Edge",
                device="cpu",
                dtype=_SERVED_DTYPE,
            )

    def test_the_refusal_precedes_the_device_copy(self, monkeypatch):
        """``pipe.to(device)`` moves the whole checkpoint (9 GB for Cosmos3-Edge)
        and is where diffusers' bare meta-tensor error surfaced; refusing first
        also catches the case where the copy happens to succeed on random weights.
        """
        pipe = _mismatched_pipe()
        _install_fake_diffusers(monkeypatch, pipe)
        with pytest.raises(RuntimeError):
            Cosmos3DiffusersBackend(embodiment=get_embodiment("umi"), device="cpu", dtype=_SERVED_DTYPE)
        assert pipe.to_calls == [], "a refused load must not copy the checkpoint to the device"

    def test_a_fully_loaded_checkpoint_still_reaches_the_device(self, monkeypatch):
        """Non-vacuity: the guard must not refuse the supported checkpoints
        (``nvidia/Cosmos3-Nano`` loads with 0 meta parameters on both versions)."""
        pipe = _loaded_pipe()
        _install_fake_diffusers(monkeypatch, pipe)
        backend = Cosmos3DiffusersBackend(embodiment=get_embodiment("umi"), device="cpu", dtype=_SERVED_DTYPE)
        assert pipe.to_calls == ["cpu"]
        assert backend._pipeline is pipe


class TestTheInstallHintNoLongerClaimsSourceOnly:
    """``Cosmos3OmniPipeline`` first shipped in diffusers 0.39.0, a PyPI release,
    so the ImportError remedy must not tell callers the symbol is source-only."""

    def test_the_import_error_hint_names_the_extra_and_not_a_source_only_claim(self, monkeypatch):
        import strands_robots.policies.cosmos3.policy_diffusers as pd

        monkeypatch.setitem(sys.modules, "diffusers", None)
        with pytest.raises(ImportError) as excinfo:
            Cosmos3DiffusersBackend(embodiment=get_embodiment("umi"), device="cpu")
        text = str(excinfo.value)
        assert "strands-robots[cosmos3-diffusers]" in text
        assert "0.39.0" in text
        assert "ships only in diffusers-from-source" not in text
        assert "_checkpoint_mismatch_hint" not in text, "a caller-facing hint must not name a private symbol"
        assert pd._install_hint() == text


class TestTheScanMatchesRealTorchSemantics:
    """The stand-in reads ``.device.type``; pin that a real torch meta parameter
    presents the same way, so the fakes above are not a private convention."""

    def test_a_real_meta_parameter_is_detected(self):
        torch = pytest.importorskip("torch")

        class TorchModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.on_meta = torch.nn.Parameter(torch.empty(2, device="meta"))
                self.on_cpu = torch.nn.Parameter(torch.zeros(2))

        assert _unloaded_checkpoint_tensors(FakePipe({"transformer": TorchModule()})) == ["transformer.on_meta"]
