"""The torch stand-in either serves an attribute or skips with the reason.

``conftest`` installs a numpy-backed stand-in when real torch is not importable.
It covers a subset, so a test can always reach past it -- directly, or through a
package it imports: ``lerobot`` reads ``torch.dtype`` at import time. What that
reach costs is the contract pinned here.

Before, a reach produced ``AttributeError: module 'torch' has no attribute
'is_tensor'``, which names neither the stand-in nor the missing dependency, so
the first move is to debug the diff. At module scope it was worse than a
failure: a collection error, and collection errors abort the whole run rather
than one module.

The contract now is serve-or-skip, and both halves of the exception type it
rests on are load-bearing, so both are pinned:

* it is still an ``AttributeError``, so every ``hasattr`` probe and
  ``except AttributeError`` fallback behaves as it does against real torch --
  making a reach visible must not turn a graceful path into a skip;
* it is additionally a pytest skip, so an *unguarded* reach is reported as one,
  naming the attribute and both remedies.

Everything here builds the stand-in in isolation through the module rather than
installing it, so the contract is verified in the environment CI actually runs:
one with real torch present.
"""

import ast
import inspect
import pathlib

import pytest

from tests.mocks import torch_mock as mock_mod

# The stand-in's documented subset -- policy logic, observation mapping and
# action conversion. Serving these is what makes it worth installing at all.
SERVED = ("Tensor", "device", "tensor", "zeros", "ones", "from_numpy", "stack", "cat", "no_grad")

# Attributes measured reaching past the stand-in on a full torch-less run of
# ``tests/``. ``dtype`` is the largest class by a wide margin and is not reached
# by this repository at all -- it is read inside ``lerobot`` during import, which
# is why completing the subset is not a way out.
MEASURED_REACHES = (
    "dtype",
    "is_tensor",
    "arange",
    "version",
    "save",
    "optim",
    "utils",
    "int",
    "uint8",
    "float16",
    "bfloat16",
    "zeros_like",
    "full",
    "isfinite",
    "get_device_name",
)


@pytest.fixture
def stand_in():
    """The stand-in module tree, built but not registered in ``sys.modules``."""
    return mock_mod._build_torch_mock()


class TestTheStandInServesItsDocumentedSubset:
    """The subset still works: a skip for something it covers would be a loss."""

    @pytest.mark.parametrize("name", SERVED)
    def test_a_served_attribute_is_returned(self, stand_in, name):
        assert getattr(stand_in["torch"], name) is not None

    def test_the_served_tensor_surface_still_computes(self, stand_in):
        torch = stand_in["torch"]
        t = torch.tensor([[0.1], [0.9]])
        assert t.detach().to("cpu").flatten().tolist() == pytest.approx([0.1, 0.9], abs=1e-6)

    def test_every_module_the_stand_in_registers_is_guarded(self, stand_in):
        # A submodule left unguarded is a hole in the same shape as the original
        # defect, so the guard is asserted over the registry rather than over a
        # hand-listed subset.
        assert set(stand_in) == {
            "torch",
            "torch.nn",
            "torch.nn.functional",
            "torch.cuda",
            "torch.backends",
            "torch.backends.mps",
            "torch.backends.cudnn",
            "torch.amp",
            "torchvision",
            "torchvision.transforms",
        }
        for name, module in stand_in.items():
            with pytest.raises(AttributeError, match="numpy-backed torch stand-in"):
                getattr(module, "definitely_not_served")
            assert module.__name__ == name


class TestAReachPastTheSubsetIsSkippedNotFailed:
    @pytest.mark.parametrize("name", MEASURED_REACHES)
    def test_a_measured_reach_is_reported_as_a_skip(self, stand_in, name):
        with pytest.raises(mock_mod.MissingMockAttribute) as excinfo:
            getattr(stand_in["torch"], name)
        assert isinstance(excinfo.value, pytest.skip.Exception)

    def test_a_reach_remains_an_attributeerror(self, stand_in):
        # The half that keeps graceful paths graceful. Several production
        # fallbacks catch AttributeError around a torch probe; if the skip were
        # not also one, those would stop being fallbacks.
        with pytest.raises(AttributeError):
            getattr(stand_in["torch"], "is_tensor")

    def test_a_graceful_fallback_still_absorbs_it(self, stand_in):
        absorbed = False
        try:
            stand_in["torch"].is_tensor([1.0])
        except AttributeError:
            absorbed = True
        assert absorbed, "except AttributeError no longer absorbs a reach past the subset"

    def test_a_hasattr_probe_still_answers_false(self, stand_in):
        assert hasattr(stand_in["torch"], "tensor") is True
        assert hasattr(stand_in["torch"], "is_tensor") is False

    def test_the_message_names_the_attribute_the_stand_in_and_both_remedies(self, stand_in):
        with pytest.raises(AttributeError) as excinfo:
            getattr(stand_in["torch"], "optim")
        text = str(excinfo.value)
        # Opens with real torch's own wording, so anything matching on that
        # prefix is unaffected by the added context.
        assert text.startswith("module 'torch' has no attribute 'optim'")
        assert "numpy-backed torch stand-in" in text
        assert 'pip install -e ".[all,dev]"' in text
        assert "real_torch_installed()" in text
        assert 'pytest.importorskip("torch") cannot skip' in text

    def test_the_skip_carries_that_message(self, stand_in):
        with pytest.raises(mock_mod.MissingMockAttribute) as excinfo:
            getattr(stand_in["torch"], "utils")
        # pytest reports ``msg``; a skip whose reason were empty would be as
        # unactionable as the failure this replaces.
        assert "numpy-backed torch stand-in" in excinfo.value.msg
        assert excinfo.value.allow_module_level is True, "a reach during module import must skip, not error"

    def test_the_typed_skip_class_is_the_documented_one(self):
        # The stand-in subclasses ``_pytest.outcomes.Skipped`` because
        # ``pytest.skip.Exception`` cannot be a base class under a type checker.
        # That is only sound while the two are the same object, so the premise is
        # executed rather than asserted in a comment.
        from _pytest.outcomes import Skipped

        assert pytest.skip.Exception is Skipped
        assert issubclass(mock_mod.MissingMockAttribute, pytest.skip.Exception)


class TestDunderLookupIsUntouched:
    """Import machinery and introspection must not see a skip.

    ``types.ModuleType`` genuinely lacks ``__path__``, ``__file__`` and
    ``__all__``, so the guard sees those probes. Answering them with a skip
    would change behaviour that has nothing to do with a test reaching past the
    tensor surface.
    """

    @pytest.mark.parametrize("name", ["__version__", "__path__", "__file__", "__all__"])
    def test_a_dunder_reach_stays_a_plain_attributeerror(self, stand_in, name):
        with pytest.raises(AttributeError) as excinfo:
            getattr(stand_in["torch"], name)
        assert not isinstance(excinfo.value, pytest.skip.Exception)
        assert "numpy-backed torch stand-in" not in str(excinfo.value)

    def test_the_dunders_python_sets_are_still_readable(self, stand_in):
        assert stand_in["torch"].__spec__ is None
        assert stand_in["torch"].__name__ == "torch"


class TestTheDiscriminatorHasOneHome:
    """``real_torch_installed`` is the one answer to "is this torch real?"."""

    def test_it_agrees_with_an_independent_witness(self):
        torch = pytest.importorskip("torch")
        # Non-circular, and it holds in both environments: ``optim`` is one of
        # the attributes measured reaching past the stand-in, so serving it is
        # something only real torch does. Asserting the equivalence rather than
        # re-deriving ``hasattr(torch, "__version__")`` is also what keeps this
        # module clean under the scan below.
        assert mock_mod.real_torch_installed() is hasattr(torch, "optim")

    def test_it_is_false_while_the_stand_in_is_registered(self, stand_in, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "torch", stand_in["torch"])
        assert mock_mod.real_torch_installed() is False

    def test_installing_is_a_no_op_when_real_torch_is_present(self):
        import sys

        if not mock_mod.real_torch_installed():
            pytest.skip("the stand-in is active in this environment")
        before = sys.modules["torch"]
        mock_mod.install_torch_mock()
        assert sys.modules["torch"] is before, "installing replaced real torch"


def _test_sources():
    """Every test module, rooted from a symbol rather than a path literal."""
    root = pathlib.Path(inspect.getfile(mock_mod)).resolve().parent.parent
    assert root.name == "tests", root
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _inlines_the_discriminator(source):
    """Find ``hasattr(<anything>, "__version__")`` calls, the discriminator's body."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "__version__"
        ):
            found.append(node.lineno)
    return found


class TestNoTestModuleReimplementsTheDiscriminator:
    """The knowledge was correct and private to one file; now it has one home.

    A module that re-derives ``hasattr(torch, "__version__")`` inline is the
    shape this change removes: the reason ``importorskip`` cannot answer the
    question has to be written down once, not per site.
    """

    def test_the_scan_sees_the_test_tree(self):
        sources = _test_sources()
        assert len(sources) > 500, f"scan root resolved somewhere unexpected: {len(sources)} modules"

    def test_no_module_inlines_it(self):
        owner = pathlib.Path(inspect.getfile(mock_mod)).resolve()
        offenders = {}
        for path in _test_sources():
            if path.resolve() == owner:
                continue  # the one home, which is where the body belongs
            lines = _inlines_the_discriminator(path.read_text(encoding="utf-8"))
            if lines:
                offenders[str(path.name)] = lines
        assert not offenders, f"call real_torch_installed() instead of re-deriving it: {offenders}"

    def test_the_scan_detects_a_planted_one(self):
        planted = 'import torch\n\n\ndef f():\n    return hasattr(torch, "__version__")\n'
        assert _inlines_the_discriminator(planted) == [5]
