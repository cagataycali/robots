"""Numpy-backed torch stand-in for environments without PyTorch.

``conftest`` installs this when ``import torch`` fails, so the parts of the suite
that need only a thin tensor surface still run without the ~2GB dependency. It is
a *subset*, not a replacement: it covers policy logic, observation mapping and
action conversion.

The contract is serve-or-skip. An attribute the mock does not provide raises
:class:`MissingMockAttribute`, which is both an ``AttributeError`` -- so every
existing ``hasattr`` probe and ``except AttributeError`` fallback behaves exactly
as it does against real torch -- and a pytest skip naming the attribute and the
remedy. So a test that needs real torch is reported as skipped rather than as a
failure whose text mentions neither the mock nor the absent extra, and a module
whose imports need more of the surface than the mock has (``lerobot`` reads
``torch.dtype`` at import time) is skipped rather than erroring during
collection.

Provides numpy-backed replacements for:
- torch.Tensor (MockTensor) - arithmetic, reshaping, device, slicing
- torch.nn.Parameter (MockParameter) - with requires_grad and device
- torch.device (MockDevice) - type string, equality, hashing
- Factory functions: tensor, zeros, ones, randint, rand, from_numpy, stack, cat
- Context managers: no_grad, inference_mode
- Submodules: torch.nn, torch.cuda, torch.backends, torch.amp

A test that knows up front that it needs real torch should say so with
:func:`real_torch_installed`, which is the one discriminator:
``pytest.importorskip("torch")`` cannot answer the question, because the mock
registers a module in ``sys.modules`` and the import therefore succeeds.

Usage:
    from tests.mocks.torch_mock import install_torch_mock
    install_torch_mock()  # no-op if real torch is available
"""

import logging
import sys
import types
from unittest.mock import MagicMock

import numpy as np

# ``pytest.skip.Exception`` is the documented handle for this class, but it is a
# runtime attribute on a function, so it cannot be used in a base-class
# position under a type checker. This import is the same object -- pinned as an
# executable premise by the contract tests rather than left as a claim here.
from _pytest.outcomes import Skipped

logger = logging.getLogger(__name__)


class MockTensor:
    """Minimal torch.Tensor replacement backed by numpy."""

    def __init__(self, data=None, dtype=None, device=None):
        if isinstance(data, MockTensor):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data, dtype=np.float32)
        elif isinstance(data, (int, float)):
            self._data = np.array([data], dtype=np.float32)
        elif data is None:
            self._data = np.array([], dtype=np.float32)
        else:
            self._data = np.array(data, dtype=np.float32)

    # Properties

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self):
        return MockDevice("cpu")

    # Shape / size helpers

    def dim(self):
        return self._data.ndim

    def size(self, dim=None):
        if dim is not None:
            return self._data.shape[dim]
        return self._data.shape

    def numel(self):
        return int(self._data.size)

    # Conversion

    def item(self):
        return float(self._data.flat[0])

    def tolist(self):
        return self._data.tolist()

    def numpy(self):
        return self._data.copy()

    def cpu(self):
        return self

    def detach(self):
        return self

    def clone(self):
        return MockTensor(self._data.copy())

    def float(self):
        return self

    def bool(self):
        return MockTensor(self._data.astype(np.bool_).astype(np.float32))

    def long(self):
        return MockTensor(self._data.astype(np.int64).astype(np.float32))

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    # Reshaping

    def unsqueeze(self, dim):
        return MockTensor(np.expand_dims(self._data, axis=dim))

    def squeeze(self, dim=None):
        if dim is not None:
            return MockTensor(np.squeeze(self._data, axis=dim))
        return MockTensor(np.squeeze(self._data))

    def view(self, *shape):
        return MockTensor(self._data.reshape(shape))

    def reshape(self, *shape):
        return MockTensor(self._data.reshape(shape))

    def flatten(self, *args, **kwargs):
        return MockTensor(self._data.reshape(-1))

    def permute(self, *dims):
        return MockTensor(np.transpose(self._data, dims))

    # Reduction

    def max(self):
        return float(self._data.max()) if self._data.size > 0 else 0.0

    def min(self):
        return float(self._data.min()) if self._data.size > 0 else 0.0

    # Dunder methods

    def __len__(self):
        return self._data.shape[0] if self._data.ndim > 0 else 1

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, np.ndarray):
            return MockTensor(result)
        return MockTensor(np.array([result]))

    def __repr__(self):
        return f"MockTensor({self._data})"

    def __float__(self):
        return float(self._data.flat[0])

    def __eq__(self, other):
        if isinstance(other, MockTensor):
            return np.array_equal(self._data, other._data)
        return np.array_equal(self._data, other)

    def __abs__(self):
        return MockTensor(np.abs(self._data))

    def __sub__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data - other._data)
        return MockTensor(self._data - other)

    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data + other._data)
        return MockTensor(self._data + other)

    def __truediv__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data / other._data)
        return MockTensor(self._data / other)

    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data * other._data)
        return MockTensor(self._data * other)


class MockParameter(MockTensor):
    """torch.nn.Parameter replacement."""

    def __init__(self, data=None, requires_grad=True):
        super().__init__(data)
        self.requires_grad = requires_grad

    @property
    def device(self):
        return MockDevice("cpu")


class MockDevice:
    """torch.device replacement."""

    def __init__(self, device_str="cpu"):
        if isinstance(device_str, MockDevice):
            device_str = device_str.type
        self.type = str(device_str).split(":")[0]

    def __repr__(self):
        return f"device(type='{self.type}')"

    def __str__(self):
        return self.type

    def __eq__(self, other):
        if isinstance(other, str):
            return self.type == other
        if isinstance(other, MockDevice):
            return self.type == other.type
        return False

    def __hash__(self):
        return hash(self.type)


class _NoGrad:
    """torch.no_grad / torch.inference_mode replacement."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, func):
        return func


# Factory functions


def _tensor(data, dtype=None, device=None):
    return MockTensor(data, dtype=dtype, device=device)


def _zeros(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return MockTensor(np.zeros(shape, dtype=np.float32))


def _ones(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return MockTensor(np.ones(shape, dtype=np.float32))


def _from_numpy(arr):
    return MockTensor(arr)


def _stack(tensors, dim=0):
    arrays = [t._data if isinstance(t, MockTensor) else np.array(t) for t in tensors]
    return MockTensor(np.stack(arrays, axis=dim))


def _cat(tensors, dim=0):
    arrays = [t._data if isinstance(t, MockTensor) else np.array(t) for t in tensors]
    return MockTensor(np.concatenate(arrays, axis=dim))


def _randint(low, high, size, dtype=None):
    return MockTensor(np.random.randint(low, high, size=size).astype(np.float32))


def _rand(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return MockTensor(np.random.rand(*shape).astype(np.float32))


def _randn(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return MockTensor(np.random.randn(*shape).astype(np.float32))


# Public API


class MissingMockAttribute(AttributeError, Skipped):
    """A torch attribute this mock does not provide.

    Both base classes are load-bearing:

    - ``AttributeError`` keeps every ``hasattr`` probe and
      ``except AttributeError`` fallback behaving as it does against real torch,
      so making a miss visible cannot turn a graceful path into a skip.
    - ``Skipped`` (which is ``pytest.skip.Exception``) means an *unguarded* miss
      is reported as a skip
      that names the attribute and the remedy, rather than as a failure whose
      text names neither the mock nor the missing extra.

    ``AttributeError`` is first in the MRO, so its ``__init__`` runs and the
    fields pytest reads off a skip have to be set here rather than delegated.
    """

    def __init__(self, message: str) -> None:
        AttributeError.__init__(self, message)
        self.msg = message
        self.pytrace = False
        # Permit the skip during module import: without this, a module whose
        # imports touch an unsupported attribute is a collection error, and
        # collection errors abort the whole run rather than one module.
        self.allow_module_level = True
        self._use_item_location = False


def _missing_attribute(module_name, attribute):
    """Build the :class:`MissingMockAttribute` for one miss.

    The message opens with the wording real torch uses, so anything matching on
    that prefix is unaffected, and then states what the reader cannot otherwise
    know: that a stand-in answered, what it covers, and both remedies.
    """
    return MissingMockAttribute(
        f"module {module_name!r} has no attribute {attribute!r}: this is the "
        "numpy-backed torch stand-in the test suite installs when real torch is "
        "not importable, and it covers policy logic, observation mapping and "
        "action conversion only. Install the real dependency to run this test "
        '(pip install -e ".[all,dev]"), or -- if the test needs real torch by '
        "nature -- gate it on real_torch_installed(), because "
        'pytest.importorskip("torch") cannot skip while the stand-in is '
        "registered in sys.modules."
    )


def _guard_missing_attributes(module):
    """Make every miss on ``module`` explain itself, and return the module."""

    def __getattr__(name):
        # Dunders keep plain lookup semantics: the import machinery probes
        # ``__path__``, pytest introspects ``__spec__``, and
        # ``real_torch_installed`` probes ``__version__`` -- none of which is a
        # test touching an unsupported part of the tensor surface, so none of
        # them should change behaviour here.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {module.__name__!r} has no attribute {name!r}")
        raise _missing_attribute(module.__name__, name)

    module.__getattr__ = __getattr__
    return module


def real_torch_installed():
    """Return True when the importable ``torch`` is real rather than this mock.

    The one discriminator, so that the reason it cannot be
    ``pytest.importorskip("torch")`` is written down once: the mock registers a
    module in ``sys.modules``, so the import succeeds and only attribute access
    fails. The mock never sets ``__version__``.
    """
    try:
        import torch
    except ImportError:
        return False
    return hasattr(torch, "__version__")


def _build_torch_mock():
    """Build the mock module tree; return ``{module name: module}``."""
    # Root module
    torch_mock = types.ModuleType("torch")
    torch_mock.Tensor = MockTensor
    torch_mock.device = MockDevice
    torch_mock.float32 = np.float32
    torch_mock.float64 = np.float64
    torch_mock.int32 = np.int32
    torch_mock.int64 = np.int64
    torch_mock.long = np.int64
    torch_mock.bool = np.bool_

    torch_mock.tensor = _tensor
    torch_mock.zeros = _zeros
    torch_mock.ones = _ones
    torch_mock.from_numpy = _from_numpy
    torch_mock.stack = _stack
    torch_mock.cat = _cat
    torch_mock.as_tensor = _tensor
    torch_mock.randint = _randint
    torch_mock.rand = _rand
    torch_mock.randn = _randn

    torch_mock.no_grad = _NoGrad
    torch_mock.inference_mode = _NoGrad
    torch_mock.manual_seed = lambda seed: None

    # torch.nn
    nn_mock = types.ModuleType("torch.nn")
    nn_mock.Parameter = MockParameter
    nn_mock.Module = MagicMock
    torch_mock.nn = nn_mock

    nn_functional_mock = types.ModuleType("torch.nn.functional")
    torch_mock.nn.functional = nn_functional_mock

    # torch.cuda
    cuda_mock = types.ModuleType("torch.cuda")
    cuda_mock.is_available = lambda: False
    cuda_mock.device_count = lambda: 0
    cuda_mock.manual_seed_all = lambda seed: None
    torch_mock.cuda = cuda_mock

    # torch.backends
    backends_mock = types.ModuleType("torch.backends")
    mps_mock = types.ModuleType("torch.backends.mps")
    mps_mock.is_available = lambda: False
    backends_mock.mps = mps_mock
    cudnn_mock = types.ModuleType("torch.backends.cudnn")
    cudnn_mock.deterministic = False
    cudnn_mock.benchmark = True
    backends_mock.cudnn = cudnn_mock
    torch_mock.backends = backends_mock

    # torch.amp
    amp_mock = types.ModuleType("torch.amp")
    amp_mock.autocast = MagicMock
    torch_mock.amp = amp_mock

    # torchvision
    torchvision_mock = types.ModuleType("torchvision")
    torchvision_transforms = types.ModuleType("torchvision.transforms")
    torchvision_mock.transforms = torchvision_transforms

    modules = {
        "torch": torch_mock,
        "torch.nn": nn_mock,
        "torch.nn.functional": nn_functional_mock,
        "torch.cuda": cuda_mock,
        "torch.backends": backends_mock,
        "torch.backends.mps": mps_mock,
        "torch.backends.cudnn": cudnn_mock,
        "torch.amp": amp_mock,
        "torchvision": torchvision_mock,
        "torchvision.transforms": torchvision_transforms,
    }
    for module in modules.values():
        _guard_missing_attributes(module)
    return modules


def install_torch_mock():
    """Install the torch stand-in into ``sys.modules``.

    No-op if real torch is already importable.
    """
    try:
        import torch  # noqa: F401

        logger.info("Real torch is available (version=%s) - mock not installed", torch.__version__)
        return  # Real torch available - nothing to do
    except Exception as exc:  # noqa: BLE001 - diagnostics: any import failure means we mock
        # IMPORTANT: print to stderr (not just logging.info, which pytest captures
        # and hides) so CI logs ALWAYS show WHY the mock was installed. A silent
        # fallback here previously masked an env-resolution bug (a CUDA torch
        # wheel that failed to import) for hours of log-archaeology.
        _msg = (
            f"[torch_mock] real torch import FAILED ({type(exc).__name__}: {exc}); "
            "installing numpy mock. If this is unexpected, the torch wheel in this "
            "env is broken/unimportable (e.g. wrong CUDA build)."
        )
        # pytest captures stdout/stderr, so ALSO write to a sentinel file that
        # CI can cat unconditionally -- this is what makes the diagnosis a
        # one-line grep instead of log-archaeology.
        print(_msg, file=sys.stderr)
        try:
            import os as _os

            with open(
                _os.environ.get("TORCH_MOCK_SENTINEL", "/tmp/torch_mock_active.txt"),
                "w",
            ) as _fh:
                _fh.write(_msg + "\n")
        except OSError:
            # Sentinel-file write is best-effort diagnostics only (read-only or
            # full /tmp, restricted CI sandbox, etc.). The stderr message above
            # already conveys why the mock was installed, so a failed write must
            # never abort mock installation or fail the test run.
            pass

    logger.info("Installing torch mock (real torch not available)")
    sys.modules.update(_build_torch_mock())
