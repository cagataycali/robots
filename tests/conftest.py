"""Shared test fixtures and configuration.

Installs a comprehensive torch mock into sys.modules so that tests can run
without PyTorch installed (CI environments). The mock supports:
- torch.Tensor creation (zeros, tensor, from_numpy, etc.)
- torch.nn.Parameter, torch.nn.Module
- torch.device
- torch.no_grad context manager
- Tensor operations (.item(), .numpy(), .shape, slicing, etc.)
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Build a realistic torch mock BEFORE any test imports policy modules
# ---------------------------------------------------------------------------


def _install_torch_mock():
    """Install a torch mock into sys.modules if torch is not available."""
    try:
        import torch  # noqa: F401

        return  # Real torch is available, no mock needed
    except ImportError:
        pass

    # --- MockTensor: behaves enough like torch.Tensor for unit tests ---
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

        @property
        def shape(self):
            return self._data.shape

        @property
        def ndim(self):
            return self._data.ndim

        @property
        def dtype(self):
            return self._data.dtype

        def dim(self):
            return self._data.ndim

        def size(self, dim=None):
            if dim is not None:
                return self._data.shape[dim]
            return self._data.shape

        def item(self):
            return float(self._data.flat[0])

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

        def to(self, *args, **kwargs):
            return self

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

        def permute(self, *dims):
            return MockTensor(np.transpose(self._data, dims))

        def contiguous(self):
            return self

        def max(self):
            return float(self._data.max()) if self._data.size > 0 else 0.0

        def min(self):
            return float(self._data.min()) if self._data.size > 0 else 0.0

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

        def numel(self):
            return int(self._data.size)

        @property
        def device(self):
            return MockDevice("cpu")

    # --- MockParameter: torch.nn.Parameter ---
    class MockParameter(MockTensor):
        def __init__(self, data=None, requires_grad=True):
            super().__init__(data)
            self.requires_grad = requires_grad

        @property
        def device(self):
            return MockDevice("cpu")

    # --- MockDevice ---
    class MockDevice:
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

    # --- Build the mock torch module ---
    torch_mock = types.ModuleType("torch")

    # Core types
    torch_mock.Tensor = MockTensor
    torch_mock.device = MockDevice
    torch_mock.float32 = np.float32
    torch_mock.float64 = np.float64
    torch_mock.int32 = np.int32
    torch_mock.int64 = np.int64
    torch_mock.bool = np.bool_

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

    torch_mock.tensor = _tensor
    torch_mock.zeros = _zeros
    torch_mock.ones = _ones
    torch_mock.from_numpy = _from_numpy
    torch_mock.stack = _stack
    torch_mock.cat = _cat
    torch_mock.as_tensor = _tensor

    def _randint(low, high, size, dtype=None):
        return MockTensor(np.random.randint(low, high, size=size).astype(np.float32))

    def _rand(*shape, dtype=None, device=None):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return MockTensor(np.random.rand(*shape).astype(np.float32))

    torch_mock.randint = _randint
    torch_mock.rand = _rand
    torch_mock.long = np.int64

    # no_grad context manager
    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __call__(self, func):
            return func

    torch_mock.no_grad = _NoGrad
    torch_mock.inference_mode = _NoGrad

    # --- torch.nn ---
    nn_mock = types.ModuleType("torch.nn")
    nn_mock.Parameter = MockParameter
    nn_mock.Module = MagicMock
    torch_mock.nn = nn_mock

    # torch.nn.functional
    nn_functional_mock = types.ModuleType("torch.nn.functional")
    torch_mock.nn.functional = nn_functional_mock

    # --- torch.cuda ---
    cuda_mock = types.ModuleType("torch.cuda")
    cuda_mock.is_available = lambda: False
    cuda_mock.device_count = lambda: 0
    torch_mock.cuda = cuda_mock

    # --- torch.backends ---
    backends_mock = types.ModuleType("torch.backends")
    mps_mock = types.ModuleType("torch.backends.mps")
    mps_mock.is_available = lambda: False
    backends_mock.mps = mps_mock
    torch_mock.backends = backends_mock

    # --- torch.amp ---
    amp_mock = types.ModuleType("torch.amp")
    amp_mock.autocast = MagicMock
    torch_mock.amp = amp_mock

    # --- Register in sys.modules ---
    sys.modules["torch"] = torch_mock
    sys.modules["torch.nn"] = nn_mock
    sys.modules["torch.nn.functional"] = nn_functional_mock
    sys.modules["torch.cuda"] = cuda_mock
    sys.modules["torch.backends"] = backends_mock
    sys.modules["torch.backends.mps"] = mps_mock
    sys.modules["torch.amp"] = amp_mock

    # Also mock torchvision if needed
    torchvision_mock = types.ModuleType("torchvision")
    torchvision_transforms = types.ModuleType("torchvision.transforms")
    torchvision_mock.transforms = torchvision_transforms
    sys.modules["torchvision"] = torchvision_mock
    sys.modules["torchvision.transforms"] = torchvision_transforms


# Install the mock at conftest load time (before any test collection)
_install_torch_mock()
