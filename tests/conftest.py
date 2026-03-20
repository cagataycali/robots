"""Shared test fixtures and configuration.

Installs a torch mock (if real torch is unavailable) so CI can run
all unit tests without PyTorch installed.
"""

from tests.mocks.torch_mock import install_torch_mock

# Must run before any test imports policy modules
install_torch_mock()
