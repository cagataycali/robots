"""Pytest plugin: make `import torch` fail so tests/conftest.py installs the mock.

Faithful emulation of an install without the [all] extra: a MetaPathFinder that
refuses torch (and its submodules) before any conftest is imported.
"""

import sys


class _RefuseTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(f"No module named {fullname!r} (blocked by blocktorch plugin)")
        return None


for _name in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
    del sys.modules[_name]
sys.meta_path.insert(0, _RefuseTorch())
