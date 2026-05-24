"""Ephemeral CA + leaf-cert minting for fleet pentests.

This is a thin wrapper around the in-tree ``tests/mesh/_pki.py`` so the
fleet can reuse the audited PKI helper without copy-pasting cryptography
calls. We deliberately do NOT vendor or re-implement -- if the test
helper changes (e.g. switches to ECDSA, adds SAN), we inherit the
change.

The `attacker_ca` mode mints a *separate* CA so we can demonstrate the
rogue-CA insider attack (AV-02 / rogue_02): same CN, different chain.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Use the in-repo PKI helper -- single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests" / "mesh"))
from _pki import TestCA, make_test_ca  # noqa: E402


@dataclass
class EphemeralCA:
    """Ephemeral test CA with an output directory.

    Use :meth:`mint` to issue a leaf cert+key pair. The key file is
    written with mode 0o600 so the F-series mTLS guard
    (``_assert_key_perms`` in ``_zenoh_config``) does not refuse it.
    """

    inner: TestCA
    out_dir: Path

    @classmethod
    def make(cls, out_dir: Path) -> "EphemeralCA":
        out_dir.mkdir(parents=True, exist_ok=True)
        return cls(inner=make_test_ca(out_dir), out_dir=out_dir)

    @property
    def ca_cert(self) -> Path:
        return self.inner.cert_path

    def mint(self, common_name: str, sub: str | None = None) -> tuple[Path, Path]:
        """Mint ``(cert_path, key_path)`` for *common_name*.

        Files land under ``{out_dir}/{sub or common_name}/``. Reusing the
        same ``sub`` overwrites; reusing the same ``common_name`` with a
        different ``sub`` issues two distinct cert/key pairs sharing the
        CN (used by the role-violation rogue).
        """
        leaf_dir = self.out_dir / (sub or common_name.replace("/", "_"))
        return self.inner.issue(common_name, leaf_dir)
