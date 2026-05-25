"""Tests for Amazon Root CA1 pin verification + size-cap fetch.

The provisioner downloads ``AmazonRootCA1.pem`` over HTTPS and pins its
SHA-256 fingerprint before writing the file to disk. These tests exercise:

* :func:`_verify_ca_bytes` accepts the canonical bytes and rejects any
  one-byte modification.
* :func:`_ensure_ca` raises on a rogue download, on a tampered on-disk
  copy, and on responses larger than :data:`_CA_FETCH_MAX_BYTES`.
* The :envvar:`STRANDS_MESH_DISABLE_CA_PIN` break-glass env var bypasses
  the check (with a WARNING log) for proxy environments that legitimately
  re-encode the cert.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.iot import provision

# Known-good copy of AmazonRootCA1.pem. If Amazon rotates this root the value
# below + provision._AMAZON_ROOT_CA1_PINS must both update together.
_REAL_CA = b"""-----BEGIN CERTIFICATE-----
MIIDQTCCAimgAwIBAgITBmyfz5m/jAo54vB4ikPmljZbyjANBgkqhkiG9w0BAQsF
ADA5MQswCQYDVQQGEwJVUzEPMA0GA1UEChMGQW1hem9uMRkwFwYDVQQDExBBbWF6
b24gUm9vdCBDQSAxMB4XDTE1MDUyNjAwMDAwMFoXDTM4MDExNzAwMDAwMFowOTEL
MAkGA1UEBhMCVVMxDzANBgNVBAoTBkFtYXpvbjEZMBcGA1UEAxMQQW1hem9uIFJv
b3QgQ0EgMTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBALJ4gHHKeNXj
ca9HgFB0fW7Y14h29Jlo91ghYPl0hAEvrAIthtOgQ3pOsqTQNroBvo3bSMgHFzZM
9O6II8c+6zf1tRn4SWiw3te5djgdYZ6k/oI2peVKVuRF4fn9tBb6dNqcmzU5L/qw
IFAGbHrQgLKm+a/sRxmPUDgH3KKHOVj4utWp+UhnMJbulHheb4mjUcAwhmahRWa6
VOujw5H5SNz/0egwLX0tdHA114gk957EWW67c4cX8jJGKLhD+rcdqsq08p8kDi1L
93FcXmn/6pUCyziKrlA4b9v7LWIbxcceVOF34GfID5yHI9Y/QCB/IIDEgEw+OyQm
jgSubJrIqg0CAwEAAaNCMEAwDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMC
AYYwHQYDVR0OBBYEFIQYzIU07LwMlJQuCFmcx7IQTgoIMA0GCSqGSIb3DQEBCwUA
A4IBAQCY8jdaQZChGsV2USggNiMOruYou6r4lK5IpDB/G/wkjUu0yKGX9rbxenDI
U5PMCCjjmCXPI6T53iHTfIUJrU6adTrCC2qJeHZERxhlbI1Bjjt/msv0tadQ1wUs
N+gDS63pYaACbvXy8MWy7Vu33PqUXHeeE6V/Uq2V8viTO96LXFvKWlJbYK8U90vv
o/ufQJVtMVT8QtPHRh8jrdkPSHCa2XV4cdFyQzR1bldZwgJcJmApzyMZFo6IQ6XU
5MsI+yMRQ+hDKXJioaldXgjUkK642M4UwtBV8ob2xJNDd2ZhwLnoQdeXeGADbkpy
rqXRfboQnoZsG4q5WTP468SQvvG5
-----END CERTIFICATE-----
"""


class TestPinVerification:
    def test_real_bytes_match(self):
        # Real bytes must match the constant in provision (else the constant
        # is wrong and every deployment will fail closed -- which is good).
        assert provision._verify_ca_bytes(_REAL_CA) is True

    def test_modified_bytes_rejected(self):
        # Flip one byte -> must fail.
        tampered = bytearray(_REAL_CA)
        tampered[100] ^= 0x01
        assert provision._verify_ca_bytes(bytes(tampered)) is False

    def test_empty_bytes_rejected(self):
        assert provision._verify_ca_bytes(b"") is False

    def test_breakglass_override_bypasses(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_DISABLE_CA_PIN", "true")
        assert provision._verify_ca_bytes(b"any garbage") is True

    def test_verify_ca_pin_path_helper(self, tmp_path):
        good = tmp_path / "ca.pem"
        good.write_bytes(_REAL_CA)
        assert provision.verify_ca_pin(good) is True

        bad = tmp_path / "rogue.pem"
        bad.write_bytes(b"-----FAKE CA-----")
        assert provision.verify_ca_pin(bad) is False

    def test_verify_missing_file_returns_false(self, tmp_path):
        assert provision.verify_ca_pin(tmp_path / "nope.pem") is False


class TestEnsureCA:
    def test_existing_clean_file_skipped(self, tmp_path):
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(_REAL_CA)
        # Should NOT make a network call
        with patch("strands_robots.mesh.iot.provision.urllib.request.urlopen") as mock_url:
            provision._ensure_ca(ca_path)
            mock_url.assert_not_called()

    def test_existing_tampered_file_raises(self, tmp_path):
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(b"rogue cert content")
        with patch("strands_robots.mesh.iot.provision.urllib.request.urlopen") as mock_url:
            with pytest.raises(RuntimeError, match="failed pin check"):
                provision._ensure_ca(ca_path)
            mock_url.assert_not_called()

    def test_download_writes_when_pin_matches(self, tmp_path):
        ca_path = tmp_path / "ca.pem"
        mock_resp = MagicMock()
        mock_resp.read.return_value = _REAL_CA
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda self, *a: None
        with patch("strands_robots.mesh.iot.provision.urllib.request.urlopen", return_value=mock_resp):
            provision._ensure_ca(ca_path)
        assert ca_path.read_bytes() == _REAL_CA

    def test_download_rogue_cert_rejected(self, tmp_path):
        # the download path is _download_with_per_socket_timeout,
        # which builds its own opener (no setdefaulttimeout). Patch the
        # helper directly so the test stays focused on the pin-mismatch
        # rejection rather than urllib internals.
        ca_path = tmp_path / "ca.pem"
        with patch(
            "strands_robots.mesh.iot.provision._download_with_per_socket_timeout",
            return_value=b"-----BEGIN ROGUE CERTIFICATE-----\n",
        ):
            with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
                provision._ensure_ca(ca_path)
        assert not ca_path.exists(), "rogue cert must NOT be written to disk"

    def test_download_oversized_rejected(self, tmp_path):
        # patch _download_with_per_socket_timeout directly. The
        # body-size cap is enforced after the download returns.
        ca_path = tmp_path / "ca.pem"
        big = b"X" * (provision._CA_FETCH_MAX_BYTES + 100)
        with patch(
            "strands_robots.mesh.iot.provision._download_with_per_socket_timeout",
            return_value=big,
        ):
            with pytest.raises(RuntimeError, match="exceeded"):
                provision._ensure_ca(ca_path)
        assert not ca_path.exists()


class TestVerifyCaPinSymlink:
    """The public verify_ca_pin must not follow symlinks (asymmetric
    with _ensure_ca's the prior fix defence was the actual gap).
    """

    def test_symlinked_ca_path_returns_false(self, tmp_path):
        from strands_robots.mesh.iot.provision import verify_ca_pin

        target = tmp_path / "real_ca.pem"
        target.write_bytes(b"-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")
        symlink = tmp_path / "ca_link.pem"
        symlink.symlink_to(target)

        # verify_ca_pin must refuse to read through the symlink
        assert verify_ca_pin(symlink) is False


# ---------------------------------------------------------------------
# the prior fix-2: STRANDS_MESH_POLICY_HOST_ALLOW operator validation
# ---------------------------------------------------------------------
