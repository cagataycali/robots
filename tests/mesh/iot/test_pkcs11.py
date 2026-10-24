"""A device key can live in a PKCS#11 token (RFC 7512 ``pkcs11:`` URI), never a PEM on disk (readers E5)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from strands_robots.mesh.transports import iot_link, pkcs11


def test_is_pkcs11_tells_a_token_uri_from_a_pem_path():
    assert pkcs11.is_pkcs11("pkcs11:token=lab;object=arm-1")
    assert not pkcs11.is_pkcs11("/etc/strands/arm-1/key.pem")
    assert not pkcs11.is_pkcs11(None)
    assert not pkcs11.is_pkcs11("")


@pytest.mark.parametrize(
    "uri, expected",
    [
        (
            "pkcs11:token=lab;object=arm-1?module-path=/opt/hsm.so&pin-value=4321",
            {"module": "/opt/hsm.so", "user_pin": "4321", "token_label": "lab", "private_key_label": "arm-1"},
        ),
        (
            "pkcs11:slot-id=7;object=arm-2?module-name=/opt/hsm.so",
            {"module": "/opt/hsm.so", "user_pin": None, "private_key_label": "arm-2", "slot_id": 7},
        ),
        (  # percent-encoded, as provision.register embeds it in an iot:// query
            "pkcs11:token=a%20lab;object=arm-3?module-path=%2Fopt%2Fhsm.so",
            {"module": "/opt/hsm.so", "user_pin": None, "token_label": "a lab", "private_key_label": "arm-3"},
        ),
    ],
)
def test_parse_reads_rfc7512_path_and_query_attributes(uri, expected, monkeypatch):
    monkeypatch.delenv("PKCS11_MODULE", raising=False)
    monkeypatch.delenv("PKCS11_PIN", raising=False)
    assert pkcs11.parse(uri) == expected


def test_parse_takes_the_module_and_pin_from_the_environment(monkeypatch):
    monkeypatch.setenv("PKCS11_MODULE", "/env/hsm.so")
    monkeypatch.setenv("PKCS11_PIN", "9999")
    assert pkcs11.parse("pkcs11:token=lab;object=arm-1") == {
        "module": "/env/hsm.so", "user_pin": "9999", "token_label": "lab", "private_key_label": "arm-1",
    }  # fmt: skip


def test_parse_without_a_module_is_refused(monkeypatch):
    monkeypatch.delenv("PKCS11_MODULE", raising=False)
    with pytest.raises(ValueError, match="module-path"):
        pkcs11.parse("pkcs11:token=lab;object=arm-1")


def test_a_pkcs11_key_builds_a_token_client_a_pem_key_builds_a_file_client(monkeypatch):
    import awsiot.mqtt5_client_builder as builder

    calls: dict[str, object] = {}
    monkeypatch.setattr(pkcs11, "client", lambda cert, key, **kw: calls.update(token=(cert, key)) or "TOKEN")
    monkeypatch.setattr(builder, "mtls_from_path", lambda **kw: calls.update(file=kw) or "FILE")

    uri = "pkcs11:token=lab;object=arm-1?module-path=/opt/hsm.so"
    assert iot_link._device_client("c.pem", uri, endpoint="e", client_id="arm-1") == "TOKEN"
    assert iot_link._device_client("c.pem", "/k.pem", endpoint="e", client_id="arm-1") == "FILE"
    assert calls["token"] == ("c.pem", uri)
    assert calls["file"]["pri_key_filepath"] == "/k.pem" and calls["file"]["cert_filepath"] == "c.pem"


# --- the real thing: a SoftHSM token, no PEM key on disk --------------------------------------------------------


def _softhsm() -> tuple[str, str]:
    """The PKCS#11 driver and the setup tool, or skip. ``SOFTHSM2_MODULE`` / ``SOFTHSM2_UTIL`` override."""
    module = os.getenv("SOFTHSM2_MODULE") or next(
        (
            c
            for c in (
                "/usr/lib/softhsm/libsofthsm2.so",
                "/usr/lib/aarch64-linux-gnu/softhsm/libsofthsm2.so",
                "/usr/lib/x86_64-linux-gnu/softhsm/libsofthsm2.so",
                "/usr/local/lib/softhsm/libsofthsm2.so",
            )
            if Path(c).exists()
        ),
        "",
    )
    util = os.getenv("SOFTHSM2_UTIL") or shutil.which("softhsm2-util") or ""
    if not module or not Path(module).exists() or not util:
        pytest.skip("SoftHSM (libsofthsm2.so + softhsm2-util) is not installed")
    return module, util


@pytest.fixture
def token(tmp_path, monkeypatch):
    """A SoftHSM token holding one RSA key (label ``devkey``) and a matching cert, with no private PEM on disk.

    Yields ``(pkcs11_uri, cert_path, tmp_path)``."""
    module, util = _softhsm()
    conf = tmp_path / "softhsm2.conf"
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    conf.write_text(f"directories.tokendir = {tokens}\nobjectstore.backend = file\nlog.level = ERROR\n")
    monkeypatch.setenv("SOFTHSM2_CONF", str(conf))

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_pem = tmp_path / "key.pem"
    key_pem.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "arm-1")])
    import datetime

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2035, 1, 1))
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    run = [util, "--module", module, "--init-token", "--free", "--label", "e5tok",
           "--so-pin", "1234", "--pin", "5678"]  # fmt: skip
    subprocess.run(run, check=True, capture_output=True)
    subprocess.run(
        [
            util,
            "--module",
            module,
            "--import",
            str(key_pem),
            "--token",
            "e5tok",
            "--label",
            "devkey",
            "--id",
            "01",
            "--pin",
            "5678",
        ],  # fmt: skip
        check=True,
        capture_output=True,
    )
    key_pem.unlink()  # the private key now lives only in the token
    uri = f"pkcs11:token=e5tok;object=devkey?module-path={module}&pin-value=5678"
    return uri, str(cert_path), tmp_path


def test_a_token_key_builds_an_mqtt_client_with_no_pem_on_disk(token):
    from awscrt.mqtt5 import Client

    uri, cert, tmp_path = token
    assert not (tmp_path / "key.pem").exists()  # no private key as a file
    client = pkcs11.client(cert, uri, endpoint="example-ats.iot.us-west-2.amazonaws.com", client_id="arm-1")
    assert isinstance(client, Client)  # awscrt found and loaded the key from the token
