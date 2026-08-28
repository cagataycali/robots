"""Grade the AWS IoT credential reference against the transport's env surface.

``docs/security.md`` owns the cross-network fleet section, and that section is
where an operator configures the AWS IoT Core path. Three properties are graded
here, each derived from the package rather than from a list kept beside it, so a
variable added later is graded on arrival:

- **Every variable the transport reads is documented.** Both the ``iot`` and
  ``bridge`` backends construct :class:`IotMqttTransport` with no arguments, so
  these variables are the whole of its configuration. A variable the code
  honours and the page omits is a setting nobody can find: all four were
  undocumented across ``docs/`` and ``README.md`` while this very section told
  the reader to provision, scope and rotate the credentials they point at.
- **The credentials of one link are documented in one place.** They configure a
  single channel - one names the Thing, one the broker, two locate the
  certificate material - so a reader configuring that link should find them
  together rather than assemble it from two sections.
- **The provisioner and the page agree.** ``ProvisionedThing.env_vars`` is the
  package's own answer to "what must an operator export", so a name it hands out
  and the page does not explain is a gap between two in-tree surfaces.

The behaviour that makes the omission matter is asserted directly too: with the
credentials unset, ``connect()`` returns ``False`` and reports the missing
variable by name. That is a fact about the code, true on both sides of this
documentation change - it is why a reader who cannot find the names discovers
them one ERROR at a time, and why the peer is simply absent from the fleet until
they do.

Scope: the selector that chooses this transport, ``STRANDS_MESH_BACKEND``, is
documented by the change that names it and is deliberately not graded here.
These rules cover the credentials the selected transport then reads.
"""

import ast
import logging
import pathlib
import re

import pytest

import strands_robots

_PACKAGE = pathlib.Path(strands_robots.__file__).parent
_PAGE = _PACKAGE.parent / "docs" / "security.md"

#: The module that owns the AWS IoT broker link. Its variables configure one
#: channel, so the reference documents them together.
_TRANSPORT = _PACKAGE / "mesh" / "transport" / "iot_transport.py"

#: The credential prefix this reference covers.
_CREDENTIAL = re.compile(r"STRANDS_IOT_[A-Z0-9_]*")

#: A documented bullet's variable: ``- `VAR` - ...`` or ``- `VAR=value` - ...``,
#: which is how every environment variable on this page is already written.
_BULLET_VAR = re.compile(r"^-\s+`([A-Z][A-Z0-9_]*)(?:=[^`]*)?`")

#: Floors so a scan that silently reads nothing fails instead of passing.
_MINIMUM_CREDENTIALS = 3
_MINIMUM_DOCUMENTED_BULLETS = 4


def _literal_env_reads(path: pathlib.Path) -> set[str]:
    """Return every literal environment variable *path* reads.

    Args:
        path: The module to scan.

    Returns:
        The variable names read through ``os.getenv`` / ``os.environ.get`` /
        ``os.environ[...]`` with a literal key.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        name = None
        if isinstance(node, ast.Call) and ast.unparse(node.func) in (
            "os.getenv",
            "os.environ.get",
        ):
            if node.args and isinstance(node.args[0], ast.Constant):
                name = node.args[0].value
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
            if isinstance(node.slice, ast.Constant):
                name = node.slice.value
        if isinstance(name, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            found.add(name)
    return found


def _credentials_read() -> set[str]:
    """Return the IoT credentials the transport reads from the environment."""
    return {var for var in _literal_env_reads(_TRANSPORT) if _CREDENTIAL.fullmatch(var)}


def _provisioner_exports() -> set[str]:
    """Return the IoT credentials the provisioner tells an operator to export."""
    from strands_robots.mesh.iot.provision import ProvisionedThing

    provisioned = ProvisionedThing(
        thing_name="probe-thing",
        thing_arn="arn:aws:iot:us-west-2:000000000000:thing/probe-thing",
        cert_arn="arn:aws:iot:us-west-2:000000000000:cert/probe",
        cert_id="probe",
        cert_path=pathlib.Path("/probe/certs/probe-thing.cert.pem"),
        key_path=pathlib.Path("/probe/certs/probe-thing.private.key"),
        ca_path=pathlib.Path("/probe/certs/AmazonRootCA1.pem"),
        endpoint="probe-ats.iot.us-west-2.amazonaws.com",
        policy_name="probe-policy",
        region="us-west-2",
    )
    return {var for var in provisioned.env_vars() if _CREDENTIAL.fullmatch(var)}


def _documented_bullets(page: str) -> dict[str, str]:
    """Map each variable the page documents to the heading it sits under.

    Args:
        page: The markdown source to read.

    Returns:
        ``{"VAR": "heading text"}`` for every bullet naming a variable.
    """
    bullets: dict[str, str] = {}
    heading = ""
    for line in page.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            continue
        match = _BULLET_VAR.match(line)
        if match:
            bullets.setdefault(match.group(1), heading)
    return bullets


def _credential_sections(page: str) -> dict[str, str]:
    """Return the heading each credential is documented under.

    A credential the page omits entirely maps to ``""``, so the caller sees one
    verdict for "documented elsewhere" and "not documented at all".
    """
    bullets = _documented_bullets(page)
    return {var: bullets.get(var, "") for var in sorted(_credentials_read())}


@pytest.fixture
def page() -> str:
    """The shipped security reference."""
    return _PAGE.read_text(encoding="utf-8")


class TestTheScanReachesTheSurface:
    """The derivations read something, so a clean result is not a silent zero."""

    def test_the_transport_scan_finds_the_credentials(self) -> None:
        """A scan that read no credential would pass every rule below."""
        found = _credentials_read()
        assert len(found) >= _MINIMUM_CREDENTIALS, (
            f"expected at least {_MINIMUM_CREDENTIALS} IoT credentials read by "
            f"{_TRANSPORT.name}, found {sorted(found)}. A scan that reads nothing "
            "reports the same clean result as a fully documented page."
        )

    def test_the_page_documents_variables_as_bullets(self, page: str) -> None:
        """The bullet form is how this page already writes its variables."""
        bullets = _documented_bullets(page)
        assert len(bullets) >= _MINIMUM_DOCUMENTED_BULLETS, (
            f"expected at least {_MINIMUM_DOCUMENTED_BULLETS} documented variables "
            f"on {_PAGE.name}, parsed {sorted(bullets)}. A parser that matches no "
            "bullet cannot tell a documented variable from an omitted one."
        )

    def test_the_provisioner_hands_out_credentials(self) -> None:
        """The export list is the operator-facing set the page must explain."""
        exported = _provisioner_exports()
        assert len(exported) >= _MINIMUM_CREDENTIALS, (
            f"expected at least {_MINIMUM_CREDENTIALS} IoT credentials from "
            f"ProvisionedThing.env_vars, found {sorted(exported)}."
        )


class TestTheReferenceCoversTheCredentials:
    """Every credential the code honours is documented, and one link reads as one."""

    def test_every_credential_the_transport_reads_is_documented(self, page: str) -> None:
        """A credential the transport reads but the page omits is unreachable config."""
        missing = sorted(_credentials_read() - set(_documented_bullets(page)))
        assert not missing, (
            f"docs/security.md documents no bullet for IoT credentials the transport "
            f"reads: {missing}. Both the iot and bridge backends construct "
            "IotMqttTransport with no arguments, so a variable the code honours and "
            "the page omits is the whole of a setting the operator cannot find - "
            "connect() then returns False and the peer never appears on the fleet."
        )

    def test_every_credential_the_provisioner_exports_is_documented(self, page: str) -> None:
        """The page and the provisioner cannot disagree about the operator's set."""
        missing = sorted(_provisioner_exports() - set(_documented_bullets(page)))
        assert not missing, (
            f"ProvisionedThing.env_vars tells an operator to export {missing}, which "
            "docs/security.md does not explain. The provisioner's export list is the "
            "package's own answer to what must be set, so the page has to name it."
        )

    def test_the_credentials_are_documented_in_one_section(self, page: str) -> None:
        """The variables of one link belong under one heading."""
        sections = _credential_sections(page)
        assert len(set(sections.values())) == 1, (
            "the credentials that configure the AWS IoT broker link are documented "
            f"under different headings (empty = not documented at all): {sections}. "
            "One link's variables belong in one place: two name the broker and the "
            "Thing, two locate the certificate material, so a reader who finds only "
            "some of them configures half a connection."
        )


class TestTheSilentOffIsWhyTheOmissionMatters:
    """Facts about the code, true on both sides of this documentation change."""

    def test_a_missing_thing_name_is_reported_by_name_and_the_mesh_stays_off(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With nothing set, the operator learns the first name from a log line."""
        pytest.importorskip("awsiot")
        from strands_robots.mesh.transport import iot_transport

        for var in _credentials_read():
            monkeypatch.delenv(var, raising=False)
        transport = iot_transport.IotMqttTransport()

        with caplog.at_level(logging.ERROR, logger=iot_transport.__name__):
            assert transport.connect() is False

        assert any("STRANDS_IOT_THING_NAME" in record.getMessage() for record in caplog.records), (
            f"connect() refused without naming the missing credential: "
            f"{[record.getMessage() for record in caplog.records]}"
        )

    def test_a_missing_endpoint_is_reported_by_name(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each credential is discovered separately, one refusal at a time."""
        pytest.importorskip("awsiot")
        from strands_robots.mesh.transport import iot_transport

        for var in _credentials_read():
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("STRANDS_IOT_THING_NAME", "probe-thing")
        transport = iot_transport.IotMqttTransport()

        with caplog.at_level(logging.ERROR, logger=iot_transport.__name__):
            assert transport.connect() is False

        assert any("STRANDS_IOT_ENDPOINT" in record.getMessage() for record in caplog.records), (
            f"connect() refused without naming the missing endpoint: "
            f"{[record.getMessage() for record in caplog.records]}"
        )
