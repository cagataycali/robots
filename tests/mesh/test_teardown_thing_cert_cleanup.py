"""Behavioural pins for ``teardown_thing``'s cert + policy cleanup sequence.

``teardown_thing`` is the privileged unprovisioning path: it must detach and
delete every certificate attached to a Thing (paginating when the Thing has
more than one page of principals), detach each cert's attached policies first,
deactivate-then-force-delete each cert, and finally delete the Thing -- all
idempotently, so a missing Thing or a per-cert AWS error never aborts the
overall teardown.

These tests drive the whole sequence against an in-memory fake IoT client that
records the ordered call log, so they assert the observable AWS contract
(which APIs ran, with which arguments, in which order) rather than internal
state. No real AWS, no boto3 required.
"""

from __future__ import annotations

import pytest


class _NotFound(Exception):
    """Stand-in for ``iot.exceptions.ResourceNotFoundException``."""


class FakeIoT:
    """Records every IoT API call teardown_thing makes, in order.

    ``principals`` seeds the cert ARNs returned for the Thing; ``policies``
    maps a cert ARN to the policy names attached to it. ``use_paginator``
    toggles whether the client exposes ``get_paginator`` (real boto3 clients
    do; minimal test shims may not -- teardown must handle both).
    """

    def __init__(self, principals, policies=None, *, use_paginator=True):
        self._principals = list(principals)
        self._policies = policies or {}
        self._use_paginator = use_paginator
        self.calls: list[tuple[str, dict]] = []
        self.exceptions = type("Exc", (), {"ResourceNotFoundException": _NotFound})()
        if use_paginator:
            self.get_paginator = self._get_paginator

    def _record(self, name, **kw):
        self.calls.append((name, kw))

    def _get_paginator(self, op):
        self._record("get_paginator", op=op)
        principals = self._principals

        class _Pager:
            def paginate(self, **kw):
                # Two pages to prove pagination is actually consumed.
                mid = len(principals) // 2
                yield {"principals": principals[:mid]}
                yield {"principals": principals[mid:]}

        return _Pager()

    def list_thing_principals(self, **kw):
        self._record("list_thing_principals", **kw)
        return {"principals": self._principals}

    def detach_thing_principal(self, **kw):
        self._record("detach_thing_principal", **kw)

    def list_attached_policies(self, **kw):
        self._record("list_attached_policies", **kw)
        target = kw.get("target")
        return {"policies": [{"policyName": p} for p in self._policies.get(target, [])]}

    def detach_policy(self, **kw):
        self._record("detach_policy", **kw)

    def update_certificate(self, **kw):
        self._record("update_certificate", **kw)

    def delete_certificate(self, **kw):
        self._record("delete_certificate", **kw)

    def delete_thing(self, **kw):
        self._record("delete_thing", **kw)


def _names(client):
    return [name for name, _ in client.calls]


@pytest.fixture
def stub_boto3(monkeypatch):
    """Patch ``_require_boto3`` to hand teardown_thing a given FakeIoT."""
    from strands_robots.mesh.iot import provision

    def _install(client):
        fake_boto3 = type("FakeBoto3", (), {})()
        fake_boto3.client = lambda *a, **kw: client
        monkeypatch.setattr(provision, "_require_boto3", lambda: fake_boto3)
        return provision

    return _install


def test_teardown_detaches_deactivates_and_deletes_each_cert(stub_boto3):
    """A Thing with two certs: each cert is detached from the Thing, its
    policies detached, then deactivated and force-deleted; the Thing is
    deleted last."""
    arn_a = "arn:aws:iot:us-east-1:123:cert/CERTAAA"
    arn_b = "arn:aws:iot:us-east-1:123:cert/CERTBBB"
    client = FakeIoT(
        principals=[arn_a, arn_b],
        policies={arn_a: ["RobotPolicy"], arn_b: ["RobotPolicy", "EstopPolicy"]},
    )
    provision = stub_boto3(client)

    provision.teardown_thing("robot-alpha", region="us-east-1")

    names = _names(client)
    # Both certs detached from the Thing by ARN.
    detached = [kw["principal"] for n, kw in client.calls if n == "detach_thing_principal"]
    assert detached == [arn_a, arn_b]
    # Policies detached before cert deletion -- cert B had two policies.
    detached_pols = [(kw["policyName"], kw["target"]) for n, kw in client.calls if n == "detach_policy"]
    assert ("RobotPolicy", arn_a) in detached_pols
    assert ("RobotPolicy", arn_b) in detached_pols
    assert ("EstopPolicy", arn_b) in detached_pols
    # Each cert deactivated then force-deleted by its cert id (ARN tail).
    deactivated = [kw["certificateId"] for n, kw in client.calls if n == "update_certificate"]
    deleted = [(kw["certificateId"], kw["forceDelete"]) for n, kw in client.calls if n == "delete_certificate"]
    assert deactivated == ["CERTAAA", "CERTBBB"]
    assert deleted == [("CERTAAA", True), ("CERTBBB", True)]
    # Thing deleted exactly once, after the certs.
    assert names.count("delete_thing") == 1
    assert names.index("delete_thing") > names.index("delete_certificate")
    assert {"thingName": "robot-alpha"} == next(kw for n, kw in client.calls if n == "delete_thing")


def test_teardown_paginates_principals(stub_boto3):
    """When the client exposes a paginator, teardown consumes every page so a
    Thing with more than one page of certs is fully cleaned (no orphans)."""
    arns = [f"arn:aws:iot:us-east-1:123:cert/CERT{i:03d}" for i in range(6)]
    client = FakeIoT(principals=arns)
    provision = stub_boto3(client)

    provision.teardown_thing("robot-many")

    assert "get_paginator" in _names(client)
    deleted = {kw["certificateId"] for n, kw in client.calls if n == "delete_certificate"}
    assert deleted == {a.rsplit("/", 1)[-1] for a in arns}


def test_teardown_without_paginator_falls_back_to_single_call(stub_boto3):
    """A minimal client shim lacking ``get_paginator`` still tears down via the
    single-call ``list_thing_principals`` fallback."""
    arn = "arn:aws:iot:us-east-1:123:cert/CERTONE"
    client = FakeIoT(principals=[arn], use_paginator=False)
    provision = stub_boto3(client)

    provision.teardown_thing("robot-shim")

    names = _names(client)
    assert "get_paginator" not in names
    assert "list_thing_principals" in names
    assert [kw["certificateId"] for n, kw in client.calls if n == "delete_certificate"] == ["CERTONE"]


def test_teardown_missing_thing_is_silent_success(stub_boto3):
    """If the Thing is gone (ResourceNotFoundException on principal lookup),
    teardown treats it as already-clean: no cert calls, no raise."""
    client = FakeIoT(principals=[], use_paginator=False)

    def _raise(**kw):
        raise _NotFound("no such thing")

    client.list_thing_principals = _raise
    provision = stub_boto3(client)

    provision.teardown_thing("ghost-robot")  # must not raise

    names = _names(client)
    assert "detach_thing_principal" not in names
    assert "delete_certificate" not in names


def test_teardown_swallows_per_cert_errors_and_finishes(stub_boto3):
    """Every per-cert AWS step (detach principal, detach policy, deactivate /
    delete cert) is best-effort: an error on the first cert is swallowed so the
    second cert and the Thing itself are still cleaned up (idempotent
    teardown)."""
    arn_a = "arn:aws:iot:us-east-1:123:cert/BADCERT"
    arn_b = "arn:aws:iot:us-east-1:123:cert/GOODCERT"
    client = FakeIoT(
        principals=[arn_a, arn_b],
        policies={arn_a: ["RobotPolicy"], arn_b: ["RobotPolicy"]},
        use_paginator=False,
    )

    real_detach = client.detach_thing_principal
    real_detach_pol = client.detach_policy
    real_delete = client.delete_certificate

    def _flaky_detach(**kw):
        if kw["principal"] == arn_a:
            raise RuntimeError("AWS throttled")
        return real_detach(**kw)

    def _flaky_detach_pol(**kw):
        if kw["target"] == arn_a:
            raise RuntimeError("AWS throttled")
        return real_detach_pol(**kw)

    def _flaky_delete(**kw):
        if kw["certificateId"] == "BADCERT":
            raise RuntimeError("AWS throttled")
        return real_delete(**kw)

    client.detach_thing_principal = _flaky_detach
    client.detach_policy = _flaky_detach_pol
    client.delete_certificate = _flaky_delete
    provision = stub_boto3(client)

    provision.teardown_thing("robot-flaky")  # must not raise

    # The healthy cert still got fully cleaned and the Thing was removed.
    deleted = [kw["certificateId"] for n, kw in client.calls if n == "delete_certificate"]
    assert "GOODCERT" in deleted
    assert "delete_thing" in _names(client)


def test_teardown_tolerates_thing_already_deleted(stub_boto3):
    """If the Thing vanishes before ``delete_thing`` (ResourceNotFoundException),
    teardown still returns cleanly rather than propagating the AWS error."""
    client = FakeIoT(principals=[], use_paginator=False)

    def _gone(**kw):
        raise _NotFound("already deleted")

    client.delete_thing = _gone
    provision = stub_boto3(client)

    provision.teardown_thing("robot-vanished")  # must not raise
