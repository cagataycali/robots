"""End to end: two Mesh peers over AWS IoT Core Direct Messaging, against a real account.

Requires (skips cleanly otherwise):
    - awsiotsdk + boto3 (``pip install 'strands-robots[mesh-iot]'``)
    - AWS credentials with iot:* in the configured region
    - ``STRANDS_IOT_DIRECT_E2E=1`` (this module creates Things, certificates and
      a policy named ``dm-e2e-*`` in the account and removes them at the end)

What it proves, in order:

1. ``provision_robot`` / ``provision_operator`` issue CSR certificates with
   ``CN=<thing>`` and publish the current policy documents (the ones carrying
   the direct messaging grants).
2. A robot Mesh in a second process and an operator Mesh here: every
   ``Mesh.send`` is delivered as a direct message (the operator transport's
   ``direct_stats``), every reply comes back, and the robot's replies went out
   as direct messages too (its ``direct_stats`` printed at exit). The median
   round trip is recorded.
3. With the robot gone, ``Mesh.send`` answers ``peer offline (iot 404)`` in
   under one second instead of spending its budget.
4. Fallback: the robot's certificate is moved to a policy WITHOUT the direct
   reply grant; its reply gets 403, the robot publishes on the computed key
   instead, and the operator still receives the answer.

Run explicitly::

    STRANDS_IOT_DIRECT_E2E=1 hatch run test-integ tests_integ/test_iot_direct_messaging_e2e.py -s
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import textwrap
import time
from typing import Any

import pytest

pytest.importorskip("awsiot", reason="awsiotsdk not installed")
boto3 = pytest.importorskip("boto3", reason="boto3 not installed")

from strands_robots.mesh.core import Mesh  # noqa: E402
from strands_robots.mesh.iot.provision import (  # noqa: E402
    ROBOT_POLICY_NAME,
    _ensure_policy,
    _robot_policy_doc,
    provision_operator,
    provision_robot,
    teardown_thing,
)

pytestmark = pytest.mark.skipif(
    os.getenv("STRANDS_IOT_DIRECT_E2E", "") != "1",
    reason="set STRANDS_IOT_DIRECT_E2E=1 to create dm-e2e-* resources in the configured AWS account",
)

ROBOT = "dm-e2e-robot"
OPERATOR = "dm-e2e-operator"
NO_DIRECT_POLICY = "dm-e2e-robot-no-direct"
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"

_ROBOT_CHILD = textwrap.dedent(
    """
    import json, os, sys, time, logging
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    from strands_robots.mesh.core import Mesh

    class Bot:
        tool_name_str = "e2e-bot"
        def get_task_status(self):
            return {"status": "idle", "thing": os.environ["STRANDS_IOT_THING_NAME"]}

    m = Mesh(Bot(), peer_id=os.environ["STRANDS_IOT_THING_NAME"], peer_type="robot")
    m.start()
    print(json.dumps({"event": "started", "alive": m.alive}), flush=True)
    deadline = time.time() + float(sys.argv[1])
    while time.time() < deadline:
        time.sleep(0.2)
    stats = getattr(m._direct, "direct_stats", None)
    m.stop()
    print(json.dumps({"event": "stopped", "direct_stats": stats}), flush=True)
    """
)


class _Bot:
    tool_name_str = "e2e-operator"

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "idle"}


def _env_for(thing: Any) -> dict[str, str]:
    env = dict(os.environ)
    env.update(thing.env_vars())
    env.setdefault("STRANDS_MESH_AUTH_MODE", "none")
    env.setdefault("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "1")
    env.setdefault("STRANDS_MESH_LOCAL_DEV", "true")
    return env


def _start_robot(thing: Any, seconds: float) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _ROBOT_CHILD, str(seconds)],
        env=_env_for(thing),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()
    started = json.loads(line)
    assert started["alive"] is True, started
    time.sleep(1.5)  # let the presence and the persistent HTTPS leg settle
    return proc


def _robot_exit_stats(proc: subprocess.Popen[str]) -> dict[str, int]:
    out, _ = proc.communicate(timeout=60)
    for line in out.splitlines():
        rec = json.loads(line)
        if rec.get("event") == "stopped":
            return rec["direct_stats"] or {}
    raise AssertionError(f"robot child printed no stop record: {out!r}")


@pytest.fixture(scope="module")
def fleet(tmp_path_factory):
    cert_dir = tmp_path_factory.mktemp("dm-e2e-certs")
    robot = provision_robot(ROBOT, region=REGION, cert_dir=cert_dir)
    operator = provision_operator(OPERATOR, region=REGION, cert_dir=cert_dir)
    iot = boto3.client("iot", region_name=REGION)
    yield {"robot": robot, "operator": operator, "iot": iot}
    for thing in (ROBOT, OPERATOR):
        teardown_thing(thing, region=REGION, cert_dir=cert_dir)
    try:
        iot.delete_policy(policyName=NO_DIRECT_POLICY)
    except Exception:  # noqa: BLE001 - best-effort cleanup of a policy the fallback test may not have created
        pass


@pytest.fixture
def operator_mesh(fleet, monkeypatch):
    for k, v in _env_for(fleet["operator"]).items():
        monkeypatch.setenv(k, v)
    m = Mesh(_Bot(), peer_id=OPERATOR, peer_type="operator")
    m.start()
    assert m.alive
    time.sleep(1.0)
    yield m
    m.stop()


def test_1_certificates_carry_the_thing_name_and_the_policies_carry_the_grants(fleet):
    from strands_robots.mesh.iot.provision import OPERATOR_POLICY_NAME

    assert fleet["robot"].subject_cn == ROBOT
    assert fleet["operator"].subject_cn == OPERATOR
    iot = fleet["iot"]
    robot_doc = json.loads(iot.get_policy(policyName=ROBOT_POLICY_NAME)["policyDocument"])
    op_doc = json.loads(iot.get_policy(policyName=OPERATOR_POLICY_NAME)["policyDocument"])
    assert "AllowDirectResponseToAnyOperator" in {s["Sid"] for s in robot_doc["Statement"]}
    assert "AllowDirectCommandToAnyRobot" in {s["Sid"] for s in op_doc["Statement"]}
    subject = subprocess.run(
        ["openssl", "x509", "-in", str(fleet["robot"].cert_path), "-noout", "-subject"],
        capture_output=True,
        text=True,
        check=False,
    )
    if subject.returncode == 0:
        assert f"CN={ROBOT}" in subject.stdout and "O=strands-robots" in subject.stdout


def test_2_every_command_and_reply_is_a_direct_message(fleet, operator_mesh):
    proc = _start_robot(fleet["robot"], seconds=25)
    try:
        latencies = []
        for _ in range(5):
            t0 = time.time()
            reply = operator_mesh.send(ROBOT, {"action": "status"}, timeout=10)
            latencies.append((time.time() - t0) * 1000)
            assert reply.get("type") == "response", reply
            assert reply["result"]["thing"] == ROBOT
        op_stats = operator_mesh._direct.direct_stats  # type: ignore[union-attr]
        assert op_stats["sent"] == 5 and op_stats["delivered"] == 5 and op_stats["failed"] == 0
    finally:
        robot_stats = _robot_exit_stats(proc)
    assert robot_stats["delivered"] == 5 and robot_stats["failed"] == 0, robot_stats
    print(f"\n[iot-direct e2e] RPC p50 {statistics.median(latencies):.0f} ms, max {max(latencies):.0f} ms")


def test_3_an_offline_robot_answers_at_once(fleet, operator_mesh):
    proc = _start_robot(fleet["robot"], seconds=3)
    _robot_exit_stats(proc)
    time.sleep(2.0)
    t0 = time.time()
    reply = operator_mesh.send(ROBOT, {"action": "status"}, timeout=30)
    elapsed = time.time() - t0
    assert reply == {"status": "error", "error": "peer offline (iot 404)", "peer": ROBOT}
    assert elapsed < 1.0, f"offline verdict took {elapsed:.2f}s"
    print(f"\n[iot-direct e2e] offline verdict in {elapsed * 1000:.0f} ms")


def test_4_a_robot_without_the_direct_grant_falls_back_to_publish(fleet, operator_mesh):
    iot = fleet["iot"]
    robot = fleet["robot"]
    doc = _robot_policy_doc(allow_estop_publish=True)
    doc["Statement"] = [s for s in doc["Statement"] if s["Sid"] != "AllowDirectResponseToAnyOperator"]
    _ensure_policy(iot, NO_DIRECT_POLICY, doc)
    iot.attach_policy(policyName=NO_DIRECT_POLICY, target=robot.cert_arn)
    iot.detach_policy(policyName=ROBOT_POLICY_NAME, target=robot.cert_arn)
    time.sleep(3.0)  # authorizer cache
    try:
        proc = _start_robot(robot, seconds=20)
        try:
            reply = operator_mesh.send(ROBOT, {"action": "status"}, timeout=10)
            assert reply.get("type") == "response", reply
        finally:
            robot_stats = _robot_exit_stats(proc)
        # The robot tried the direct reply, was refused, and published instead.
        assert robot_stats["failed"] >= 1 and robot_stats["delivered"] == 0, robot_stats
    finally:
        iot.attach_policy(policyName=ROBOT_POLICY_NAME, target=robot.cert_arn)
        iot.detach_policy(policyName=NO_DIRECT_POLICY, target=robot.cert_arn)
