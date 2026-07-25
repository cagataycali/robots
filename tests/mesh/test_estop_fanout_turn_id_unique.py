"""Regression: the cloud e-stop fan-out Lambda must stamp each fan-out with a
unique turn_id (and a numeric timestamp).

Robots dedup inbound commands on ``(sender_id, turn_id)`` in ``_exec_cmd``.
The fan-out Lambda published a constant ``turn_id="estop-fanout"`` with a
constant ``sender_id``, so the SECOND-ever cloud e-stop -- and every one after
-- was dropped fleet-wide as a replay of the first. ``timestamp`` was also fed
``context.aws_request_id`` (a UUID) into a numeric epoch-seconds field.

We exec the deployed Lambda source with AWS mocked and invoke the handler for
two consecutive e-stops, asserting each robot command carries a distinct
turn_id equal to that invocation's aws_request_id, and a numeric timestamp.
"""

from __future__ import annotations

import json
import types

import boto3

from strands_robots.mesh.iot import bootstrap as boot_mod


class _FakePaginator:
    def paginate(self, **_kw):
        return [
            {
                "things": [
                    {"thingName": "robot-a", "attributes": {"strands-mesh-role": "robot"}},
                    {"thingName": "robot-b", "attributes": {"strands-mesh-role": "robot"}},
                ]
            }
        ]


def _run_fanout(monkeypatch, *, request_id: str, estop_t: str) -> list[dict]:
    """Exec the deployed Lambda source and invoke it once. Returns the list of
    decoded command payloads published to robot /cmd inboxes."""
    published: list[dict] = []

    iot = types.SimpleNamespace(get_paginator=lambda _name: _FakePaginator())

    def _publish(topic, qos, payload):
        published.append(json.loads(payload.decode()))

    iot_data = types.SimpleNamespace(publish=_publish)
    # ddb.put_item succeeds (no ConditionalCheckFailed) -> not a duplicate.
    ddb = types.SimpleNamespace(
        put_item=lambda **_kw: {},
        exceptions=types.SimpleNamespace(ConditionalCheckFailedException=type("CCFE", (Exception,), {})),
    )
    fakes = {"iot": iot, "iot-data": iot_data, "dynamodb": ddb}
    monkeypatch.setattr(boto3, "client", lambda name, *a, **k: fakes[name])

    ns: dict = {}
    exec(boot_mod._ESTOP_LAMBDA_SOURCE, ns)  # noqa: S102 - deployed source under test
    handler = ns["lambda_handler"]

    ctx = types.SimpleNamespace(aws_request_id=request_id)
    event = {"peer_id": "op-1", "t": estop_t, "responses_received": 3}
    handler(event, ctx)
    return published


def test_two_consecutive_fanouts_carry_distinct_turn_ids(monkeypatch):
    first = _run_fanout(monkeypatch, request_id="req-aaaa", estop_t="100.0")
    second = _run_fanout(monkeypatch, request_id="req-bbbb", estop_t="200.0")

    assert first and second
    # Every command in one invocation shares that invocation's request id...
    assert {p["turn_id"] for p in first} == {"req-aaaa"}
    assert {p["turn_id"] for p in second} == {"req-bbbb"}
    # ...and the two invocations use DIFFERENT turn_ids, so a robot deduping on
    # (sender_id, turn_id) does not reject the second e-stop as a replay.
    assert first[0]["turn_id"] != second[0]["turn_id"]


def test_fanout_timestamp_is_numeric(monkeypatch):
    published = _run_fanout(monkeypatch, request_id="req-cccc", estop_t="300.0")
    assert published
    for p in published:
        assert isinstance(p["timestamp"], (int, float))
        # Not the request-id UUID that used to leak into the numeric field.
        assert p["timestamp"] != "req-cccc"
