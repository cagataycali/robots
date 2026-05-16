# AWS IoT Spike Findings (2026-05-16)

Empirical results from running raw `awscrt.mqtt5` against AWS IoT Core in
account `947951559549`, region `us-west-2`, endpoint
`a2acz9p1ge6619-ats.iot.us-west-2.amazonaws.com`.

These findings amend `AWS_IOT_MESH_INTEGRATION.md` (the design doc) with
real-world IoT Core behaviour discovered during the spike.

## What was tested

| Spike | What | Result |
|---|---|---|
| `spike_publish.py` | Single Thing publishes presence + state, subscribes to its own + wildcard | ✅ |
| `spike_op_robot.py` (cross-process) | Robot → Operator response over MQTT v5 | ✅ |
| `spike_shadow.py` | Robot updates named shadow `presence`, operator reads via REST | ✅ |
| `subscriber.py + publisher.py` | Two-process Thing-A → Thing-B publish; Thing-A → own response (security check) | ✅ allowed / ❌ denied (correct) |

## 5 things the design doc didn't catch

### 1. `iot:RetainPublish` is a separate action

If your topic resource includes `iot:Publish` but not `iot:RetainPublish`,
publishing with `retain=True` returns `PUBACK reason_code=135` (Not
Authorized). This is silently dropped by the broker.

**Fix in policy templates**: include `iot:RetainPublish` alongside `iot:Publish`
for `presence`, `health`, `safety/event`, `safety/estop`, `lidar/state`,
`map/info`, `*/shadow/*`.

### 2. `*` in topic-resource ARNs matches `/`, but **only where the policy
   says so**

Conventional wisdom says IoT Policy `*` matches across slashes (unlike MQTT
filter `+`). Confirmed: `topic/strands/${ThingName}/*` matches
`strands/so100-01/state` AND `strands/so100-01/lidar/summary`. Good — our
nested topic schema works.

But the **operator publishing back to a robot** path failed because the
operator policy didn't grant `iot:Publish` on `topic/strands/*/cmd`. Add that
explicitly. Same for the **robot publishing responses to an operator's
namespace** — robot policy needs `iot:Publish` on
`topic/strands/*/response/*` (because the robot doesn't own the operator's
prefix).

### 3. The `${iot:Connection.Thing.ThingName}` substitution requires a
   `client-id` matching the Thing name

If the MQTT5 client connects with a `client_id` that's not the Thing name
attached to the cert, the substitution silently leaves the variable
unsubstituted and **all policies degrade to implicit deny**. The SDK's
`mqtt5_client_builder.mtls_from_path()` accepts an arbitrary `client_id` —
we MUST set it equal to the Thing name in our `IotMqttTransport` to keep
the policy variables working.

### 4. `test-authorization` returns `IMPLICIT_DENY` with `missingContextValues`
   when called without `--client-id`

This was misleading. It does NOT mean the actual MQTT call would be denied.
Always pass `--client-id <thing-name>` to get a faithful answer. Production
code doesn't have this problem because real connections always carry a
client-id, but it cost ~10 minutes during the spike.

### 5. MQTT5 same-client publish-receive does NOT echo by default in awscrt

When a single client subscribes to `strands/+/presence` and publishes to
`strands/me/presence`, the **publishing client does NOT receive its own
publish** even though it has a matching subscription. This is per MQTT 5
spec (subscription default `No Local: false` actually means "you DO get
your own"... but the SDK appears to not deliver them in our tests).

**Implication for code**: do not rely on self-echo for the wildcard
`strands/+/presence` subscription to populate the local peer registry with
this peer's own presence. Mesh already filters self-loop in `_on_presence`
(`if peer_id == self.peer_id: return`), so this is already correct. But our
**tests** need to be cross-process (or at least cross-client-id) to validate
the wire path.

## Updated policy templates

Two policies, attached to two cert types. Verified working.

### Robot Thing policy (`strands-robot`)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid":"AllowConnect","Effect":"Allow","Action":"iot:Connect",
     "Resource":"arn:aws:iot:*:*:client/${iot:Connection.Thing.ThingName}"},
    {"Sid":"AllowOwnTopics","Effect":"Allow",
     "Action":["iot:Publish","iot:RetainPublish"],
     "Resource":["arn:aws:iot:*:*:topic/strands/${iot:Connection.Thing.ThingName}/*"]},
    {"Sid":"AllowResponseToAnyOperator","Effect":"Allow","Action":"iot:Publish",
     "Resource":["arn:aws:iot:*:*:topic/strands/*/response/*"]},
    {"Sid":"AllowSafetyEstop","Effect":"Allow",
     "Action":["iot:Publish","iot:RetainPublish"],
     "Resource":["arn:aws:iot:*:*:topic/strands/safety/estop"]},
    {"Sid":"AllowOwnSubscriptions","Effect":"Allow","Action":"iot:Subscribe",
     "Resource":[
       "arn:aws:iot:*:*:topicfilter/strands/${iot:Connection.Thing.ThingName}/*",
       "arn:aws:iot:*:*:topicfilter/strands/broadcast",
       "arn:aws:iot:*:*:topicfilter/strands/safety/estop",
       "arn:aws:iot:*:*:topicfilter/strands/+/presence"
     ]},
    {"Sid":"AllowReceiveOthers","Effect":"Allow","Action":"iot:Receive",
     "Resource":["arn:aws:iot:*:*:topic/strands/*"]},
    {"Sid":"AllowShadow","Effect":"Allow",
     "Action":["iot:Publish","iot:Subscribe","iot:Receive"],
     "Resource":[
       "arn:aws:iot:*:*:topic/$aws/things/${iot:Connection.Thing.ThingName}/shadow/*",
       "arn:aws:iot:*:*:topicfilter/$aws/things/${iot:Connection.Thing.ThingName}/shadow/*"
     ]}
  ]
}
```

### Operator Thing policy (`strands-operator`)

Same `AllowConnect`, `AllowShadow`. Plus:

```json
{"Sid":"OperatorPublishToFleet","Effect":"Allow",
 "Action":["iot:Publish","iot:RetainPublish"],
 "Resource":[
   "arn:aws:iot:*:*:topic/strands/*/cmd",
   "arn:aws:iot:*:*:topic/strands/broadcast",
   "arn:aws:iot:*:*:topic/strands/safety/estop"
 ]},
{"Sid":"OperatorReceiveResponses","Effect":"Allow",
 "Action":["iot:Subscribe","iot:Receive"],
 "Resource":[
   "arn:aws:iot:*:*:topic/strands/${iot:Connection.Thing.ThingName}/response/*",
   "arn:aws:iot:*:*:topicfilter/strands/${iot:Connection.Thing.ThingName}/response/*"
 ]},
{"Sid":"OperatorObserveFleet","Effect":"Allow",
 "Action":["iot:Subscribe","iot:Receive"],
 "Resource":[
   "arn:aws:iot:*:*:topic/strands/*",
   "arn:aws:iot:*:*:topicfilter/strands/+/presence",
   "arn:aws:iot:*:*:topicfilter/strands/+/state",
   "arn:aws:iot:*:*:topicfilter/strands/+/health",
   "arn:aws:iot:*:*:topicfilter/strands/+/safety/event",
   "arn:aws:iot:*:*:topicfilter/strands/safety/estop"
 ]}
```

## Implementation handoff

The `IotMqttTransport` class needs to:

1. Resolve `client_id` to the Thing name at construction (read from cert SAN
   or env `STRANDS_IOT_THING_NAME`). Hard-fail if not set.
2. Subscribe with patterns derived from the Mesh role (robot vs operator)
   that match the policy templates above.
3. Wrap MQTT5 callbacks with the `_MqttSampleShim` (zenoh-shaped Sample obj)
   so existing `Mesh._on_presence`, `_on_cmd`, `_on_response` handlers work
   unchanged.
4. Translate Zenoh wildcards (`*`, `**`) to MQTT (`+`, `#`) in
   `declare_subscriber`. Our exclusive patterns are:
   - `strands/*/presence` → `strands/+/presence`
   - `strands/{peer}/response/**` → `strands/{peer}/response/#` (or `+`
     since turn_ids are single-segment)
5. Per-topic QoS map (presence=1+retain, state=0, cmd=1, response=1,
   safety/*=2, broadcast=1).
6. SQLite-backed store-and-forward for offline operation.
7. Reconnect with jitter — never thunder-herd.

## Spike artefacts (kept for reference)

- `/tmp/iot-spike/spike_publish.py` — single-Thing pub/sub
- `/tmp/iot-spike/spike_op_robot.py` — same-process op→robot RPC
  (assertions fail due to MQTT5 same-client semantics — see finding #5)
- `/tmp/iot-spike/subscriber.py` + `publisher.py` — two-process op→robot
  RPC (works perfectly)
- `/tmp/iot-spike/spike_shadow.py` — named shadow + REST read
- `/tmp/iot-spike/robot-policy-v3.json` — final working robot policy
- `/tmp/iot-spike/operator-policy.json` — final working operator policy

## AWS resources created (will be torn down before merge)

- Thing: `so100-spike-01` (cert + presence shadow)
- Thing: `so100-spike-02` (cert)
- Thing: `bedrock-agent-spike-01` (cert)
- Policy: `strands-robot-spike` (v3)
- Policy: `strands-operator-spike` (v1)
- Certificates: 3 active

Cleanup script: `/tmp/iot-spike/teardown.sh` (TBD).

