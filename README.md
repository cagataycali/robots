# Artifact: the IoT transport routes that decide broker delivery

Measurement for the PR that pins `strands_robots/mesh/transport/iot_transport.py`'s
five unexecuted lines. Tests only -- no library line changes -- so the artifact is
the coverage + mechanism + mutation measurement rather than a rollout.

| file | what it is |
| --- | --- |
| `broker_delivery_routes.png` | the figure |
| `capture.py` | reads the two coverage JSONs and calls the two drop mechanisms directly |
| `compose.py` | draws the figure; every rendered number is asserted against `facts.json` |
| `mutate.py` | the mutation table: 6 regressions x (new module, 470 pre-existing cases) |
| `facts.json` | every measured value in the figure |
| `mutations.json` | raw mutation results |

Reproduce (from a checkout of the PR branch):

```
MUJOCO_GL=egl python3 -m pytest tests/mesh -q --no-cov \
  --cov=strands_robots --cov-report=json:/tmp/cov-after-X.json --cov-fail-under=0
MUJOCO_GL=egl python3 -m pytest tests/mesh -q --no-cov \
  --ignore=tests/mesh/test_iot_broker_delivery_routes.py \
  --cov=strands_robots --cov-report=json:/tmp/cov-before-X.json --cov-fail-under=0
python3 mutate.py X && python3 capture.py X && python3 compose.py X
```

`mutate.py` AST-scopes every anchor to its enclosing function, asserts the anchor
occurs exactly once inside it, and restores the source byte-identically in a
`finally`.
