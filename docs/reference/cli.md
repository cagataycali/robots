---
description: Every flag of strands-robots doctor, verify-dataset, dashboard and iot, what each prints, and its exit codes.
---

# Command line

One console script, `strands-robots`, with four subcommands; `python -m strands_robots <command>` is the same entry point. After this page you know every flag each subcommand takes, what it prints, and its exit codes.

```bash
strands-robots --help       # usage and the command list
strands-robots --version    # strands-robots <installed version>
```

The first token is the subcommand; a missing or unknown one prints the list (`doctor`, `verify-dataset`, `dashboard`, `iot`) and exits 1. Flags after it belong to that command's own parser.

## doctor

Checks the machine for a working install; exit 0 when every check passes, 1 when any fails.

```bash
strands-robots doctor
strands-robots doctor --list
```

| flag | effect |
|---|---|
| `--list` | print the check names and exit 0 without probing anything |
| `-h`, `--help` | usage and exit |

An unknown argument exits 2 with the usage line. Each row prints `PASS`, `WARN`, `SKIP` or `FAIL` with one line of detail and, for a failure, the fix; only `FAIL` rows change the exit code. `NO_COLOR` or `TERM=dumb` turns colour off. [Doctor](../start/doctor.md#the-probes) lists the probes in order, with expected output.

## verify-dataset

Validates the episode integrity of a recorded LeRobot dataset on disk; exit 0 when the report is `ok`, 1 when a check fails.

```bash
strands-robots verify-dataset ~/.cache/huggingface/lerobot/you/so101_pick --expected 20
strands-robots verify-dataset ./my_dataset --json --no-check-videos
```

| argument | default | effect |
|---|---|---|
| `root` | required | dataset root directory, the one that contains `meta/` |
| `-e`, `--expected N` | none | require exactly N distinct episodes |
| `--min-frames N` | `1` | every episode must hold at least N frames; `0` disables the check |
| `--json` | off | print the report as JSON instead of the human summary |
| `--no-check-videos` | videos checked | skip the per-episode video-file checks |
| `--no-check-stats` | stats checked | skip the dead-control-column check (an all-zero action or state column) |

The summary lists the episode count, frame totals and every problem, one per line; `strands_robots.verify_dataset.verify_dataset(root, ...)` returns the dict `--json` prints. [Verify](../learn/data/verify.md) says what each check catches.

## dashboard

Serves the operator dashboard with uvicorn; 0 on a clean shutdown, 2 when it refuses to start.

```bash
strands-robots dashboard --open
strands-robots dashboard --host 0.0.0.0 --port 8090 --log-level debug
```

| flag | default | effect |
|---|---|---|
| `--host ADDR` | `127.0.0.1` | bind address |
| `--port N` | `8090` | TCP port, 1 to 65535; else exit 2 |
| `--open` | off | open the default browser after binding |
| `--log-level L` | `info` | uvicorn log level: `critical`, `error`, `warning`, `info`, `debug` |

A bind beyond `127.0.0.1`, `::1` or `localhost` is refused with exit 2 until the API is guarded: enrol a passkey on `http://127.0.0.1` first, or set `DASHBOARD_AUTH_TOKEN`. Needs the `dashboard` extra; pages and endpoints: [dashboard](../learn/dashboard.md).

## iot

AWS IoT identities for the `iot` and `bridge` mesh backends (`[mesh-iot]` extra, AWS credentials in the environment). Exit 0 on success, 1 when AWS refused or the Thing is missing, 2 on usage error.

```bash
strands-robots iot clear-retained-safety --apply
```

| verb | effect |
|---|---|
| `provision-robot THING [--estop-publish]` | Thing, CSR certificate with `CN=THING`, `strands-robot-no-estop` policy (or `strands-robot`) |
| `provision-operator THING` | the same, `strands-operator` policy |
| `reprovision THING [--estop-publish {keep,drop}]` | rotate the certificate (Thing, attributes, policies stay); `strands-robot` needs the flag; restart the peer |
| `withdraw-estop-publish [--keep THING] [--apply]` | move `strands-robot` certificates to `strands-robot-no-estop`, except `--keep`; dry run without `--apply` |
| `clear-retained-safety [--apply]` | post-rollout, delete retained `strands/safety/` messages; dry run without `--apply` |
| `teardown THING` | delete the Thing, its certificates and local files |

Every verb takes `--region`; Thing verbs take `--cert-dir` (default `~/.strands_robots/iot`) and print `export` lines. `reprovision` gives a pre-CSR robot (certificate CN `AWS IoT Certificate`) the direct-reply grant ([direct messaging](../learn/mesh/direct.md)).
