# Command line

The package installs one console script, `strands-robots`, with four subcommands. `python -m strands_robots <command>` is the same entry point. After this page you know every flag each subcommand takes, what it prints, and its exit codes.

```bash
strands-robots --help       # usage and the command list
strands-robots --version    # strands-robots <installed version>
```

The first token is the subcommand. A missing or unknown command prints the list (`doctor`, `verify-dataset`, `dashboard`, `iot`) and exits 1. Flags after the command belong to that command's own parser.

## doctor

Checks the machine for a working install and exits 0 when every check passes, 1 when any fails.

```bash
strands-robots doctor
strands-robots doctor --list
```

| flag | effect |
|---|---|
| `--list` | print the check names and exit 0 without probing anything |
| `-h`, `--help` | usage and exit |

An argument the parser does not know exits 2 with the usage line. The checks run in this order: Python, Package, Strands SDK, MuJoCo, MuJoCo GL, LeRobot, Torchcodec, CUDA/GPU, Torch Arch, Warp Arch, Serial, HF Auth, Device Connect, Mesh, Sim Test. Each row prints `PASS`, `WARN`, `SKIP` or `FAIL` with one line of detail and, for a failure, the fix; only `FAIL` rows change the exit code. `NO_COLOR` or `TERM=dumb` turns the colour off. The walk-through with expected output is [start/doctor.md](../start/doctor.md).

## verify-dataset

Validates the episode integrity of a recorded LeRobot dataset on disk. Exit 0 when the report is `ok`, 1 when any check fails.

```bash
strands-robots verify-dataset ~/.cache/huggingface/lerobot/you/so101_pick --expected 20
strands-robots verify-dataset ./my_dataset --json --no-check-videos
```

| argument | default | effect |
|---|---|---|
| `root` | required | dataset root directory, the one that contains `meta/` |
| `-e`, `--expected N` | none | require exactly N distinct episodes, the count you meant to record |
| `--min-frames N` | `1` | every episode must hold at least N frames; `0` disables the check |
| `--json` | off | print the report as JSON instead of the human summary |
| `--no-check-videos` | videos checked | skip the per-episode video-file exists and non-empty checks |
| `--no-check-stats` | stats checked | skip the dead-control-column check (an action or state column that is all zeros) |

The summary lists the episode count, frame totals and every problem found, one per line. The same report is available in Python as `strands_robots.verify_dataset.verify_dataset(root, ...)`, which returns the dict the `--json` flag prints. See [learn/data/verify.md](../learn/data/verify.md) for what each check catches.

## dashboard

Serves the operator dashboard with uvicorn. Returns 0 on a clean shutdown, 2 when it refuses to start.

```bash
strands-robots dashboard --open
strands-robots dashboard --host 0.0.0.0 --port 8090 --log-level debug
```

| flag | default | effect |
|---|---|---|
| `--host ADDR` | `127.0.0.1` | bind address |
| `--port N` | `8090` | TCP port, 1 to 65535; anything else exits 2 |
| `--open` | off | open the dashboard in the default browser after binding |
| `--log-level L` | `info` | uvicorn log level: `critical`, `error`, `warning`, `info`, `debug` |

A bind to anything other than `127.0.0.1`, `::1` or `localhost` is refused with exit 2 until the API is guarded: enrol a passkey on `http://127.0.0.1` first, or set `DASHBOARD_AUTH_TOKEN`. The dashboard needs the `dashboard` extra (`pip install "strands-robots[dashboard]"`). Its pages and endpoints are described in [learn/dashboard.md](../learn/dashboard.md).

## iot

Provisions, re-provisions and tears down AWS IoT identities for the `iot` and `bridge` mesh backends (`[mesh-iot]` extra, AWS credentials in the environment). Exit 0 on success, 1 when AWS refused or the Thing does not exist, 2 on a usage error.

```bash
strands-robots iot provision-robot so101-arm-01
strands-robots iot provision-operator ops-console-1
strands-robots iot reprovision so101-arm-01
strands-robots iot teardown so101-arm-01
```

| verb | effect |
|---|---|
| `provision-robot THING [--no-estop-publish]` | Thing, CSR certificate with `CN=THING`, `strands-robot` policy (or `strands-robot-no-estop`) |
| `provision-operator THING` | the same, `strands-operator` policy |
| `reprovision THING` | rotate the certificate: the new one is active before the old ones are deleted; the Thing, its attributes and policies stay. A running peer's MQTT session ends, so restart it |
| `teardown THING` | delete the Thing, its certificates and the local files |

Every verb takes `--region` and `--cert-dir` (default `~/.strands_robots/iot`) and prints the `export` lines. `reprovision` is how a robot provisioned before the CSR default (certificate CN `AWS IoT Certificate`) gains the direct-reply grant ([direct messaging](../learn/mesh/direct.md)).
