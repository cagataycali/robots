Title: fix(v1): a feetech bus read costs the reply, not the 50 ms read timeout

Fix to part 5 of #4749 · roadmap #3818
<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cagataycali/robots/pr-media/v1/feetech-bus-reads-the-reply-size/d06_bus_frame.dark.svg"><img alt="The bus reads FF FF ID LEN, then exactly LEN bytes; asking for two more waits out the timeout; an echoed request and a stray byte are skipped." src="https://raw.githubusercontent.com/cagataycali/robots/pr-media/v1/feetech-bus-reads-the-reply-size/d06_bus_frame.paper.svg"></picture>

## In one paragraph
`FeetechBus._txrx` asked pyserial for `6 + nparams + 2` bytes and `sync_read` for `(6 + n) * len(ids) + 2`. A non-echoing adapter sends exactly `6 + nparams`, so every `ping`, `read` and `sync_read` sat until the 50 ms serial timeout: a real SO-101's `observe()` costs 54 ms and the 30 Hz loop cannot hold its period. The `+ 2` did not help an echoing adapter either; on dev that wire fails with `reply from servo 254, expected 1`. Now the bus reads the 4-byte head `FF FF ID LEN`, then exactly `LEN` bytes, skips an echoed request and a stray byte ahead of the header, and the timeout is 20 ms.

## What it looks like
<img alt="Two SO-101 twins replaying 3 s of observes: dev left, fixed right" src="https://raw.githubusercontent.com/cagataycali/robots/pr-media/v1/feetech-bus-reads-the-reply-size/so101_bus_before_after.gif">

Rendered headless in MuJoCo from what `Robot("so101", port=<pty>)` observed in 3 s against six emulated STS3215 servos over real pyserial: dev 30 observes, fixed 8650.

## Measured
| gate | dev | this PR |
|---|---|---|
| `bus.ping(1)` p50 / p95 | 50.10 / 50.23 ms | 0.07 / 0.07 ms |
| `bus.sync_read(6 x 6 B)` | 50.16 / 50.39 ms | 0.15 / 0.16 ms |
| `FeetechDriver.read()` | 50.20 / 50.60 ms | 0.19 / 0.20 ms |
| constructor's 6 pings | 300.82 ms | 0.43 ms |
| `pytest tests/test_drivers.py` (3 wires x 4 verbs, never waits) | 13 failed | 27 passed |
| acceptance 9 `Driver.read()` p50 < 3 ms on a pty | 50.4 ms, failed | passed |
| ruff, format, mypy, lint-imports; `pytest` | | clean; 253 passed |

Refs #3818
