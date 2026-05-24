#!/usr/bin/env python3
"""Rogue-fleet orchestrator.

For each rogue scenario:

1. Build a per-scenario tempdir (CA, certs, ACL, audit dir).
2. Compose the victim's posture (env vars).
3. Fork the victim subprocess; block on its ``READY`` line.
4. Fork the rogue subprocess with its own posture + the victim's listen
   endpoint + path to a result JSONL file.
5. Wait for the rogue to exit; collect its result.
6. SIGTERM the victim; collect its ``GOODBYE``.
7. Append to the global summary.

Each scenario is **fully isolated** -- new tempdir, new ports, new audit
log. Scenarios run sequentially by default (--parallel toggles); a flaky
rogue cannot pollute the next one.

Usage::

    python run_fleet.py                     # run all rogues sequentially
    python run_fleet.py --rogue rogue_05    # run one rogue
    python run_fleet.py --filter safety     # run rogues whose dir name contains 'safety'
    python run_fleet.py --keep-tmp          # don't rm scenario tempdirs (debugging)

Exit code 0 = every defence held. Non-zero = at least one bypass.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ROBOTS_REPO = Path(__file__).resolve().parents[2]

# Each rogue dir has:
#   attack.py    -- the attack script (subprocess entry-point)
#   README.md    -- humans-readable explanation
#   av_id.txt    -- canonical AV id (one line) used for cross-ref to harness
# The orchestrator discovers rogues by listing dirs named ``rogue_*``.


@dataclass
class Scenario:
    rogue_dir: Path
    rogue_id: str
    av_id: str
    title: str
    needs_victim: bool  # set in metadata.json: do we fork victim_robot.py?
    # Posture overrides (env applied to the victim before spawn)
    victim_env: dict[str, str] = field(default_factory=dict)
    # Posture overrides for the rogue (env applied to rogue subprocess)
    rogue_env: dict[str, str] = field(default_factory=dict)


def _load_metadata(rogue_dir: Path) -> Scenario:
    """Each rogue ships ``metadata.json`` describing its scenario."""
    meta_path = rogue_dir / "metadata.json"
    if not meta_path.exists():
        # Fallback: best-effort inference
        return Scenario(
            rogue_dir=rogue_dir,
            rogue_id=rogue_dir.name,
            av_id="",
            title=rogue_dir.name,
            needs_victim=True,
        )
    meta = json.loads(meta_path.read_text())
    return Scenario(
        rogue_dir=rogue_dir,
        rogue_id=rogue_dir.name,
        av_id=meta.get("av_id", ""),
        title=meta.get("title", rogue_dir.name),
        needs_victim=meta.get("needs_victim", True),
        victim_env=dict(meta.get("victim_env", {})),
        rogue_env=dict(meta.get("rogue_env", {})),
    )


def _wait_for_ready(proc: subprocess.Popen, timeout: float = 8.0) -> dict | None:
    """Read the victim's stdout until we see READY or timeout."""
    deadline = time.time() + timeout
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return None
            time.sleep(0.05)
            continue
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "READY":
            return event
        if event.get("event") == "FAILED":
            return event
    return None


def _provision_pki(scenario_dir: Path) -> dict[str, str]:
    """Create a CA + victim leaf cert + a separate operator cert.

    Returns env vars the victim needs to come up in mtls posture.
    """
    sys.path.insert(0, str(ROOT.parent))
    from rogue_fleet._lib.pki import EphemeralCA

    ca = EphemeralCA.make(scenario_dir / "pki")
    cert, key = ca.mint("victim-r1", sub="victim")
    # Operator cert (used by some rogues to act as a peer)
    op_cert, op_key = ca.mint("operator-1", sub="operator")
    return {
        "CA_CERT": str(ca.ca_cert),
        "VICTIM_CERT": str(cert),
        "VICTIM_KEY": str(key),
        "OPERATOR_CERT": str(op_cert),
        "OPERATOR_KEY": str(op_key),
        "PKI_DIR": str(scenario_dir / "pki"),
    }


def _victim_env_for(scenario: Scenario, pki: dict[str, str], port: int, audit_dir: Path) -> dict[str, str]:
    """Compose the victim env: production defaults + scenario overrides."""
    base = {
        # Make sure subprocesses can find the in-tree mesh.
        "PYTHONPATH": f"{ROBOTS_REPO}:{ROOT.parent}",
        # Known-safe defaults: full mtls + ACL on permissive shape.
        "STRANDS_MESH_AUTH_MODE": "mtls",
        "STRANDS_MESH_TLS_CA": pki["CA_CERT"],
        "STRANDS_MESH_TLS_CERT": pki["VICTIM_CERT"],
        "STRANDS_MESH_TLS_KEY": pki["VICTIM_KEY"],
        "STRANDS_MESH_NAMESPACE": "strands",
        "STRANDS_MESH_MULTICAST": "false",
        "STRANDS_MESH_PORT": str(port),
        "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL": "1",
        "STRANDS_MESH_AUDIT_DIR": str(audit_dir),
        # Tighten DoS caps for clean test signals.
        "STRANDS_MESH_MAX_CMD_BYTES": "4096",
        "STRANDS_MESH_MAX_SAFETY_BYTES": "2048",
        "STRANDS_MESH_CMD_RATE_HZ": "20",
        "STRANDS_MESH_SAFETY_RATE_HZ": "2",
        "VICTIM_PEER_ID": "victim-r1",
        "VICTIM_LISTEN_PORT": str(port),
    }
    # Apply scenario overrides (string values; ``__UNSET__`` means delete).
    env = {**os.environ, **base, **scenario.victim_env}
    for k, v in list(env.items()):
        if v == "__UNSET__":
            env.pop(k, None)
    # Surface PKI vars to the rogue too (they read them via env).
    env.update(pki)
    return env


def _rogue_env_for(scenario: Scenario, base: dict[str, str], result_file: Path, victim_listen: str) -> dict[str, str]:
    env = {**base, **scenario.rogue_env}
    env["ROGUE_RESULT_FILE"] = str(result_file)
    env["VICTIM_LISTEN"] = victim_listen
    for k, v in list(env.items()):
        if v == "__UNSET__":
            env.pop(k, None)
    return env


def run_scenario(scenario: Scenario, *, keep_tmp: bool = False) -> list[dict[str, Any]]:
    """Run one scenario and return the list of result records the rogue wrote."""
    scenario_dir = Path(tempfile.mkdtemp(prefix=f"rogue_{scenario.rogue_id}_"))
    try:
        pki = _provision_pki(scenario_dir)
        # Free port for this scenario (random high ephemeral port).
        sys.path.insert(0, str(ROOT.parent))
        from rogue_fleet._lib.zenoh_helpers import free_port  # noqa: WPS433

        port = free_port()
        audit_dir = scenario_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        result_file = scenario_dir / "result.jsonl"

        victim_env = _victim_env_for(scenario, pki, port, audit_dir)
        rogue_env = _rogue_env_for(
            scenario,
            victim_env,
            result_file,
            victim_listen=f"localhost:{port}",
        )

        victim_proc: subprocess.Popen | None = None
        if scenario.needs_victim:
            victim_proc = subprocess.Popen(
                [sys.executable, str(ROOT / "target" / "victim_robot.py")],
                env=victim_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            ready = _wait_for_ready(victim_proc, timeout=10.0)
            if not ready or ready.get("event") != "READY":
                victim_proc.kill()
                victim_proc.wait(timeout=3)
                stderr = (victim_proc.stderr.read() if victim_proc.stderr else "") or ""
                return [
                    {
                        "rogue_id": scenario.rogue_id,
                        "av_id": scenario.av_id,
                        "title": scenario.title,
                        "posture": "<victim-failed-to-start>",
                        "defence_held": False,
                        "observed": f"victim never became ready: {ready!r}",
                        "error": stderr[-2000:],
                        "duration_s": 0.0,
                    }
                ]

        # Run the rogue.
        try:
            rogue_proc = subprocess.run(
                [sys.executable, str(scenario.rogue_dir / "attack.py")],
                env=rogue_env,
                capture_output=True,
                text=True,
                timeout=60,
            )
        finally:
            if victim_proc is not None:
                try:
                    victim_proc.send_signal(signal.SIGTERM)
                    victim_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    victim_proc.kill()
                    victim_proc.wait(timeout=3)

        # Read results.
        records: list[dict[str, Any]] = []
        if result_file.exists():
            for line in result_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not records:
            # The rogue printed nothing usable -- emit a synthetic failure.
            records = [
                {
                    "rogue_id": scenario.rogue_id,
                    "av_id": scenario.av_id,
                    "title": scenario.title,
                    "posture": "<no-result>",
                    "defence_held": False,
                    "observed": (rogue_proc.stdout or "")[-1500:],
                    "error": (rogue_proc.stderr or "")[-2000:],
                    "duration_s": 0.0,
                }
            ]
        return records
    finally:
        if not keep_tmp:
            shutil.rmtree(scenario_dir, ignore_errors=True)


def discover_rogues(filter_substr: str | None = None) -> list[Path]:
    rogues = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("rogue_"))
    if filter_substr:
        rogues = [r for r in rogues if filter_substr in r.name]
    return rogues


def main() -> int:
    parser = argparse.ArgumentParser(description="strands-robots rogue-fleet pentest orchestrator")
    parser.add_argument("--rogue", help="Run a single rogue by directory name")
    parser.add_argument("--filter", help="Run rogues whose name contains this substring")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep per-scenario tempdirs")
    parser.add_argument("--summary", default=str(ROOT / "FLEET_RESULTS.md"), help="Markdown summary path")
    args = parser.parse_args()

    if args.rogue:
        target = ROOT / args.rogue
        if not target.exists():
            print(f"no such rogue: {args.rogue}", file=sys.stderr)
            return 2
        rogues = [target]
    else:
        rogues = discover_rogues(args.filter)

    if not rogues:
        print("no rogues found", file=sys.stderr)
        return 2

    print(f"⚡ launching {len(rogues)} rogue(s)\n")
    all_records: list[dict[str, Any]] = []
    for rogue in rogues:
        scenario = _load_metadata(rogue)
        print(f"→ {scenario.rogue_id}  ({scenario.av_id}) -- {scenario.title}")
        t0 = time.time()
        records = run_scenario(scenario, keep_tmp=args.keep_tmp)
        dt = time.time() - t0
        for r in records:
            mark = "✅" if r.get("defence_held") else "❌"
            print(f"   {mark} {r.get('observed', '')[:90]}")
        print(f"   ({dt:.1f}s)")
        all_records.extend(records)

    held = sum(1 for r in all_records if r.get("defence_held"))
    total = len(all_records)
    print(f"\n=== {held}/{total} defences held ===")

    # Write a Markdown summary.
    lines = [
        "# Rogue Fleet Results",
        "",
        f"**Run at**: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"**Total**: {held}/{total} defences held",
        "",
        "| Rogue | AV | Title | Posture | Held | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in all_records:
        held_mark = "✅" if r.get("defence_held") else "❌"
        notes = (r.get("observed") or "")[:120].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{r.get('rogue_id', '')}` | {r.get('av_id', '')} | {r.get('title', '')} "
            f"| {r.get('posture', '')[:60].replace('|', '\\|')} | {held_mark} | {notes} |"
        )
    Path(args.summary).write_text("\n".join(lines) + "\n")
    print(f"\nsummary -> {args.summary}")

    return 0 if held == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
