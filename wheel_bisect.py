import io, json, re, subprocess, sys, zipfile
from pathlib import Path

VERSIONS = ["0.36.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"]
out = {}
for v in VERSIONS:
    d = Path(f"/tmp/hubprobe/w{v}")
    d.mkdir(parents=True, exist_ok=True)
    whl = list(d.glob("*.whl"))
    if not whl:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "download", f"huggingface_hub=={v}",
             "--no-deps", "-q", "-d", str(d)],
            capture_output=True, text=True)
        if r.returncode != 0:
            out[v] = {"error": r.stderr.strip()[-200:]}
            continue
        whl = list(d.glob("*.whl"))
    z = zipfile.ZipFile(whl[0])
    names = set(z.namelist())
    has_buckets_mod = "huggingface_hub/cli/buckets.py" in names
    # locate the CLI entry module (name moved across versions)
    entry = None
    for cand in ("huggingface_hub/cli/hf.py", "huggingface_hub/commands/huggingface_cli.py"):
        if cand in names:
            entry = cand
            break
    reg_buckets = reg_sync = False
    if entry:
        src = z.read(entry).decode("utf-8", "replace")
        reg_buckets = bool(re.search(r'name\s*=\s*["\']buckets["\']|buckets_cli', src))
        reg_sync = bool(re.search(r'\bsync\b', src))
    out[v] = {
        "cli/buckets.py": has_buckets_mod,
        "entry": entry,
        "registers_buckets": reg_buckets,
        "registers_sync": reg_sync,
        "cli_files": sorted(n.split("/")[-1] for n in names
                            if n.startswith("huggingface_hub/cli/") and n.endswith(".py"))[:6],
    }
print(json.dumps(out, indent=1))
