"""Mutation table: this round's module vs the pre-round one.

Arm A = the reshaped module (this round). Arm B = the module as pushed, whose
`except ImportError: pass` is the alert. The reshape must lose nothing.
"""
from __future__ import annotations

import ast, json, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
import strands_robots
assert pathlib.Path(strands_robots.__file__).parents[1] == ROOT
RUN = sys.argv[1]

SRC = ROOT / "strands_robots" / "registry" / "policies.py"
TESTF = ROOT / "tests" / "registry" / "test_provider_import_error_names_its_remedy.py"
POST, PRE = pathlib.Path(f"/tmp/postround-{RUN}.py"), pathlib.Path(f"/tmp/preround-{RUN}.py")

prod = SRC.read_text(encoding="utf-8")
tree = ast.parse(prod)
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "import_policy_class")
region = "\n".join(prod.splitlines()[fn.lineno - 1: fn.end_lineno]) + "\n"

RAISE = """        except ImportError as exc:
            # A provider whose module needs an optional dependency at import
            # time (lerobot_local imports torch) otherwise raises a bare
            # "No module named 'torch'" naming neither this provider nor the
            # remedy - the dead end _provider_import_error exists to close.
            raise _provider_import_error(canonical, exc, config.get("extra")) from exc
"""
MUTS = {
    "M1 the funnel substitutes MockPolicy instead of reporting": (
        RAISE,
        "        except ImportError:\n            from strands_robots.policies.mock import MockPolicy\n\n            return MockPolicy\n",
    ),
    "M2 the funnel returns None instead of reporting": (
        RAISE, "        except ImportError:\n            return type(None)\n",
    ),
    "M3 the auto-discovery branch misreports a missing dep as unknown": (
        '        if getattr(exc, "name", None) != f"strands_robots.policies.{provider}":\n'
        "            raise _provider_import_error(provider, exc, None) from exc\n",
        "        pass\n",
    ),
}
for label, (old, _new) in MUTS.items():
    n_fn, n_file = region.count(old), prod.count(old)
    print(f"anchor [{label[:2]}]: in_fn={n_fn} in_file={n_file}")
    assert n_fn == 1, f"anchor not unique inside import_policy_class for {label}"


def run(label: str) -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTF), "-q", "--no-cov", "-p", "no:randomly", "--tb=no"],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    out = proc.stdout
    n = len([l for l in out.splitlines() if l.startswith("FAILED")])
    return n


rows = []
try:
    for label, (old, new) in MUTS.items():
        SRC.write_text(prod.replace(old, new, 1), encoding="utf-8")
        assert SRC.read_text(encoding="utf-8") != prod, f"{label}: mutation did not apply"
        shutil.copy(POST, TESTF); a = run("A")
        shutil.copy(PRE, TESTF);  b = run("B")
        rows.append({"mutation": label, "this_round": a, "pre_round": b})
        print(f"  {label:<62} this-round={a:>2}  pre-round={b:>2}")
    # control: unmutated
    SRC.write_text(prod, encoding="utf-8")
    shutil.copy(POST, TESTF); a = run("A")
    shutil.copy(PRE, TESTF);  b = run("B")
    rows.append({"mutation": "control (unmutated)", "this_round": a, "pre_round": b})
    print(f"  {'control (unmutated)':<62} this-round={a:>2}  pre-round={b:>2}")
finally:
    SRC.write_text(prod, encoding="utf-8")
    shutil.copy(POST, TESTF)
    assert SRC.read_text(encoding="utf-8") == prod, "PROD RESTORE FAILED"
    assert TESTF.read_text(encoding="utf-8") == POST.read_text(encoding="utf-8"), "TEST RESTORE FAILED"
    print("\nrestored: production and test file byte-identical")

json.dump({"tree": str(ROOT), "rows": rows}, open(f"/tmp/mutations-{RUN}.json", "w"), indent=2)
