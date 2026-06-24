"""Native-pipeline argv parity tests (run only where the real checkouts exist).

These assert that every ``--flag`` our trainers emit in ``build_command`` is a
flag the *real* native pipeline actually accepts -- catching drift between our
wrapper and Isaac-GR00T / cosmos-framework without launching a full finetune.

Skipped automatically unless GR00T_ROOT / COSMOS_ROOT point at real checkouts
(set by CI on a GPU box; absent on laptops -> skipped, never failing).
"""

import ast
import os
import re

import pytest

from strands_robots.training import TrainSpec, create_trainer

GR00T_ROOT = os.environ.get("GR00T_ROOT")
COSMOS_ROOT = os.environ.get("COSMOS_ROOT")


def _flag_names(cmd):
    """Extract the set of --flag names (without =value) from an argv list."""
    names = set()
    for tok in cmd:
        m = re.match(r"^--([a-zA-Z0-9_\-]+)=", tok)
        if m:
            names.add(m.group(1))
        elif tok.startswith("--"):
            names.add(tok[2:])
    return names


@pytest.mark.skipif(
    not (GR00T_ROOT and os.path.isfile(os.path.join(GR00T_ROOT, "gr00t", "configs", "finetune_config.py"))),
    reason="GR00T_ROOT not set to a real Isaac-GR00T checkout",
)
def test_groot_flags_match_real_finetune_config():
    """Every --flag we emit must be a real FinetuneConfig dataclass field."""
    cfg_path = os.path.join(GR00T_ROOT, "gr00t", "configs", "finetune_config.py")
    with open(cfg_path) as f:
        src = f.read()
    tree = ast.parse(src)
    fields = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FinetuneConfig":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
    assert fields, "could not parse FinetuneConfig fields"

    spec = TrainSpec(
        dataset_root="/tmp/ds",
        base_model="nvidia/GR00T-N1.5-3B",
        output_dir="/tmp/out",
        embodiment="GR1",
        steps=500,
        save_freq=250,
        global_batch_size=8,
        learning_rate=1e-4,
        extra={"groot_root": GR00T_ROOT},
    )
    cmd = create_trainer("groot").build_command(spec)
    emitted = _flag_names(cmd)
    # Drop the torchrun/launcher meta-flags that aren't FinetuneConfig fields.
    emitted -= {"nproc_per_node", "master_port"}
    unknown = emitted - fields
    assert not unknown, f"flags not in real FinetuneConfig: {sorted(unknown)}"


@pytest.mark.skipif(
    not (COSMOS_ROOT and os.path.isfile(os.path.join(COSMOS_ROOT, "cosmos_framework", "scripts", "train.py"))),
    reason="COSMOS_ROOT not set to a real cosmos-framework checkout",
)
def test_cosmos_train_accepts_sft_toml():
    """The real cosmos train.py must accept --sft-toml (our sole driver flag)."""
    train_py = os.path.join(COSMOS_ROOT, "cosmos_framework", "scripts", "train.py")
    with open(train_py) as f:
        src = f.read()
    assert '"--sft-toml"' in src or "'--sft-toml'" in src, "real cosmos train.py no longer accepts --sft-toml"

    spec = TrainSpec(
        dataset_root="/tmp/ds",
        base_model="nvidia/Cosmos3",
        output_dir="/tmp/out",
        steps=10,
        save_freq=5,
        global_batch_size=1,
        extra={"cosmos_root": COSMOS_ROOT, "sft_toml": train_py},  # any real file
    )
    cmd = create_trainer("cosmos3").build_command(spec)
    assert any(t.startswith("--sft-toml=") for t in cmd), cmd
    assert "cosmos_framework.scripts.train" in cmd


@pytest.mark.skipif(
    not (COSMOS_ROOT and os.path.isdir(os.path.join(COSMOS_ROOT, "cosmos_framework", "scripts"))),
    reason="COSMOS_ROOT not set to a real cosmos-framework checkout",
)
def test_cosmos_convert_and_export_scripts_exist():
    """prepare()/export() target real cosmos scripts that still exist."""
    scripts = os.path.join(COSMOS_ROOT, "cosmos_framework", "scripts")
    assert os.path.isfile(os.path.join(scripts, "convert_model_to_dcp.py"))
    assert os.path.isfile(os.path.join(scripts, "export_model.py"))
