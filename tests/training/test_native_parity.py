"""Native-pipeline argv parity tests (run only where the real checkouts exist).

These assert that every ``--flag`` our trainers emit in ``build_command`` is a
flag the *real* native pipeline actually accepts -- catching drift between our
wrapper and cosmos-framework without launching a full finetune.

Skipped automatically unless COSMOS_ROOT points at a real checkout
(set by CI on a GPU box; absent on laptops -> skipped, never failing).
"""

import os
import re

import pytest

from strands_robots.training import TrainSpec, create_trainer

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
