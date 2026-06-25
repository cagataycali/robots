"""Container launcher: install the offline ckpt resolver, then run the VERA server.

Equivalent to ``python -m vera.server.start_vera_server <args>`` but imports
``wandb_offline_resolve`` FIRST so the IDM wandb-run-id is resolved to the locally
mounted checkpoint (provenance.json match) instead of hitting the network.

Usage (from entrypoint.sh):
    python /opt/launch_server.py --embodiment mimicgen --port 8800 [...]
"""

import runpy
import sys

import wandb_offline_resolve

# Patch download_checkpoint to resolve the IDM wandb-run-id against the locally
# mounted checkpoints before the server module is imported (idempotent; the module
# also self-installs on import, this explicit call documents the ordering contract).
wandb_offline_resolve.install()

# Hand the remaining argv to the server module as if it were invoked with -m.
sys.argv = ["vera.server.start_vera_server", *sys.argv[1:]]
runpy.run_module("vera.server.start_vera_server", run_name="__main__")
