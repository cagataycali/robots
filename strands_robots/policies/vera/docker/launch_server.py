"""Container launcher: install the offline ckpt resolver, then run the VERA server.

Equivalent to ``python -m vera.server.start_vera_server <args>`` but imports
``wandb_offline_resolve`` FIRST so the IDM wandb-run-id is resolved to the locally
mounted checkpoint (provenance.json match) instead of hitting the network.

Usage (from entrypoint.sh):
    python /opt/launch_server.py --embodiment mimicgen --port 8800 [...]
"""
import runpy
import sys

import wandb_offline_resolve  # noqa: F401 — import side effect patches download_checkpoint

# Hand the remaining argv to the server module as if it were invoked with -m.
sys.argv = ["vera.server.start_vera_server", *sys.argv[1:]]
runpy.run_module("vera.server.start_vera_server", run_name="__main__")
