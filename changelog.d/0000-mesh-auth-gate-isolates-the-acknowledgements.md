### Fixed: the wire-auth refusal is pinned against a leaked acknowledgement, not by luck of test order

`tests/mesh/test_session_config.py` pins that `STRANDS_MESH_AUTH_MODE=none`
refuses to build a config without a second factor. Its isolation fixture
cleared eleven env vars but neither acknowledgement the gate actually reads, so
the refusal was only exercised when nothing earlier in the process had set one.
Seven examples do `os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")` at
import and the example smoke tests load them in process: run the fleet
transport smoke test first and the cell reports `DID NOT RAISE`, because the
dev preset is itself an acknowledgement. The module now runs with a leaked
preset as its standing condition, clears both factors per cell, drops the keys
its own raw `os.environ` writes leave behind, and grades the isolation roster
against the env vars the gate reads - so a third factor cannot silently retire
the refusal.
