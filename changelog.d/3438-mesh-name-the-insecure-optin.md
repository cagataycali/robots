### Fixed:
- The `WIRE SECURITY DISABLED` error line now names the variable that actually disabled wire auth: `STRANDS_MESH_LOCAL_DEV` when the localhost preset opened the gate, otherwise `STRANDS_MESH_AUTH_MODE=none + STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1`. It used to credit `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1` even when the operator never set it.
