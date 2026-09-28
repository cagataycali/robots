### Fixed: the Isaac-on-AWS smoke no longer silently skips the asset install on a fresh instance

`examples/isaac_on_aws/run_smoke.sh` fetches the `unitree_g1` asset on the
instance host with `python3 -m pip install --target ... | tail -1`. That block
is shipped to the instance as one `AWS-RunShellScript` document and ran under
`set -e` only - not the script's top-level `set -euo pipefail` - so the pipe
returned `tail`'s status (0) and a missing `pip` printed "No module named pip"
without aborting. `provision.sh`'s cloud-init never installed `python3-pip`, and
Ubuntu's `python3` ships without pip or `ensurepip`, so on a fresh instance the
asset install was silently skipped and `add_robot("g1")` later refused for a
model file that was merely absent - two errors downstream of the cause.

The captured package path was corrupted the same way: `robot_descriptions`
1.23.0 prints a `Cloning ...`/`Found commit ...` notice to **stdout** on the
first import that triggers the clone, and a bare `PKG=$(python3 -c ... 2>/dev/null)`
folded that notice into `PKG`, so `cp -rL "$PKG"` got a multi-line non-path.

cloud-init now installs `python3-pip`; the staged block enables `pipefail` so the
pip step fails loudly if pip is ever absent; and the package path is emitted
behind a `PKGPATH=` sentinel that `sed` pulls back out, so any notice on stdout
is tolerated.
