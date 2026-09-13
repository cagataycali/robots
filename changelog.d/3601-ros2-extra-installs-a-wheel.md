### Fixed: the `[ros2]` extra installs a wheel, instead of a source build on every supported Python

`pip install 'strands-robots[ros2]'` could not install on any interpreter this
project supports. The extra declared `cyclonedds>=0.10.2,<1.0.0` while
`requires-python` is `>=3.12`, and cyclonedds publishes wheels for cp37-cp310 up
to 0.10.5 and for cp310-cp313 from 11.0.1 - so nothing that ceiling admitted had
a wheel for a supported Python, and the install fell through to the sdist:

    Failed to build `cyclonedds==0.10.5`
    Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH

That C install is the one thing the extra exists to make unnecessary. The RTPS
transport is the rclpy-free half of the ROS 2 story - `use_rtps`,
`HardwareRtpsBridge` and `docs/rtps-integration.md` all offer it as "a
self-contained pip wheel", no sourced distro - and an install that needs a
CycloneDDS toolchain leaves it no easier to provision than the distro it was
built to avoid.

The ceiling now admits the wheel-bearing series (`>=0.10.2,<12`, capping the
major per the manifest's own bound convention) and `uv.lock` records cyclonedds
11.0.1 with its cp312/cp313 wheels. The RTPS suites and a two-participant DDS
loopback (write, take, exact round-trip) pass unchanged on it; the API the
package uses - `idl`, `idl.annotations`, `idl.types`, `domain`, `pub`, `sub`,
`topic`, `qos` - is the same in both series.

`uv lock --check` cannot report this: the sdist it resolved *is* a resolution
inside the declared bounds, which is what `--check` grades. So
`tests/test_lockfile_parity_gate.py` gains it as a third offline drift class -
a dependency an extra documents as a wheel whose only locked artifact is a
source distribution - beside the below-floor and missing-from-the-lock rules it
already covers.

Unchanged, and upstream's to fix: cyclonedds has published no linux-aarch64
wheel in any release, so an aarch64 host still builds from source. The rule is
phrased over the artifact kind rather than per-platform coverage for that
reason.
