"""AWS IoT Core integration for the strands-robots mesh.

This subpackage owns the cloud-side concerns of the mesh:

- :mod:`provision` — one-command bootstrap of a Thing + cert + policy.
  Implements the 5-line out-of-box experience: a customer runs
  ``strands-robots iot provision`` once and the next ``Robot()`` call
  joins their AWS account's mesh with mTLS authentication.
- (future) ``shadow`` — Named-shadow mirror of presence / state.
- (future) ``camera_offload`` — S3-backed camera frame transport.

The wire-level transport (``IotMqttTransport``) lives in
:mod:`strands_robots.mesh.transport.iot_transport` and is independent of
this package — you can use it without ever calling :mod:`provision` if you
already have certs.
"""

from strands_robots.mesh.iot.provision import provision_robot, provision_operator

__all__ = ["provision_robot", "provision_operator"]
