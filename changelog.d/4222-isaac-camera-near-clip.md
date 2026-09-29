### Fixed: Isaac cameras see objects closer than one metre

A camera created by the Isaac backend's `add_camera` kept USD's default near clipping plane of 1 m, so a camera placed under a metre from a tabletop scene rendered only the far floor. The near plane is now 1 cm, matching the Kit viewport camera and MuJoCo's default.
