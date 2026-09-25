### Fixed

- `examples/04_mesh_peer_discovery.py` gives each copy its own mesh identity.
  Every copy joined as `example-arm-01`, and a record is filed under its
  `peer_id`, so two terminals overwrote one row and each read the other as
  itself - both printed `Discovered mesh peers: 0` beside the hint to start a
  second terminal. Pass an id on the command line for a readable name; the
  default cannot be shared with another process.
