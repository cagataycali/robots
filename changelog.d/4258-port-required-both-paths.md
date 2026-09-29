### Fixed: a serial arm with no usable port is refused when Robot() is called, on both drivers

`port=""` on the lerobot path constructed and failed at the first bus action; an omitted `port` on `driver="strands"` returned a driver that failed there too. Both now raise the lerobot path's `ValueError` at construction, naming this host's serial devices. The twin transport still needs no port.
