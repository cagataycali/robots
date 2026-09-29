### Fixed: Isaac render() returns the size it was asked for

`render(width=, height=)` returned the camera's native resolution on Isaac, where MuJoCo returns the requested one. The frame is now resampled to the requested size, and the envelope reports both sizes.
