### Fixed: an Isaac camera warming up no longer logs errors

`add_camera` polls the new camera until it returns a frame; each not-ready poll was logged at ERROR, so every healthy camera printed two error lines on Isaac Sim 6.1. Warm-up polls now log at DEBUG; real render failures still log at ERROR.
