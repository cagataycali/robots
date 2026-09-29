### Fixed: Isaac keeps MJCF joint names USD cannot spell, and their drives and damping

A robot whose MJCF joints are not identifiers (so101: "1".."6") came out of Isaac's converter with mangled joint names and limp joints, and every MJCF robot lost its joints' passive damping. The backend now maps the names back to MuJoCo's and authors each drive with the actuator's gains plus the joint's damping.
