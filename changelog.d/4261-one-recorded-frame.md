### Changed: every recording path writes its frame through one `RecordedFrame`

The MuJoCo, Isaac and Newton single-policy recording hooks and the MuJoCo and
Isaac `run_multi_policy` loops each built the dataset frame themselves: the
`<robot>__<key>` prefixing once a scene holds two robots, the undriven robots'
measured state, and the action columns the driven robots owe. Those five copies
are now `strands_robots.simulation.recording.RecordedFrame`; a backend supplies
only its state, action and camera arrays. The recorded frames are unchanged.
