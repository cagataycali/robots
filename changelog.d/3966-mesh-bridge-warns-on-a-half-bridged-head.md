### Fixed: a topic in `STRANDS_MESH_BRIDGE_TOPICS` whose per-turn tails are not bridged now says so

`STRANDS_MESH_BRIDGE_TOPICS` matches a topic suffix exactly and
`STRANDS_MESH_BRIDGE_TOPICS_PREFIX` matches a head and its tails. An operator
who added a head to the exact list and then ran tail-suffixed traffic
(`<head>/<turn>`) on it got the bare topic on MQTT and every per-turn message
left on the LAN: the change looked like it took while half of it silently did
not. `_should_bridge` still refuses the tail, as documented, and now logs one
warning per head naming the topic, the unbridged tail and the prefix variable
that bridges it. `tests/mesh/test_bridge_half_bridged_head_warning.py` pins the
unchanged verdict, the once-per-head throttle, and silence for heads that are in
the prefix list or in neither list.
