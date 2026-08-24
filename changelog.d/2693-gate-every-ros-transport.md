### Fixed: the ROS 2 operator-approval gate covers every transport onto the graph, not only `use_ros`

Three agent-callable tools reach a ROS 2 graph - `use_ros` over in-process rclpy, `use_rtps`
over raw RTPS and `use_rosbridge` over a rosbridge WebSocket - and all three can carry a
command to a physical robot. The gate that holds safety-critical surfaces behind an operator
interrupt was consulted from `use_ros` alone, and neither sibling took a `tool_context` at all,
so an agent declined at a drive topic could re-issue the identical command under another tool
name and it went out with no prompt, no allowlist check and no audit row. Measured against the
same blocklisted surface, `use_ros` prompted and refused on a decline while `use_rtps` reported
`published 1 message(s) to /probe_only/cmd_vel` - the tool name was the whole difference. This
is the defect #2313 fixed across verbs, one level up: one tool gated, three tools reachable.

The blocklist, the two env knobs and the approval decision now have a single owner,
`strands_robots.tools._command_gate`, which all three tools consult - the same shape
`_numeric_options` already uses for the same three tools, and for the reason its own docstring
gives: a rule about a physical surface cannot differ between two transports onto that surface.
Which verbs carry a command stays per transport at each call site, because only `use_ros`
speaks all three protocols. `use_rtps` and `use_rosbridge` are now `@tool(context=True)` and
thread the context in, so they prompt rather than inheriting the fail-closed path, and
`use_ros` keeps its own wrapper so its call sites, its `use_ros:`-prefixed refusals and its
`use_ros_tool` audit source are unchanged.

Both new call sites sit after the backend probe and before the transport lock, so a transport
that cannot publish never prompts and a human deciding never holds the DDS lock or the
WebSocket connection, and after argument validation, so an incomplete call is reported without
asking an operator about it. Reading stays ungated everywhere, as does `use_rtps`'s `advertise`,
which creates a publisher without writing a sample; both boundaries are pinned. The new suite's
structural half derives the set of commanding transports from the tree rather than listing
them, so a fourth transport is graded on arrival instead of shipping un-gated.
