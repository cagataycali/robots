### Quality: pin the `robot_mesh` tool's two opposite optional-dependency arms

`_audit_tool_action` swallows an unavailable `strands_robots.mesh.audit` and
leaves a DEBUG breadcrumb, so a broken audit path stays discoverable to an
operator asking why LLM tool actions stopped appearing in the safety log; the
mesh-module arm refuses with a structured error instead, because an action that
needs the fleet cannot be served without it. Neither arm was driven, so the
forensic record of every tool action could have gone quiet with the suite green.
Adds cases pinning the breadcrumb's presence, its level and its content, that a
broken audit path leaves a tool call's answer byte-identical, that the mesh-module
refusal names the module and carries the import failure, and that the two channels
stay opposite so the asymmetry reads as deliberate.
