### Changed: a relative `export_xml(output_path=...)` lands in `~/.strands_robots/scenes`

`export_xml` resolved a relative destination against the process working
directory, so the most natural agent call - `export_xml {"output_path":
"scene.xml"}` when asked to save the scene - dropped the file wherever the
process was started, the user's git checkout included, while the same agent's
`render` was confined to `~/.strands_robots/renders`. A relative path (a bare
name or one with directories) is now anchored to the scenes directory
(`STRANDS_ROBOTS_SCENE_ROOT`, default `~/.strands_robots/scenes`); an absolute
path is written as given, as before. Every guard (traversal, symlinked target,
metacharacters) runs on the anchored destination, and the success text reports
the resolved path. Scripts that passed relative paths and read the file back
from the CWD need the returned path or an absolute destination.
