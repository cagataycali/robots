### Fixed: `export_xml` without `output_path` labels a capped reply as a preview

The inline reply is capped at 2,000 characters, but its header named the full
length (`Model XML (12130 chars)` for an SO-100 scene) and the body stopped
mid-attribute behind a bare `...`, so it read as the whole scene and failed to
parse when saved and loaded. A capped reply now says
`Model XML preview (first N of M chars; pass output_path=... for the full MJCF)`,
ends on a complete tag followed by `<!-- truncated -->`, and carries a json
block `{"chars", "shown", "truncated"}`. A scene under the cap is returned whole,
as before.
